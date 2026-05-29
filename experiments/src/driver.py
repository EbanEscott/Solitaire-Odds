"""CLI for resumable local experiment orchestration."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .registry import (
    ExperimentRegistry,
    RUN_INTERRUPTED,
    RUN_RUNNING,
    RUN_SUCCEEDED,
    TASK_INTERRUPTED,
    TASK_RETRYABLE,
    TASK_RUNNING,
    TASK_SUCCEEDED,
    TaskRecord,
)
from .specs import (
    REPO_ROOT,
    ExperimentSpec,
    SpecValidationError,
    flatten_run_parameters,
    load_experiment_spec,
    resolve_working_directory,
    task_payload,
)


DEFAULT_STALE_AFTER_SECONDS = 300
DEFAULT_HEARTBEAT_SECONDS = 5
DEFAULT_SERVICE_STARTUP_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class RuntimeLayout:
    """Stable on-disk layout for the experiments control plane."""

    root: Path
    registry_dir: Path
    artifacts_dir: Path
    work_dir: Path
    database_path: Path

    @classmethod
    def create(cls) -> "RuntimeLayout":
        """Create the runtime directory structure expected by the driver."""

        root = REPO_ROOT / "experiments" / "runtime"
        registry_dir = root / "registry"
        artifacts_dir = root / "artifacts"
        work_dir = root / "work"
        # Creating the directories eagerly keeps later task execution paths simple and makes the
        # runtime layout deterministic regardless of which command is run first.
        for directory in (root, registry_dir, artifacts_dir, work_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            registry_dir=registry_dir,
            artifacts_dir=artifacts_dir,
            work_dir=work_dir,
            database_path=registry_dir / "experiments.sqlite3",
        )


@dataclass(frozen=True)
class PlannedTask:
    """Resolved task plan passed from spec parsing into registry creation."""

    task_name: str
    task_order: int
    task_kind: str
    payload: Mapping[str, Any]
    command: tuple[str, ...]
    working_directory: Path
    artifact_dir: Path


def _slugify(value: str) -> str:
    """Convert a task name into a stable filesystem-friendly slug."""

    safe = [char.lower() if char.isalnum() else "-" for char in value]
    collapsed = "".join(safe).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed or "task"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON payload with consistent formatting and parent directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write one JSON record per line for later ingestion by analysis tooling."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON file into a dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _relative_to_repo(path: Path) -> str:
    """Render a path relative to the repository root for readable manifests and CLI output."""

    return str(path.resolve().relative_to(REPO_ROOT))


def _plan_tasks(spec: ExperimentSpec, run_id: str, layout: RuntimeLayout) -> list[PlannedTask]:
    """Resolve workflow tasks into concrete task plans with artifact locations."""

    planned: list[PlannedTask] = []
    for index, workflow_task in enumerate(spec.workflow_tasks, start=1):
        # Each task receives its own stable artifact root so retries can accumulate under the same
        # logical task while still writing each attempt into a separate immutable subdirectory.
        task_root = layout.artifacts_dir / run_id / f"{index:02d}-{_slugify(workflow_task.name)}"
        planned.append(
            PlannedTask(
                task_name=workflow_task.name,
                task_order=index,
                task_kind=workflow_task.kind,
                payload=task_payload(
                    spec,
                    workflow_task.payload_key,
                    workflow_task.payload_overrides,
                ),
                command=workflow_task.command,
                working_directory=resolve_working_directory(workflow_task.working_directory),
                artifact_dir=task_root,
            )
        )
    return planned


def _task_row_payload(task: PlannedTask) -> dict[str, str]:
    """Serialize a planned task into the registry row format."""

    return {
        "task_name": task.task_name,
        "task_order": str(task.task_order),
        "task_kind": task.task_kind,
        "payload_json": json.dumps(task.payload, sort_keys=True),
        "command_json": json.dumps(list(task.command), sort_keys=True),
        "working_directory": str(task.working_directory),
        "artifact_dir": str(task.artifact_dir),
    }


def _build_manifest(
    *,
    spec: ExperimentSpec,
    run_id: str,
    task: TaskRecord,
    attempt_number: int,
    task_status: str,
    final_dir: Path,
) -> dict[str, Any]:
    """Build the manifest persisted beside each finalized task attempt."""

    files = [
        str(path.relative_to(final_dir))
        for path in sorted(final_dir.rglob("*"))
        if path.is_file()
    ]
    return {
        "manifest_version": "v1",
        "run_id": run_id,
        "task_name": task.task_name,
        "task_kind": task.task_kind,
        "attempt_number": attempt_number,
        "task_status": task_status,
        "artifact_dir": _relative_to_repo(final_dir),
        "spec_path": _relative_to_repo(spec.spec_path),
        "spec_hash": spec.spec_hash,
        "files": files,
    }


def _create_noop_payload(temp_dir: Path, task: TaskRecord) -> None:
    """Materialize the payload for a noop task so it still produces reviewable artifacts."""

    payload = json.loads(task.payload_json or "{}")
    _write_json(temp_dir / "task_payload.json", payload)


def _archive_attempt_artifact(
    *,
    registry: ExperimentRegistry,
    spec: ExperimentSpec,
    run_id: str,
    task: TaskRecord,
    attempt_id: int,
    attempt_number: int,
    task_status: str,
    temp_dir: Path,
    final_dir: Path,
) -> None:
    """Finalize a task attempt directory with a manifest and register it in the registry."""

    if not temp_dir.exists():
        return

    manifest = _build_manifest(
        spec=spec,
        run_id=run_id,
        task=task,
        attempt_number=attempt_number,
        task_status=task_status,
        final_dir=temp_dir,
    )
    _write_json(temp_dir / "manifest.json", manifest)
    temp_dir.rename(final_dir)
    registry.register_artifact(
        run_id=run_id,
        task_id=task.task_id,
        attempt_id=attempt_id,
        artifact_kind="task_attempt",
        relative_path=_relative_to_repo(final_dir),
        manifest_json=json.dumps(manifest, sort_keys=True),
    )


def _get_git_commit(repo_path: Path) -> str:
    """Return the current git commit for a repo path, or 'unknown' if unavailable."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return "unknown"

    if result.returncode == 0:
        return result.stdout.strip()
    return "unknown"


def _list_episode_logs(log_dir: Path) -> list[Path]:
    """List the top-level episode log files produced by the engine working directory."""

    if not log_dir.exists():
        return []
    return sorted(path for path in log_dir.glob("episode*.log") if path.is_file())


def _move_episode_logs(source_dir: Path, destination_dir: Path) -> list[Path]:
    """Move working episode logs into a controlled destination and return their new paths."""

    moved: list[Path] = []
    paths = _list_episode_logs(source_dir)
    if not paths:
        return moved

    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = destination_dir / path.name
        path.replace(target)
        moved.append(target)
    return moved


def _restore_episode_logs(backup_dir: Path, destination_dir: Path) -> None:
    """Restore any pre-existing working logs after one collection attempt finishes."""

    if not backup_dir.exists():
        return

    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(backup_dir.glob("episode*.log")):
        path.replace(destination_dir / path.name)


def _count_episode_summaries(log_paths: list[Path]) -> int:
    """Count EPISODE_SUMMARY records across a set of captured engine log files."""

    summaries = 0
    for path in log_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("EPISODE_SUMMARY "):
                    summaries += 1
    return summaries


def _build_endgame_collection_command(payload: Mapping[str, Any]) -> list[str]:
    """Construct the Gradle command used to generate one endgame collection shard."""

    command = [
        "./gradlew",
        "test",
        "--tests",
        str(payload["engine_test"]),
        "--rerun-tasks",
        "--console=plain",
        "-Dlog.episodes=true",
        f"-Dendgame.games.difficulty.level={payload['level']}",
        f"-Dendgame.games.per.level={payload['requested_games']}",
    ]
    if payload.get("randomise"):
        command.append("-Dendgame.randomise=true")
    shard_seed = payload.get("shard_seed")
    if shard_seed is not None:
        command.append(f"-Dendgame.random.seed={shard_seed}")
    return command


def _resolve_repo_relative_path(path_text: str | None) -> Path | None:
    """Resolve a potentially relative repository path used in experiment payloads."""

    if path_text is None:
        return None
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _resolve_training_resume_checkpoint(task: TaskRecord, payload: Mapping[str, Any]) -> Path | None:
    """Resolve the checkpoint a training task should resume from, if any."""

    explicit_resume_from = payload.get("resume_from")
    if explicit_resume_from:
        return _resolve_repo_relative_path(str(explicit_resume_from))

    checkpoint_prefix = str(payload.get("checkpoint_prefix", "policy_value"))
    attempt_root = Path(task.artifact_dir)
    latest_candidates = sorted(
        attempt_root.glob(f"attempt-*/checkpoints/{checkpoint_prefix}_latest.pt"),
        reverse=True,
    )
    if latest_candidates:
        return latest_candidates[0]

    epoch_candidates = sorted(
        attempt_root.glob(f"attempt-*/checkpoints/{checkpoint_prefix}_epoch_*.pt"),
        reverse=True,
    )
    if epoch_candidates:
        return epoch_candidates[0]
    return None


def _resolve_evaluation_checkpoint(
    *,
    layout: RuntimeLayout,
    run_id: str,
    payload: Mapping[str, Any],
) -> Path | None:
    """Resolve the checkpoint an evaluation shard should load through the model service."""

    explicit_checkpoint = payload.get("checkpoint")
    if explicit_checkpoint:
        return _resolve_repo_relative_path(str(explicit_checkpoint))

    checkpoint_prefix = str(payload.get("checkpoint_prefix", "policy_value"))
    run_root = layout.artifacts_dir / run_id
    latest_candidates = sorted(
        run_root.glob(f"*/attempt-*/checkpoints/{checkpoint_prefix}_latest.pt"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    if latest_candidates:
        return latest_candidates[0]

    epoch_candidates = sorted(
        run_root.glob(f"*/attempt-*/checkpoints/{checkpoint_prefix}_epoch_*.pt"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    if epoch_candidates:
        return epoch_candidates[0]
    return None


def _build_policy_value_train_command(
    payload: Mapping[str, Any],
    task: TaskRecord,
    temp_dir: Path,
) -> tuple[list[str], Path, Path, Path | None]:
    """Construct the Python training command and its output locations for one task attempt."""

    architecture_params = payload.get("architecture_params", {})
    if not isinstance(architecture_params, dict):
        raise ValueError("training payload is missing architecture_params")

    dataset_sources = payload.get("dataset_sources", [])
    if not isinstance(dataset_sources, list) or not dataset_sources:
        raise ValueError("training payload is missing dataset_sources")

    checkpoint_dir = temp_dir / "checkpoints"
    metrics_output = temp_dir / str(payload.get("metrics_filename", "epoch_metrics.jsonl"))
    resume_checkpoint = _resolve_training_resume_checkpoint(task, payload)

    command = [
        sys.executable,
        "-m",
        "src.train_policy_value",
        "--hidden-dim",
        str(architecture_params.get("hidden_dim", 256)),
        "--num-layers",
        str(architecture_params.get("num_layers", 2)),
        "--epochs",
        str(payload["epochs"]),
        "--batch-size",
        str(payload["batch_size"]),
        "--learning-rate",
        str(payload["learning_rate"]),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-prefix",
        str(payload.get("checkpoint_prefix", "policy_value")),
        "--save-every-epochs",
        str(payload.get("checkpoint_every_epochs", 1)),
        "--metrics-output",
        str(metrics_output),
    ]

    if architecture_params.get("batch_norm"):
        command.append("--batch-norm")
    if architecture_params.get("residual"):
        command.append("--residual")
    if resume_checkpoint is not None:
        command.extend(["--resume-from", str(resume_checkpoint)])

    simulate_interrupt_after_epoch = payload.get("simulate_interrupt_after_epoch")
    if simulate_interrupt_after_epoch is not None:
        command.extend([
            "--simulate-interrupt-after-epoch",
            str(simulate_interrupt_after_epoch),
        ])

    command.extend(str(_resolve_repo_relative_path(str(source))) for source in dataset_sources)
    return command, checkpoint_dir, metrics_output, resume_checkpoint


def _build_alpha_level_eval_command(
    payload: Mapping[str, Any],
    summary_output: Path,
) -> tuple[list[str], str]:
    """Construct the Gradle command used to run one AlphaSolitaire evaluation shard."""

    service_base_url = f"http://{payload['service_host']}:{payload['service_port']}"
    command = [
        "./gradlew",
        "test",
        "--tests",
        str(payload["engine_test"]),
        "--rerun-tasks",
        "--console=plain",
        f"-Dendgame.games.difficulty.level={payload['level']}",
        f"-Dendgame.games.per.level={payload['requested_games']}",
        f"-Dendgame.games.start.index={payload['game_start_index']}",
        f"-Dalphasolitaire.service.baseUrl={service_base_url}",
        f"-Dalphasolitaire.summary.json={summary_output}",
        f"-Dalphasolitaire.mcts.simulations={payload['mcts_simulations']}",
        f"-Dalphasolitaire.mcts.maxDepth={payload['mcts_max_depth']}",
        f"-Dalphasolitaire.mcts.cpuct={payload['mcts_cpuct']}",
    ]
    return command, service_base_url


def _wait_for_service_ready(service_process: subprocess.Popen[str], service_base_url: str) -> str | None:
    """Wait until the model service answers health probes or fails to start."""

    deadline = time.monotonic() + DEFAULT_SERVICE_STARTUP_TIMEOUT_SECONDS
    health_url = f"{service_base_url}/health"

    while time.monotonic() < deadline:
        exit_code = service_process.poll()
        if exit_code is not None:
            return f"model service exited before becoming ready (exit code {exit_code})"

        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    return None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass

        time.sleep(1)

    return f"model service did not become ready within {DEFAULT_SERVICE_STARTUP_TIMEOUT_SECONDS} seconds"


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    """Terminate a subprocess and wait briefly before forcing a kill."""

    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_endgame_collect_shard(
    *,
    registry: ExperimentRegistry,
    spec: ExperimentSpec,
    run_id: str,
    task: TaskRecord,
    attempt_id: int,
    temp_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    heartbeat_seconds: int,
) -> tuple[int, str | None]:
    """Run one endgame collection shard and archive its generated episode logs."""

    payload = json.loads(task.payload_json or "{}")
    engine_dir = Path(task.working_directory or REPO_ROOT)
    engine_logs_dir = engine_dir / "logs"
    backup_dir = temp_dir / "preexisting-engine-logs"
    collected_logs_dir = temp_dir / "collected-logs"

    command = _build_endgame_collection_command(payload)
    _write_json(temp_dir / "command.json", {"command": command})

    registry.connection.execute(
        """
        UPDATE task_attempts
        SET command_json = ?, working_directory = ?
        WHERE attempt_id = ?
        """,
        (json.dumps(command), str(engine_dir), attempt_id),
    )
    registry.connection.commit()

    # The engine currently writes episode logs to a shared working directory. Move any existing
    # files out of the way first so the shard can capture only the logs produced by this attempt.
    _move_episode_logs(engine_logs_dir, backup_dir)

    generated_logs: list[Path] = []
    try:
        exit_code = _run_command_task(
            registry=registry,
            run_id=run_id,
            task=task,
            attempt_id=attempt_id,
            command=command,
            working_directory=engine_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            heartbeat_seconds=heartbeat_seconds,
        )
    finally:
        generated_logs = _move_episode_logs(engine_logs_dir, collected_logs_dir)
        _restore_episode_logs(backup_dir, engine_logs_dir)

    actual_games = _count_episode_summaries(generated_logs)
    summary = {
        "collection_kind": payload.get("kind"),
        "level": payload.get("level"),
        "requested_games": payload.get("requested_games"),
        "actual_games": actual_games,
        "shard_index": payload.get("shard_index"),
        "shard_count": payload.get("shard_count"),
        "randomise": payload.get("randomise", False),
        "shard_seed": payload.get("shard_seed"),
        "engine_test": payload.get("engine_test"),
        "spec_hash": spec.spec_hash,
        "repo_git_commit": _get_git_commit(REPO_ROOT),
        "engine_git_commit": _get_git_commit(engine_dir),
        "log_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
            }
            for path in generated_logs
        ],
    }
    _write_json(temp_dir / "collection_summary.json", summary)

    if exit_code != 0:
        return exit_code, f"endgame collection command exited with status {exit_code}"
    if not generated_logs:
        return 1, "endgame collection produced no episode logs"
    if actual_games <= 0:
        return 1, "endgame collection produced logs but no EPISODE_SUMMARY records"
    return 0, None


def _run_policy_value_train(
    *,
    registry: ExperimentRegistry,
    spec: ExperimentSpec,
    run_id: str,
    task: TaskRecord,
    attempt_id: int,
    temp_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    heartbeat_seconds: int,
) -> tuple[int, str | None]:
    """Run one resumable policy-value training attempt and summarize its checkpoint outputs."""

    payload = json.loads(task.payload_json or "{}")
    working_directory = Path(task.working_directory or REPO_ROOT)
    command, checkpoint_dir, metrics_output, resume_checkpoint = _build_policy_value_train_command(
        payload,
        task,
        temp_dir,
    )
    _write_json(temp_dir / "command.json", {"command": command})

    registry.connection.execute(
        """
        UPDATE task_attempts
        SET command_json = ?, working_directory = ?
        WHERE attempt_id = ?
        """,
        (json.dumps(command), str(working_directory), attempt_id),
    )
    registry.connection.commit()

    exit_code = _run_command_task(
        registry=registry,
        run_id=run_id,
        task=task,
        attempt_id=attempt_id,
        command=command,
        working_directory=working_directory,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        heartbeat_seconds=heartbeat_seconds,
    )

    checkpoint_files = sorted(path for path in checkpoint_dir.glob("*.pt") if path.is_file())
    latest_checkpoint = checkpoint_dir / f"{payload.get('checkpoint_prefix', 'policy_value')}_latest.pt"
    training_summary = {
        "training_kind": payload.get("kind"),
        "architecture_family": payload.get("architecture_family"),
        "architecture_params": payload.get("architecture_params", {}),
        "dataset_kind": payload.get("dataset_kind"),
        "dataset_sources": payload.get("dataset_sources", []),
        "checkpoint_prefix": payload.get("checkpoint_prefix", "policy_value"),
        "checkpoint_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
            }
            for path in checkpoint_files
        ],
        "latest_checkpoint": str(latest_checkpoint.relative_to(temp_dir)) if latest_checkpoint.exists() else None,
        "metrics_output": str(metrics_output.relative_to(temp_dir)) if metrics_output.exists() else None,
        "resume_from": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "spec_hash": spec.spec_hash,
        "repo_git_commit": _get_git_commit(REPO_ROOT),
        "neural_git_commit": _get_git_commit(working_directory),
        "simulated_interrupt_after_epoch": payload.get("simulate_interrupt_after_epoch"),
    }
    _write_json(temp_dir / "training_summary.json", training_summary)

    if exit_code == 99:
        return exit_code, "training interrupted after saving a resumable checkpoint"
    if exit_code != 0:
        return exit_code, f"policy-value training exited with status {exit_code}"
    if not latest_checkpoint.exists():
        return 1, "training completed but no latest checkpoint was produced"
    if not metrics_output.exists():
        return 1, "training completed but no epoch metrics file was produced"
    return 0, None


def _run_alpha_level_eval_shard(
    *,
    registry: ExperimentRegistry,
    spec: ExperimentSpec,
    run_id: str,
    task: TaskRecord,
    attempt_id: int,
    temp_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    heartbeat_seconds: int,
    layout: RuntimeLayout,
) -> tuple[int, str | None]:
    """Run one AlphaSolitaire evaluation shard against a driver-managed model service."""

    payload = json.loads(task.payload_json or "{}")
    engine_dir = Path(task.working_directory or REPO_ROOT)
    neural_dir = REPO_ROOT / "neural-network"
    checkpoint_path = _resolve_evaluation_checkpoint(layout=layout, run_id=run_id, payload=payload)
    if checkpoint_path is None or not checkpoint_path.exists():
        return 1, "evaluation could not resolve a checkpoint to serve"

    raw_summary_output = temp_dir / "engine_evaluation_summary.json"
    evaluation_command, service_base_url = _build_alpha_level_eval_command(payload, raw_summary_output)
    service_command = [
        sys.executable,
        "-m",
        "src.service",
        "--checkpoint",
        str(checkpoint_path),
        "--host",
        str(payload["service_host"]),
        "--port",
        str(payload["service_port"]),
    ]
    _write_json(
        temp_dir / "command.json",
        {
            "command": evaluation_command,
            "service_command": service_command,
        },
    )

    registry.connection.execute(
        """
        UPDATE task_attempts
        SET command_json = ?, working_directory = ?
        WHERE attempt_id = ?
        """,
        (json.dumps(evaluation_command), str(engine_dir), attempt_id),
    )
    registry.connection.commit()

    service_stdout_path = temp_dir / "service_stdout.log"
    service_stderr_path = temp_dir / "service_stderr.log"
    service_process: subprocess.Popen[str] | None = None

    try:
        with service_stdout_path.open("w", encoding="utf-8") as service_stdout_handle, service_stderr_path.open(
            "w", encoding="utf-8"
        ) as service_stderr_handle:
            service_process = subprocess.Popen(
                service_command,
                cwd=str(neural_dir),
                stdout=service_stdout_handle,
                stderr=service_stderr_handle,
                text=True,
            )

            service_error = _wait_for_service_ready(service_process, service_base_url)
            if service_error is not None:
                return 1, service_error

            exit_code = _run_command_task(
                registry=registry,
                run_id=run_id,
                task=task,
                attempt_id=attempt_id,
                command=evaluation_command,
                working_directory=engine_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                heartbeat_seconds=heartbeat_seconds,
            )
    finally:
        _stop_process(service_process)

    enriched_summary = {
        **(_read_json(raw_summary_output) if raw_summary_output.exists() else {}),
        "experiment_id": spec.experiment_id,
        "evaluation_kind": payload.get("kind"),
        "architecture_family": payload.get("architecture_family"),
        "architecture_params": json.dumps(payload.get("architecture_params", {}), sort_keys=True),
        "training_kind": payload.get("training_kind"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_source": "explicit" if payload.get("checkpoint") else "training_artifact",
        "service_base_url": service_base_url,
        "engine_test": payload.get("engine_test"),
        "mcts_simulations": payload.get("mcts_simulations"),
        "mcts_max_depth": payload.get("mcts_max_depth"),
        "mcts_cpuct": payload.get("mcts_cpuct"),
        "requested_games": payload.get("requested_games"),
        "game_start_index": payload.get("game_start_index"),
        "game_end_index_exclusive": payload.get("game_end_index_exclusive"),
        "shard_index": payload.get("shard_index"),
        "shard_count": payload.get("shard_count"),
        "run_id": run_id,
        "task_name": task.task_name,
        "task_kind": task.task_kind,
        "spec_hash": spec.spec_hash,
        "repo_git_commit": _get_git_commit(REPO_ROOT),
        "engine_git_commit": _get_git_commit(engine_dir),
        "neural_git_commit": _get_git_commit(neural_dir),
    }
    _write_json(temp_dir / "evaluation_summary.json", enriched_summary)

    if exit_code != 0:
        return exit_code, f"AlphaSolitaire evaluation exited with status {exit_code}"
    if not raw_summary_output.exists():
        return 1, "evaluation completed but produced no structured summary"
    if int(enriched_summary.get("games_tested", 0)) <= 0:
        return 1, "evaluation completed but reported zero tested games"
    return 0, None


def _format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple GitHub-flavored markdown table."""

    header_row = "| " + " | ".join(headers) + " |"
    divider_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, divider_row, *body_rows])


def _load_latest_task_json_artifact(task: TaskRecord, filename: str) -> dict[str, Any]:
    """Load one JSON artifact from the most recent archived attempt for a task."""

    attempt_dirs = sorted(Path(task.artifact_dir).glob("attempt-*"), reverse=True)
    if not attempt_dirs:
        raise ValueError(f"could not find an archived attempt for task '{task.task_name}'")

    artifact_path = attempt_dirs[0] / filename
    if not artifact_path.exists():
        raise ValueError(f"could not find {filename} for task '{task.task_name}'")
    return _read_json(artifact_path)


def _enrich_evaluation_summary_with_registry(
    *,
    registry: ExperimentRegistry,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Backfill stable run-level fields for summaries created before the current schema."""

    run_id = summary.get("run_id")
    if isinstance(run_id, str) and run_id:
        summary.setdefault("experiment_id", registry.get_run(run_id).experiment_id)
    return summary


def _build_evaluation_rollup_sql(view_name: str, source_relation: str) -> str:
    """Build the DuckDB view definition used for evaluation rollups."""

    return f"""
CREATE OR REPLACE VIEW {view_name} AS
WITH grouped AS (
    SELECT
        run_id,
        experiment_id,
        level,
        architecture_family,
        architecture_params,
        training_kind,
        checkpoint_path,
        checkpoint_source,
        COUNT(*) AS shard_count,
        SUM(games_tested) AS games_tested,
        SUM(games_won) AS games_won,
        SUM(games_lost) AS games_lost,
        SUM(avg_moves * games_tested) / NULLIF(SUM(games_tested), 0) AS avg_moves,
        SUM(avg_time_seconds * games_tested) / NULLIF(SUM(games_tested), 0) AS avg_time_seconds,
        SUM(total_time_seconds) AS total_time_seconds,
        SUM(games_won) * 100.0 / NULLIF(SUM(games_tested), 0) AS win_percent
    FROM {source_relation}
    GROUP BY
        run_id,
        experiment_id,
        level,
        architecture_family,
        architecture_params,
        training_kind,
        checkpoint_path,
        checkpoint_source
),
wilson AS (
    SELECT
        *,
        games_won * 1.0 / NULLIF(games_tested, 0) AS win_ratio,
        1.96 AS z_score
    FROM grouped
)
SELECT
    *,
    CASE
        WHEN games_tested = 0 THEN 0.0
        ELSE 100.0 * (
            (
                win_ratio + (z_score * z_score) / (2.0 * games_tested)
            ) / (1.0 + (z_score * z_score) / games_tested)
            - z_score * sqrt(
                (
                    (win_ratio * (1.0 - win_ratio) + (z_score * z_score) / (4.0 * games_tested))
                    / games_tested
                )
            ) / (1.0 + (z_score * z_score) / games_tested)
        )
    END AS win_rate_ci_low,
    CASE
        WHEN games_tested = 0 THEN 0.0
        ELSE 100.0 * (
            (
                win_ratio + (z_score * z_score) / (2.0 * games_tested)
            ) / (1.0 + (z_score * z_score) / games_tested)
            + z_score * sqrt(
                (
                    (win_ratio * (1.0 - win_ratio) + (z_score * z_score) / (4.0 * games_tested))
                    / games_tested
                )
            ) / (1.0 + (z_score * z_score) / games_tested)
        )
    END AS win_rate_ci_high
FROM wilson
""".strip()


def _build_evaluation_report_markdown(
    *,
    run_id: str,
    spec: ExperimentSpec,
    rollup_rows: list[dict[str, Any]],
    shard_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> str:
    """Build a markdown summary for one evaluated run."""

    lines = [
        f"# Evaluation Report: {run_id}",
        "",
        f"Experiment: {spec.experiment_id}",
        "",
    ]

    if rollup_rows:
        rollup_table = _format_markdown_table(
            [
                "Level",
                "Games",
                "Wins",
                "Win %",
                "95% CI",
                "Avg Moves",
                "Avg Time/Game",
                "Total Time",
                "Checkpoint",
            ],
            [
                [
                    str(row["level"]),
                    str(row["games_tested"]),
                    str(row["games_won"]),
                    f"{float(row['win_percent']):.2f}",
                    f"{float(row['win_rate_ci_low']):.2f} - {float(row['win_rate_ci_high']):.2f}",
                    f"{float(row['avg_moves']):.2f}",
                    f"{float(row['avg_time_seconds']):.3f}s",
                    f"{float(row['total_time_seconds']):.3f}s",
                    str(row["checkpoint_path"]),
                ]
                for row in rollup_rows
            ],
        )
        lines.extend(["## Rollup", "", rollup_table, ""])

    if shard_rows:
        shard_table = _format_markdown_table(
            ["Shard", "Game Range", "Games", "Wins", "Win %", "Avg Moves", "Avg Time"],
            [
                [
                    str(int(row["shard_index"]) + 1),
                    f"{row['game_start_index']} - {int(row['game_end_index_exclusive']) - 1}",
                    str(row["games_tested"]),
                    str(row["games_won"]),
                    f"{float(row['win_percent']):.2f}",
                    f"{float(row['avg_moves']):.2f}",
                    f"{float(row['avg_time_seconds']):.3f}s",
                ]
                for row in shard_rows
            ],
        )
        lines.extend(["## Shards", "", shard_table, ""])

    if comparison_rows:
        comparison_table = _format_markdown_table(
            ["Run", "Experiment", "Level", "Win %", "Delta vs Current", "Avg Moves", "Avg Time/Game", "Checkpoint"],
            [
                [
                    str(row["compared_run_id"]),
                    str(row["experiment_id"]),
                    str(row["level"]),
                    f"{float(row['win_percent']):.2f}",
                    f"{float(row['win_percent_delta']):+.2f}",
                    f"{float(row['avg_moves']):.2f}",
                    f"{float(row['avg_time_seconds']):.3f}s",
                    str(row["checkpoint_path"]),
                ]
                for row in comparison_rows
            ],
        )
        lines.extend(["## Cross-Run Comparison", "", comparison_table, ""])

    return "\n".join(lines).strip() + "\n"


def _run_evaluation_report(
    *,
    registry: ExperimentRegistry,
    spec: ExperimentSpec,
    run_id: str,
    task: TaskRecord,
    temp_dir: Path,
) -> tuple[int, str | None]:
    """Aggregate evaluation shard summaries into Parquet, DuckDB views, and markdown."""

    try:
        import duckdb  # type: ignore[import-not-found]
    except ImportError:
        return 1, "evaluation reporting requires the 'duckdb' Python package"

    payload = json.loads(task.payload_json or "{}")
    current_run_shard_summaries: list[dict[str, Any]] = []
    for run_task in registry.list_tasks(run_id):
        if run_task.task_kind != "alpha_level_eval_shard" or run_task.status != TASK_SUCCEEDED:
            continue

        try:
            current_run_shard_summaries.append(
                _enrich_evaluation_summary_with_registry(
                    registry=registry,
                    summary=_load_latest_task_json_artifact(run_task, "evaluation_summary.json"),
                )
            )
        except ValueError as exc:
            return 1, f"evaluation report {exc}"

    if not current_run_shard_summaries:
        return 1, "evaluation report found no successful evaluation shard summaries"

    historical_shard_summaries: list[dict[str, Any]] = []
    for historical_task in registry.list_tasks_by_kind(
        task_kind="alpha_level_eval_shard",
        task_status=TASK_SUCCEEDED,
        run_status=RUN_SUCCEEDED,
        exclude_run_id=run_id,
    ):
        try:
            historical_shard_summaries.append(
                _enrich_evaluation_summary_with_registry(
                    registry=registry,
                    summary=_load_latest_task_json_artifact(historical_task, "evaluation_summary.json"),
                )
            )
        except ValueError:
            # Historical comparison data is best-effort. If an older run is missing its structured
            # summary artifact, keep the current run report usable instead of failing the new run.
            continue

    all_run_shard_summaries = [*historical_shard_summaries, *current_run_shard_summaries]

    jsonl_path = temp_dir / "evaluation_shards.jsonl"
    all_runs_jsonl_path = temp_dir / "evaluation_shards_all_runs.jsonl"
    duckdb_path = temp_dir / str(payload.get("duckdb_filename", "evaluation.duckdb"))
    shards_parquet_path = temp_dir / str(payload.get("shards_parquet_filename", "evaluation_shards.parquet"))
    rollups_parquet_path = temp_dir / str(payload.get("rollups_parquet_filename", "evaluation_rollups.parquet"))
    all_runs_rollups_parquet_path = temp_dir / "evaluation_rollups_all_runs.parquet"
    comparison_parquet_path = temp_dir / "evaluation_run_comparison.parquet"
    queries_path = temp_dir / str(payload.get("queries_filename", "evaluation_queries.sql"))
    report_markdown_path = temp_dir / str(payload.get("report_markdown_filename", "evaluation_report.md"))

    _write_jsonl(jsonl_path, current_run_shard_summaries)
    _write_jsonl(all_runs_jsonl_path, all_run_shard_summaries)

    quoted_current_jsonl = str(jsonl_path).replace("'", "''")
    quoted_all_runs_jsonl = str(all_runs_jsonl_path).replace("'", "''")
    quoted_shards_parquet = str(shards_parquet_path).replace("'", "''")
    quoted_rollups_parquet = str(rollups_parquet_path).replace("'", "''")
    quoted_all_runs_rollups_parquet = str(all_runs_rollups_parquet_path).replace("'", "''")
    quoted_comparison_parquet = str(comparison_parquet_path).replace("'", "''")
    relative_current_jsonl = jsonl_path.name.replace("'", "''")
    relative_all_runs_jsonl = all_runs_jsonl_path.name.replace("'", "''")
    relative_shards_parquet = shards_parquet_path.name.replace("'", "''")
    relative_rollups_parquet = rollups_parquet_path.name.replace("'", "''")
    relative_all_runs_rollups_parquet = all_runs_rollups_parquet_path.name.replace("'", "''")
    relative_comparison_parquet = comparison_parquet_path.name.replace("'", "''")

    rollups_current_run_sql = _build_evaluation_rollup_sql(
        "evaluation_rollups_current_run",
        "evaluation_shards_current_run",
    )
    rollups_all_runs_sql = _build_evaluation_rollup_sql(
        "evaluation_rollups_all_runs",
        "evaluation_shards_all_runs",
    )
    comparison_sql = f"""
CREATE OR REPLACE VIEW evaluation_run_comparison AS
WITH current_baseline AS (
    SELECT
        level,
        AVG(win_percent) AS baseline_win_percent,
        AVG(avg_moves) AS baseline_avg_moves,
        AVG(avg_time_seconds) AS baseline_avg_time_seconds
    FROM evaluation_rollups_current_run
    GROUP BY level
)
SELECT
    current_baseline.level,
    evaluation_rollups_all_runs.run_id AS compared_run_id,
    evaluation_rollups_all_runs.experiment_id,
    evaluation_rollups_all_runs.architecture_family,
    evaluation_rollups_all_runs.architecture_params,
    evaluation_rollups_all_runs.training_kind,
    evaluation_rollups_all_runs.checkpoint_path,
    evaluation_rollups_all_runs.checkpoint_source,
    evaluation_rollups_all_runs.games_tested,
    evaluation_rollups_all_runs.games_won,
    evaluation_rollups_all_runs.games_lost,
    evaluation_rollups_all_runs.shard_count,
    evaluation_rollups_all_runs.win_percent,
    evaluation_rollups_all_runs.win_rate_ci_low,
    evaluation_rollups_all_runs.win_rate_ci_high,
    evaluation_rollups_all_runs.avg_moves,
    evaluation_rollups_all_runs.avg_time_seconds,
    evaluation_rollups_all_runs.total_time_seconds,
    current_baseline.baseline_win_percent,
    evaluation_rollups_all_runs.win_percent - current_baseline.baseline_win_percent AS win_percent_delta,
    evaluation_rollups_all_runs.avg_moves - current_baseline.baseline_avg_moves AS avg_moves_delta,
    evaluation_rollups_all_runs.avg_time_seconds - current_baseline.baseline_avg_time_seconds AS avg_time_seconds_delta
FROM evaluation_rollups_all_runs
JOIN current_baseline ON current_baseline.level = evaluation_rollups_all_runs.level
ORDER BY current_baseline.level, evaluation_rollups_all_runs.win_percent DESC, evaluation_rollups_all_runs.run_id ASC
""".strip()

    queries_sql = f"""
CREATE OR REPLACE TABLE evaluation_shards_current_run AS
SELECT *
FROM read_json_auto('{relative_current_jsonl}', records = true);

CREATE OR REPLACE VIEW evaluation_shards AS
SELECT *
FROM evaluation_shards_current_run;

CREATE OR REPLACE TABLE evaluation_shards_all_runs AS
SELECT *
FROM read_json_auto('{relative_all_runs_jsonl}', records = true);

{rollups_current_run_sql};

CREATE OR REPLACE VIEW evaluation_rollups AS
SELECT *
FROM evaluation_rollups_current_run;

{rollups_all_runs_sql};

{comparison_sql};

COPY (SELECT * FROM evaluation_shards_current_run ORDER BY shard_index) TO '{relative_shards_parquet}' (FORMAT PARQUET);
COPY (SELECT * FROM evaluation_rollups_current_run ORDER BY level, checkpoint_path) TO '{relative_rollups_parquet}' (FORMAT PARQUET);
COPY (SELECT * FROM evaluation_rollups_all_runs ORDER BY level, win_percent DESC, run_id) TO '{relative_all_runs_rollups_parquet}' (FORMAT PARQUET);
COPY (SELECT * FROM evaluation_run_comparison ORDER BY level, win_percent DESC, compared_run_id) TO '{relative_comparison_parquet}' (FORMAT PARQUET);

SELECT * FROM evaluation_run_comparison ORDER BY level, win_percent DESC, compared_run_id;
""".strip()

    connection = duckdb.connect(str(duckdb_path))
    try:
        connection.execute(
            "CREATE OR REPLACE TABLE evaluation_shards_current_run AS SELECT * FROM read_json_auto(?, records = true)",
            [str(jsonl_path)],
        )
        connection.execute(
            "CREATE OR REPLACE VIEW evaluation_shards AS SELECT * FROM evaluation_shards_current_run"
        )
        connection.execute(
            "CREATE OR REPLACE TABLE evaluation_shards_all_runs AS SELECT * FROM read_json_auto(?, records = true)",
            [str(all_runs_jsonl_path)],
        )
        connection.execute(rollups_current_run_sql)
        connection.execute("CREATE OR REPLACE VIEW evaluation_rollups AS SELECT * FROM evaluation_rollups_current_run")
        connection.execute(rollups_all_runs_sql)
        connection.execute(comparison_sql)

        shard_cursor = connection.execute(
            "SELECT shard_index, game_start_index, game_end_index_exclusive, games_tested, games_won, win_percent, avg_moves, avg_time_seconds FROM evaluation_shards_current_run ORDER BY shard_index"
        )
        shard_columns = [column[0] for column in shard_cursor.description]
        shard_rows = [dict(zip(shard_columns, row)) for row in shard_cursor.fetchall()]

        rollup_cursor = connection.execute(
            "SELECT level, games_tested, games_won, win_percent, win_rate_ci_low, win_rate_ci_high, avg_moves, avg_time_seconds, total_time_seconds, checkpoint_path FROM evaluation_rollups_current_run ORDER BY level, checkpoint_path"
        )
        rollup_columns = [column[0] for column in rollup_cursor.description]
        rollup_rows = [dict(zip(rollup_columns, row)) for row in rollup_cursor.fetchall()]

        comparison_cursor = connection.execute(
            "SELECT compared_run_id, experiment_id, level, win_percent, win_percent_delta, avg_moves, avg_time_seconds, checkpoint_path FROM evaluation_run_comparison ORDER BY level, win_percent DESC, compared_run_id"
        )
        comparison_columns = [column[0] for column in comparison_cursor.description]
        comparison_rows = [dict(zip(comparison_columns, row)) for row in comparison_cursor.fetchall()]

        shards_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        rollups_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        connection.execute(
            f"COPY (SELECT * FROM evaluation_shards_current_run ORDER BY shard_index) TO '{quoted_shards_parquet}' (FORMAT PARQUET)"
        )
        connection.execute(
            f"COPY (SELECT * FROM evaluation_rollups_current_run ORDER BY level, checkpoint_path) TO '{quoted_rollups_parquet}' (FORMAT PARQUET)"
        )
        connection.execute(
            f"COPY (SELECT * FROM evaluation_rollups_all_runs ORDER BY level, win_percent DESC, run_id) TO '{quoted_all_runs_rollups_parquet}' (FORMAT PARQUET)"
        )
        connection.execute(
            f"COPY (SELECT * FROM evaluation_run_comparison ORDER BY level, win_percent DESC, compared_run_id) TO '{quoted_comparison_parquet}' (FORMAT PARQUET)"
        )
    finally:
        connection.close()

    queries_path.write_text(queries_sql + "\n", encoding="utf-8")
    report_markdown = _build_evaluation_report_markdown(
        run_id=run_id,
        spec=spec,
        rollup_rows=rollup_rows,
        shard_rows=shard_rows,
        comparison_rows=comparison_rows,
    )
    report_markdown_path.write_text(report_markdown, encoding="utf-8")

    return 0, None


def _run_command_task(
    *,
    registry: ExperimentRegistry,
    run_id: str,
    task: TaskRecord,
    attempt_id: int,
    command: list[str],
    working_directory: Path,
    stdout_path: Path,
    stderr_path: Path,
    heartbeat_seconds: int,
) -> int:
    """Run one command task while continuously refreshing registry heartbeats."""

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(working_directory),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )

        while True:
            exit_code = process.poll()
            if exit_code is not None:
                # Emit a final heartbeat on completion so stale-recovery logic does not briefly
                # see a finished command as abandoned if the process exits between poll cycles.
                registry.heartbeat(run_id, task.task_id, attempt_id)
                return int(exit_code)
            registry.heartbeat(run_id, task.task_id, attempt_id)
            time.sleep(max(1, heartbeat_seconds))


def _execute_task(
    *,
    spec: ExperimentSpec,
    run_id: str,
    task: TaskRecord,
    layout: RuntimeLayout,
    registry: ExperimentRegistry,
    heartbeat_seconds: int,
) -> None:
    """Execute one task attempt and finalize its artifact directory atomically."""

    command = json.loads(task.command_json or "[]")
    attempt_base = Path(task.artifact_dir)
    next_attempt = registry.start_task_attempt(
        run_id=run_id,
        task=task,
        artifact_dir=str(attempt_base / "pending"),
        stdout_path=None,
        stderr_path=None,
    )

    final_dir = attempt_base / f"attempt-{next_attempt.attempt_number:04d}"
    temp_dir = attempt_base / f".attempt-{next_attempt.attempt_number:04d}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=False)

    # Each attempt writes into a temporary directory first. Only once the manifest is complete do
    # we rename it into place, which keeps partially written artifacts out of the success path.
    stdout_path = temp_dir / "stdout.log"
    stderr_path = temp_dir / "stderr.log"

    registry.connection.execute(
        """
        UPDATE task_attempts
        SET artifact_dir = ?, stdout_path = ?, stderr_path = ?
        WHERE attempt_id = ?
        """,
        (str(final_dir), str(stdout_path), str(stderr_path), next_attempt.attempt_id),
    )
    registry.connection.commit()

    try:
        exit_code = 0
        error_message: str | None = None
        if task.task_kind == "noop":
            _create_noop_payload(temp_dir, task)
        elif task.task_kind == "command":
            _write_json(temp_dir / "command.json", {"command": command})
            exit_code = _run_command_task(
                registry=registry,
                run_id=run_id,
                task=task,
                attempt_id=next_attempt.attempt_id,
                command=command,
                working_directory=Path(task.working_directory or REPO_ROOT),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                heartbeat_seconds=heartbeat_seconds,
            )
        elif task.task_kind == "endgame_collect_shard":
            exit_code, error_message = _run_endgame_collect_shard(
                registry=registry,
                spec=spec,
                run_id=run_id,
                task=task,
                attempt_id=next_attempt.attempt_id,
                temp_dir=temp_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                heartbeat_seconds=heartbeat_seconds,
            )
        elif task.task_kind == "policy_value_train":
            exit_code, error_message = _run_policy_value_train(
                registry=registry,
                spec=spec,
                run_id=run_id,
                task=task,
                attempt_id=next_attempt.attempt_id,
                temp_dir=temp_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                heartbeat_seconds=heartbeat_seconds,
            )
        elif task.task_kind == "alpha_level_eval_shard":
            exit_code, error_message = _run_alpha_level_eval_shard(
                registry=registry,
                spec=spec,
                run_id=run_id,
                task=task,
                attempt_id=next_attempt.attempt_id,
                temp_dir=temp_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                heartbeat_seconds=heartbeat_seconds,
                layout=layout,
            )
        elif task.task_kind == "evaluation_report":
            exit_code, error_message = _run_evaluation_report(
                registry=registry,
                spec=spec,
                run_id=run_id,
                task=task,
                temp_dir=temp_dir,
            )
        else:
            raise ValueError(f"unsupported task kind: {task.task_kind}")

        if exit_code != 0:
            # Failed command attempts are kept as retryable artifacts rather than deleted. That
            # makes debugging easier and preserves the evidence needed to understand the failure.
            _archive_attempt_artifact(
                registry=registry,
                spec=spec,
                run_id=run_id,
                attempt_id=next_attempt.attempt_id,
                task=task,
                attempt_number=next_attempt.attempt_number,
                task_status=TASK_RETRYABLE,
                temp_dir=temp_dir,
                final_dir=final_dir,
            )
            registry.finish_task_attempt(
                run_id=run_id,
                task_id=task.task_id,
                attempt_id=next_attempt.attempt_id,
                task_status=TASK_RETRYABLE,
                run_status=RUN_INTERRUPTED,
                exit_code=exit_code,
                error_message=error_message or f"command exited with status {exit_code}",
                status_message=f"Task '{task.task_name}' failed and can be retried",
            )
            raise RuntimeError(f"task '{task.task_name}' failed with exit code {exit_code}")

        _archive_attempt_artifact(
            registry=registry,
            spec=spec,
            run_id=run_id,
            attempt_id=next_attempt.attempt_id,
            task=task,
            attempt_number=next_attempt.attempt_number,
            task_status=TASK_SUCCEEDED,
            temp_dir=temp_dir,
            final_dir=final_dir,
        )
        registry.finish_task_attempt(
            run_id=run_id,
            task_id=task.task_id,
            attempt_id=next_attempt.attempt_id,
            task_status=TASK_SUCCEEDED,
            run_status=RUN_RUNNING,
            exit_code=0,
            error_message=None,
            status_message=None,
        )
    except KeyboardInterrupt:
        # Interrupted attempts are still archived with a manifest so resume logic and post-mortem
        # inspection have a durable record of what had already been written to disk.
        _archive_attempt_artifact(
            registry=registry,
            spec=spec,
            run_id=run_id,
            task=task,
            attempt_id=next_attempt.attempt_id,
            attempt_number=next_attempt.attempt_number,
            task_status=TASK_INTERRUPTED,
            temp_dir=temp_dir,
            final_dir=final_dir,
        )
        registry.finish_task_attempt(
            run_id=run_id,
            task_id=task.task_id,
            attempt_id=next_attempt.attempt_id,
            task_status=TASK_INTERRUPTED,
            run_status=RUN_INTERRUPTED,
            exit_code=None,
            error_message="Interrupted by user",
            status_message=f"Task '{task.task_name}' interrupted",
        )
        raise
    except RuntimeError:
        # Expected task-level failures are already finalized above. Re-raising here avoids trying
        # to archive the same attempt twice through the generic unexpected-exception path.
        raise
    except Exception as exc:
        # Any unexpected exception during task setup or execution must still close out the
        # attempt, otherwise the run remains stuck in a perpetual running state.
        _write_json(
            temp_dir / "unexpected_error.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        _archive_attempt_artifact(
            registry=registry,
            spec=spec,
            run_id=run_id,
            task=task,
            attempt_id=next_attempt.attempt_id,
            attempt_number=next_attempt.attempt_number,
            task_status=TASK_RETRYABLE,
            temp_dir=temp_dir,
            final_dir=final_dir,
        )
        registry.finish_task_attempt(
            run_id=run_id,
            task_id=task.task_id,
            attempt_id=next_attempt.attempt_id,
            task_status=TASK_RETRYABLE,
            run_status=RUN_INTERRUPTED,
            exit_code=None,
            error_message=f"{type(exc).__name__}: {exc}",
            status_message=f"Task '{task.task_name}' failed and can be retried",
        )
        raise RuntimeError(f"task '{task.task_name}' failed with unexpected error: {exc}") from exc


def _cmd_plan(args: argparse.Namespace) -> int:
    """Validate a spec and print the concrete task plan without mutating runtime state."""

    spec = load_experiment_spec(args.spec)
    layout = RuntimeLayout.create()
    run_id = args.run_id or spec.experiment_id
    tasks = _plan_tasks(spec, run_id, layout)

    print(f"Run ID: {run_id}")
    print(f"Spec: {_relative_to_repo(spec.spec_path)}")
    for task in tasks:
        print(
            f"- {task.task_order:02d} {task.task_name} [{task.task_kind}] "
            f"artifact_root={_relative_to_repo(task.artifact_dir)}"
        )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Create or resume a run, executing incomplete tasks in order."""

    spec = load_experiment_spec(args.spec)
    layout = RuntimeLayout.create()
    run_id = args.run_id or spec.experiment_id
    tasks = _plan_tasks(spec, run_id, layout)

    with ExperimentRegistry(layout.database_path) as registry:
        registry.register_spec(
            spec_hash=spec.spec_hash,
            experiment_id=spec.experiment_id,
            api_version=spec.api_version,
            spec_path=str(spec.spec_path),
            spec_json=spec.to_json(),
        )

        run = registry.create_or_resume_run(
            run_id=run_id,
            experiment_id=spec.experiment_id,
            spec_hash=spec.spec_hash,
            spec_path=str(spec.spec_path),
            artifact_root=str(layout.artifacts_dir / run_id),
        )
        registry.replace_run_parameters(run_id, flatten_run_parameters(spec.raw))

        recovered = registry.recover_stale_running_tasks(run_id, args.stale_after_seconds)
        if recovered:
            print(f"Recovered {recovered} stale task(s) before resuming run {run_id}.")

        registry.ensure_tasks(run_id, (_task_row_payload(task) for task in tasks))

        executed = 0
        task_rows = registry.list_tasks(run_id)
        for task in task_rows:
            if task.status == TASK_SUCCEEDED:
                # Completed tasks are never re-executed for the same run id. This is the core of
                # the resume guarantee and keeps reruns idempotent at the task boundary.
                print(f"Skipping completed task: {task.task_name}")
                continue
            if task.status == TASK_RUNNING:
                raise RuntimeError(
                    f"task '{task.task_name}' is already running; wait or lower --stale-after-seconds"
                )
            if args.max_tasks is not None and executed >= args.max_tasks:
                # max_tasks exists primarily to validate resumability during development. It forces
                # the driver to stop cleanly without pretending the run has failed.
                registry.set_run_status(
                    run_id,
                    RUN_INTERRUPTED,
                    "Run paused after reaching the requested max task limit",
                )
                print(
                    f"Paused run {run_id} after {executed} task(s). "
                    "Re-run the same command to resume."
                )
                return 0

            print(f"Running task {task.task_order:02d} {task.task_name} [{task.task_kind}]")
            _execute_task(
                spec=spec,
                run_id=run_id,
                task=task,
                layout=layout,
                registry=registry,
                heartbeat_seconds=args.heartbeat_seconds,
            )
            executed += 1

        registry.set_run_status(run_id, RUN_SUCCEEDED, "Run completed successfully")
        print(f"Run {run_id} completed successfully.")
        return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Print the current persisted status of a run and its tasks."""

    layout = RuntimeLayout.create()
    with ExperimentRegistry(layout.database_path) as registry:
        run = registry.get_run(args.run_id)
        print(f"Run ID: {run.run_id}")
        print(f"Experiment: {run.experiment_id}")
        print(f"Status: {run.status}")
        if run.status_message:
            print(f"Message: {run.status_message}")
        if run.current_task_name:
            print(f"Current Task: {run.current_task_name}")
        print("Tasks:")
        for task in registry.list_tasks(run.run_id):
            print(f"- {task.task_order:02d} {task.task_name}: {task.status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the experiments driver."""

    parser = argparse.ArgumentParser(
        prog="python -m experiments.src",
        description="AlphaSolitaire experiment driver",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Validate a spec and print the planned tasks")
    plan_parser.add_argument("spec", help="Path to a JSON experiment spec")
    plan_parser.add_argument("--run-id", help="Override the default run id")
    plan_parser.set_defaults(func=_cmd_plan)

    run_parser = subparsers.add_parser("run", help="Create or resume a run from a spec")
    run_parser.add_argument("spec", help="Path to a JSON experiment spec")
    run_parser.add_argument("--run-id", help="Override the default run id")
    run_parser.add_argument(
        "--max-tasks",
        type=int,
        help="Run at most this many incomplete tasks before pausing for resume validation",
    )
    run_parser.add_argument(
        "--stale-after-seconds",
        type=int,
        default=DEFAULT_STALE_AFTER_SECONDS,
        help="Recover running tasks whose heartbeat is older than this threshold",
    )
    run_parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=DEFAULT_HEARTBEAT_SECONDS,
        help="Heartbeat cadence while command tasks are running",
    )
    run_parser.set_defaults(func=_cmd_run)

    status_parser = subparsers.add_parser("status", help="Show run and task status")
    status_parser.add_argument("run_id", help="Run identifier to inspect")
    status_parser.set_defaults(func=_cmd_status)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the experiments CLI and exit with a non-zero code on validation or run errors."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        raise SystemExit(args.func(args))
    except SpecValidationError as exc:
        # Validation errors are user-fixable input issues, so reserve a distinct exit code.
        print(f"Spec validation error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except (KeyError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
