"""CLI for resumable local experiment orchestration."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
from statistics import median
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .architectures import get_adapter_for_family
from .architectures import (
    ARCHIVED_EPISODE_LOGS_DATASET_KIND,
    RUN_COLLECTION_EPISODE_LOGS_DATASET_KIND,
)
from .registry import (
    ArtifactRecord,
    AttemptRecord,
    ExperimentRegistry,
    RUN_INTERRUPTED,
    RUN_RUNNING,
    RUN_SUCCEEDED,
    RunRecord,
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
DEFAULT_DOCTOR_OUTPUT_FILENAME = "runtime_health_report.md"
DEFAULT_DOCTOR_JSON_OUTPUT_FILENAME = "runtime_health_report.json"
DEFAULT_TEMP_ATTEMPT_RETENTION_HOURS = 24
DEFAULT_WORK_FILE_RETENTION_DAYS = 14
SQLITE_ASSESSMENT_TEXT = (
    "SQLite remains sufficient for the current single-machine control plane. "
    "The registry is local, the task volume is modest, and Phase 6 hardening still relies "
    "on direct point queries and ordered scans rather than concurrent multi-host scheduling."
)


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


@dataclass(frozen=True)
class CleanupCandidate:
    """One removable runtime path discovered by the cleanup policy."""

    path: Path
    category: str
    age_text: str


@dataclass(frozen=True)
class TaskRuntimeEstimate:
    """Estimated runtime for one remaining task before the run starts."""

    task_name: str
    task_kind: str
    estimated_seconds: float | None
    basis: str


@dataclass(frozen=True)
class RunPreflightEstimate:
    """Aggregate runtime estimate for the remaining tasks in one run."""

    task_estimates: tuple[TaskRuntimeEstimate, ...]

    @property
    def remaining_task_count(self) -> int:
        return len(self.task_estimates)

    @property
    def known_task_count(self) -> int:
        return sum(1 for estimate in self.task_estimates if estimate.estimated_seconds is not None)

    @property
    def unknown_task_count(self) -> int:
        return sum(1 for estimate in self.task_estimates if estimate.estimated_seconds is None)

    @property
    def estimated_seconds(self) -> float:
        return sum(estimate.estimated_seconds or 0.0 for estimate in self.task_estimates)


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
    registry.connection.execute(
        """
        UPDATE task_attempts
        SET artifact_dir = ?,
            stdout_path = ?,
            stderr_path = ?
        WHERE attempt_id = ?
        """,
        (
            str(final_dir),
            str(final_dir / "stdout.log"),
            str(final_dir / "stderr.log"),
            attempt_id,
        ),
    )
    registry.connection.commit()
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


def _resolve_training_dataset_sources(
    registry: ExperimentRegistry,
    run_id: str,
    payload: Mapping[str, Any],
) -> list[Path]:
    """Resolve the concrete log files a training task should consume."""

    dataset_kind = payload.get("dataset_kind")
    if dataset_kind == ARCHIVED_EPISODE_LOGS_DATASET_KIND:
        dataset_sources = payload.get("dataset_sources", [])
        if not isinstance(dataset_sources, list) or not dataset_sources:
            raise ValueError("training payload is missing dataset_sources")
        return [
            _resolve_repo_relative_path(str(source))
            for source in dataset_sources
            if _resolve_repo_relative_path(str(source)) is not None
        ]

    if dataset_kind != RUN_COLLECTION_EPISODE_LOGS_DATASET_KIND:
        raise ValueError(f"unsupported training dataset kind: {dataset_kind}")

    dataset_sources: list[Path] = []
    for collection_task in registry.list_tasks(run_id):
        if collection_task.task_kind != "endgame_collect_shard" or collection_task.status != TASK_SUCCEEDED:
            continue
        latest_attempt = _latest_attempt(registry, collection_task.task_id)
        if latest_attempt is None or latest_attempt.status != TASK_SUCCEEDED:
            continue
        dataset_sources.extend(_list_episode_logs(Path(latest_attempt.artifact_dir) / "collected-logs"))

    if not dataset_sources:
        raise ValueError(
            "training dataset kind 'run_collection_episode_logs' found no successful collected logs in the current run"
        )
    return sorted(dataset_sources, key=lambda path: str(path))


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
    dataset_sources: list[Path],
) -> tuple[list[str], Path, Path, Path | None]:
    """Construct the Python training command and its output locations for one task attempt."""

    architecture_params = payload.get("architecture_params", {})
    if not isinstance(architecture_params, dict):
        raise ValueError("training payload is missing architecture_params")

    architecture_family = payload.get("architecture_family")
    if not isinstance(architecture_family, str) or not architecture_family:
        raise ValueError("training payload is missing architecture_family")
    adapter = get_adapter_for_family(architecture_family)

    checkpoint_dir = temp_dir / "checkpoints"
    metrics_output = temp_dir / str(payload.get("metrics_filename", "epoch_metrics.jsonl"))
    resume_checkpoint = _resolve_training_resume_checkpoint(task, payload)

    command = [
        sys.executable,
        "-m",
        "src.train_policy_value",
        *adapter.build_training_command_args(architecture_params),
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
    if resume_checkpoint is not None:
        command.extend(["--resume-from", str(resume_checkpoint)])

    simulate_interrupt_after_epoch = payload.get("simulate_interrupt_after_epoch")
    if simulate_interrupt_after_epoch is not None:
        command.extend([
            "--simulate-interrupt-after-epoch",
            str(simulate_interrupt_after_epoch),
        ])

    command.extend(str(source) for source in dataset_sources)
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
    resolved_dataset_sources = _resolve_training_dataset_sources(registry, run_id, payload)
    command, checkpoint_dir, metrics_output, resume_checkpoint = _build_policy_value_train_command(
        payload,
        task,
        temp_dir,
        resolved_dataset_sources,
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
        "dataset_sources": [
            _format_repo_or_absolute_path(path)
            for path in resolved_dataset_sources
        ],
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


def _parse_utc_timestamp(timestamp_text: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp recorded by the registry, if present."""

    if timestamp_text is None or not timestamp_text:
        return None
    try:
        return datetime.fromisoformat(timestamp_text)
    except ValueError:
        return None


def _format_age(delta: timedelta) -> str:
    """Render a coarse human-readable duration for operator-facing reports."""

    total_seconds = max(0, int(delta.total_seconds()))
    if total_seconds < 60:
        return f"{total_seconds}s"
    if total_seconds < 3600:
        return f"{total_seconds // 60}m"
    if total_seconds < 86400:
        return f"{total_seconds // 3600}h"
    return f"{total_seconds // 86400}d"


def _format_duration_seconds(seconds: float | None) -> str:
    """Render a wall-clock duration estimate in a compact human-readable form."""

    if seconds is None:
        return "unknown"

    total_seconds = max(0, int(round(seconds)))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def _format_repo_or_absolute_path(path: Path) -> str:
    """Render a path relative to the repo when possible, otherwise as an absolute path."""

    try:
        return _relative_to_repo(path)
    except ValueError:
        return str(path)


def _latest_attempt(registry: ExperimentRegistry, task_id: int) -> AttemptRecord | None:
    """Return the newest recorded attempt for one task, if any exist."""

    attempts = registry.list_task_attempts(task_id)
    return attempts[0] if attempts else None


def _attempt_duration_seconds(attempt: AttemptRecord | None) -> float | None:
    """Return one attempt's wall-clock duration in seconds when both timestamps exist."""

    if attempt is None:
        return None

    started_at = _parse_utc_timestamp(attempt.started_at)
    completed_at = _parse_utc_timestamp(attempt.completed_at)
    if started_at is None or completed_at is None or completed_at < started_at:
        return None
    return (completed_at - started_at).total_seconds()


def _matching_history_entries(
    history_entries: list[dict[str, Any]],
    payload: Mapping[str, Any],
    match_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Select historical timings whose key parameters match the current payload."""

    matched: list[dict[str, Any]] = []
    for entry in history_entries:
        historical_payload = entry["payload"]
        if all(historical_payload.get(key) == payload.get(key) for key in match_keys):
            matched.append(entry)
    return matched


def _estimate_task_duration_from_history(
    *,
    payload: Mapping[str, Any],
    history_entries: list[dict[str, Any]],
    unit_key: str,
    unit_label: str,
    match_keys: tuple[str, ...],
) -> tuple[float | None, str]:
    """Estimate a task duration from historical per-unit timings when possible."""

    requested_units = payload.get(unit_key)
    if not isinstance(requested_units, (int, float)) or requested_units <= 0:
        return None, f"missing '{unit_key}' in task payload"

    matched_entries = _matching_history_entries(history_entries, payload, match_keys)
    candidate_entries = matched_entries or history_entries
    unit_rates: list[float] = []
    for entry in candidate_entries:
        historical_units = entry["payload"].get(unit_key)
        if not isinstance(historical_units, (int, float)) or historical_units <= 0:
            continue
        unit_rates.append(entry["duration_seconds"] / float(historical_units))

    if not unit_rates:
        return None, "no historical timing data"

    estimated_seconds = median(unit_rates) * float(requested_units)
    if matched_entries:
        return (
            estimated_seconds,
            f"median historical {unit_label} rate from {len(unit_rates)} matching successful attempt(s)",
        )
    return (
        estimated_seconds,
        f"median historical {unit_label} rate from {len(unit_rates)} successful attempt(s)",
    )


def _estimate_task_duration_from_totals(
    history_entries: list[dict[str, Any]],
) -> tuple[float | None, str]:
    """Estimate a task duration from total runtimes when no better model exists."""

    durations = [entry["duration_seconds"] for entry in history_entries]
    if not durations:
        return None, "no historical timing data"
    return median(durations), f"median historical total runtime from {len(durations)} successful attempt(s)"


def _successful_history_by_kind(
    registry: ExperimentRegistry,
    task_kind: str,
    cache: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return cached successful historical attempt timings for one task kind."""

    if task_kind in cache:
        return cache[task_kind]

    entries: list[dict[str, Any]] = []
    for historical_task in registry.list_tasks_by_kind(task_kind=task_kind, task_status=TASK_SUCCEEDED):
        latest_attempt = _latest_attempt(registry, historical_task.task_id)
        if latest_attempt is None or latest_attempt.status != TASK_SUCCEEDED:
            continue
        duration_seconds = _attempt_duration_seconds(latest_attempt)
        if duration_seconds is None:
            continue
        entries.append(
            {
                "payload": json.loads(historical_task.payload_json or "{}"),
                "duration_seconds": duration_seconds,
            }
        )

    cache[task_kind] = entries
    return entries


def _estimate_task_runtime(
    *,
    registry: ExperimentRegistry,
    task_name: str,
    task_kind: str,
    payload: Mapping[str, Any],
    history_cache: dict[str, list[dict[str, Any]]],
) -> TaskRuntimeEstimate:
    """Estimate one task runtime using successful historical attempts of the same kind."""

    history_entries = _successful_history_by_kind(registry, task_kind, history_cache)
    estimated_seconds: float | None
    basis: str

    if task_kind == "endgame_collect_shard":
        estimated_seconds, basis = _estimate_task_duration_from_history(
            payload=payload,
            history_entries=history_entries,
            unit_key="requested_games",
            unit_label="game",
            match_keys=("engine_test", "level", "randomise"),
        )
    elif task_kind == "alpha_level_eval_shard":
        estimated_seconds, basis = _estimate_task_duration_from_history(
            payload=payload,
            history_entries=history_entries,
            unit_key="requested_games",
            unit_label="game",
            match_keys=(
                "engine_test",
                "level",
                "architecture_family",
                "mcts_simulations",
                "mcts_max_depth",
                "mcts_cpuct",
            ),
        )
    elif task_kind == "policy_value_train":
        estimated_seconds, basis = _estimate_task_duration_from_history(
            payload=payload,
            history_entries=history_entries,
            unit_key="epochs",
            unit_label="epoch",
            match_keys=("architecture_family", "training_kind", "dataset_kind"),
        )
    else:
        estimated_seconds, basis = _estimate_task_duration_from_totals(history_entries)

    return TaskRuntimeEstimate(
        task_name=task_name,
        task_kind=task_kind,
        estimated_seconds=estimated_seconds,
        basis=basis,
    )


def _estimate_remaining_runtime(
    registry: ExperimentRegistry,
    remaining_tasks: list[tuple[str, str, Mapping[str, Any]]],
) -> RunPreflightEstimate:
    """Estimate total runtime for the remaining tasks in the requested run."""

    history_cache: dict[str, list[dict[str, Any]]] = {}
    task_estimates = tuple(
        _estimate_task_runtime(
            registry=registry,
            task_name=task_name,
            task_kind=task_kind,
            payload=payload,
            history_cache=history_cache,
        )
        for task_name, task_kind, payload in remaining_tasks
    )
    return RunPreflightEstimate(task_estimates=task_estimates)


def _print_run_preflight(run_id: str, preflight: RunPreflightEstimate) -> None:
    """Print the pre-launch runtime estimate for the remaining tasks in one run."""

    print(f"Preflight estimate for run {run_id}:")
    if preflight.remaining_task_count == 0:
        print("- No remaining tasks. The run is already complete.")
        return

    if preflight.unknown_task_count == 0:
        print(f"- Estimated remaining runtime: {_format_duration_seconds(preflight.estimated_seconds)}")
    elif preflight.known_task_count == 0:
        print(
            f"- Estimated remaining runtime: unknown; no historical timing data for {preflight.unknown_task_count} remaining task(s)"
        )
    else:
        print(
            f"- Estimated remaining runtime: at least {_format_duration_seconds(preflight.estimated_seconds)} "
            f"plus {preflight.unknown_task_count} task(s) without historical timing data"
        )

    grouped: dict[str, dict[str, Any]] = {}
    for estimate in preflight.task_estimates:
        group = grouped.setdefault(
            estimate.task_kind,
            {
                "count": 0,
                "estimated_seconds": 0.0,
                "unknown_count": 0,
                "bases": Counter(),
            },
        )
        group["count"] += 1
        if estimate.estimated_seconds is None:
            group["unknown_count"] += 1
        else:
            group["estimated_seconds"] += estimate.estimated_seconds
        group["bases"][estimate.basis] += 1

    for task_kind, group in grouped.items():
        count = int(group["count"])
        unknown_count = int(group["unknown_count"])
        basis, _ = group["bases"].most_common(1)[0]
        if unknown_count == 0:
            estimate_text = _format_duration_seconds(float(group["estimated_seconds"]))
        elif unknown_count == count:
            estimate_text = "unknown"
        else:
            estimate_text = f"at least {_format_duration_seconds(float(group['estimated_seconds']))}"
        if unknown_count:
            estimate_text = f"{estimate_text} ({unknown_count} task(s) without historical timing data)"
        print(f"- {task_kind}: {count} task(s), {estimate_text}. Basis: {basis}.")


def _prompt_for_run_confirmation(run_id: str, preflight: RunPreflightEstimate) -> bool:
    """Ask the operator to confirm the estimated run before launching tasks."""

    if preflight.remaining_task_count == 0:
        return True
    response = input(f"Continue with run {run_id} using the estimate above? [y/N] ").strip().lower()
    return response in {"y", "yes"}


def _resolve_attempt_log_path(attempt: AttemptRecord | None, filename: str) -> Path | None:
    """Resolve one attempt log path, tolerating older rows that still point at temp dirs."""

    if attempt is None:
        return None

    direct_path_text = attempt.stderr_path if filename == "stderr.log" else attempt.stdout_path
    if direct_path_text:
        direct_path = Path(direct_path_text)
        if direct_path.exists():
            return direct_path

    artifact_dir = Path(attempt.artifact_dir)
    fallback_path = artifact_dir / filename
    if fallback_path.exists():
        return fallback_path
    return None


def _build_runtime_health_report_markdown(
    *,
    generated_at: str,
    recovered_stale_tasks: int,
    run_status_counts: Mapping[str, int],
    run_rows: list[list[str]],
    stale_task_rows: list[list[str]],
    problematic_task_rows: list[list[str]],
    missing_artifact_rows: list[list[str]],
) -> str:
    """Build a markdown dashboard summarizing runtime health across all runs."""

    total_runs = sum(run_status_counts.values())
    summary_rows = [
        ["Generated At", generated_at],
        ["Total Runs", str(total_runs)],
        ["Succeeded Runs", str(run_status_counts.get(RUN_SUCCEEDED, 0))],
        ["Running Runs", str(run_status_counts.get(RUN_RUNNING, 0))],
        ["Interrupted Runs", str(run_status_counts.get(RUN_INTERRUPTED, 0))],
        ["Recovered Stale Tasks", str(recovered_stale_tasks)],
        ["Stale Tasks", str(len(stale_task_rows))],
        ["Problematic Tasks", str(len(problematic_task_rows))],
        ["Missing Artifacts", str(len(missing_artifact_rows))],
    ]

    lines = [
        "# Runtime Health Report",
        "",
        "## Summary",
        "",
        _format_markdown_table(["Metric", "Value"], summary_rows),
        "",
    ]

    if run_rows:
        lines.extend(
            [
                "## Recent Runs",
                "",
                _format_markdown_table(
                    ["Run", "Experiment", "Status", "Updated", "Current Task", "Message"],
                    run_rows,
                ),
                "",
            ]
        )

    if stale_task_rows:
        lines.extend(
            [
                "## Stale Tasks",
                "",
                _format_markdown_table(
                    ["Run", "Task", "Heartbeat Age", "Last Heartbeat", "Artifact Root"],
                    stale_task_rows,
                ),
                "",
            ]
        )

    if problematic_task_rows:
        lines.extend(
            [
                "## Problematic Tasks",
                "",
                _format_markdown_table(
                    ["Run", "Task", "Status", "Attempts", "Exit", "Error", "Stderr"],
                    problematic_task_rows,
                ),
                "",
            ]
        )

    if missing_artifact_rows:
        lines.extend(
            [
                "## Missing Artifacts",
                "",
                _format_markdown_table(
                    ["Run", "Task", "Issue", "Path"],
                    missing_artifact_rows,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## SQLite Assessment",
            "",
            SQLITE_ASSESSMENT_TEXT,
            "",
        ]
    )

    return "\n".join(lines)


def _default_doctor_json_output_path(markdown_output_path: Path) -> Path:
    """Derive the default JSON companion path for a markdown doctor report."""

    if markdown_output_path.name == DEFAULT_DOCTOR_OUTPUT_FILENAME:
        return markdown_output_path.with_name(DEFAULT_DOCTOR_JSON_OUTPUT_FILENAME)
    return markdown_output_path.with_suffix(".json")


def _build_runtime_health_report_payload(
    *,
    generated_at: str,
    recovered_stale_tasks: int,
    run_status_counts: Mapping[str, int],
    recent_runs: list[dict[str, Any]],
    stale_tasks: list[dict[str, Any]],
    problematic_tasks: list[dict[str, Any]],
    missing_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the machine-readable runtime health payload written beside the markdown report."""

    return {
        "report_version": "v1",
        "generated_at": generated_at,
        "summary": {
            "total_runs": int(sum(run_status_counts.values())),
            "run_status_counts": {key: int(value) for key, value in sorted(run_status_counts.items())},
            "recovered_stale_tasks": int(recovered_stale_tasks),
            "stale_tasks": len(stale_tasks),
            "problematic_tasks": len(problematic_tasks),
            "missing_artifacts": len(missing_artifacts),
        },
        "recent_runs": recent_runs,
        "stale_tasks": stale_tasks,
        "problematic_tasks": problematic_tasks,
        "missing_artifacts": missing_artifacts,
        "sqlite_assessment": SQLITE_ASSESSMENT_TEXT,
    }


def _find_cleanup_candidates(
    *,
    layout: RuntimeLayout,
    temp_attempt_older_than_hours: int,
    work_file_older_than_days: int,
) -> list[CleanupCandidate]:
    """Discover runtime paths eligible for cleanup under the current retention policy."""

    now = datetime.now(timezone.utc)
    temp_cutoff = now - timedelta(hours=temp_attempt_older_than_hours)
    work_cutoff = now - timedelta(days=work_file_older_than_days)
    candidates: list[CleanupCandidate] = []

    for path in sorted(layout.artifacts_dir.glob("**/.attempt-*.tmp")):
        if not path.is_dir():
            continue
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified_at < temp_cutoff:
            candidates.append(
                CleanupCandidate(
                    path=path,
                    category="temporary attempt directory",
                    age_text=_format_age(now - modified_at),
                )
            )

    for path in sorted(layout.work_dir.rglob("*")):
        if not path.is_file():
            continue
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified_at < work_cutoff:
            candidates.append(
                CleanupCandidate(
                    path=path,
                    category="runtime work file",
                    age_text=_format_age(now - modified_at),
                )
            )

    return candidates


def _prune_empty_directories(root: Path) -> int:
    """Remove empty directories under one runtime root after file cleanup."""

    if not root.exists():
        return 0

    removed = 0
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        if path == root or not path.exists():
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


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
        persisted_tasks_by_name: dict[str, TaskRecord] = {}
        try:
            registry.get_run(run_id)
        except KeyError:
            pass
        else:
            persisted_tasks_by_name = {
                task.task_name: task for task in registry.list_tasks(run_id)
            }

        remaining_tasks: list[tuple[str, str, Mapping[str, Any]]] = []
        for planned_task in tasks:
            persisted_task = persisted_tasks_by_name.get(planned_task.task_name)
            if persisted_task is not None and persisted_task.status == TASK_SUCCEEDED:
                continue
            if persisted_task is not None:
                remaining_tasks.append(
                    (
                        persisted_task.task_name,
                        persisted_task.task_kind,
                        json.loads(persisted_task.payload_json or "{}"),
                    )
                )
            else:
                remaining_tasks.append(
                    (
                        planned_task.task_name,
                        planned_task.task_kind,
                        planned_task.payload,
                    )
                )

        preflight = _estimate_remaining_runtime(registry, remaining_tasks)
        _print_run_preflight(run_id, preflight)
        if remaining_tasks and not args.yes:
            if not sys.stdin.isatty():
                raise RuntimeError(
                    "run confirmation requires an interactive terminal; rerun with --yes to accept the estimate"
                )
            if not _prompt_for_run_confirmation(run_id, preflight):
                print(f"Cancelled run {run_id} before launching any tasks.")
                return 0

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


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Inspect runtime health across all runs and write a markdown dashboard report."""

    layout = RuntimeLayout.create()
    output_path = args.output or (layout.work_dir / DEFAULT_DOCTOR_OUTPUT_FILENAME)
    json_output_path = args.json_output or _default_doctor_json_output_path(output_path)

    with ExperimentRegistry(layout.database_path) as registry:
        recovered_stale_tasks = 0
        if args.recover_stale:
            for run in registry.list_runs():
                recovered_stale_tasks += registry.recover_stale_running_tasks(
                    run.run_id,
                    args.stale_after_seconds,
                )

        runs = registry.list_runs()
        now = datetime.now(timezone.utc)
        stale_cutoff = now - timedelta(seconds=args.stale_after_seconds)
        run_status_counts = Counter(run.status for run in runs)
        generated_at = now.replace(microsecond=0).isoformat()

        recent_runs: list[dict[str, Any]] = []
        stale_tasks: list[dict[str, Any]] = []
        problematic_tasks: list[dict[str, Any]] = []
        missing_artifacts: list[dict[str, Any]] = []

        for run in runs[: args.limit_runs]:
            recent_runs.append(
                {
                    "run_id": run.run_id,
                    "experiment_id": run.experiment_id,
                    "status": run.status,
                    "updated_at": run.updated_at,
                    "current_task_name": run.current_task_name,
                    "status_message": run.status_message,
                }
            )

        for run in runs:
            tasks = registry.list_tasks(run.run_id)
            task_by_id = {task.task_id: task for task in tasks}
            artifact_rows: list[ArtifactRecord] = registry.list_artifacts(run.run_id)
            artifact_rows_by_task_id: dict[int, list[ArtifactRecord]] = {}
            for artifact in artifact_rows:
                artifact_rows_by_task_id.setdefault(artifact.task_id, []).append(artifact)

            artifact_root = Path(run.artifact_root)
            run_artifact_root_exists = artifact_root.exists()
            if not run_artifact_root_exists:
                missing_artifacts.append(
                    {
                        "run_id": run.run_id,
                        "task_name": None,
                        "issue": "run artifact root missing",
                        "path": _format_repo_or_absolute_path(artifact_root),
                    }
                )

            for task in tasks:
                last_heartbeat = _parse_utc_timestamp(task.last_heartbeat_at)
                if (
                    task.status == TASK_RUNNING
                    and last_heartbeat is not None
                    and args.stale_after_seconds > 0
                    and last_heartbeat < stale_cutoff
                ):
                    stale_tasks.append(
                        {
                            "run_id": run.run_id,
                            "task_name": task.task_name,
                            "heartbeat_age": _format_age(now - last_heartbeat),
                            "last_heartbeat_at": task.last_heartbeat_at,
                            "artifact_root": _format_repo_or_absolute_path(Path(task.artifact_dir)),
                        }
                    )

                latest_attempt: AttemptRecord | None = _latest_attempt(registry, task.task_id)
                if task.status in {TASK_INTERRUPTED, TASK_RETRYABLE}:
                    attempt_count = len(registry.list_task_attempts(task.task_id))
                    stderr_log_path = _resolve_attempt_log_path(latest_attempt, "stderr.log")
                    problematic_tasks.append(
                        {
                            "run_id": run.run_id,
                            "task_name": task.task_name,
                            "status": task.status,
                            "attempt_count": attempt_count,
                            "exit_code": latest_attempt.exit_code if latest_attempt is not None else None,
                            "error": (latest_attempt.error_message or run.status_message) if latest_attempt is not None else run.status_message,
                            "stderr_path": _format_repo_or_absolute_path(stderr_log_path) if stderr_log_path is not None else None,
                        }
                    )

                if task.status == TASK_SUCCEEDED and not artifact_rows_by_task_id.get(task.task_id):
                    missing_artifacts.append(
                        {
                            "run_id": run.run_id,
                            "task_name": task.task_name,
                            "issue": "succeeded task has no registered artifact",
                            "path": _format_repo_or_absolute_path(Path(task.artifact_dir)),
                        }
                    )

            if run_artifact_root_exists:
                for artifact in artifact_rows:
                    artifact_path = (REPO_ROOT / artifact.relative_path).resolve()
                    if not artifact_path.exists():
                        task = task_by_id.get(artifact.task_id)
                        missing_artifacts.append(
                            {
                                "run_id": run.run_id,
                                "task_name": task.task_name if task is not None else f"task_id={artifact.task_id}",
                                "issue": "registered artifact path missing",
                                "path": _format_repo_or_absolute_path(artifact_path),
                            }
                        )

    run_rows = [
        [
            row["run_id"],
            row["experiment_id"],
            row["status"],
            row["updated_at"],
            row["current_task_name"] or "-",
            row["status_message"] or "-",
        ]
        for row in recent_runs
    ]
    stale_task_rows = [
        [
            row["run_id"],
            row["task_name"],
            row["heartbeat_age"],
            row["last_heartbeat_at"] or "-",
            row["artifact_root"],
        ]
        for row in stale_tasks
    ]
    problematic_task_rows = [
        [
            row["run_id"],
            row["task_name"],
            row["status"],
            str(row["attempt_count"]),
            str(row["exit_code"]) if row["exit_code"] is not None else "-",
            row["error"] or "-",
            row["stderr_path"] or "-",
        ]
        for row in problematic_tasks
    ]
    missing_artifact_rows = [
        [
            row["run_id"],
            row["task_name"] or "-",
            row["issue"],
            row["path"],
        ]
        for row in missing_artifacts
    ]

    report_markdown = _build_runtime_health_report_markdown(
        generated_at=generated_at,
        recovered_stale_tasks=recovered_stale_tasks,
        run_status_counts=run_status_counts,
        run_rows=run_rows,
        stale_task_rows=stale_task_rows,
        problematic_task_rows=problematic_task_rows,
        missing_artifact_rows=missing_artifact_rows,
    )
    report_payload = _build_runtime_health_report_payload(
        generated_at=generated_at,
        recovered_stale_tasks=recovered_stale_tasks,
        run_status_counts=run_status_counts,
        recent_runs=recent_runs,
        stale_tasks=stale_tasks,
        problematic_tasks=problematic_tasks,
        missing_artifacts=missing_artifacts,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_markdown, encoding="utf-8")
    _write_json(json_output_path, report_payload)

    print(
        "Runtime health summary: "
        f"runs={sum(run_status_counts.values())}, "
        f"stale_tasks={len(stale_tasks)}, "
        f"problematic_tasks={len(problematic_tasks)}, "
        f"missing_artifacts={len(missing_artifacts)}"
    )
    if recovered_stale_tasks:
        print(f"Recovered {recovered_stale_tasks} stale task(s) before writing the report.")
    print(f"Wrote runtime health report to {_format_repo_or_absolute_path(output_path)}")
    print(f"Wrote runtime health JSON to {_format_repo_or_absolute_path(json_output_path)}")
    return 0


def _cmd_cleanup(args: argparse.Namespace) -> int:
    """Apply the runtime cleanup policy to stale temp directories and old work files."""

    layout = RuntimeLayout.create()
    candidates = _find_cleanup_candidates(
        layout=layout,
        temp_attempt_older_than_hours=args.temp_attempt_older_than_hours,
        work_file_older_than_days=args.work_file_older_than_days,
    )

    if not candidates:
        print("No cleanup candidates found.")
        return 0

    action_text = "Would remove" if args.dry_run else "Removing"
    for candidate in candidates:
        print(
            f"{action_text} {_format_repo_or_absolute_path(candidate.path)} "
            f"[{candidate.category}; age={candidate.age_text}]"
        )

    if args.dry_run:
        print(f"Found {len(candidates)} cleanup candidate(s).")
        return 0

    removed_count = 0
    for candidate in candidates:
        if not candidate.path.exists():
            continue
        if candidate.path.is_dir():
            shutil.rmtree(candidate.path)
        else:
            candidate.path.unlink()
        removed_count += 1

    pruned_empty_directories = _prune_empty_directories(layout.work_dir)
    pruned_empty_directories += _prune_empty_directories(layout.artifacts_dir)
    print(
        f"Removed {removed_count} runtime path(s) and pruned {pruned_empty_directories} empty directory(ies)."
    )
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
    run_parser.add_argument(
        "--yes",
        action="store_true",
        help="Accept the preflight runtime estimate without an interactive confirmation prompt",
    )
    run_parser.set_defaults(func=_cmd_run)

    status_parser = subparsers.add_parser("status", help="Show run and task status")
    status_parser.add_argument("run_id", help="Run identifier to inspect")
    status_parser.set_defaults(func=_cmd_status)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Write a runtime health dashboard covering failed runs and missing artifacts",
    )
    doctor_parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Markdown report output path "
            f"(default: experiments/runtime/work/{DEFAULT_DOCTOR_OUTPUT_FILENAME})"
        ),
    )
    doctor_parser.add_argument(
        "--json-output",
        type=Path,
        help=(
            "JSON report output path "
            f"(default: experiments/runtime/work/{DEFAULT_DOCTOR_JSON_OUTPUT_FILENAME})"
        ),
    )
    doctor_parser.add_argument(
        "--stale-after-seconds",
        type=int,
        default=DEFAULT_STALE_AFTER_SECONDS,
        help="Treat running tasks older than this heartbeat threshold as stale in the report",
    )
    doctor_parser.add_argument(
        "--recover-stale",
        action="store_true",
        help="Recover stale running tasks across all runs before writing the report",
    )
    doctor_parser.add_argument(
        "--limit-runs",
        type=int,
        default=20,
        help="Maximum number of recent runs to include in the dashboard table",
    )
    doctor_parser.set_defaults(func=_cmd_doctor)

    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Apply retention policies to stale temp attempt directories and old work files",
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the runtime paths that would be removed without deleting them",
    )
    cleanup_parser.add_argument(
        "--temp-attempt-older-than-hours",
        type=int,
        default=DEFAULT_TEMP_ATTEMPT_RETENTION_HOURS,
        help="Delete .attempt-*.tmp directories older than this many hours",
    )
    cleanup_parser.add_argument(
        "--work-file-older-than-days",
        type=int,
        default=DEFAULT_WORK_FILE_RETENTION_DAYS,
        help="Delete files under experiments/runtime/work older than this many days",
    )
    cleanup_parser.set_defaults(func=_cmd_cleanup)

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
