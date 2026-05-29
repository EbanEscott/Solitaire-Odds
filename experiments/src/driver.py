"""Phase 1 CLI for resumable local experiment orchestration."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
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


@dataclass(frozen=True)
class RuntimeLayout:
    """Stable on-disk layout for the Phase 1 experiment control plane."""

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
    """Serialize a planned task into the registry row format used by Phase 1."""

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
    """Materialize the payload for a noop task so Phase 1 still produces reviewable artifacts."""

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
                # the Phase 1 resume guarantee and keeps reruns idempotent at the task boundary.
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
    """Construct the CLI parser for the Phase 1 experiments driver."""

    parser = argparse.ArgumentParser(
        prog="python -m experiments.src",
        description="Phase 1 AlphaSolitaire experiment driver",
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
