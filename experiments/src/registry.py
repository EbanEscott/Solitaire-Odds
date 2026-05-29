"""SQLite-backed registry for experiment runs, tasks, attempts, and artifacts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping


RUN_PENDING = "pending"
RUN_RUNNING = "running"
RUN_INTERRUPTED = "interrupted"
RUN_FAILED = "failed"
RUN_SUCCEEDED = "succeeded"

TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_SUCCEEDED = "succeeded"
TASK_INTERRUPTED = "interrupted"
TASK_RETRYABLE = "retryable"
TASK_CANCELLED = "cancelled"


def utc_now() -> str:
    """Return a UTC timestamp in a stable text format for SQLite storage."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class RunRecord:
    """Persisted run state returned from the registry."""

    run_id: str
    experiment_id: str
    status: str
    artifact_root: str
    spec_hash: str
    current_task_name: str | None
    status_message: str | None


@dataclass(frozen=True)
class TaskRecord:
    """Persisted task state returned from the registry."""

    task_id: int
    run_id: str
    task_name: str
    task_order: int
    task_kind: str
    payload_json: str
    command_json: str
    working_directory: str | None
    status: str
    artifact_dir: str
    last_heartbeat_at: str | None


@dataclass(frozen=True)
class AttemptRecord:
    """Persisted task attempt state returned from the registry."""

    attempt_id: int
    task_id: int
    attempt_number: int
    artifact_dir: str


class ExperimentRegistry:
    """Manage the persistent SQLite registry for the experiments control plane."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(database_path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._initialize_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.connection.close()

    def __enter__(self) -> "ExperimentRegistry":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _initialize_schema(self) -> None:
        """Create the registry schema if it does not already exist."""

        # The schema keeps lineage, task state, and artifact metadata separate so the execution
        # model can evolve without rewriting the basic persistence contract.
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiment_specs (
                spec_hash TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                api_version TEXT NOT NULL,
                spec_path TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                spec_hash TEXT NOT NULL,
                spec_path TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_root TEXT NOT NULL,
                current_task_name TEXT,
                status_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                last_heartbeat_at TEXT,
                FOREIGN KEY (spec_hash) REFERENCES experiment_specs(spec_hash)
            );

            CREATE TABLE IF NOT EXISTS run_parameters (
                run_id TEXT NOT NULL,
                parameter_key TEXT NOT NULL,
                parameter_value TEXT NOT NULL,
                PRIMARY KEY (run_id, parameter_key),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS tasks (
                task_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                task_name TEXT NOT NULL,
                task_order INTEGER NOT NULL,
                task_kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                command_json TEXT NOT NULL,
                working_directory TEXT,
                status TEXT NOT NULL,
                artifact_dir TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                last_heartbeat_at TEXT,
                UNIQUE (run_id, task_name),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS task_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                attempt_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                artifact_dir TEXT NOT NULL,
                command_json TEXT NOT NULL,
                working_directory TEXT,
                stdout_path TEXT,
                stderr_path TEXT,
                exit_code INTEGER,
                error_message TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                heartbeat_at TEXT,
                UNIQUE (task_id, attempt_number),
                FOREIGN KEY (task_id) REFERENCES tasks(task_id)
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                task_id INTEGER NOT NULL,
                attempt_id INTEGER NOT NULL,
                artifact_kind TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id),
                FOREIGN KEY (task_id) REFERENCES tasks(task_id),
                FOREIGN KEY (attempt_id) REFERENCES task_attempts(attempt_id)
            );
            """
        )
        self.connection.commit()

    def register_spec(
        self,
        *,
        spec_hash: str,
        experiment_id: str,
        api_version: str,
        spec_path: str,
        spec_json: str,
    ) -> None:
        """Persist the canonical spec snapshot once for lineage and repeatability."""

        self.connection.execute(
            """
            INSERT OR IGNORE INTO experiment_specs (
                spec_hash,
                experiment_id,
                api_version,
                spec_path,
                spec_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (spec_hash, experiment_id, api_version, spec_path, spec_json, utc_now()),
        )
        self.connection.commit()

    def create_or_resume_run(
        self,
        *,
        run_id: str,
        experiment_id: str,
        spec_hash: str,
        spec_path: str,
        artifact_root: str,
    ) -> RunRecord:
        """Create a run record or reuse an existing compatible run for resume."""

        row = self.connection.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is not None:
            # Reusing the run is only safe when the persisted spec hash matches the current one.
            # This prevents a new experiment configuration from silently inheriting old state.
            if row["spec_hash"] != spec_hash:
                raise ValueError(
                    f"run_id '{run_id}' already exists with a different spec hash"
                )
            return self._row_to_run(row)

        now = utc_now()
        self.connection.execute(
            """
            INSERT INTO runs (
                run_id,
                experiment_id,
                spec_hash,
                spec_path,
                status,
                artifact_root,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                experiment_id,
                spec_hash,
                spec_path,
                RUN_PENDING,
                artifact_root,
                now,
                now,
            ),
        )
        self.connection.commit()
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        """Fetch a single persisted run by identifier."""

        row = self.connection.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"run not found: {run_id}")
        return self._row_to_run(row)

    def replace_run_parameters(self, run_id: str, parameters: Mapping[str, str]) -> None:
        """Persist flattened run dimensions for later querying and reporting."""

        self.connection.execute("DELETE FROM run_parameters WHERE run_id = ?", (run_id,))
        self.connection.executemany(
            "INSERT INTO run_parameters (run_id, parameter_key, parameter_value) VALUES (?, ?, ?)",
            ((run_id, key, value) for key, value in sorted(parameters.items())),
        )
        self.connection.commit()

    def ensure_tasks(self, run_id: str, planned_tasks: Iterable[Mapping[str, str]]) -> None:
        """Create any missing tasks for the run without duplicating existing rows."""

        now = utc_now()
        for task in planned_tasks:
            # INSERT OR IGNORE lets the driver rebuild its task plan on every invocation while
            # preserving the status of tasks that were already created or even completed earlier.
            self.connection.execute(
                """
                INSERT OR IGNORE INTO tasks (
                    run_id,
                    task_name,
                    task_order,
                    task_kind,
                    payload_json,
                    command_json,
                    working_directory,
                    status,
                    artifact_dir,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task["task_name"],
                    task["task_order"],
                    task["task_kind"],
                    task["payload_json"],
                    task["command_json"],
                    task["working_directory"],
                    TASK_PENDING,
                    task["artifact_dir"],
                    now,
                    now,
                ),
            )
            # Refresh task metadata for resumable tasks so a restarted driver can benefit from
            # newer planning logic or corrected payload normalization without mutating work that
            # already completed successfully or is genuinely still running elsewhere.
            self.connection.execute(
                """
                UPDATE tasks
                SET task_order = ?,
                    task_kind = ?,
                    payload_json = ?,
                    command_json = ?,
                    working_directory = ?,
                    artifact_dir = ?,
                    updated_at = ?
                WHERE run_id = ?
                  AND task_name = ?
                  AND status != ?
                  AND status != ?
                """,
                (
                    task["task_order"],
                    task["task_kind"],
                    task["payload_json"],
                    task["command_json"],
                    task["working_directory"],
                    task["artifact_dir"],
                    now,
                    run_id,
                    task["task_name"],
                    TASK_SUCCEEDED,
                    TASK_RUNNING,
                ),
            )
        self.connection.commit()

    def list_tasks(self, run_id: str) -> list[TaskRecord]:
        """List all tasks for a run in stable execution order."""

        rows = self.connection.execute(
            "SELECT * FROM tasks WHERE run_id = ? ORDER BY task_order ASC, task_id ASC",
            (run_id,),
        ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def list_tasks_by_kind(
        self,
        *,
        task_kind: str,
        task_status: str | None = None,
        run_status: str | None = None,
        exclude_run_id: str | None = None,
    ) -> list[TaskRecord]:
        """List tasks across runs filtered by task kind and optional task or run state."""

        query = [
            "SELECT tasks.*",
            "FROM tasks",
            "JOIN runs ON runs.run_id = tasks.run_id",
            "WHERE tasks.task_kind = ?",
        ]
        parameters: list[str] = [task_kind]

        if task_status is not None:
            query.append("AND tasks.status = ?")
            parameters.append(task_status)
        if run_status is not None:
            query.append("AND runs.status = ?")
            parameters.append(run_status)
        if exclude_run_id is not None:
            query.append("AND tasks.run_id != ?")
            parameters.append(exclude_run_id)

        query.append("ORDER BY tasks.run_id ASC, tasks.task_order ASC, tasks.task_id ASC")
        rows = self.connection.execute("\n".join(query), parameters).fetchall()
        return [self._row_to_task(row) for row in rows]

    def recover_stale_running_tasks(self, run_id: str, stale_after_seconds: int) -> int:
        """Mark stale running tasks interrupted so the next driver invocation can resume."""

        if stale_after_seconds <= 0:
            return 0

        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)).replace(
            microsecond=0
        ).isoformat()
        rows = self.connection.execute(
            """
            SELECT task_id
            FROM tasks
            WHERE run_id = ?
              AND status = ?
              AND last_heartbeat_at IS NOT NULL
              AND last_heartbeat_at < ?
            """,
            (run_id, TASK_RUNNING, cutoff),
        ).fetchall()

        recovered = 0
        for row in rows:
            recovered += 1
            task_id = int(row["task_id"])
            now = utc_now()
            # Stale tasks are treated as interrupted rather than failed. That preserves the
            # operator's ability to retry the work once the driver has reclaimed the run.
            self.connection.execute(
                """
                UPDATE tasks
                SET status = ?,
                    updated_at = ?,
                    completed_at = ?,
                    last_heartbeat_at = ?
                WHERE task_id = ?
                """,
                (TASK_INTERRUPTED, now, now, now, task_id),
            )
            self.connection.execute(
                """
                UPDATE task_attempts
                SET status = ?,
                    completed_at = ?,
                    heartbeat_at = ?
                WHERE task_id = ? AND status = ?
                """,
                (TASK_INTERRUPTED, now, now, task_id, TASK_RUNNING),
            )

        if recovered:
            self.connection.execute(
                """
                UPDATE runs
                SET status = ?,
                    status_message = ?,
                    updated_at = ?,
                    last_heartbeat_at = ?
                WHERE run_id = ?
                """,
                (
                    RUN_INTERRUPTED,
                    f"Recovered {recovered} stale running task(s)",
                    utc_now(),
                    utc_now(),
                    run_id,
                ),
            )

        self.connection.commit()
        return recovered

    def start_task_attempt(
        self,
        *,
        run_id: str,
        task: TaskRecord,
        artifact_dir: str,
        stdout_path: str | None,
        stderr_path: str | None,
    ) -> AttemptRecord:
        """Transition a task into running state and open a fresh attempt record."""

        row = self.connection.execute(
            "SELECT COALESCE(MAX(attempt_number), 0) + 1 AS next_attempt FROM task_attempts WHERE task_id = ?",
            (task.task_id,),
        ).fetchone()
        attempt_number = int(row["next_attempt"])
        now = utc_now()

        # The first started_at value is preserved across retries so the task retains a stable
        # historical start time even if individual attempts are interrupted or retried.
        self.connection.execute(
            """
            UPDATE tasks
            SET status = ?,
                updated_at = ?,
                started_at = COALESCE(started_at, ?),
                completed_at = NULL,
                last_heartbeat_at = ?
            WHERE task_id = ?
            """,
            (TASK_RUNNING, now, now, now, task.task_id),
        )
        cursor = self.connection.execute(
            """
            INSERT INTO task_attempts (
                task_id,
                attempt_number,
                status,
                artifact_dir,
                command_json,
                working_directory,
                stdout_path,
                stderr_path,
                started_at,
                heartbeat_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.task_id,
                attempt_number,
                TASK_RUNNING,
                artifact_dir,
                task.command_json,
                task.working_directory,
                stdout_path,
                stderr_path,
                now,
                now,
            ),
        )
        self.connection.execute(
            """
            UPDATE runs
            SET status = ?,
                current_task_name = ?,
                status_message = NULL,
                updated_at = ?,
                started_at = COALESCE(started_at, ?),
                last_heartbeat_at = ?
            WHERE run_id = ?
            """,
            (RUN_RUNNING, task.task_name, now, now, now, run_id),
        )
        self.connection.commit()
        return AttemptRecord(
            attempt_id=int(cursor.lastrowid),
            task_id=task.task_id,
            attempt_number=attempt_number,
            artifact_dir=artifact_dir,
        )

    def heartbeat(self, run_id: str, task_id: int, attempt_id: int) -> None:
        """Refresh heartbeat timestamps while a task attempt is active."""

        # Heartbeats are written to the run, task, and attempt rows together so stale recovery
        # can reason at whichever level is most convenient without additional joins.
        now = utc_now()
        self.connection.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, updated_at = ? WHERE task_id = ?",
            (now, now, task_id),
        )
        self.connection.execute(
            "UPDATE task_attempts SET heartbeat_at = ? WHERE attempt_id = ?",
            (now, attempt_id),
        )
        self.connection.execute(
            "UPDATE runs SET last_heartbeat_at = ?, updated_at = ? WHERE run_id = ?",
            (now, now, run_id),
        )
        self.connection.commit()

    def finish_task_attempt(
        self,
        *,
        run_id: str,
        task_id: int,
        attempt_id: int,
        task_status: str,
        run_status: str,
        exit_code: int | None,
        error_message: str | None,
        status_message: str | None,
    ) -> None:
        """Finalize task and run state after one attempt completes."""

        now = utc_now()
        completed_at = now if task_status != TASK_RUNNING else None
        # The task status and the run status intentionally move independently. A task can finish
        # as retryable or interrupted while the overall run remains resumable rather than terminal.
        self.connection.execute(
            """
            UPDATE tasks
            SET status = ?,
                updated_at = ?,
                completed_at = ?,
                last_heartbeat_at = ?
            WHERE task_id = ?
            """,
            (task_status, now, completed_at, now, task_id),
        )
        self.connection.execute(
            """
            UPDATE task_attempts
            SET status = ?,
                exit_code = ?,
                error_message = ?,
                completed_at = ?,
                heartbeat_at = ?
            WHERE attempt_id = ?
            """,
            (task_status, exit_code, error_message, completed_at, now, attempt_id),
        )
        self.connection.execute(
            """
            UPDATE runs
            SET status = ?,
                status_message = ?,
                current_task_name = CASE WHEN ? = ? THEN NULL ELSE current_task_name END,
                updated_at = ?,
                completed_at = CASE WHEN ? = ? THEN ? ELSE completed_at END,
                last_heartbeat_at = ?
            WHERE run_id = ?
            """,
            (
                run_status,
                status_message,
                run_status,
                RUN_SUCCEEDED,
                now,
                run_status,
                RUN_SUCCEEDED,
                now,
                now,
                run_id,
            ),
        )
        self.connection.commit()

    def register_artifact(
        self,
        *,
        run_id: str,
        task_id: int,
        attempt_id: int,
        artifact_kind: str,
        relative_path: str,
        manifest_json: str,
    ) -> None:
        """Persist a manifest pointer for a finalized artifact directory."""

        # The manifest is duplicated in SQLite so operators can inspect artifact lineage without
        # having to traverse the filesystem first, which is especially useful after interruptions.
        self.connection.execute(
            """
            INSERT INTO artifacts (
                run_id,
                task_id,
                attempt_id,
                artifact_kind,
                relative_path,
                manifest_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, task_id, attempt_id, artifact_kind, relative_path, manifest_json, utc_now()),
        )
        self.connection.commit()

    def set_run_status(self, run_id: str, status: str, status_message: str | None) -> None:
        """Update top-level run status outside of task finalization."""

        now = utc_now()
        # This method is used for coarse lifecycle transitions such as operator-requested pause
        # or final success, where there is no single task attempt acting as the source of truth.
        self.connection.execute(
            """
            UPDATE runs
            SET status = ?,
                status_message = ?,
                current_task_name = CASE WHEN ? = ? THEN NULL ELSE current_task_name END,
                updated_at = ?,
                completed_at = CASE WHEN ? = ? THEN ? ELSE completed_at END,
                last_heartbeat_at = ?
            WHERE run_id = ?
            """,
            (
                status,
                status_message,
                status,
                RUN_SUCCEEDED,
                now,
                status,
                RUN_SUCCEEDED,
                now,
                now,
                run_id,
            ),
        )
        self.connection.commit()

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=str(row["run_id"]),
            experiment_id=str(row["experiment_id"]),
            status=str(row["status"]),
            artifact_root=str(row["artifact_root"]),
            spec_hash=str(row["spec_hash"]),
            current_task_name=row["current_task_name"],
            status_message=row["status_message"],
        )

    def _row_to_task(self, row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=int(row["task_id"]),
            run_id=str(row["run_id"]),
            task_name=str(row["task_name"]),
            task_order=int(row["task_order"]),
            task_kind=str(row["task_kind"]),
            payload_json=str(row["payload_json"]),
            command_json=str(row["command_json"]),
            working_directory=row["working_directory"],
            status=str(row["status"]),
            artifact_dir=str(row["artifact_dir"]),
            last_heartbeat_at=row["last_heartbeat_at"],
        )
