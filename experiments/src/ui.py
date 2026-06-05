"""Read-only local web UI for the experiments control plane."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from .driver import RuntimeLayout, _estimate_remaining_runtime, _format_duration_seconds, _parse_utc_timestamp
from .registry import AttemptRecord, ArtifactRecord, ExperimentRegistry, RunRecord, TASK_SUCCEEDED, TaskRecord
from .specs import REPO_ROOT


LOG = logging.getLogger(__name__)
TEXT_VIEW_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".sql", ".txt", ".csv"}


@dataclass(frozen=True)
class UiConfig:
    """Runtime configuration for the local experiments UI."""

    layout: RuntimeLayout
    host: str
    port: int
    refresh_seconds: int
    limit_runs: int
    log_tail_lines: int
    file_line_limit: int = 2000


def _task_stage_label(task_kind: str) -> str:
    """Map one task kind to a short stage label for the operator UI."""

    return {
        "endgame_collect_shard": "Collect",
        "policy_value_train": "Train",
        "alpha_level_eval_shard": "Evaluate",
        "evaluation_report": "Report",
    }.get(task_kind, task_kind.replace("_", " ").title())


def _format_age(timestamp_text: str | None) -> str:
    """Render a compact relative age from one stored UTC timestamp."""

    timestamp = _parse_utc_timestamp(timestamp_text)
    if timestamp is None:
        return "-"

    total_seconds = max(0, int((datetime.now(timezone.utc) - timestamp).total_seconds()))
    if total_seconds < 60:
        return f"{total_seconds}s ago"
    if total_seconds < 3600:
        return f"{total_seconds // 60}m ago"
    if total_seconds < 86400:
        return f"{total_seconds // 3600}h ago"
    return f"{total_seconds // 86400}d ago"


def _format_timestamp(timestamp_text: str | None) -> str:
    """Normalize one stored UTC timestamp for display."""

    timestamp = _parse_utc_timestamp(timestamp_text)
    if timestamp is None:
        return "-"
    return timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _resolve_attempt_log_path(attempt: AttemptRecord | None, filename: str) -> Path | None:
    """Resolve one archived attempt log path when it exists."""

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


def _latest_attempt(registry: ExperimentRegistry, task_id: int) -> AttemptRecord | None:
    """Return the newest attempt for one task when present."""

    attempts = registry.list_task_attempts(task_id)
    return attempts[0] if attempts else None


def _read_tail(path: Path | None, max_lines: int) -> str:
    """Read the tail of one text file for compact log views."""

    if path is None or not path.exists():
        return ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=max_lines))


def _read_text(path: Path, max_lines: int) -> str:
    """Read a bounded number of lines from one text artifact."""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = list(deque(handle, maxlen=max_lines))
    return "".join(lines)


def _resolve_repo_file(path_text: str) -> Path:
    """Resolve one repo-relative file path while preventing traversal outside the workspace."""

    candidate = (REPO_ROOT / path_text).resolve()
    candidate.relative_to(REPO_ROOT)
    return candidate


def _estimated_remaining_text(registry: ExperimentRegistry, tasks: list[TaskRecord]) -> str:
    """Estimate remaining runtime using the same history model as the CLI."""

    remaining_tasks: list[tuple[str, str, dict[str, Any]]] = []
    for task in tasks:
        if task.status == TASK_SUCCEEDED:
            continue
        remaining_tasks.append((task.task_name, task.task_kind, json.loads(task.payload_json or "{}")))

    preflight = _estimate_remaining_runtime(registry, remaining_tasks)
    if preflight.remaining_task_count == 0:
        return "0s"
    if preflight.unknown_task_count == 0:
        return _format_duration_seconds(preflight.estimated_seconds)
    if preflight.known_task_count == 0:
        return f"unknown ({preflight.unknown_task_count} task(s) without timing data)"
    return (
        f"at least {_format_duration_seconds(preflight.estimated_seconds)} plus "
        f"{preflight.unknown_task_count} task(s) without timing data"
    )


def _progress_percent(completed_tasks: int, total_tasks: int) -> int:
    """Convert completed task counts into a bounded integer percentage."""

    if total_tasks <= 0:
        return 0
    return max(0, min(100, int(round((completed_tasks / total_tasks) * 100))))


def _artifact_links_for_attempt(
    registry: ExperimentRegistry,
    task: TaskRecord,
    attempt: AttemptRecord | None,
) -> list[tuple[str, str]]:
    """Collect the most useful text artifact links for one task attempt."""

    if attempt is None:
        return []

    links: list[tuple[str, str]] = []
    for artifact in registry.list_artifacts(task.run_id):
        if artifact.task_id != task.task_id or artifact.attempt_id != attempt.attempt_id:
            continue
        artifact_path = _resolve_repo_file(artifact.relative_path)
        if artifact_path.is_file() and artifact_path.suffix in TEXT_VIEW_SUFFIXES:
            links.append((artifact.artifact_kind, f"/file?path={quote(artifact.relative_path)}"))

    attempt_dir = Path(attempt.artifact_dir)
    if attempt_dir.exists():
        for path in sorted(attempt_dir.iterdir()):
            if not path.is_file() or path.suffix not in TEXT_VIEW_SUFFIXES:
                continue
            relative_path = str(path.resolve().relative_to(REPO_ROOT))
            links.append((path.name, f"/file?path={quote(relative_path)}"))

    unique_links: dict[str, str] = {}
    for label, href in links:
        unique_links.setdefault(label, href)
    return sorted(unique_links.items())


def _run_summary(registry: ExperimentRegistry, run: RunRecord) -> dict[str, Any]:
    """Assemble one run summary for the overview and detail pages."""

    tasks = registry.list_tasks(run.run_id)
    completed_tasks = sum(1 for task in tasks if task.status == TASK_SUCCEEDED)
    status_counts = Counter(task.status for task in tasks)
    task_by_name = {task.task_name: task for task in tasks}
    current_task = task_by_name.get(run.current_task_name or "")
    current_stage = _task_stage_label(current_task.task_kind) if current_task is not None else "-"

    return {
        "run": run,
        "tasks": tasks,
        "completed_tasks": completed_tasks,
        "total_tasks": len(tasks),
        "remaining_tasks": max(0, len(tasks) - completed_tasks),
        "progress_percent": _progress_percent(completed_tasks, len(tasks)),
        "current_stage": current_stage,
        "eta_text": _estimated_remaining_text(registry, tasks),
        "elapsed_text": _format_age(run.started_at).replace(" ago", "") if run.started_at else "-",
        "heartbeat_age": _format_age(run.last_heartbeat_at),
        "updated_age": _format_age(run.updated_at),
        "status_counts": status_counts,
    }


def _base_styles() -> str:
    """Return the shared CSS for the local operator UI."""

    return """
        :root {
            --paper: #f2f2f2;
            --panel: #ffffff;
            --ink: #111111;
            --muted: #555555;
            --line: #d0d0d0;
            --accent: #111111;
            --accent-soft: #e8e8e8;
            --good: #111111;
            --warn: #333333;
            --bad: #000000;
            --mono: Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            --serif: Georgia, "Times New Roman", Times, serif;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: linear-gradient(180deg, #fcfcfc 0%, var(--paper) 100%);
            color: var(--ink);
            font-family: var(--serif);
        }
        a { color: var(--accent); text-decoration: none; }
        a:hover { text-decoration: underline; }
        .shell { max-width: 1380px; margin: 0 auto; padding: 24px; }
        .masthead {
            display: flex; justify-content: space-between; align-items: end; gap: 16px;
            padding-bottom: 16px; border-bottom: 2px solid var(--line); margin-bottom: 20px;
        }
        .masthead h1 { margin: 0; font-size: 2rem; letter-spacing: 0.02em; }
        .masthead .sub { color: var(--muted); font-size: 0.95rem; }
        .grid { display: grid; gap: 16px; }
        .grid.cards { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
        .grid.focus { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
        }
        .panel h2, .panel h3 { margin-top: 0; }
        .metric-label { color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; }
        .metric-value { font-size: 1.5rem; margin-top: 4px; }
        .badge {
            display: inline-block; border-radius: 4px; padding: 4px 10px; font-size: 0.8rem;
            font-family: var(--mono); text-transform: uppercase; letter-spacing: 0.08em;
            border: 1px solid currentColor;
        }
        .badge.succeeded, .badge.running, .badge.pending, .badge.interrupted, .badge.retryable, .badge.failed {
            color: var(--ink);
            background: #ffffff;
        }
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff;
        }
        th, td {
            padding: 12px 10px;
            border-bottom: 1px solid var(--line);
            vertical-align: top;
        }
        th {
            text-align: left;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            background: #f6f6f6;
            border-bottom: 1px solid #bfbfbf;
        }
        tbody tr:nth-child(even) td { background: #fafafa; }
        tbody tr:hover td { background: #f0f0f0; }
        tbody tr:last-child td { border-bottom: 0; }
        td .progress { margin-top: 6px; }
        .mono { font-family: var(--mono); }
        .progress {
            width: 100%; height: 10px; background: #e5e5e5; border-radius: 3px; overflow: hidden;
        }
        .progress > span { display: block; height: 100%; background: #111111; }
        .split { display: grid; gap: 16px; grid-template-columns: 1.4fr 1fr; }
        pre {
            margin: 0; white-space: pre-wrap; word-break: break-word; font-family: var(--mono);
            background: #121212; color: #f4f4f4; padding: 16px; border-radius: 8px; overflow: auto;
            max-height: 520px;
        }
        .toolbar { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
        .toolbar a { font-family: var(--mono); }
        ul.link-list { list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }
        .run-focus-title { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; }
        .run-focus-title p { margin: 0; color: var(--muted); }
        .run-focus-card h3 { margin-bottom: 8px; }
        .run-focus-meta { display: grid; gap: 6px; margin: 12px 0; }
        .run-focus-meta div { color: var(--muted); }
        .run-focus-actions { margin-top: 14px; }
        .run-focus-actions a {
            display: inline-block;
            border: 1px solid var(--ink);
            border-radius: 4px;
            padding: 7px 10px;
            font-family: var(--mono);
            background: #fff;
        }
        @media (max-width: 980px) {
            .split { grid-template-columns: 1fr; }
        }
    """


def _status_badge(status: str) -> str:
    """Render one status value as a colored badge."""

    safe_status = html.escape(status)
    return f'<span class="badge {safe_status}">{safe_status}</span>'


def _render_page(title: str, body: str, *, refresh_seconds: int | None) -> str:
    """Wrap one page body in the shared HTML shell."""

    refresh_meta = (
        f'<meta http-equiv="refresh" content="{refresh_seconds}">' if refresh_seconds is not None else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh_meta}
  <title>{html.escape(title)}</title>
  <style>{_base_styles()}</style>
</head>
<body>
  <div class="shell">
    <div class="masthead">
      <div>
        <h1>Experiments UI</h1>
        <div class="sub">Thin local operator view over the registry and artifact files.</div>
      </div>
      <div class="sub mono">{html.escape(title)}</div>
    </div>
    {body}
  </div>
</body>
</html>"""


def _overview_body(config: UiConfig, registry: ExperimentRegistry) -> str:
        """Build the runs overview page."""

        runs = registry.list_runs()[: config.limit_runs]
        summaries = [_run_summary(registry, run) for run in runs]
        run_status_counts = Counter(summary["run"].status for summary in summaries)
        active_summaries = [summary for summary in summaries if summary["run"].status != "succeeded"]
        historical_summaries = [summary for summary in summaries if summary["run"].status == "succeeded"]

        cards = "".join(
                f'''
                <div class="panel">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value mono">{value}</div>
                </div>
                '''
                for label, value in (
                        ("Runs", len(summaries)),
                        ("Running", run_status_counts.get("running", 0)),
                        ("Interrupted", run_status_counts.get("interrupted", 0)),
                        ("Succeeded", run_status_counts.get("succeeded", 0)),
                )
        )

        rows = "".join(
                f"""
                <tr>
                    <td class="mono"><a href="/runs/{quote(summary['run'].run_id)}">{html.escape(summary['run'].run_id)}</a></td>
                    <td>{html.escape(summary['run'].experiment_id)}</td>
                    <td>{_status_badge(summary['run'].status)}</td>
                    <td>{html.escape(summary['current_stage'])}</td>
                    <td>{html.escape(summary['run'].current_task_name or '-')}</td>
                    <td>
                        <div class="mono">{summary['completed_tasks']}/{summary['total_tasks']}</div>
                        <div class="progress"><span style="width: {summary['progress_percent']}%"></span></div>
                    </td>
                    <td>{html.escape(summary['elapsed_text'])}</td>
                    <td>{html.escape(summary['eta_text'])}</td>
                    <td>{html.escape(summary['heartbeat_age'])}</td>
                    <td>{html.escape(summary['run'].status_message or '-')}</td>
                </tr>
                """
                for summary in historical_summaries
        )

        active_cards = "".join(
                f"""
                <div class="panel run-focus-card">
                    <div class="run-focus-title">
                        <h3 class="mono"><a href="/runs/{quote(summary['run'].run_id)}">{html.escape(summary['run'].run_id)}</a></h3>
                        {_status_badge(summary['run'].status)}
                    </div>
                    <p>{html.escape(summary['run'].experiment_id)}</p>
                    <div class="run-focus-meta mono">
                        <div>Stage: {html.escape(summary['current_stage'])}</div>
                        <div>Current task: {html.escape(summary['run'].current_task_name or '-')}</div>
                        <div>Progress: {summary['completed_tasks']}/{summary['total_tasks']}</div>
                        <div>Elapsed: {html.escape(summary['elapsed_text'])}</div>
                        <div>Remaining: {html.escape(summary['eta_text'])}</div>
                        <div>Heartbeat: {html.escape(summary['heartbeat_age'])}</div>
                    </div>
                    <div class="progress"><span style="width: {summary['progress_percent']}%"></span></div>
                    <div class="run-focus-actions"><a href="/runs/{quote(summary['run'].run_id)}">Open run detail</a></div>
                </div>
                """
                for summary in active_summaries
        )

        history_markup = (
                f"""
            <div class="panel" style="margin-top: 18px;">
                <h2>Recent History</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Run</th>
                            <th>Experiment</th>
                            <th>Status</th>
                            <th>Stage</th>
                            <th>Current Task</th>
                            <th>Progress</th>
                            <th>Elapsed</th>
                            <th>Remaining</th>
                            <th>Heartbeat</th>
                            <th>Message</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        """
                if historical_summaries
                else ""
        )

        return f"""
            <div class="grid cards">{cards}</div>
            <div class="panel" style="margin-top: 18px;">
                <div class="run-focus-title">
                    <h2>Current Focus</h2>
                    <p>{'Click a run to open its detail page.' if active_summaries else 'No active or interrupted runs at the moment.'}</p>
                </div>
                <div class="grid focus">{active_cards or '<div class="mono">No active runs.</div>'}</div>
            </div>
            {history_markup}
        """


def _run_detail_body(config: UiConfig, registry: ExperimentRegistry, run: RunRecord) -> str:
    """Build the detail page for one run."""

    summary = _run_summary(registry, run)
    tasks: list[TaskRecord] = summary["tasks"]

    cards = "".join(
        f'''
        <div class="panel">
          <div class="metric-label">{label}</div>
          <div class="metric-value mono">{value}</div>
        </div>
        '''
        for label, value in (
            ("Status", run.status),
            ("Current Stage", summary["current_stage"]),
            ("Progress", f"{summary['completed_tasks']}/{summary['total_tasks']}"),
            ("Elapsed", summary["elapsed_text"]),
            ("Remaining", summary["eta_text"]),
            ("Heartbeat", summary["heartbeat_age"]),
        )
    )

    task_rows = []
    for task in tasks:
        attempt = _latest_attempt(registry, task.task_id)
        stdout_href = f"/attempts/{attempt.attempt_id}/log/stdout" if attempt is not None else None
        stderr_href = f"/attempts/{attempt.attempt_id}/log/stderr" if attempt is not None else None
        task_rows.append(
            f"""
            <tr>
              <td class="mono">{task.task_order:02d}</td>
              <td>{html.escape(_task_stage_label(task.task_kind))}</td>
              <td><a href="/runs/{quote(run.run_id)}/tasks/{task.task_id}">{html.escape(task.task_name)}</a></td>
              <td>{_status_badge(task.status)}</td>
              <td>{html.escape(_format_age(task.last_heartbeat_at))}</td>
              <td class="mono">{attempt.attempt_number if attempt is not None else '-'}</td>
              <td>
                {'<a href="' + stdout_href + '">stdout</a>' if stdout_href else '-'}
                {' | <a href="' + stderr_href + '">stderr</a>' if stderr_href else ''}
              </td>
            </tr>
            """
        )

    counts_markup = "".join(
        f'<span class="badge {html.escape(status)}">{html.escape(status)} {count}</span>'
        for status, count in sorted(summary["status_counts"].items())
    )

    return f"""
      <div class="toolbar">
        <a href="/runs">All runs</a>
      </div>
      <div class="grid cards">{cards}</div>
      <div class="panel" style="margin-top: 18px;">
        <h2>{html.escape(run.run_id)}</h2>
        <div class="toolbar">{counts_markup}</div>
        <div class="progress"><span style="width: {summary['progress_percent']}%"></span></div>
        <p class="mono" style="margin-top: 12px;">Current task: {html.escape(run.current_task_name or '-')} | Updated: {html.escape(_format_timestamp(run.updated_at))}</p>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Stage</th>
              <th>Task</th>
              <th>Status</th>
              <th>Heartbeat</th>
              <th>Attempt</th>
              <th>Logs</th>
            </tr>
          </thead>
          <tbody>{''.join(task_rows)}</tbody>
        </table>
      </div>
    """


def _task_detail_body(config: UiConfig, registry: ExperimentRegistry, run: RunRecord, task: TaskRecord) -> str:
    """Build the drill-down page for one task and its latest attempt."""

    attempt = _latest_attempt(registry, task.task_id)
    stdout_path = _resolve_attempt_log_path(attempt, "stdout.log")
    stderr_path = _resolve_attempt_log_path(attempt, "stderr.log")
    artifact_links = _artifact_links_for_attempt(registry, task, attempt)

    return f"""
      <div class="toolbar">
        <a href="/runs">All runs</a>
        <a href="/runs/{quote(run.run_id)}">{html.escape(run.run_id)}</a>
      </div>
      <div class="split">
        <div class="grid">
          <div class="panel">
            <h2>{html.escape(task.task_name)}</h2>
            <p class="mono">Stage: {html.escape(_task_stage_label(task.task_kind))}</p>
            <p class="mono">Status: {html.escape(task.status)}</p>
            <p class="mono">Latest attempt: {attempt.attempt_number if attempt is not None else '-'}</p>
            <p class="mono">Started: {html.escape(_format_timestamp(attempt.started_at if attempt is not None else None))}</p>
            <p class="mono">Heartbeat: {html.escape(_format_timestamp(attempt.heartbeat_at if attempt is not None else None))}</p>
            <p class="mono">Exit code: {attempt.exit_code if attempt is not None and attempt.exit_code is not None else '-'}</p>
            <p>{html.escape(attempt.error_message if attempt is not None and attempt.error_message else '-')}</p>
          </div>
          <div class="panel">
            <h3>Artifact Links</h3>
            <ul class="link-list">
              {''.join(f'<li><a href="{href}">{html.escape(label)}</a></li>' for label, href in artifact_links) or '<li>-</li>'}
            </ul>
          </div>
        </div>
        <div class="grid">
          <div class="panel">
            <h3>Stdout Tail</h3>
            <div class="toolbar"><a href="/attempts/{attempt.attempt_id}/log/stdout">Open full stdout view</a></div>
            <pre>{html.escape(_read_tail(stdout_path, config.log_tail_lines) or 'No stdout available.')}</pre>
          </div>
          <div class="panel">
            <h3>Stderr Tail</h3>
            <div class="toolbar"><a href="/attempts/{attempt.attempt_id}/log/stderr">Open full stderr view</a></div>
            <pre>{html.escape(_read_tail(stderr_path, config.log_tail_lines) or 'No stderr available.')}</pre>
          </div>
        </div>
      </div>
    """


def _log_body(config: UiConfig, registry: ExperimentRegistry, attempt_id: int, stream_name: str) -> str:
    """Build the full log page for one attempt stream."""

    row = registry.connection.execute(
        "SELECT * FROM task_attempts WHERE attempt_id = ?",
        (attempt_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"attempt not found: {attempt_id}")

    attempt = registry._row_to_attempt(row)
    filename = "stdout.log" if stream_name == "stdout" else "stderr.log"
    path = _resolve_attempt_log_path(attempt, filename)
    content = _read_tail(path, max(config.log_tail_lines * 10, config.file_line_limit)) or "No log output available."

    return f"""
      <div class="toolbar">
        <a href="/runs">All runs</a>
      </div>
      <div class="panel">
        <h2 class="mono">Attempt {attempt_id} {html.escape(stream_name)}</h2>
        <pre>{html.escape(content)}</pre>
      </div>
    """


def _file_body(config: UiConfig, path: Path, path_text: str) -> str:
    """Build the file viewer page for one text artifact."""

    return f"""
      <div class="toolbar">
        <a href="/runs">All runs</a>
      </div>
      <div class="panel">
        <h2 class="mono">{html.escape(path_text)}</h2>
        <pre>{html.escape(_read_text(path, config.file_line_limit))}</pre>
      </div>
    """


def _build_handler(config: UiConfig) -> type[BaseHTTPRequestHandler]:
    """Build the HTTP handler bound to one UI runtime configuration."""

    class UiHandler(BaseHTTPRequestHandler):
        """Serve the thin local operator UI backed by the experiments registry."""

        def do_GET(self) -> None:  # noqa: N802
            """Serve one read-only UI page."""

            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            try:
                if path in {"/", "/runs"}:
                    with ExperimentRegistry(config.layout.database_path) as registry:
                        self._send_html(_render_page("Runs Overview", _overview_body(config, registry), refresh_seconds=config.refresh_seconds))
                    return

                if path.startswith("/runs/") and "/tasks/" not in path:
                    run_id = path.split("/", 2)[2]
                    with ExperimentRegistry(config.layout.database_path) as registry:
                        run = registry.get_run(run_id)
                        body = _run_detail_body(config, registry, run)
                        self._send_html(_render_page(f"Run {run_id}", body, refresh_seconds=config.refresh_seconds))
                    return

                if path.startswith("/runs/") and "/tasks/" in path:
                    parts = path.strip("/").split("/")
                    run_id = parts[1]
                    task_id = int(parts[3])
                    with ExperimentRegistry(config.layout.database_path) as registry:
                        run = registry.get_run(run_id)
                        task = next((item for item in registry.list_tasks(run_id) if item.task_id == task_id), None)
                        if task is None:
                            raise KeyError(f"task not found: {task_id}")
                        body = _task_detail_body(config, registry, run, task)
                        self._send_html(_render_page(f"Task {task.task_name}", body, refresh_seconds=config.refresh_seconds))
                    return

                if path.startswith("/attempts/") and "/log/" in path:
                    parts = path.strip("/").split("/")
                    attempt_id = int(parts[1])
                    stream_name = parts[3]
                    if stream_name not in {"stdout", "stderr"}:
                        raise ValueError(f"unsupported log stream: {stream_name}")
                    with ExperimentRegistry(config.layout.database_path) as registry:
                        body = _log_body(config, registry, attempt_id, stream_name)
                        self._send_html(_render_page(f"Attempt {attempt_id} {stream_name}", body, refresh_seconds=config.refresh_seconds))
                    return

                if path == "/file":
                    path_text = query.get("path", [""])[0]
                    if not path_text:
                        raise ValueError("missing path query parameter")
                    artifact_path = _resolve_repo_file(path_text)
                    if not artifact_path.exists() or artifact_path.suffix not in TEXT_VIEW_SUFFIXES:
                        raise FileNotFoundError(path_text)
                    body = _file_body(config, artifact_path, path_text)
                    self._send_html(_render_page(path_text, body, refresh_seconds=None))
                    return

                if path == "/health":
                    self._send_text("ok\n")
                    return

                self.send_error(HTTPStatus.NOT_FOUND, "page not found")
            except FileNotFoundError as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            except KeyError as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            except ValueError as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def log_message(self, format: str, *args: object) -> None:
            """Send request logging through the experiments logger."""

            LOG.debug("UI %s - %s", self.address_string(), format % args)

        def _send_html(self, payload: str) -> None:
            """Write one HTML response."""

            encoded = payload.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_text(self, payload: str) -> None:
            """Write one plain-text response."""

            encoded = payload.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return UiHandler


def run_ui_server(*, host: str, port: int, refresh_seconds: int, limit_runs: int, log_tail_lines: int) -> int:
    """Start the thin local operator UI and block until interrupted."""

    config = UiConfig(
        layout=RuntimeLayout.create(),
        host=host,
        port=port,
        refresh_seconds=refresh_seconds,
        limit_runs=limit_runs,
        log_tail_lines=log_tail_lines,
    )

    server = ThreadingHTTPServer((host, port), _build_handler(config))
    LOG.info("Experiments UI listening on http://%s:%s/runs", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Experiments UI stopped.")
    finally:
        server.server_close()
    return 0