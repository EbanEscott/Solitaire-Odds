# Experiments

This folder is the control plane for long-running AlphaSolitaire experiments.

The recommendation is to keep the experiments stack Python-based while treating the Java engine as a worker process and source of deterministic game data. That keeps training, analytics, and orchestration close to the current neural-network code without forcing Java to become the owner of Python model workflows.

Start with [DESIGN.md](DESIGN.md) for the current architecture, storage model, resumability rules, and implementation shape.
See [RESULTS.md](RESULTS.md) for result presentation, runtime housekeeping, and promotion rules.
Use notebooks under `experiments/notebooks/` for active research questions, goals, and experiment notes.

Run ID note:

- The example commands in this README use fixed `run_id` values for readability.
- Re-running `python -m experiments.src run ... --run-id <same-id>` resumes the existing persisted run and skips tasks that already succeeded.
- To replay an example from scratch, use a fresh `run_id` such as `resume-demo-2`.

Dataset sourcing note:

- `dataset.kind=archived_episode_logs` reuses existing archived logs listed in `dataset.sources`.
- `dataset.kind=run_collection_episode_logs` trains from successful `endgame_collect_shard` artifacts produced earlier in the same run.

## Current Status

This control plane currently supports:

- JSON experiment specs under `experiments/specs/`.
- A Python CLI at `python -m experiments.src`.
- A SQLite registry for runs, tasks, attempts, artifact manifests, and resumability state.
- Resume-safe execution keyed by `run_id`, with task-level heartbeats and stale-task recovery.
- Real collection, training, evaluation, reporting, and maintenance commands.
- Ignored runtime output under `experiments/runtime/`.

## Current Capabilities

### Planning And Resume

- `plan` validates a spec and expands it into the concrete task graph for a run.
- `run` creates or resumes a persisted run and skips tasks that already succeeded.
- `status` shows run and task state from the registry.
- `ui` starts a thin local read-only web UI for overall run progress, current task state, and log drill-down.
- `run` prints a coarse preflight runtime estimate before launch and prompts for confirmation unless `--yes` is supplied.
- `run --live-output` prints run and task progress headings, the active log level, rolling ETA context, and a summarized live view of child task output while still writing the full per-attempt stdout and stderr log files.
- `--log-level DEBUG` restores the raw child-process stream when you want full detail; the default `INFO` view keeps the summarized operator-focused output and filters a few known noisy Java/Gradle stderr lines.

### Collection

- A `collection` spec section expands into real `endgame_collect_shard` tasks.
- Collection runs through `./gradlew test` against the Java engine generator.
- Each shard captures stdout, stderr, command metadata, manifest data, and a collection summary artifact.
- Generated `episode*.log` files are moved into immutable attempt artifact directories.
- Multi-shard collection supports reproducible random seeds through `-Dendgame.random.seed=<seed>`.

### Training

- A `training` spec section expands into a real `policy_value_train` task.
- `mlp_policy_value` and `gnn_policy_value` are both supported through the same driver contract.
- Training datasets can come from archived logs or from collection artifacts produced earlier in the same run.
- Training writes resumable checkpoints, epoch metrics, and a structured training summary artifact.
- Resume uses the latest valid checkpoint produced by earlier attempts of the same training task.

### Evaluation And Reporting

- An `evaluation` spec section expands into restartable `alpha_level_eval_shard` tasks plus a follow-on `evaluation_report` task.
- Use `evaluation.level` for one target level or `evaluation.levels` for a full level ladder in one run.
- Evaluation starts the model service with the resolved checkpoint for each shard.
- Deterministic game blocks are controlled through `seed_start`, `seed_end`, and `games_per_shard`.
- Per-shard summaries are captured as JSON artifacts.
- Reporting emits DuckDB, Parquet, JSONL, SQL, and markdown outputs.
- Cross-run comparison views include successful completed historical runs, not just the current run.

The report task writes artifacts such as:

- `evaluation_shards.jsonl`
- `evaluation_shards_all_runs.jsonl`
- `evaluation_shards.parquet`
- `evaluation_rollups.parquet`
- `evaluation_rollups_all_runs.parquet`
- `evaluation_run_comparison.parquet`
- `evaluation.duckdb`
- `evaluation_queries.sql`
- `evaluation_report.md`

The DuckDB output now includes both the current-run rollups and cross-run comparison views:

- `evaluation_rollups_current_run`
- `evaluation_rollups_all_runs`
- `evaluation_run_comparison`

For result presentation, promotion guidance, and runtime housekeeping, see [RESULTS.md](RESULTS.md).

### Operations And Runtime Health

- `doctor` writes a markdown runtime health dashboard and a JSON companion report.
- `doctor --recover-stale` can reclaim stale running tasks across all runs before writing the report.
- The health report highlights recent runs, stale tasks, retryable or interrupted tasks, and missing artifacts.
- `cleanup` applies conservative retention rules to stale `.attempt-*.tmp` directories and aged files under `experiments/runtime/work`.
- Archived attempt stdout and stderr paths are persisted in their finalized locations for later inspection.
- SQLite remains the recommended registry store for the current single-machine control plane.

## Example Workflows

```bash
source neural-network/.venv/bin/activate

# Validate and inspect a minimal resume demo
python -m experiments.src plan experiments/specs/mlp_phase1_demo.json

# Pause after one task to simulate interruption
python -m experiments.src run experiments/specs/mlp_phase1_demo.json --run-id resume-demo --max-tasks 1

# Resume the same run without duplicating completed tasks
python -m experiments.src run experiments/specs/mlp_phase1_demo.json --run-id resume-demo

# Run sharded collection through the Java engine
python -m experiments.src run experiments/specs/endgame_phase2_demo.json --run-id collection-demo --yes

# Train from archived logs
python -m experiments.src run experiments/specs/mlp_phase3_demo.json --run-id archived-train-demo --yes

# Collect fresh logs, train on those collected artifacts, and evaluate in one run
python -m experiments.src run experiments/specs/mlp_fresh_data_demo.json --run-id fresh-data-demo --yes

# Run the GNN training + evaluation workflow
python -m experiments.src run experiments/specs/gnn_phase5_demo.json --run-id gnn-demo --yes

# Run the evaluation + reporting workflow
python -m experiments.src run experiments/specs/alpha_phase4_demo.json --run-id evaluation-demo --yes

# Inspect run state
python -m experiments.src status fresh-data-demo

# Start the thin local UI and open http://127.0.0.1:8765/runs
python -m experiments.src ui

# Write a runtime health dashboard covering all runs
python -m experiments.src doctor

# Accept the preflight estimate without a prompt for unattended launches
python -m experiments.src run experiments/specs/alpha_phase4_demo.json --run-id unattended-demo --yes

# Stream the underlying task output while the run is executing
python -m experiments.src run experiments/specs/baseline_mlp_level2_dev.json --run-id baseline-verbose-demo --yes --live-output

# Override the JSON companion output path when automation wants a fixed location
python -m experiments.src doctor --json-output experiments/runtime/work/runtime_health_custom.json

# Reclaim stale tasks across all runs before writing the report
python -m experiments.src doctor --recover-stale --stale-after-seconds 300

# Inspect cleanup candidates without deleting anything
python -m experiments.src cleanup --dry-run

# Apply the default retention policy
python -m experiments.src cleanup
```

The default `doctor` output now writes both:

- `experiments/runtime/work/runtime_health_report.md`
- `experiments/runtime/work/runtime_health_report.json`

The preflight runtime estimate is intentionally approximate. It uses successful historical attempt timings for similar task kinds and payloads when that data exists, then falls back to broader task-kind medians when it does not.

Unattended maintenance note:

Cron example:

```bash
15 * * * * cd /Users/ebo/Code/Solitaire-Odds && source neural-network/.venv/bin/activate && python -m experiments.src doctor --recover-stale && python -m experiments.src cleanup >> experiments/runtime/work/maintenance.log 2>&1
```

Launchd example:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>com.solitaireodds.experiments.maintenance</string>
	<key>ProgramArguments</key>
	<array>
		<string>/bin/zsh</string>
		<string>-lc</string>
		<string>cd /Users/ebo/Code/Solitaire-Odds &amp;&amp; source neural-network/.venv/bin/activate &amp;&amp; python -m experiments.src doctor --recover-stale &amp;&amp; python -m experiments.src cleanup</string>
	</array>
	<key>StartCalendarInterval</key>
	<dict>
		<key>Minute</key>
		<integer>15</integer>
	</dict>
	<key>StandardOutPath</key>
	<string>/Users/ebo/Code/Solitaire-Odds/experiments/runtime/work/maintenance.log</string>
	<key>StandardErrorPath</key>
	<string>/Users/ebo/Code/Solitaire-Odds/experiments/runtime/work/maintenance.log</string>
</dict>
</plist>
```

If you launch experiment runs from other unattended entry points, pass `--yes` so the preflight confirmation does not block waiting for operator input.

Current responsibilities for this folder:

- Experiment specs and sweep definitions.
- Driver and scheduler code for long-running jobs.
- Experiment registry metadata.
- Structured analysis outputs and generated summaries.

Current non-responsibilities for this folder:

- Core Solitaire gameplay logic. That stays in `engine/`.
- Model implementations and training primitives. Those stay in `neural-network/`.
