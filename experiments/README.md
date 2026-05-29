# Experiments

This folder is the control plane for long-running AlphaSolitaire experiments.

The recommendation is to keep the experiments stack Python-based while treating the Java engine as a worker process and source of deterministic game data. That keeps training, analytics, and orchestration close to the current neural-network code without forcing Java to become the owner of Python model workflows.

Start with [DESIGN.md](DESIGN.md) for the proposed architecture, storage model, resumability rules, and phased proof-of-concept plan.

Run ID note:

- The example commands in this README use fixed `run_id` values for readability.
- Re-running `python -m experiments.src run ... --run-id <same-id>` resumes the existing persisted run and skips tasks that already succeeded.
- To replay an example from scratch, use a fresh `run_id` such as `phase3-demo-2`.

## Phase 1 Status

Phase 1 is now scaffolded.

Current capabilities:

- JSON experiment specs under `experiments/specs/`.
- A Python CLI at `python -m experiments.src`.
- A SQLite registry for runs, tasks, attempts, and artifact manifests.
- Resume-safe task execution keyed by `run_id`.
- Ignored runtime output under `experiments/runtime/`.

Current limitation:

- The default `collect`, `train`, and `evaluate` stages are intentionally `noop` tasks in Phase 1. They record payloads, create artifact manifests, and prove resumability, but they do not yet invoke the Java engine or neural-network trainer. That wiring begins in later phases.

Useful commands:

```bash
source neural-network/.venv/bin/activate

# Validate and inspect the demo spec
python -m experiments.src plan experiments/specs/mlp_phase1_demo.json

# Pause after one task to simulate interruption
python -m experiments.src run experiments/specs/mlp_phase1_demo.json --run-id phase1-demo --max-tasks 1

# Resume the same run without duplicating completed tasks
python -m experiments.src run experiments/specs/mlp_phase1_demo.json --run-id phase1-demo

# Inspect persisted run state
python -m experiments.src status phase1-demo
```

## Phase 2 Status

Phase 2 is now wired for endgame data collection through the Java engine.

Current Phase 2 capabilities:

- A `collection` spec section that expands into real shard tasks.
- Built-in `endgame_collect_shard` task execution through `./gradlew test`.
- Per-shard stdout, stderr, command, manifest, and collection summary artifacts.
- Per-shard reproducible random seeds via `-Dendgame.random.seed=<seed>`.
- Safe capture of generated `episode*.log` files into immutable artifact directories.
- Resume behavior that skips completed shards and continues the remaining shard tasks.

Example Phase 2 commands:

```bash
source neural-network/.venv/bin/activate

# Inspect the sharded collection plan
python -m experiments.src plan experiments/specs/endgame_phase2_demo.json --run-id phase2-demo

# Run one shard and stop
python -m experiments.src run experiments/specs/endgame_phase2_demo.json --run-id phase2-demo --max-tasks 1

# Resume the remaining shard(s)
python -m experiments.src run experiments/specs/endgame_phase2_demo.json --run-id phase2-demo

# Recover a stale run after an interrupted or crashed attempt
python -m experiments.src run experiments/specs/endgame_phase2_demo.json --run-id phase2-demo --stale-after-seconds 1

# Inspect final run state
python -m experiments.src status phase2-demo
```

## Phase 3 Status

Phase 3 is now wired for resumable MLP training through the Python trainer.

Current Phase 3 capabilities:

- A `training` spec section that expands into a real `policy_value_train` task.
- Periodic epoch checkpoints with model state, optimizer state, RNG state, and metrics.
- Automatic resume from the latest valid checkpoint created by a prior attempt of the same task.
- Per-attempt JSONL epoch metrics and a structured training summary artifact.
- Validation support through an intentional interrupt path for checkpoint-resume testing.

Example Phase 3 commands:

```bash
source neural-network/.venv/bin/activate

# Inspect the training plan
python -m experiments.src plan experiments/specs/mlp_phase3_demo.json --run-id phase3-demo

# Run until the demo intentionally interrupts after epoch 1
python -m experiments.src run experiments/specs/mlp_phase3_demo.json --run-id phase3-demo

# Resume from the checkpoint written by the first attempt
python -m experiments.src run experiments/specs/mlp_phase3_demo.json --run-id phase3-demo --stale-after-seconds 1

# Inspect final run state
python -m experiments.src status phase3-demo
```

## Phase 4 Status

Phase 4 is now wired for sharded AlphaSolitaire evaluation and generated analysis artifacts.

Current Phase 4 capabilities:

- An `evaluation` spec section that expands into restartable `alpha_level_eval_shard` tasks.
- Driver-managed model service startup using the evaluation checkpoint for each shard.
- Deterministic evaluation game blocks via `seed_start`, `seed_end`, and `games_per_shard`.
- Structured per-shard evaluation summaries captured as JSON artifacts.
- A follow-on `evaluation_report` task that emits DuckDB, Parquet, JSONL, SQL, and markdown outputs.
- Cross-run comparison views and artifacts that include completed historical evaluation runs, not just the current run.
- Resume behavior that skips completed evaluation shards and only reruns missing work.

Example Phase 4 commands:

```bash
source neural-network/.venv/bin/activate

# Inspect the evaluation plan
python -m experiments.src plan experiments/specs/alpha_phase4_demo.json --run-id phase4-demo

# Run the sharded evaluation and report task
python -m experiments.src run experiments/specs/alpha_phase4_demo.json --run-id phase4-demo

# Inspect final run state
python -m experiments.src status phase4-demo
```

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

Planned responsibilities for this folder:

- Experiment specs and sweep definitions.
- Driver and scheduler code for long-running jobs.
- Experiment registry metadata.
- Structured analysis outputs and generated summaries.

Planned non-responsibilities for this folder:

- Core Solitaire gameplay logic. That stays in `engine/`.
- Model implementations and training primitives. Those stay in `neural-network/`.
