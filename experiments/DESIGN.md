# Experiments Design

## Current Stack Choice

The experiments stack is Python-based.

Gameplay, rules, generators, and engine-side tests remain in Java. The orchestration layer that plans, resumes, records, and analyses experiments lives in Python and calls the Java engine through stable CLI boundaries.

Current module ownership:

- `experiments/` owns orchestration, run state, artifact management, and analysis outputs.
- `engine/` owns Solitaire rules, data generation, and engine-side evaluation.
- `neural-network/` owns datasets, model code, training code, and model serving.

This split matches the current implementation: training and analysis are Python-first, while the game engine remains Java-first.

## Goals

- Run long-lived experiments across architecture families such as MLP and later GNN.
- Resume safely after interruption such as power loss, host reboot, or process crash.
- Track full lineage for datasets, checkpoints, code revisions, configs, and evaluations.
- Replace ad hoc markdown-only findings with structured, queryable experiment data.
- Keep the engine and neural-network modules focused on their existing concerns.

## Current Non-Goals

- Multi-machine distributed scheduling.
- Real-time dashboards served to multiple users.
- Heavy external orchestration platforms such as Airflow, Ray, MLflow, or Celery.
- Replacing the existing engine-to-model HTTP evaluation path.

## Repository Layout

The experiments control plane lives in a third top-level folder:

```text
experiments/
  README.md
  DESIGN.md
  RESULTS.md
  specs/
  src/
  runtime/
    registry/
    artifacts/
    parquet/
    work/
```

Current folder ownership:

- `engine/` should stay focused on Solitaire rules, players, generators, and engine-side tests.
- `neural-network/` should stay focused on datasets, model code, training code, and model serving.
- `experiments/` is a control-plane concern that coordinates both modules but should not be owned by either.

Tracked versus ignored content is split as:

- Track: docs, specs, small schemas, generated summary reports worth reviewing.
- Ignore: runtime DB files, temporary workspaces, large logs, checkpoints, parquet exports, and large generated artifacts.

## Current Architecture

### Control Plane

The experiments driver owns:

- Reading experiment specs.
- Expanding sweeps into concrete runs.
- Scheduling resumable tasks.
- Recording task state, heartbeats, and artifact manifests.
- Launching engine collection, training, and evaluation subprocesses.
- Materialising normalized analytics outputs.

### Worker Boundaries

The driver calls existing module entry points rather than reimplementing them:

- Java engine for episode generation and evaluation.
- Python trainer for model training.
- Python service only when live neural inference is part of evaluation.

This keeps the contracts narrow and prevents the orchestration layer from reaching into internals too early.

## Current Protocol And Storage Model

The current stack uses different technologies for different jobs.

### 1. Experiment Specs

- Format: JSON.
- Purpose: human-authored, versioned experiment definitions.
- Scope: architecture family, dataset lineage, training config, evaluation config, and sweep params.

### 2. Execution Protocol

- Format: local subprocess invocation with structured stdout/stderr capture.
- Purpose: launch engine and neural-network tasks with stable CLI arguments.
- Rule: avoid in-process Java/Python coupling.

### 3. Raw Artifact Storage

- Format: filesystem.
- Purpose: immutable logs, checkpoints, reports, plots, and captured command output.
- Rule: every task writes into a unique artifact directory and completes via atomic rename.

### 4. Experiment Registry

- Format: SQLite.
- Purpose: source of truth for run state, resumability, task status, lineage pointers, and heartbeats.
- Current fit: embedded, portable, simple to inspect, and sufficient for the current single-machine stack.

### 5. Analysis Store

- Format: Parquet queried through DuckDB.
- Purpose: slice-and-dice analysis across large runs without loading everything into memory.
- Reason: much better for analytics than raw JSONL or ad hoc markdown.

### 6. Inference Protocol

- Format: HTTP/JSON.
- Purpose: keep only for the live engine-to-model evaluation boundary that already exists.
- Rule: do not use HTTP as the orchestration protocol for local experiments.

## Resumability Model

Resumability comes from small idempotent work units, not from one giant process staying alive indefinitely.

The current hierarchy is:

- Experiment: a named intent such as `mlp-hidden-dim-sweep-v1`.
- Run: one concrete config generated from a spec.
- Stage: collect, train, evaluate, analyze.
- Shard: the smallest restartable work unit inside a stage.
- Attempt: one execution of a shard.

Examples of good shard boundaries:

- Data collection for seed range `2000-2499`.
- Evaluation over a fixed game block such as 100 games.
- Training checkpoint intervals such as every epoch or every N optimizer steps.

Rules for resumability:

- Never mark a shard complete until its outputs are fully written and validated.
- Keep a heartbeat timestamp for active work so stale locks can be reclaimed.
- Write a manifest for every completed artifact directory.
- Make output directories immutable after successful completion.
- Record command line, code revision, parent checkpoint, and data sources for every run.

## Current Registry Model

The registry stays intentionally small. Current core tables:

- `experiment_specs`
- `runs`
- `run_parameters`
- `tasks`
- `task_attempts`
- `artifacts`
- `metrics_rollups`

Useful task statuses:

- `pending`
- `running`
- `succeeded`
- `failed`
- `interrupted`
- `retryable`
- `cancelled`

Useful run dimensions to persist:

- Architecture family and params.
- Dataset identity and source logs.
- Code revisions for root repo, engine, and neural-network.
- Training hyperparameters.
- Evaluation suite and seed ranges.
- Parent checkpoint and produced checkpoint.

## Architecture-Aware Experiment Contract

The driver does not assume every model is an MLP.

The current contract nests architecture-specific parameters under a family key.

Example:

```json
{
  "api_version": "v1",
  "experiment_id": "mlp_baseline_sweep_v1",
  "architecture": {
    "family": "mlp",
    "params": {
      "hidden_dim": 512,
      "num_layers": 3,
      "batch_norm": false,
      "residual": false
    }
  },
  "dataset": {
    "kind": "archived_episode_logs",
    "sources": [
      "engine/logs/archive/level2/20260529T071709Z/episode.log",
      "engine/logs/archive/level3/20260529T071711Z/episode.log"
    ]
  },
  "training": {
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.001,
    "resume_from": null
  },
  "evaluation": {
    "suite": "alpha_seeded_levels",
    "games_per_shard": 100,
    "seed_start": 0,
    "seed_end": 5000
  }
}
```

The same contract now supports both `family: mlp` and `family: gnn`, with each family owning its own parameter validation and trainer arguments.

Dataset sourcing now has two supported modes:

- `dataset.kind: archived_episode_logs` reuses existing archived logs listed in `dataset.sources`.
- `dataset.kind: run_collection_episode_logs` resolves successful collection shard artifacts from the same run so one experiment can collect fresh data, train on it, and then evaluate the resulting checkpoint.

## Current Implementation Approach

The current control plane stays lightweight.

- The orchestration core uses Python standard library modules such as `argparse`, `subprocess`, `sqlite3`, `json`, `pathlib`, and `dataclasses`.
- Structured evaluation reporting uses DuckDB and Parquet outputs.
- The stack does not depend on heavier orchestration platforms such as Airflow, Ray, MLflow, or Celery.

## Current Implementation Status

### Orchestration And Registry

The current driver is a Python CLI under `experiments/src` that reads versioned JSON specs, expands them into concrete tasks, records run state in SQLite, and resumes work by `run_id`.

Current implementation details:

- Runs, tasks, attempts, artifact manifests, and flattened run parameters are persisted in the registry.
- Every task attempt writes into a temporary artifact directory and only becomes durable after validation and finalization.
- Task heartbeats are written while work is active so stale attempts can be reclaimed.
- Resume semantics are task-oriented: successful tasks are skipped, interrupted or retryable tasks are eligible to run again, and the run record remains the source of truth for overall status.

### Collection And Dataset Sourcing

The current collection path generates deterministic endgame datasets through the Java engine in restartable shards.

Current implementation details:

- `collection.shard_count` and `collection.games_per_shard` define the shard plan.
- Multi-shard collection requires `collection.randomise=true` so shards do not reproduce the same deterministic data.
- Each shard can receive an explicit `endgame.random.seed`, preserving reproducibility while allowing shards to differ.
- Generated `episode*.log` files are moved into finalized task artifacts together with stdout, stderr, command metadata, and a collection summary.
- Dataset sourcing supports both archived logs and same-run collection artifacts.

### Training

The current training path drives the Python trainer through one `policy_value_train` task per run.

Current implementation details:

- Training writes epoch checkpoints, a rolling `*_latest.pt` checkpoint, JSONL metrics, and a structured training summary.
- Resume uses optimizer state, RNG state, completed epoch, and global step from the latest valid checkpoint of the same task.
- Training supports both `mlp_policy_value` and `gnn_policy_value`.
- Training datasets can come from existing archived logs or from successful collection artifacts produced earlier in the same run.

### Evaluation And Analysis

The current evaluation path runs seeded AlphaSolitaire evaluation in restartable shards and then materializes structured reporting outputs.

Current implementation details:

- Evaluation expands into one task per deterministic game block using `seed_start`, `seed_end`, and `games_per_shard`.
- Each shard starts the model service from the resolved checkpoint, runs `AlphaSolitaireLevelTest`, and captures a structured JSON summary artifact.
- A follow-on report task aggregates shard summaries into JSONL, Parquet, DuckDB views, SQL, and markdown.
- Cross-run analysis includes successful completed historical evaluation runs so DuckDB can expose run-to-run comparison views.

### Architecture Support

The current experiment contract is architecture-aware rather than MLP-specific.

Current implementation details:

- `experiments/src/architectures.py` owns family-specific validation and trainer CLI construction.
- Specs support `architecture.family` values `mlp` and `gnn`.
- Matching training kinds are `mlp_policy_value` and `gnn_policy_value`.
- The current GNN family is a legal-move graph model over the encoded board state and the active legal-move mask.
- MLP and GNN use the same evaluation workflow and checkpoint resolution model.

### Operations And Hardening

The current single-machine stack includes operator-facing runtime health and maintenance commands.

Current implementation details:

- `python -m experiments.src doctor` writes both a markdown runtime dashboard and a JSON companion report.
- `doctor --recover-stale` exposes stale-task recovery across all runs.
- `python -m experiments.src cleanup` applies conservative retention rules to stale temporary attempt directories and aged work files.
- `python -m experiments.src run` prints a coarse preflight runtime estimate and prompts for confirmation unless `--yes` is supplied.
- Small `cron` and `launchd` workflows are sufficient for unattended maintenance on one machine.
- SQLite remains sufficient because the workload is still single-machine and the registry access pattern is low-concurrency.

## Current End-To-End Slice

The implemented end-to-end slice now looks like this:

1. Read a versioned experiment spec.
2. Collect fresh logs or reference archived logs.
3. Train an MLP or GNN checkpoint with resumable attempts.
4. Evaluate that checkpoint in deterministic shards.
5. Materialize structured analysis outputs and runtime health reports.
