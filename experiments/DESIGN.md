# Experiments Design

## Recommendation

The experiments stack should be Python-based.

That does not mean moving gameplay or engine logic out of Java. It means the orchestration layer that plans, resumes, records, and analyses experiments should live in Python and call the Java engine through stable CLI boundaries.

## Why Python Over Java

Python is the better fit for this layer because:

- The current model training, checkpoints, and service already live in `neural-network/`.
- Experiment analysis will naturally lean on Python tooling such as notebooks, DuckDB, Parquet, pandas, or polars.
- Orchestration work is mostly subprocess control, artifact management, spec parsing, and metrics aggregation, which is simpler in Python.
- A Java-based driver would still need to shell out to Python for training and model-side analysis, creating a split-brain control plane.
- Future GNN work is much more likely to stay Python-first than Java-first.

Java would be the better choice only if the new layer were primarily an engine benchmark harness with deep in-process access to engine internals and almost no model-side workflow. That is not the direction here.

## Goals

- Run long-lived experiments across architecture families such as MLP and later GNN.
- Resume safely after interruption such as power loss, host reboot, or process crash.
- Track full lineage for datasets, checkpoints, code revisions, configs, and evaluations.
- Replace ad hoc markdown-only findings with structured, queryable experiment data.
- Keep the engine and neural-network modules focused on their existing concerns.

## Non-Goals For The First Proof Of Concept

- Multi-machine distributed scheduling.
- Real-time dashboards served to multiple users.
- Heavy external orchestration platforms such as Airflow, Ray, MLflow, or Celery.
- Replacing the existing engine-to-model HTTP evaluation path.

## Placement In The Repository

The new driver belongs in a third top-level folder:

```text
experiments/
  README.md
  DESIGN.md
  specs/
  src/
  runtime/
    registry/
    artifacts/
    parquet/
    work/
```

Reasoning:

- `engine/` should stay focused on Solitaire rules, players, generators, and engine-side tests.
- `neural-network/` should stay focused on datasets, model code, training code, and model serving.
- `experiments/` is a control-plane concern that coordinates both modules but should not be owned by either.

Tracked versus ignored content should be split cleanly:

- Track: docs, specs, small schemas, generated summary reports worth reviewing.
- Ignore: runtime DB files, temporary workspaces, large logs, checkpoints, parquet exports, and large generated artifacts.

## Proposed Architecture

### Control Plane

The experiments driver owns:

- Reading experiment specs.
- Expanding sweeps into concrete runs.
- Scheduling resumable tasks.
- Recording task state, heartbeats, and artifact manifests.
- Launching engine collection, training, and evaluation subprocesses.
- Materialising normalized analytics outputs.

### Worker Boundaries

The driver should call existing module entry points rather than reimplement them:

- Java engine for episode generation and evaluation.
- Python trainer for model training.
- Python service only when live neural inference is part of evaluation.

This keeps the contracts narrow and prevents the orchestration layer from reaching into internals too early.

## Protocol And Storage Recommendations

Use different technologies for different jobs instead of forcing one tool to do everything.

### 1. Experiment Specs

- Format: YAML or JSON.
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
- Reason: embedded, portable, simple to inspect, and enough for a single-machine proof of concept.

### 5. Analysis Store

- Format: Parquet queried through DuckDB.
- Purpose: slice-and-dice analysis across large runs without loading everything into memory.
- Reason: much better for analytics than raw JSONL or ad hoc markdown.

### 6. Inference Protocol

- Format: HTTP/JSON.
- Purpose: keep only for the live engine-to-model evaluation boundary that already exists.
- Rule: do not use HTTP as the orchestration protocol for local experiments.

## Resumability Model

Resumability should come from small idempotent work units, not from hoping one giant process never fails.

The basic hierarchy should be:

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

## Suggested Data Model

The registry schema can stay small at first. These tables are enough for a proof of concept:

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

The driver should not assume every model is an MLP.

Use a common experiment contract with architecture-specific parameters nested under a family key.

Example:

```yaml
api_version: v1
experiment_id: mlp_baseline_sweep_v1

architecture:
  family: mlp
  params:
    hidden_dim: 512
    num_layers: 3
    batch_norm: false
    residual: false

dataset:
  kind: archived_episode_logs
  sources:
    - ../engine/logs/archive/level2/20260529T071709Z/episode.log
    - ../engine/logs/archive/level3/20260529T071711Z/episode.log

training:
  epochs: 20
  batch_size: 64
  learning_rate: 0.001
  resume_from: null

evaluation:
  suite: alpha_seeded_levels
  games_per_shard: 100
  seed_start: 0
  seed_end: 5000
```

Later, the same contract can support `family: gnn` with graph-specific params and data preparation rules.

## Recommended Implementation Style

Keep the proof of concept lightweight.

- Start with Python standard library where possible: `argparse`, `subprocess`, `sqlite3`, `json`, `pathlib`, `dataclasses`.
- Add DuckDB when the structured analytics phase starts.
- Add pandas or polars only when notebook or reporting ergonomics justify it.
- Avoid heavy experiment platforms until the single-machine workflow is clearly insufficient.

## Proof Of Concept Phases

### Phase 1: Driver Skeleton

Goal: run one MLP experiment from a spec and resume it after interruption.

Checklist:

- [x] Create `experiments/specs/` and define a versioned experiment spec format.
- [x] Create `experiments/src/` with a small driver CLI.
- [x] Implement SQLite registry tables for runs, tasks, attempts, and artifacts.
- [x] Implement run status transitions and heartbeat handling.
- [x] Define artifact directory naming and manifest rules.
- [x] Prove that a stopped run can be resumed without duplicating completed tasks.

Implementation note:

- The current Phase 1 workflow derives `collect`, `train`, and `evaluate` tasks from the spec and executes them as `noop` tasks by default. That is intentional. The goal of this phase is to prove spec loading, registry behavior, artifact layout, heartbeat tracking, and resume semantics before wiring the real engine and trainer subprocesses in later phases.

### Phase 2: Sharded Data Collection

Goal: generate reproducible episode datasets from the Java engine in restartable shards.

Checklist:

- [ ] Define a shard unit for collection such as fixed seed blocks or game-count blocks.
- [ ] Wrap the engine invocation behind a stable driver task.
- [ ] Archive shard outputs into immutable artifact directories.
- [ ] Capture stdout, stderr, command line, and code revision for each shard.
- [ ] Validate shard completeness before marking it successful.
- [ ] Support retry of failed or interrupted shards only.

### Phase 3: Resumable MLP Training

Goal: train the current MLP through the driver with periodic checkpoints and restart support.

Checklist:

- [ ] Extend training to save periodic checkpoints, not just a final checkpoint.
- [ ] Persist optimizer state, scheduler state if added, RNG state, and current epoch or step.
- [ ] Resume training from the latest valid checkpoint.
- [ ] Emit structured per-epoch metrics for later analysis.
- [ ] Record produced checkpoints and lineage in the registry.
- [ ] Validate recovery from a simulated interruption mid-training.

### Phase 4: Structured Evaluation And Analysis

Goal: replace manual findings with structured experiment outputs that can be queried and compared.

Checklist:

- [ ] Run seeded evaluation in restartable shards.
- [ ] Normalize evaluation summaries into Parquet tables.
- [ ] Add DuckDB queries or views for win rate, confidence interval, runtime, and lineage comparisons.
- [ ] Create one notebook or local report that can slice results by architecture and hyperparameters.
- [ ] Generate markdown summaries from structured data instead of writing them by hand.

### Phase 5: Architecture Adapter Layer

Goal: support MLP and GNN under one experiment interface.

Checklist:

- [ ] Introduce an architecture adapter contract in the driver.
- [ ] Move current MLP support behind that contract.
- [ ] Define GNN-specific dataset preparation and training parameters.
- [ ] Support architecture-specific validation rules in experiment specs.
- [ ] Run comparable experiments across MLP and GNN using the same evaluation suites.

### Phase 6: Hardening And Scale-Up

Goal: make the single-machine stack reliable enough for long unattended runs.

Checklist:

- [ ] Add stale-heartbeat recovery rules.
- [ ] Add retention and cleanup policies for temporary workspaces.
- [ ] Add summary reports for failed runs and missing artifacts.
- [ ] Add simple local dashboards or reports if notebooks are no longer enough.
- [ ] Reassess whether SQLite is still sufficient before considering a heavier service stack.

## Recommended First Build Slice

The smallest useful slice is:

1. A Python driver that reads one MLP experiment spec.
2. A SQLite registry that tracks collection, training, and evaluation tasks.
3. Sharded engine collection.
4. Resumable MLP training.
5. One DuckDB-backed report that compares completed runs.

That slice is enough to prove the architecture before adding GNN-specific complexity.
