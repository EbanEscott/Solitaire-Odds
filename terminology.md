# Terminology

This file defines the experiment and modeling terms used across the repo.

## Experiment Terms

### Experiment Spec

A JSON file under `experiments/specs/` that declares what to collect, train, evaluate, and report.

### Experiment ID

The stable identity of a spec. This is the logical name of the experiment design, such as `archived-mlp-train-l2to4-eval-level4-dev`.

### Run ID

The identity of one execution of a spec. Reusing the same `run_id` resumes that run; using a new `run_id` starts a fresh run.

### Run

One persisted execution record in the experiments driver. A run contains ordered tasks, attempts, artifacts, and status.

### Task

One concrete unit of work inside a run, such as collection, training, one evaluation shard, or report generation.

### Attempt

One try at a task. If a task is resumed or retried, it creates another attempt under the same logical task.

### Artifact

Any durable output written by a task attempt, such as logs, summaries, checkpoints, Parquet files, or markdown reports.

### Shard

One slice of a larger job.

In this repo, a shard usually means one of these:

- one collection invocation that generates part of a dataset
- one evaluation invocation that handles a contiguous block of games or seeds

Shards exist to keep failure scope small, support resumability, and make longer runs easier to inspect.

### Seed Band

A fixed contiguous range of evaluation seeds, such as `seed_start=0` and `seed_end=20`. Using the same seed band makes runs comparable.

### Development Band

A smaller seed band used for quick iteration and cheaper comparisons before running more expensive confirmation sweeps.

### Confirmation Band

A larger or disjoint seed band used to verify that the apparent winners from development runs still hold up.

### Sweep

A family of comparable runs where one chosen variable changes across runs, such as `hidden_dim`, `num_layers`, or evaluation level.

### Baseline

The current reference configuration used for comparison. New candidates should be compared against the baseline before promotion.

### Evaluation Ladder

A sequence of evaluation levels, usually Levels 2 through 10, used to measure the same checkpoint across a consistent difficulty range.

## Data And Training Terms

### Archived Dataset

Training logs that already exist on disk and are named explicitly in `dataset.sources`.

### Fresh Data

Training logs collected earlier in the same run by `endgame_collect_shard` tasks and then consumed by training through `dataset.kind=run_collection_episode_logs`.

### Checkpoint

A saved model state written during or after training. A checkpoint can later be evaluated, resumed from, or promoted.

### Checkpoint Lineage

The history of where a checkpoint came from: dataset sources, architecture settings, training settings, and earlier checkpoints.

### Curriculum Depth

How far up the endgame difficulty ladder the training data extends. For example, a model trained on archived Levels 2 through 6 has a shallower curriculum than one trained through Level 10.

### Curriculum Frontier

The highest level included in the current training dataset.

### Architecture Family

The broad model type, currently `mlp` or `gnn`.

### Hidden Dimension

The width of the hidden representation used by the network. Larger values usually increase capacity, runtime, and memory use.

### Number of Layers

The depth of the network. More layers can increase capacity, but also make training slower or less stable.

### Batch Normalization

An optional normalization layer used in the MLP. In this repo it is experimental and should be tested after the main capacity sweep, not treated as a first-pass default.

### Residual Connection

An optional skip connection used in the MLP so later layers can reuse earlier representations more directly. In this repo it is also experimental.

### Action Embedding Dimension

A GNN-specific width for action-node representations.

### Message Passing Steps

The number of update rounds in the GNN between connected nodes.

### Dropout

A GNN regularization setting that randomly masks part of the representation during training.

### Search Budget

The amount of tree search work allowed during evaluation, usually controlled here by settings such as `mcts_simulations`, `mcts_max_depth`, and `mcts_cpuct`.

## Reporting Terms

### Rollup

A grouped summary derived from shard-level results, such as one row per level with games, wins, win rate, and runtime.

### Cross-Run Comparison

A report view that compares the current run against historical successful runs using the same reporting schema.

### Promotion

The act of moving a result from ignored runtime outputs into tracked repo documentation, summaries, or published benchmark tables.
