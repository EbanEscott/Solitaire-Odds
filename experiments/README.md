# Experiments

This folder is the control plane for long-running AlphaSolitaire experiments.

The recommendation is to keep the experiments stack Python-based while treating the Java engine as a worker process and source of deterministic game data. That keeps training, analytics, and orchestration close to the current neural-network code without forcing Java to become the owner of Python model workflows.

Start with [DESIGN.md](DESIGN.md) for the proposed architecture, storage model, resumability rules, and phased proof-of-concept plan.

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

Planned responsibilities for this folder:

- Experiment specs and sweep definitions.
- Driver and scheduler code for long-running jobs.
- Experiment registry metadata.
- Structured analysis outputs and generated summaries.

Planned non-responsibilities for this folder:

- Core Solitaire gameplay logic. That stays in `engine/`.
- Model implementations and training primitives. Those stay in `neural-network/`.
