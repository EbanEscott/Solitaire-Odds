# Experiments

This folder is the control plane for long-running AlphaSolitaire experiments.

The recommendation is to keep the experiments stack Python-based while treating the Java engine as a worker process and source of deterministic game data. That keeps training, analytics, and orchestration close to the current neural-network code without forcing Java to become the owner of Python model workflows.

Start with [DESIGN.md](DESIGN.md) for the proposed architecture, storage model, resumability rules, and phased proof-of-concept plan.

Planned responsibilities for this folder:

- Experiment specs and sweep definitions.
- Driver and scheduler code for long-running jobs.
- Experiment registry metadata.
- Structured analysis outputs and generated summaries.

Planned non-responsibilities for this folder:

- Core Solitaire gameplay logic. That stays in `engine/`.
- Model implementations and training primitives. Those stay in `neural-network/`.
