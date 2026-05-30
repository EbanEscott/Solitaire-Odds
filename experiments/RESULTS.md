# Experiment Results

This document describes how to treat experiment outputs, how to compare runs, and which results should be promoted beyond the runtime artifact tree.

## Result Surfaces

The experiments stack has three different output surfaces, and they serve different purposes.

### Canonical Per-Run Outputs

The canonical output of one completed experiment run lives under `experiments/runtime/artifacts/<run_id>/`.

That artifact tree contains task-level outputs such as:

- collection summaries and captured `episode*.log` files
- training summaries, metrics, and checkpoints
- evaluation shard summaries
- generated markdown, JSONL, Parquet, DuckDB, and SQL report artifacts

If the question is "what did this run produce?", the first place to look is the run artifact tree.

### Operational Runtime Outputs

`experiments/runtime/work/` is an operational area rather than the canonical result surface.

It is the right place for:

- `doctor` outputs such as `runtime_health_report.md` and `runtime_health_report.json`
- maintenance logs
- temporary inspection outputs created while operating the driver locally

It is useful for operators, but it is not the place to treat as the final published result for an experiment family.

### Registry State

`experiments/runtime/registry/` is persistence and lineage state.

It records runs, tasks, attempts, heartbeats, and artifact pointers. It supports resume and inspection, but it is not itself the presentation surface for results.

## Promotion Guidance

Generated run artifacts are the source of truth for raw experiment outputs. Promotion is the next step taken only when a result is stable enough to matter outside one local run directory.

Current promotion rules:

- Treat `experiments/runtime/artifacts/` as the canonical generated output for each run.
- Treat notebooks as consumers of DuckDB and Parquet outputs, not as the source of truth.
- Promote only distilled summaries, comparisons, or conclusions that are stable enough to keep under version control.
- If a result affects the repo-wide player leaderboard, the canonical published summary belongs in the root `README.md`.
- If a result is experiment-specific, promote it to a tracked markdown summary or tracked notebook outside `experiments/runtime/`.

## Research Notebooks

This document is intentionally not the place for active experiment narratives, research questions, or working hypotheses.

Use tracked notebooks for that material. Current notebook:

- `experiments/notebooks/mlp_cliff_research.ipynb` — research questions, MLP-only cliff hypotheses, confidence-interval sizing notes, and a small data-ingest scaffold.

## Runtime Housekeeping

The runtime folders are ignored by git, so you do not need to clean them for repository hygiene. They still matter operationally.

- Keep `experiments/runtime/artifacts/` if you want to preserve canonical run outputs, cross-run comparisons, and checkpoint lineage.
- Keep `experiments/runtime/registry/` if you want resume state and historical run metadata.
- `experiments/runtime/work/` is the least valuable long-term area; it mainly holds doctor reports, maintenance logs, and temporary operator outputs.

The built-in cleanup command is conservative. It removes stale `.attempt-*.tmp` directories under `artifacts/` and old files under `work/`. It does not wipe the archived run artifacts or the registry database.

Practical rule:

- Use `python -m experiments.src cleanup` for routine housekeeping.
- Do not manually delete `artifacts/` or `registry/` unless you intentionally want to lose old runs and comparison history.