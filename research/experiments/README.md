# Research Experiments

Keep curated research records here. The root [experiments workspace](../../experiments/README.md) continues to hold long-running jobs, notebooks, and working notes.

- [001: First reproduction pass on four upstream Klondike implementations](../../../Solitaire-Repos/literature-reproduction/001-literature-reproduction/README.md) — stored in the sibling `Solitaire-Repos` directory; 29 September 2026; builds, fixture checks, bounded fresh trials and archived result recounts.

Create a folder for each study, such as `001-search-budget-vs-win-rate/`, containing a `README.md` and a `figures/` folder when needed. A study can reference multiple runs.

Each study README should record:

- **Question and hypothesis:** Link to the relevant research question.
- **Method:** Rules, information available to each player, baselines, controls, sample sizes, and metrics.
- **Reproduction:** Code commit, commands, configuration, seeds or deal identifiers, model versions, and relevant environment details.
- **Evidence:** Run IDs and links to specs, notebooks, logs, and result artifacts; identify local-only artifacts and how to regenerate them.
- **Findings:** Results, uncertainty, and interpretation.
- **Limitations and next steps:** Confounders, incomplete runs, and follow-up work.

Link to existing run artifacts and notes rather than copying them. Keep core implementations and tests in their existing modules, and orchestration in the root experiments workspace. Add figures here when they support the research write-up.
