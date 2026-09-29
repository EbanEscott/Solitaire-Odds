# Lonelybot: An Engine for Solving Thoughtful/Random Klondike Solitaire

- **Citation key:** `lonelybot`
- **Author:** `vuonghy2442` (GitHub account)
- **Year:** Not specified in the existing literature record
- **Publication:** GitHub repository and README
- **Source:** [Repository](https://github.com/vuonghy2442/lonelybot) · [Reported results](https://github.com/vuonghy2442/lonelybot#running-results)
- **Retrieved:** 2026-09-29

## Summary

Lonelybot is a Klondike solver with Thoughtful and random-play modes. Its author reports strong random-play results using hindsight optimization (HOP) and Monte Carlo tree search (MCTS).

## Key findings

The README reports **7,148 wins in 15,024 games (47.58% ± 0.80%)**, taking roughly **3.17 seconds per game**, for its random-Klondike HOP/MCTS mode. The project describes standard draw-three play with no undo.

## Conditions and limitations

These are software results, not a peer-reviewed study; the author requests further verification. Our small pilot below does not reproduce the full reported sample. Before comparison, inspect information access, action compression, pruning, software version, and seeds. The confidence level for the quoted uncertainty is not recorded in the existing survey.

## Relevance to Solitaire Odds

**Our interpretation:** a promising implementation to inspect and potentially reproduce. Its reported speed and win rate warrant investigation, but a matched rules and observation interface must precede any ranking against our players.

## Source availability

Repository and README only; no companion paper PDF is stored. No commit was recorded in the initial survey, so the source link may change over time. This note preserves the survey's claims without reproducing the full README.

## Local reproduction — 29 September 2026

Built commit `dc41b8e49c69a529816cafb7f49dc3a461c4fd7c` with Cargo 1.98.0. Tests: **16 passed, 2 ignored, 0 failed**. The checkout had no lockfile; the generated dependency lock is preserved with our results.

`./target/release/lonecli hop-loop default 0 3 100` won **48/100** draw-three games, seeds 0–99, in **248.16 seconds** (2.48 seconds/game). Wilson 95% interval: **38.46%–57.68%**. This is compatible with 47.58%, but does not confirm the original estimate or precision. The current HOP code clears hidden tableau identities before planning and checks move legality; matching its stock access and action compression to our engine remains necessary.

The encoded-deal README example matches. Default-seed examples differ, and Git history records an explicit breaking RNG change. Preserve encoded deals, not just seed numbers. The `random` command returned 1,210/10,000, but inspection shows it takes the first dominance-filtered move rather than sampling moves uniformly.

See the [reproduction record](../../../Solitaire-Repos/literature-reproduction/001-literature-reproduction/README.md#lonelybot) for commands, environment, evidence and limitations.
