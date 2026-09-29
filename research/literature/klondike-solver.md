# Klondike-Solver

- **Citation key:** `shootmeKlondikeSolver`
- **Author:** ShootMe (GitHub account)
- **Publication:** Software repository; no companion PDF
- **Source:** [Repository](https://github.com/ShootMe/Klondike-Solver)
- **Version tested:** `40c8bf93c0a66e34182b0e2ff60dc37393ac17ce`
- **Checked:** 2026-09-29

## Summary

C++ full-information Klondike solver that searches for minimum-length solutions. Its archived statistics report 91.9% draw-one and 83.6% draw-three solved across 1,000 deals each, with a 60-million-state limit.

## Local reproduction — 29 September 2026

Built with Apple Clang and ran all five supplied fixtures in draw-one and draw-three. Every expected result stated in the fixture comments matches after converting the temporary input file to CRLF line endings. The original LF-only PySol input is silently rejected and the solver reuses the preceding deal. This can produce a misleading result without an error message.

The two corrected fixture batches took 21.04 and 15.63 seconds respectively. The executable labels raw clock ticks as milliseconds; use external elapsed times. Exact commands, fixture outcomes, version details and evidence are in the [reproduction record](../../../Solitaire-Repos/literature-reproduction/001-literature-reproduction/README.md#klondike-solver).

## Conditions and limitations

These are full-information fixture checks, not a replication of the historical 1,000-deal statistics. Original benchmark seeds are not recorded in `Statistics.txt`. State-limit exhaustion means unresolved, even where that file calls it “most likely impossible.” Move counts include tableau flips, so normalize action accounting before comparing other implementations.

## Relevance to Solitaire Odds

A working source of small minimum-length examples and potential full-information reference solutions. Validate input loading before treating its output as a deal label. Related context: [Blake and Gent's winnability study](the-winnability-of-klondike-solitaire-and-many-other-patience-games.md).
