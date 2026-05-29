# Level 6 Findings

Date: May 29, 2026

## Summary

The Level 6 baseline is the original flat-board MLP trained on archived Levels 2-6 endgame episodes. It is strong through roughly Level 29, then collapses sharply at Levels 30-31. This established the original "MLP cliff" and motivated the next phase of work.

## Training Snapshot

- Model: PolicyValueNet
- Architecture: 256 hidden units, 2 layers
- Training data: Levels 2-6 endgame positions
- Dataset size: about 864k samples from about 283k episodes
- Validation accuracy: 88.9% policy, 100.0% value

## Phase 0 Sweep

Primary seeded sweep from the baseline checkpoint:

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 20 | 20 | 19 | 95.00% | 18.65 | 0.193s |
| 25 | 20 | 15 | 75.00% | 21.65 | 0.282s |
| 30 | 19 | 2 | 10.53% | 32.42 | 0.779s |
| 35 | 19 | 0 | 0.00% | 14.53 | 0.639s |
| 40 | 19 | 0 | 0.00% | 14.53 | 0.638s |
| 50 | 18 | 0 | 0.00% | 14.39 | 1.011s |

Dense follow-up near the cliff:

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 22 | 20 | 17 | 85.00% | 19.80 | 0.213s |
| 24 | 20 | 16 | 80.00% | 21.20 | 0.258s |
| 26 | 20 | 15 | 75.00% | 22.45 | 0.297s |
| 28 | 20 | 15 | 75.00% | 24.05 | 0.324s |
| 29 | 20 | 14 | 70.00% | 25.45 | 0.357s |
| 31 | 19 | 0 | 0.00% | 14.63 | 0.545s |

## Findings

- The baseline stays strong through about Level 29.
- The decline is not gradual from Level 20 onward.
- The failure is abrupt between Levels 29 and 31, with Level 30 already close to unusable.
- Deterministic seeded generation produced only 18-19 games at some deeper levels, so the deepest rows used fewer than the requested 20 games.

## Cliff Estimate

The original Level 6-trained MLP cliff is around Level 30-31.

## Why It Mattered

This was the key evidence that the flat-board MLP had hit an architectural limit. It could imitate A* well on easier seeded endgames, but it did not keep that competence once the reverse-move depth crossed into the low 30s.