# Level 10 Findings

Date: May 29, 2026

## Summary

The Level 10 checkpoint is a partial recovery from the Level 9 regression, but only a small one. The dense seeded wall moved from the Level 9 band around Level 18 up to around Level 19, yet the broad sweep also exposed a non-monotonic dip at Level 10 itself.

## Training Snapshot

- Checkpoint: `policy_value_level10_20260529T183825.pt`
- Model: PolicyValueNet
- Architecture: 256 hidden units, 2 layers
- Training data: archived Levels 2-10 episode logs
- Dataset size: 14787 training samples, 1643 validation samples
- Validation accuracy: 74.80% policy, 100.00% value

Archived training inputs:

- `../engine/logs/archive/level2/20260529T071709Z/episode.log`
- `../engine/logs/archive/level3/20260529T071711Z/episode.log`
- `../engine/logs/archive/level4/20260529T071714Z/episode.log`
- `../engine/logs/archive/level5/20260529T071718Z/episode.log`
- `../engine/logs/archive/level6/20260529T071724Z/episode.log`
- `../engine/logs/archive/level7/20260529T071736Z/episode.log`
- `../engine/logs/archive/level8/20260529T081124Z/episode.log`
- `../engine/logs/archive/level9/20260529T082013Z/episode.log`
- `../engine/logs/archive/level10/20260529T083801Z/episode.log`

## Evaluation Notes

These seeded sweeps use the shared 1000-move cap from `ResultsConfig.MAX_MOVES_PER_GAME` via `AlphaSolitaireLevelTest`.

Unlike the earlier checkpoints, the Level 10 broad sweep was not monotonic: it dipped sharply at Level 10, recovered at Level 15, and then dropped back to zero by Level 20.

## Broad Seeded Sweep

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 5 | 3 | 3 | 100.00% | 4.00 | 0.036s |
| 10 | 3 | 1 | 33.33% | 669.67 | 47.676s |
| 15 | 3 | 3 | 100.00% | 16.67 | 0.737s |
| 20 | 3 | 0 | 0.00% | 1000.00 | 26.572s |

## Dense Follow-up Near the Wall

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 16 | 1 | 1 | 100.00% | 15.00 | 0.774s |
| 17 | 1 | 1 | 100.00% | 16.00 | 0.826s |
| 18 | 1 | 1 | 100.00% | 22.00 | 1.204s |
| 19 | 1 | 0 | 0.00% | 1000.00 | 26.517s |
| 20 | 1 | 0 | 0.00% | 1000.00 | 21.636s |

## Comparison

| Checkpoint | Estimated Cliff |
|------------|-----------------|
| L2-L6 Baseline | Level 30-31 |
| L2-L7 Checkpoint | Around Level 42 |
| L2-L8 Checkpoint | Around Level 30 |
| L2-L9 Checkpoint | Between Level 17 and Level 18 |
| L2-L10 Checkpoint | Between Level 18 and Level 19 |

## Findings

- Level 10 recovers about one level of seeded depth relative to Level 9, but it remains far worse than the Level 8 and Level 7 checkpoints.
- The Level 10 broad sweep is irregular: Level 10 falls to 1/3, Level 15 rebounds to 3/3, and Level 20 drops to 0/3.
- The dense wall map is cleaner than the broad sweep and suggests the real zero boundary sits between Level 18 and Level 19.
- Validation policy accuracy improved again, but the seeded generalization result still does not track that metric reliably.

## Cliff Estimate

The best supported estimate is that the Level 10 checkpoint cliff is around Level 19.

More precisely:

- The last confirmed winning level is Level 18.
- The first confirmed zero-win level is Level 19.
- The broad sweep shows an earlier instability pocket at Level 10, so the overall curve is noisier than the earlier checkpoints.

## Practical Takeaway

Level 10 is better than Level 9 in the deep seeded band, but only slightly. It pushes the zero boundary out by about one level while still showing substantial instability and remains nowhere near the stronger Level 7 or even Level 8 behavior.