# Level 3 Findings

Date: May 29, 2026

## Summary

The cumulative Level 3 checkpoint behaves much more like a conventional flat-board MLP baseline. It stays perfect through the mid-20s, then drops abruptly between Levels 28 and 29. That places its seeded cliff almost exactly on top of the old L2-L6 baseline band.

## Training Snapshot

- Model: PolicyValueNet
- Architecture: 256 hidden units, 2 layers
- Training data: archived Levels 2-3 episode logs
- Dataset size: 810 training samples, 89 validation samples
- Validation accuracy: 33.70% policy, 100.00% value

Archived training inputs:

- `../engine/logs/archive/level2/20260529T071709Z/episode.log`
- `../engine/logs/archive/level3/20260529T071711Z/episode.log`

## Broad Seeded Sweep

Quick backfill sweep with 3 games per level:

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 2 | 3 | 3 | 100.00% | 1.00 | 0.016s |
| 5 | 3 | 3 | 100.00% | 4.00 | 0.039s |
| 10 | 3 | 3 | 100.00% | 9.00 | 0.203s |
| 15 | 3 | 3 | 100.00% | 14.00 | 0.370s |
| 20 | 3 | 3 | 100.00% | 19.00 | 0.514s |
| 25 | 3 | 3 | 100.00% | 24.00 | 0.844s |
| 30 | 3 | 0 | 0.00% | 42.33 | 1.904s |

## Dense Follow-up Near the Wall

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 26 | 1 | 1 | 100.00% | 25.00 | 1.051s |
| 27 | 1 | 1 | 100.00% | 26.00 | 1.129s |
| 28 | 1 | 1 | 100.00% | 27.00 | 1.140s |
| 29 | 1 | 0 | 0.00% | 41.00 | 2.061s |
| 30 | 1 | 0 | 0.00% | 42.00 | 2.175s |

## Comparison

| Checkpoint | Estimated Cliff |
|------------|-----------------|
| L2-only Checkpoint | No zero-win level observed through Level 50 |
| L2-L3 Checkpoint | Between Level 28 and Level 29 |
| L2-L6 Baseline | Level 30-31 |

## Findings

- The Level 3 checkpoint is far more stable than the tiny Level 2-only result, but it is also less surprising.
- It remains perfect through Level 28 in the dense pass.
- The first confirmed zero-win level is Level 29.
- This places the L3 cliff very close to the later L6 baseline cliff, suggesting that most of the basic seeded competence was already present once Level 3 data entered the curriculum.

## Cliff Estimate

The best supported estimate is that the Level 3 checkpoint cliff is around Level 29.

More precisely:

- The last confirmed winning level is Level 28.
- The first confirmed zero-win level is Level 29.

## Practical Takeaway

Level 3 already captures most of the flat-board MLP's familiar seeded range. Unlike the anomalously strong L2-only run, the L2-L3 checkpoint looks like the standard pattern: strong through the high 20s, then a sharp break right before 30.