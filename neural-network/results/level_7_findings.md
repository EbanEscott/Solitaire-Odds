# Level 7 Findings

Date: May 29, 2026

## Summary

Adding Level 7 data removed the old Level 30-31 cliff. The new checkpoint stays perfect through Level 41 in the seeded sweep, but it drops sharply at Level 42 and remains degraded at deeper points. The current best estimate places the new cliff around Level 42.

## Training Snapshot

- Checkpoint: `policy_value_level7_20260529T172306.pt`
- Model: PolicyValueNet
- Architecture: 256 hidden units, 2 layers
- Training data: archived Levels 2-7 episode logs
- Dataset size: 6293 training samples, 699 validation samples
- Validation accuracy: 61.80% policy, 100.00% value

Archived training inputs:

- `../engine/logs/archive/level2/20260529T071709Z/episode.log`
- `../engine/logs/archive/level3/20260529T071711Z/episode.log`
- `../engine/logs/archive/level4/20260529T071714Z/episode.log`
- `../engine/logs/archive/level5/20260529T071718Z/episode.log`
- `../engine/logs/archive/level6/20260529T071724Z/episode.log`
- `../engine/logs/archive/level7/20260529T071736Z/episode.log`

## Seeded Evaluation Results

Completed comparison points against the old Phase 0 baseline:

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 20 | 20 | 20 | 100.00% | 18.30 | 0.570s |
| 29 | 20 | 20 | 100.00% | 27.30 | 1.058s |
| 30 | 19 | 19 | 100.00% | 28.26 | 1.103s |
| 31 | 19 | 19 | 100.00% | 29.32 | 1.157s |
| 35 | 19 | 19 | 100.00% | 33.32 | 1.254s |
| 40 | 19 | 19 | 100.00% | 38.32 | 1.319s |
| 41 | 19 | 19 | 100.00% | 39.32 | 1.059s |
| 42 | 18 | 12 | 66.67% | 43.94 | 1.006s |
| 45 | 18 | 11 | 61.11% | 48.33 | 8.555s |
| 50 | 18 | 11 | 61.11% | 53.67 | 1.641s |

Dense follow-up around the new wall:

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 41 | 19 | 19 | 100.00% | 39.32 | 1.059s |
| 42 | 18 | 12 | 66.67% | 43.94 | 1.006s |

Direct comparison with the Level 6 baseline:

| Level | L2-L6 Baseline | L2-L7 Checkpoint |
|-------|----------------|------------------|
| 20 | 19/20 (95.00%) | 20/20 (100.00%) |
| 29 | 14/20 (70.00%) | 20/20 (100.00%) |
| 30 | 2/19 (10.53%) | 19/19 (100.00%) |
| 31 | 0/19 (0.00%) | 19/19 (100.00%) |
| 35 | 0/19 (0.00%) | 19/19 (100.00%) |
| 40 | 0/19 (0.00%) | 19/19 (100.00%) |
| 41 | not measured in the original dense sweep | 19/19 (100.00%) |
| 42 | not measured in the original dense sweep | 12/18 (66.67%) |
| 45 | not measured in the original dense sweep | 11/18 (61.11%) |
| 50 | 0/18 (0.00%) | 11/18 (61.11%) |

## Findings

- The old Level 30-31 collapse is gone.
- The checkpoint stays perfect through Level 41 in the completed seeded sweep.
- The dense follow-up shows the actual break starts immediately at Level 42, where the model drops to 12/18 wins.
- Levels 45 and 50 remain in the same degraded band, so the sharp break happens just after Level 41 rather than gradually across the 40s.
- The Level 7 dataset is much smaller than the historical Level 6 baseline corpus, so the result is a meaningful performance shift even though the retrain is not a like-for-like data volume comparison.

## Cliff Estimate

The best supported estimate from the completed sweep is that the new cliff moved from Level 30-31 to around Level 42.

More precisely:

- Lower bound: no earlier than Level 41, because Level 41 was still 19/19.
- Upper bound: no later than Level 42, because Level 42 had already dropped to 12/18.

## Practical Takeaway

Level 7 curriculum extension pushed the useful seeded endgame range by about 10-12 levels relative to the old baseline. The flat-board MLP still degrades eventually, but the failure point is now around the low 40s instead of the low 30s.