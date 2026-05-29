# Level 5 Findings

Date: May 29, 2026

## Summary

The cumulative Level 5 checkpoint is the first backfilled model that clearly pushes the flat-board MLP wall beyond the old Level 30 baseline. It stays perfect through Level 40, then drops to zero at Level 41 in the dense follow-up.

## Training Snapshot

- Model: PolicyValueNet
- Architecture: 256 hidden units, 2 layers
- Training data: archived Levels 2-5 episode logs
- Dataset size: 3,180 training samples, 353 validation samples
- Validation accuracy: 48.20% policy, 100.00% value

Archived training inputs:

- `../engine/logs/archive/level2/20260529T071709Z/episode.log`
- `../engine/logs/archive/level3/20260529T071711Z/episode.log`
- `../engine/logs/archive/level4/20260529T071714Z/episode.log`
- `../engine/logs/archive/level5/20260529T071718Z/episode.log`

## Broad Seeded Sweep

Quick backfill sweep with 3 games per level:

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 2 | 3 | 3 | 100.00% | 1.00 | 0.016s |
| 5 | 3 | 3 | 100.00% | 4.00 | 0.028s |
| 10 | 3 | 3 | 100.00% | 9.00 | 0.153s |
| 15 | 3 | 3 | 100.00% | 14.00 | 0.320s |
| 20 | 3 | 3 | 100.00% | 19.00 | 0.553s |
| 25 | 3 | 3 | 100.00% | 24.00 | 0.836s |
| 30 | 3 | 3 | 100.00% | 28.67 | 1.091s |
| 35 | 3 | 3 | 100.00% | 33.67 | 1.330s |
| 40 | 3 | 3 | 100.00% | 38.67 | 1.394s |
| 45 | 3 | 0 | 0.00% | 1000.00 | 42.254s |

## Dense Follow-up Near the Wall

The dense pass was stopped as soon as the first zero-win level was confirmed:

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 41 | 1 | 0 | 0.00% | 1000.00 | 74.176s |

## Comparison

| Checkpoint | Estimated Cliff |
|------------|-----------------|
| L2-only Checkpoint | No zero-win level observed through Level 50 |
| L2-L3 Checkpoint | Between Level 28 and Level 29 |
| L2-L4 Checkpoint | No zero-win level observed through Level 50; weak tail only |
| L2-L5 Checkpoint | Between Level 40 and Level 41 |
| L2-L6 Baseline | Level 30-31 |
| L7 Checkpoint | Around Level 42 |

## Findings

- Level 5 is the first backfilled checkpoint that clearly extends the sharp seeded wall beyond the original L6 baseline.
- The model remains perfect through Level 40 in the broad sweep.
- The first confirmed zero-win level is Level 41.
- The failure mode beyond the wall is severe: both the Level 41 dense probe and the Level 45 broad probe hit the 1000-move cap and lost.

## Cliff Estimate

The best supported estimate is that the Level 5 checkpoint cliff is between Level 40 and Level 41.

More precisely:

- the last confirmed fully winning level is Level 40,
- the first confirmed zero-win level is Level 41.

## Practical Takeaway

Level 5 is a genuine step up from Levels 3 and 4. In seeded endgame play it almost reaches the later Level 7 range, which suggests that a large part of the eventual improvement was already present once the curriculum included Level 5 data.