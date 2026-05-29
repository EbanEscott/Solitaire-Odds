# Level 4 Findings

Date: May 29, 2026

## Summary

The cumulative Level 4 checkpoint does not show a clean zero-win cliff in the backfill sweep. Instead, it stays perfect through Level 25, then falls into a weak but persistent tail that still produces occasional wins all the way out to Level 50.

## Training Snapshot

- Model: PolicyValueNet
- Architecture: 256 hidden units, 2 layers
- Training data: archived Levels 2-4 episode logs
- Dataset size: 1,879 training samples, 208 validation samples
- Validation accuracy: 46.20% policy, 100.00% value

Archived training inputs:

- `../engine/logs/archive/level2/20260529T071709Z/episode.log`
- `../engine/logs/archive/level3/20260529T071711Z/episode.log`
- `../engine/logs/archive/level4/20260529T071714Z/episode.log`

## Broad Seeded Sweep

Quick backfill sweep with 3 games per level:

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 2 | 3 | 3 | 100.00% | 1.00 | 0.016s |
| 5 | 3 | 3 | 100.00% | 4.00 | 0.030s |
| 10 | 3 | 3 | 100.00% | 9.00 | 0.125s |
| 15 | 3 | 3 | 100.00% | 14.00 | 0.182s |
| 20 | 3 | 3 | 100.00% | 19.00 | 0.295s |
| 25 | 3 | 3 | 100.00% | 24.00 | 0.522s |
| 30 | 3 | 1 | 33.33% | 38.67 | 1.089s |
| 35 | 3 | 1 | 33.33% | 43.67 | 1.260s |
| 40 | 3 | 1 | 33.33% | 48.67 | 1.360s |
| 45 | 3 | 1 | 33.33% | 52.67 | 1.561s |
| 50 | 3 | 1 | 33.33% | 57.67 | 1.677s |

## Focused Level 50 Recheck

Because the broad sweep still showed a non-zero result at Level 50, I reran Level 50 with 10 games:

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 50 | 10 | 1 | 10.00% | 248.40 | 13.400s |

## Comparison

| Checkpoint | Estimated Cliff |
|------------|-----------------|
| L2-only Checkpoint | No zero-win level observed through Level 50; 5/10 at Level 50 recheck |
| L2-L3 Checkpoint | Between Level 28 and Level 29 |
| L2-L4 Checkpoint | No zero-win level observed through Level 50; weak 1/10 tail at Level 50 |
| L2-L6 Baseline | Level 30-31 |

## Findings

- Adding Level 4 data improves validation policy accuracy substantially over L3, but it does not create a stronger clean cliff boundary.
- The model remains perfect through Level 25.
- From Level 30 onward it settles into a low-probability survival regime rather than collapsing immediately.
- The larger Level 50 recheck confirms that the deep tail is real, but weak: only 1 win in 10 games.

## Cliff Estimate

There is no confirmed zero-win level through Level 50 in the broad pass.

The best description for L4 is a weak long tail rather than a sharp wall:

- strong through Level 25,
- partial survival from Level 30 through Level 50,
- but only 10.00% wins in the focused Level 50 recheck.

## Practical Takeaway

Level 4 looks like an unstable middle ground between the anomalously strong L2-only checkpoint and the much cleaner L3 cliff. It can still solve some deep seeded boards, but that competence is sparse and inconsistent rather than reliably extended.