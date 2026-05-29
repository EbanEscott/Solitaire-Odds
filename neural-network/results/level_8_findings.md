# Level 8 Findings

Date: May 29, 2026

## Summary

The Level 8 checkpoint is a regression relative to the Level 7 checkpoint. The expanded broad sweep was configured to continue in 5-level steps through Levels 55, 60, and beyond until the first zero-win level appeared, but the first zero arrived immediately at Level 30, so the deeper 5-step levels were not needed.

## Training Snapshot

- Checkpoint: `policy_value_level8_20260529T181151.pt`
- Model: PolicyValueNet
- Architecture: 256 hidden units, 2 layers
- Training data: archived Levels 2-8 episode logs
- Dataset size: 8504 training samples, 944 validation samples
- Validation accuracy: 72.99% policy, 100.00% value

Archived training inputs:

- `../engine/logs/archive/level2/20260529T071709Z/episode.log`
- `../engine/logs/archive/level3/20260529T071711Z/episode.log`
- `../engine/logs/archive/level4/20260529T071714Z/episode.log`
- `../engine/logs/archive/level5/20260529T071718Z/episode.log`
- `../engine/logs/archive/level6/20260529T071724Z/episode.log`
- `../engine/logs/archive/level7/20260529T071736Z/episode.log`
- `../engine/logs/archive/level8/20260529T081124Z/episode.log`

## Broad Seeded Sweep

The broad sweep was set up to test Levels 20, 25, 30, 35, 40, 45, 50, 55, 60, and higher in 5-level steps, stopping at the first zero-win result.

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 20 | 20 | 15 | 75.00% | 21.15 | 0.347s |
| 25 | 20 | 15 | 75.00% | 26.10 | 0.423s |
| 30 | 19 | 0 | 0.00% | 36.53 | 0.886s |

Because the first zero-win level was already Level 30, the planned 35+ and 55/60+ broad checks were not needed to locate the zero boundary.

## Dense Follow-up Near the Wall

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 26 | 20 | 15 | 75.00% | 27.10 | 0.442s |
| 27 | 20 | 15 | 75.00% | 28.10 | 0.453s |
| 28 | 20 | 15 | 75.00% | 29.10 | 0.466s |
| 29 | 20 | 14 | 70.00% | 30.65 | 0.510s |
| 30 | 19 | 0 | 0.00% | 36.53 | 0.897s |

## Comparison

Compared with earlier checkpoints at the key boundary levels:

| Level | L2-L6 Baseline | L2-L7 Checkpoint | L2-L8 Checkpoint |
|-------|----------------|------------------|------------------|
| 20 | 19/20 (95.00%) | 20/20 (100.00%) | 15/20 (75.00%) |
| 25 | 15/20 (75.00%) | not part of the L7 broad sweep | 15/20 (75.00%) |
| 29 | 14/20 (70.00%) | 20/20 (100.00%) | 14/20 (70.00%) |
| 30 | 2/19 (10.53%) | 19/19 (100.00%) | 0/19 (0.00%) |

## Findings

- The Level 8 checkpoint did not extend the Level 7 wall.
- It regressed sharply from the Level 7 result: Level 7 stayed perfect through Level 41, while Level 8 is already at zero by Level 30.
- Relative to the old Level 6 baseline, Level 8 roughly matches the mid-20s band but is worse at Level 30.
- The training metrics looked better than Level 7 on held-out policy accuracy, but that did not translate into deeper seeded generalization.

## Cliff Estimate

The best supported estimate is that the Level 8 checkpoint cliff is around Level 30.

More precisely:

- Lower bound: no earlier than Level 29, because Level 29 still won 14/20.
- Upper bound: no later than Level 30, because Level 30 dropped to 0/19.

## Practical Takeaway

Simply adding Level 8 episodes to this flat-board MLP training pipeline did not continue the Level 7 improvement. On seeded endgame generalization, the Level 8 checkpoint falls back to roughly the old pre-Level-7 wall and should be treated as a regression candidate rather than the new default.