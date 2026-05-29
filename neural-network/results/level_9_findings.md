# Level 9 Findings

Date: May 29, 2026

## Summary

The Level 9 checkpoint is a severe regression on seeded endgame evaluation. After restoring the standard 1000-move safety cap used by the result sweeps, the seeded wall fell from the Level 8 band around Level 30 down to roughly Level 18, with instability already visible at Level 15.

## Training Snapshot

- Checkpoint: `policy_value_level9_20260529T182029.pt`
- Model: PolicyValueNet
- Architecture: 256 hidden units, 2 layers
- Training data: archived Levels 2-9 episode logs
- Dataset size: 11302 training samples, 1255 validation samples
- Validation accuracy: 73.94% policy, 100.00% value

Archived training inputs:

- `../engine/logs/archive/level2/20260529T071709Z/episode.log`
- `../engine/logs/archive/level3/20260529T071711Z/episode.log`
- `../engine/logs/archive/level4/20260529T071714Z/episode.log`
- `../engine/logs/archive/level5/20260529T071718Z/episode.log`
- `../engine/logs/archive/level6/20260529T071724Z/episode.log`
- `../engine/logs/archive/level7/20260529T071736Z/episode.log`
- `../engine/logs/archive/level8/20260529T081124Z/episode.log`
- `../engine/logs/archive/level9/20260529T082013Z/episode.log`

## Evaluation Notes

The original uncapped level sweep stalled inside the first Level 20 game because `AlphaSolitaireLevelTest` was not applying the shared move cap. The test harness now reuses `ResultsConfig.MAX_MOVES_PER_GAME` (default 1000), and all results below use that capped configuration.

Because the Level 20 capped probe was already a zero-win loss and the dense follow-up found the first zero at Level 18, deeper Level 19-20 dense checks were not needed.

## Coarse Seeded Sweep

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 5 | 3 | 3 | 100.00% | 4.00 | 0.035s |
| 10 | 3 | 3 | 100.00% | 9.00 | 0.256s |
| 15 | 3 | 1 | 33.33% | 671.33 | 51.462s |
| 20 | 1 | 0 | 0.00% | 1000.00 | 78.075s |

The Level 20 row is a capped single-game confirmation probe. It was enough to show that the broad sweep had already crossed into the zero-win band.

## Dense Follow-up Near the Wall

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 16 | 1 | 1 | 100.00% | 15.00 | 0.818s |
| 17 | 1 | 1 | 100.00% | 16.00 | 0.828s |
| 18 | 1 | 0 | 0.00% | 1000.00 | 77.594s |

Level 19 and Level 20 were not rerun in the dense pass because Level 18 already produced the first zero-win row, and Level 20 had already been separately confirmed as a capped loss.

## Comparison

| Checkpoint | Estimated Cliff |
|------------|-----------------|
| L2-L6 Baseline | Level 30-31 |
| L2-L7 Checkpoint | Around Level 42 |
| L2-L8 Checkpoint | Around Level 30 |
| L2-L9 Checkpoint | Between Level 17 and Level 18 |

## Findings

- The Level 9 checkpoint regressed far below the Level 8 checkpoint instead of extending it.
- Level 15 is already unstable, losing 2 of 3 capped games and spending an average of 671.33 moves.
- Level 17 still completes a seeded win, but Level 18 hits the 1000-move cap and loses.
- The held-out policy accuracy improved again, but that metric is now clearly disconnected from seeded endgame generalization for this flat-board MLP pipeline.

## Cliff Estimate

The best supported estimate is that the Level 9 checkpoint cliff is around Level 18.

More precisely:

- Instability begins by Level 15, because that row already drops to 1/3.
- The last confirmed winning level is Level 17.
- The first confirmed zero-win level is Level 18.

## Practical Takeaway

Adding Level 9 episodes pushed this checkpoint in the wrong direction. Relative to the Level 7 and Level 8 checkpoints, the Level 9 model is much less robust on seeded endgames and should be treated as a regression candidate rather than a stronger default.