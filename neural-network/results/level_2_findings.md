# Level 2 Findings

Date: May 29, 2026

## Summary

The Level 2-only checkpoint is a surprising result. Even though it was trained from just the archived Level 2 episodes, it stayed above zero throughout the tested seeded range and still won half of a larger Level 50 probe. Within the original Phase 0 range, this backfill run did not reveal a hard cliff.

## Training Snapshot

- Model: PolicyValueNet
- Architecture: 256 hidden units, 2 layers
- Training data: archived Level 2 episode log only
- Dataset size: 29 training samples, 3 validation samples
- Validation accuracy: 0.00% policy, 100.00% value

Archived training inputs:

- `../engine/logs/archive/level2/20260529T071709Z/episode.log`

## Broad Seeded Sweep

Quick backfill sweep with 3 games per level:

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 2 | 3 | 3 | 100.00% | 1.00 | 0.033s |
| 5 | 3 | 3 | 100.00% | 4.00 | 0.101s |
| 10 | 3 | 3 | 100.00% | 9.00 | 0.274s |
| 15 | 3 | 3 | 100.00% | 14.00 | 0.488s |
| 20 | 3 | 3 | 100.00% | 19.00 | 0.779s |
| 25 | 3 | 3 | 100.00% | 24.00 | 1.040s |
| 30 | 3 | 2 | 66.67% | 33.00 | 1.643s |
| 35 | 3 | 2 | 66.67% | 38.00 | 1.983s |
| 40 | 3 | 2 | 66.67% | 43.00 | 2.314s |
| 45 | 3 | 1 | 33.33% | 51.33 | 2.821s |
| 50 | 3 | 1 | 33.33% | 56.33 | 3.164s |

## Focused Recheck at Level 50

Because the broad sweep sample was small and the result was unexpectedly strong, Level 50 was rerun with 10 games:

| Level | Games Tested | Games Won | Win Rate | Avg Moves | Avg Time/Game |
|-------|--------------|-----------|----------|-----------|---------------|
| 50 | 10 | 5 | 50.00% | 54.00 | 2.878s |

## Comparison

| Checkpoint | Observed Behaviour in Tested Range |
|------------|------------------------------------|
| L2-only Checkpoint | No zero-win level observed through Level 50 |
| L2-L6 Baseline | Sharp cliff at Level 30-31 |
| L2-L7 Checkpoint | Cliff around Level 42 |

## Findings

- The L2-only checkpoint does not resemble the later L6 baseline curve at all.
- It weakens gradually after the mid-20s but stays above zero even at Level 50 in this backfill run.
- The larger Level 50 recheck strengthened the same conclusion: the checkpoint is weak there, but not dead.
- This suggests the later regressions are not explained by "not enough curriculum depth" alone. Some later cumulative checkpoints appear to introduce interference that this tiny Level 2 model avoids.
- The extremely small dataset means the validation policy metric is noisy and should not be treated as a stable quality estimate.

## Cliff Estimate

No hard cliff was observed within the tested Level 2-50 range.

More precisely:

- Performance is clearly degraded by Levels 45-50.
- But the checkpoint still wins non-zero fractions there, including 5/10 at Level 50.
- The true zero boundary, if it exists for this evaluation setup, is beyond the tested range.

## Practical Takeaway

The L2-only checkpoint is a useful counterexample for the curriculum story. More levels did not automatically produce a stronger flat-board MLP. In this seeded test setup, the tiny L2-only model remained viable far deeper than some of the later L8-L10 cumulative checkpoints.