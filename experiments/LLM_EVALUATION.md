# LLM Player Screening

This document defines the repeatable screening protocol for LLM-backed Klondike players. The
screen is deliberately small: it determines whether a model deserves a larger experiment; it is
not a statistically precise win-rate benchmark.

## Protocol

- Run 10 independently shuffled games per model and configuration.
- Use prompt version `P0`: the model states the rules and strategy it already knows before the
  deal, then receives complete boards and legal-move lists without engine guidance.
- Use one fresh stateful conversation per game.
- Cap each game at 200 iterations.
- Enable episode logging and assign a unique run ID.
- Change only one experimental variable between batches.
- Do not promote a 10-game win percentage to the root README as a benchmark result.

Example properties to add to either LLM result command:

```bash
-Dtest.games=10 \
-Dtest.max.moves.per.game=200 \
-Dlog.episodes=true \
-Dexperiment.run.id=ollama-qwen3-1.7b-p0-20260925
```

Use a unique run ID for each model/configuration. The engine generates an `auto-...` ID when the
property is omitted, but an explicit descriptive ID is easier to find in an append-only log. After
the run, generate the review report:

```bash
python experiments/tools/analyze_llm_episodes.py engine/logs/episode.log \
  --run-id ollama-qwen3-1.7b-p0-20260925 \
  --json-output experiments/runtime/work/qwen3-1.7b-analysis.json \
  --markdown-output experiments/runtime/work/qwen3-1.7b-analysis.md
```

Record the report and final judgment in `experiments/notebooks/llm_player_research.ipynb`.

## Ratings

Apply the first matching category. The analyzer suggests a category from objective thresholds;
the reviewer confirms or overrides it and records the reason.

### Technical failure

- Fewer than 80% of the planned games have episode summaries, or
- repeated malformed responses, timeouts, context failures, or process failures prevent a fair
  gameplay assessment.

Do not judge playing strength from this batch. Fix the integration problem and rerun it.

### Not viable

- The batch is technically valid and contains no wins, and
- median maximum foundation progress is at most four cards while median tableau discovery is at
  most four cards, or at least half the completed games are pathological repetition cases.

A pathological repetition case has at least 20 recorded turns and repeats 30% or more of its
visited state keys.

### Needs prompting

- The batch is technically valid but does not meet the promising threshold, and
- the model demonstrates coherent progress while making recurring, identifiable strategic errors.

Use the episode review to name one general failure mode before changing the prompt. Examples are
poor stock planning, premature foundation play, failure to exploit newly opened tableau space, or
unjustified reversals. Do not bundle several strategy changes into one prompt revision.

### Promising

- At least one game is won, or
- at least half the completed games reach a strong late position: 24 foundation cards or no more
  than three face-down tableau cards.

A promising result qualifies the model for a larger run. It does not establish its true win rate.

## Review Checklist

### Run validity

- Confirm the exact model, provider, reasoning/thinking setting, prompt version, move cap, and run
  ID in the `EPISODE_GAME` metadata.
- Confirm that all games use fresh conversations and random deals.
- Account for every planned game and investigate missing summaries.
- Separate gameplay failures from response-schema, timeout, context, and process failures.

### Outcome and progress

- Record wins, termination reasons, successful moves, and duration.
- Compare maximum and final foundation cards.
- Compare initial and minimum remaining face-down tableau cards.
- Identify games that reached a strong position but failed to finish.
- Treat a voluntary quit separately from a move-cap ending.

### Coherence

- Inspect games with the highest repeated-state ratio.
- Inspect immediate source/destination reversals and foundation-to-tableau moves in context; these
  moves are not automatically mistakes.
- Check whether stock turns repeatedly revisit the same state without creating opportunities.
- Check whether the model responds to newly revealed cards and changed legal moves.
- Compare play with the model's recorded pre-game strategy.

### Final judgment

- Confirm or override the suggested rating.
- State the dominant evidence in two or three sentences.
- Link representative game numbers rather than describing every move.
- Decide one next action: stop, fix a technical problem, test one prompt change, or run a larger
  sample.

## Logged Records

- `EPISODE_GAME` contains run, model, session/conversation, prompt, and pre-game strategy metadata.
- `EPISODE_STEP` contains the state before a command, legal moves, selected command, and immediate
  progress signals.
- `EPISODE_SUMMARY` contains outcome, termination reason, duration, and final board progress.

Older episode logs without `EPISODE_GAME` records remain analyzable, but their model identity and
strategy cannot be reconstructed from the episode file alone.
