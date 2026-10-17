# Configurable Klondike rules checklist

Work branch: `configurable-klondike-rules`.

Implement three independent settings: `drawCount` (`1` or `3`), `stockVisibility` (`topOnly` or `allDrawn`), and `maxStockPasses` (`1`, `3`, or `null` for unlimited). Intended defaults are `3`, `topOnly`, and `null`. Keep foundation-to-tableau moves (worry-back) enabled.

## 1. Establish the rules contract

- [x] Confirm defaults: `3`, `topOnly`, unlimited.
- [x] Define pass counting and recycle-plus-draw behavior.
- [x] Define knowledge behavior for training undo versus search rollback.
- [x] Audit historical benchmark visibility before labelling existing results.
- [x] Write [engine/RULES.md](../engine/RULES.md), including worry-back and worked examples.

The implementation inspected during stage 1 marked every drawn card as revealed and retained the entire waste in planning copies, while some model observations exposed only its top card. Verify benchmark revisions before describing historical results as uniformly `topOnly`.

Stage 1 is documentation only. The rules document describes the agreed target, not a claim that the current engine already implements it.

### Historical visibility audit

Source inspection on 2026-10-17 found:

| Revision / evidence | Finding |
| --- | --- |
| `b0ccc6a` (`A* results`), `Solitaire.turnThree()` | Draws up to three, recycles without a pass limit, and marks every drawn card as revealed. |
| `b0ccc6a`, `player/ai/astar/AStarPlayer.java` | Creates search states with `solitaire.copy()`; the historical copy implementation retains actual hidden identities on GAME boards. These results cannot be certified as knowledge-limited `topOnly`. |
| `fd12cd7` (`Record Astra model-backed results`), `Solitaire.turnThree()` | Still marks every drawn card as revealed internally. |
| `fd12cd7`, `BoardFormatter.appendStockAndTalon()` | Displays only the waste top and count, plus the stock count. |
| `fd12cd7`, `GamePrompts.buildTurnPrompt()`, Codex and Copilot players | Model turn prompts use that formatted board; covered waste identities are not listed in that board. |
| `fd12cd7`, `TypeSafePlayer.visibleBoard()` and observation format | Structured waste observations contain count and visible top; historical observations retain waste count and top. |

Paths above are relative to `engine/src/main/java/ai/games/`. Inspect historical files with `git show <revision>:<path>`. This source audit establishes a difference between model-facing waste observations and engine/search access. It does not certify every historical run artifact, side channel, or player as implementing the new contract. In particular, top-only board output is not proof of full end-to-end knowledge isolation.

Preserve existing model-backed results as historical measurements. Do not label the entire published table as uniformly `topOnly`; later README work should describe the observed interface and disclose the broader historical search access separately. No published values or test dates were changed in stage 1.

### Run constraints

- Do not rerun LLM/model-backed result suites: the user has ruled out their cost.
- Non-LLM player reruns are acceptable when implementation is ready; stage 1 performs no game runs. Check service requirements before selecting neural tests.
- Keep development reruns in logs. The user has requested the search-player results refresh in stage 6 after implementation and validation are complete.

## 2. Implement core rules

- [x] Add an immutable, validated rules configuration.
- [x] Thread configuration through game creation and copies.
- [x] Implement configurable draw count and stock pass limits.
- [x] Centralize stock-turn legality.
- [x] Preserve card order when recycling.
- [x] Include rule-relevant state in copying, undo, hashing, and comparisons.

Implemented `GameRules`, the explicit `Solitaire(deck, rules)` constructor, and JVM configuration through `game.drawCount`, `game.stockVisibility`, and `game.maxStockPasses`. Existing constructors capture runtime configuration once; copies retain it. `turnStock()` reports success, and `turnThree()` remains a compatibility wrapper using the configured count. Both move-generation paths use `canTurnStock()`. The game loop rejects blocked turns without counting them as successful moves; rule-based and greedy fallback decisions respect the limit.

Pass state survives copies and rollback. Rules and finite pass counts participate in keys/comparisons; unlimited pass counts do not prevent cycle detection. At the end of stage 2, knowledge tracking, knowledge-sensitive keys, and training-undo semantics remained for stage 3; visibility was stored but not yet enforced.

Validation: `./gradlew test --tests 'ai.games.unit.*' --console=plain` passed 278 tests on 2026-10-17, including six new rules tests covering the configuration matrix, draw/recycle order, partial draws, rejected turns, remaining waste moves, copies, rollback, and state identity. No results suites or model calls were run.

## 3. Enforce player knowledge

- [x] Implement `topOnly` and `allDrawn` observation behavior.
- [x] Reveal the next waste card after its top card is played.
- [x] Preserve observed identities across recycling.
- [x] Mask unknown stock and waste cards in planning views.
- [x] Update reveal resolution and audit search players for hidden-information access.
- [x] Align CLI, model observations, prompts, and neural-service inputs with configured rules.

Both visibility policies now control actual observations and planning reveals. Waste/tableau exposure records knowledge; search rollback restores it exactly, while `undoTrainingMove()` retains real observations and restores pass allowance. Covered knowledge participates in state keys, and comparisons distinguish knowledge differences.

All search-player entry points use masked boards. A* counts only exposed UNKNOWN slots, Monte Carlo resolves those slots or samples complete hypothetical worlds, and beam/AlphaSolitaire stop further expansion at unresolved reveal leaves. Simple search players no longer inspect real hidden identities through copies. The legacy planning move entry point delegates to the maintained generator.

CLI rendering, conversational prompts, TypeSafe observations, neural requests, and episode steps respect the same boundary. Requests describe rules and retain observed pile order. Neural service ingestion now supplies the unseen-card inventory to existing features. Its current feature vector does not separately encode rule settings, pass count, or remembered pile order; these remain request metadata, with legality enforced by the engine. No checkpoint or architecture changes were made.

Validation: the Java unit suite plus `GamePromptsTest` and mocked `TypeSafePlayerTest` passed 298 tests. The Python service observation test passed for camelCase and snake_case inventories using a local stub model. Formatter coverage verifies fresh observations when a renderer is reused. No results suites or paid/model-backed game runs were executed.

## 4. Verify behavior

- [ ] Cover all 12 configuration combinations.
- [ ] Test partial draws, waste exposure, retained knowledge, and recycling order.
- [ ] Test pass boundaries without prematurely ending otherwise playable games.
- [ ] Test copies, rollback, state keys, and blocked stock turns.
- [ ] Check observations for hidden-information leaks.
- [ ] Run relevant engine tests and player smoke tests.

## 5. Clean up documentation and reporting

- [ ] Condense `engine/README.md` into setup, configuration, commands, and testing.
- [ ] Move durable implementation details into player documentation.
- [ ] Move development commentary into existing experiment notes.
- [ ] Record rules in episode and benchmark metadata.
- [ ] Clarify root README defaults and verified historical result conditions.
- [ ] Leave published result values and test dates unchanged until stage 6.

Keep `engine/RULES.md` as the lasting rules specification. Keep interim results in experiment notes or run logs.

## 6. Refresh search-player results

After stages 2–5 are complete, rerun the six search players with `drawCount=3`, `stockVisibility=topOnly`, and `maxStockPasses=null` (unlimited), with worry-back enabled and training undo disabled. This aligns stock visibility with the historical model-facing observations for comparison. Preserve the existing LLM/model-backed results and their original dates; do not rerun those suites.

Use the corresponding tests in `engine/src/test/java/ai/games/results`, gather their logged results, and refresh each player's row in the root README.

- [ ] Rule-based Heuristics — run `RuleBasedHeuristicsPlayerResultsTest` and refresh its results.
- [ ] Greedy Search — run `GreedySearchPlayerResultsTest` and refresh its results.
- [ ] Hill-climbing Search — run `HillClimberPlayerResultsTest` and refresh its results.
- [ ] Beam Search — run `BeamSearchPlayerResultsTest` and refresh its results.
- [ ] Monte Carlo Search — run `MonteCarloPlayerResultsTest` and refresh its results.
- [ ] A* Search — run `AStarPlayerResultsTest` and refresh its results.
- [ ] Record resolved rules, code revision, game counts, search budgets, move caps, run dates, and log locations so the refreshed results are reproducible.
- [ ] Update the root README's search-results test date and rules statement; preserve historical model-result dates and disclose remaining differences in benchmark conditions rather than presenting visibility alignment as full parity.

AlphaSolitaire is a separate neural player and is outside this six-player refresh.
