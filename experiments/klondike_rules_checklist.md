# Configurable Klondike rules checklist

Work branch: `configurable-klondike-rules`.

Implement three independent settings: `drawCount` (`1` or `3`), `stockVisibility` (`topOnly` or `allDrawn`), and `maxStockPasses` (`1`, `3`, or `null` for unlimited). Intended defaults are `3`, `topOnly`, and `null`. Phase 6 adds independent `thoughtful` (`false` by default) for full initial knowledge. Keep foundation-to-tableau moves (worry-back) enabled.

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
- Keep development reruns in logs. The user has requested the search-player results refresh in stage 7 after implementation and validation are complete.

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

- [x] Cover all 12 configuration combinations.
- [x] Test partial draws, waste exposure, retained knowledge, and recycling order.
- [x] Test pass boundaries without prematurely ending otherwise playable games.
- [x] Test copies, rollback, state keys, and blocked stock turns.
- [x] Check observations for hidden-information leaks.
- [x] Run relevant engine tests and player smoke tests.

Verification completed on 2026-10-17: **382 Java tests passed, with no failures or skips**, plus the Python service observation-ingestion test. No production behavior changes were needed during this stage.

- `RulesBoundaryMatrixTest`: 12 independently reported configurations on a complete deck with an exact five-card stock. Covers partial final draws, observation counts, repeated recycling order, pass exhaustion, GAME/PLAN copies, rollback, state keys, remaining waste moves, and foundation-to-tableau worry-back.
- `ConfiguredRulesSmokeTest`: 72 cases covering six search players × 12 configurations. Each begins on the same shuffled deal (seed `20261017`) after a complete stock traversal and checks up to 12 player decisions. Checks actual command legality, board integrity, pass limits, and that searches do not mutate real cards, knowledge, history, or pass allowance. Hill/beam use seed `42`; Monte Carlo retains its normal sampling behavior. These are bounded smoke runs, not completed-game win-rate measurements.
- Existing game and knowledge tests cover exposure after waste moves, permanent observations, training undo versus search rollback, hidden-arrangement invariance, and masked CLI/model/neural payloads. Mocked model-interface tests make no external calls.

The smoke harness accepts both the engine's top-card shorthand and explicit-card commands; string equality with the canonical move list alone is not a valid legality test.

Run from `engine/`:

```bash
./gradlew test --tests 'ai.games.unit.*' --tests ai.games.player.ai.ConfiguredRulesSmokeTest --tests ai.games.player.ai.GamePromptsTest --tests ai.games.player.ai.TypeSafePlayerTest --console=plain
```

Run from `neural-network/`:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Local verification output: `engine/logs/profiles/klondike-stage4/verification.log`. Published result values and dates remain unchanged. No results-package suites, LLM games, paid model calls, or live neural-service runs were performed.

## 5. Clean up documentation and reporting

- [x] Condense `engine/README.md` into setup, configuration, commands, and testing.
- [x] Move durable implementation details into player documentation.
- [x] Move development commentary into existing experiment notes.
- [x] Record rules in episode and benchmark metadata.
- [x] Clarify root README defaults and verified historical result conditions.
- [x] Leave published result values and test dates unchanged until the search-results refresh (now stage 7).

Keep `engine/RULES.md` as the lasting rules specification. Keep interim results in experiment notes or run logs.

Validation: the Java unit suite passed 310 tests with no failures or skips, including 12 rules-metadata cases. Startup logs record actual rules even when episode logging is off; episode game, step, and summary payloads retain rules and pass counts. Documentation links were checked, and published table values and run dates were compared against the committed README and remain unchanged. No benchmark suites or model games were run. Local output: `engine/logs/profiles/klondike-stage5/verification.log`.

## 6. Add Thoughtful Klondike

Implemented on 2026-10-17. Added independent Boolean `thoughtful`, default `false`, configured through `-Dgame.thoughtful=true|false`. Existing three-setting behavior stays the default. This expands the configuration matrix from 12 to 24 combinations; it does not change the physical deal, legal moves, draw count, recycling order, pass allowance, or worry-back.

### Contract and justification

With `thoughtful=false`, initial knowledge includes only visible cards, and later observations follow `stockVisibility`. With `thoughtful=true`, all 52 identities and their exact positions are known from the start. Face-down tableau cards remain face-down and unplayable until uncovered; stock cards still require drawing; only the waste top is playable. Full knowledge persists across moves, recycling, copies, and undo.

Keep `stockVisibility` independent and recorded in thoughtful mode. Both values have the same informational effect there because all cards are already known; do not silently rewrite the selected value. `allDrawn` alone is not thoughtful mode: it does not expose the initial stock or face-down tableau.

Thoughtful mode enables full-information investigation of a deal under its exact movement rules. A validated winning sequence proves winnability. Proving unwinnability requires a complete, sound search of legal continuations; a timeout, move cap, heuristic prune, or exhausted bounded player budget means unresolved. The existing A*, Monte Carlo, beam, hill-climbing, greedy, and rule-based players are not certified decision procedures. Enabling this setting does not make their losses proofs. Relevant research: [Blake and Gent, The Winnability of Klondike Solitaire and Many Other Patience Games](https://arxiv.org/abs/1906.12314).

A matched-deal standard/thoughtful comparison can measure an agent's response to extra information. It cannot by itself identify the cause of each loss or classify deals as unwinnable. A future winnability benchmark should distinguish proven winnable, proven unwinnable, and unresolved, retain winning traces/proof evidence, and use matching draw/pass rules. Building that certifying solver is a separate project, outside this phase.

### Implementation checklist

- [x] Extend the rules contract and runtime documentation with `thoughtful=false` by default, strict Boolean validation, its interaction with visibility, and the distinction between knowledge and legal access. Preserve source compatibility for existing three-argument `GameRules` construction by delegating to `thoughtful=false`.
- [x] Initialize thoughtful knowledge from the actual dealt board after dealing, without changing face-up counts or exposing hidden identities in standard games. Keep true state and player knowledge separate; use the ordinary PLAN path for both modes. A complete thoughtful deal must have no unknown identities or unresolved reveal slots.
- [x] Preserve the setting and complete knowledge through GAME/PLAN copies, stock turns, recycling, search rollback, and training undo. Include the setting in state keys and comparisons without colliding with the existing finite-pass encoding; retain unlimited-pass cycle detection.
- [x] Audit legal move generation and all six search players. Same physical position must have the same legal actions in either mode. Full-information planning must need no hypothetical assignments; Monte Carlo must not replace known cards. Existing heuristic budgets/pruning remain bounded and must not be presented as exhaustive solving.
- [x] Extend CLI rule display, model prompts, TypeSafe rule maps, and model/neural observations with the setting. Expose ordered known stock and full tableau positions while explicitly retaining face-down status and waste playability. Verify actual payloads using local/mocked tests, with no model-backed games.
- [x] Extend `GAME_RULES` and episode metadata with `thoughtful`. Add ordered known stock/tableau positions to episode observations so thoughtful information survives logging; use the same masked planning view for standard mode. Preserve explicit ordering and face-down counts. Old logs lacking the setting have unknown historical conditions, not automatically certified standard or thoughtful behavior.
- [x] Audit AlphaSolitaire request/service ingestion and document the feature limitation: complete known positions can be transported, but current features do not fully encode remembered stock/tableau order or rule settings. Do not claim full-information neural evaluation from metadata alone. Feature redesign and checkpoint retraining remain separate unless explicitly authorized.
- [x] Extend rule-boundary coverage to all 24 configurations and search smoke coverage to six players × 24 configurations. Test complete initial knowledge, retained positions, unchanged legality, no standard-mode leaks, copy/undo/key behavior, metadata, and model/neural payloads. Verify both visibility values behave identically informationally under thoughtful mode.
- [x] Run the Java unit and targeted local smoke/interface tests, plus relevant local Python ingestion tests if changed. Record results in development logs; do not run result suites or paid/live model games in this phase.
- [x] Update implementation docs and the checklist once complete. Keep published values/dates unchanged. Make `thoughtful=false` explicit in the stage 7 search refresh; keep thoughtful experiments separate and do not relabel historical search access as a verified thoughtful benchmark.

### Implementation and validation

Implemented `game.thoughtful` with strict Boolean parsing, a backward-compatible three-argument rules constructor, and full initial knowledge recorded through the existing knowledge tracker. Planning retains all positions without changing face-up counts. The flag participates in rule comparisons, state keys, CLI/model/neural observations, and startup/episode metadata. Episode steps now preserve ordered known stock and tableau positions with face-down counts.

New legality checks exposed an existing direct-command bug: `move S …` could bypass drawing and move the stock top. The engine now rejects stock-source moves in both modes; stock cards enter play only through `turn`. Search generators already excluded these commands.

Validation: 541 Java tests passed with no failures or skips, including 24 rule-boundary configurations, 144 search-player smoke cases, full knowledge/rollback and payload tests, and 24 metadata configurations. Two local Python tests passed using a stub model, verifying unseen-inventory ingestion and the current feature limitation for thoughtful input. No result suites, live services, model-backed games, or checkpoint training ran. Published result rows and dates remain unchanged. Java output: `engine/logs/profiles/klondike-stage6/verification.log`.

### Code impact map

| Area | Expected impact |
| --- | --- |
| `GameRules` | Fourth record component, default, validated JVM property, compatibility constructor. |
| `Solitaire`, `UnknownCardTracker` | Initial complete knowledge, existing masked planning path, knowledge retention, rule hashing/comparison. Prefer populating the knowledge boundary over bypassing it with raw GAME boards. |
| `LegalMovesHelper`, search players | Verify face-up/movement constraints remain independent of known identity; no unknown sampling needed for thoughtful boards. Heuristic behavior may still change with additional information. |
| `BoardFormatter`, `GamePrompts`, `TypeSafePlayer` | Add flag to explicitly constructed rule displays/maps and clarify known-but-face-down cards; existing ordered knowledge fields can carry full positions. |
| `AlphaSolitaireRequest`, Python service/features | Record and transport the flag/knowledge; distinguish payload support from feature/model support. |
| `Game`, `EpisodeLogger` | Startup flag plus structured metadata and complete ordered observations; serializing `GameRules` adds the field automatically, but manually assembled records require changes. |
| Rules, smoke, interface, metadata tests | Expand the matrix, verify exact positions and invariant move legality, retain standard-mode leak checks. |
| READMEs and result harness | Document the fourth switch, record exact rules, keep standard and thoughtful results separate. No historical relabelling or model reruns. |

## 7. Refresh search-player results

After stages 2–6 are complete, rerun the six search players with `drawCount=3`, `stockVisibility=topOnly`, and `maxStockPasses=null` (unlimited), with `thoughtful=false`, worry-back enabled, and training undo disabled. This aligns stock visibility with the historical model-facing observations for comparison. Preserve the existing LLM/model-backed results and their original dates; do not rerun those suites.

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
