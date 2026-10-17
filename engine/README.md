# Engine

Java/Spring Boot command-line Klondike with human, search, model-backed, and neural players. See [RULES.md](RULES.md) for gameplay, visibility, retained knowledge, stock passes, and training undo.

## Setup and running

Use JDK 21 and the bundled Gradle wrapper (8.12). The build uses Spring Boot 3.5.7. Run all commands below from `engine/` and select one player profile.

```bash
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-human
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-human -Dtraining.mode=true
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-astar -Dguidance.mode=false
```

On Windows, use `gradlew.bat`. Run `chcp 65001` for Unicode suits, or use `-Dsuit.ascii=true` for ASCII suits.

## Configuration

The wrapper forwards JVM properties to the application and tests. Switch rules with `-D` options, just as you switch training mode:

```bash
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-human -Dgame.drawCount=1 -Dgame.stockVisibility=allDrawn -Dgame.maxStockPasses=3
```

| Property | Values | Default |
| --- | --- | --- |
| `game.drawCount` | `1`, `3` | `3` |
| `game.stockVisibility` | `topOnly`, `allDrawn` | `topOnly` |
| `game.maxStockPasses` | `1`, `3`, `null` (unlimited) | `null` |
| `game.thoughtful` | `true`, `false`; full initial knowledge without changing legal moves | `false` |
| `training.mode` | `true`, `false`; enables player undo | `false` |
| `guidance.mode` | `true`, `false`; recommendations and feedback | `true` |
| `log.episodes` | `true`, `false`; structured episode records | `false` |
| `max.moves.per.game` | Maximum game-loop iterations, including rejected commands | `10000` |
| `suit.ascii` | `true`, `false`; ASCII suit rendering | `false` |

Rules are independent and captured once at game construction directly from JVM system properties. They are not read from Spring configuration files. Invalid values fail at construction. Java callers can supply an explicit immutable `GameRules` through `new Solitaire(deck, rules)`; copies retain the source rules.

Enable `-Dgame.thoughtful=true` to make every card identity and position known from the start. Face-down tableau cards remain unplayable until uncovered, and stock cards still require drawing. `stockVisibility` remains recorded but reveals nothing new in thoughtful mode.

Only the waste top is playable under either visibility policy. Observed identities stay known through covering and recycling. Pass 1 is the initial traversal; a permitted `turn` on empty stock recycles and draws in one action. Reaching the pass limit leaves other legal moves available. Foundation-to-tableau moves remain enabled.

## Player profiles

| Profile | Player / documentation |
| --- | --- |
| `ai-human` | Human console input |
| `ai-rule` | Rule-based heuristics |
| `ai-greedy` | One-step greedy search |
| `ai-hill` | Hill-climbing search |
| `ai-beam` | Bounded beam search |
| `ai-mcts` | [Monte Carlo search](src/main/java/ai/games/player/ai/mcts/README.md) |
| `ai-astar` | [A* search](src/main/java/ai/games/player/ai/astar/README.md) |
| `ai-alpha-solitaire` | [AlphaSolitaire](src/main/java/ai/games/player/ai/alpha/README.md), requiring the [Python service](../neural-network/README.md) |
| `ai-openai` | OpenAI API |
| `ai-ollama` | Local Ollama service |
| `ai-codex` | Codex CLI |
| `ai-copilot` | GitHub Copilot CLI |
| `ai-typesafe` | TypeSafe AI decision API |

See [model-backed player setup](src/main/java/ai/games/player/ai/README.md) for authentication, model selection, prompt profiles, session handling, and retry settings. Rule-based foundation selection uses only exposed tableau tops; its visible-stack scan targets tableau piles only. Search internals are documented beside each player; shared copying, masking, and move-generation details are in [game state and planning](src/main/java/ai/games/game/README.md).

## Gameplay commands

- `turn` — draw up to the configured count, recycling first if permitted.
- `move FROM TO` — move the top card, for example `move W T3` or `move T7 F1`.
- `move FROM CARD TO` — select the start card of a tableau sequence.
- `undo` — restore the previous position and pass allowance while retaining observations; requires training mode.
- `quit` — end the game.

Pile codes are `T1`–`T7` for tableau, `F1`–`F4` for foundations, and `W` for waste (also called talon). Draw from stock with `turn`.

## Build and verification

Compile and package without running result suites:

```bash
./gradlew bootJar
```

Routine local verification:

```bash
./gradlew test --tests 'ai.games.unit.*' --console=plain
./gradlew test --tests ai.games.player.ai.ConfiguredRulesSmokeTest --console=plain
```

The smoke suite covers six search players across all 24 rule combinations. It checks bounded decisions and board integrity, not win rates. Select functional tests individually: some require external model or neural services. An unrestricted `test` or `build` also selects result suites.

For a recorded search-only sweep on identical seeded deals, run `python3 scripts/search_sweep.py --games 100`. Configure players, rules, seeds, and move caps through its flags; outputs include per-game results, termination reasons, deck orders, metadata, and the actual source snapshot. The wrapper validates commands, so timings include that overhead.

See [result-suite instructions](src/test/java/ai/games/results/README.md) for explicit benchmark selection and game/move limits. Development measurements belong in logs or [experiment notes](../experiments/README.md). The [root results table](../README.md) contains historical measurements; updating defaults does not retroactively change their conditions.

## Logs and metadata

`logs/game.log` contains a `GAME_RULES` startup record for every game, including benchmarks with episode logging disabled. It records the actual draw count, visibility, pass limit, thoughtful mode, starting pass, solver, run ID, optional game index, training/guidance settings, and iteration cap.

Enable `-Dlog.episodes=true` for structured JSON records in `logs/episode.log`:

- `EPISODE_GAME` records player metadata after the first response, with `metadata.game_rules` and `metadata.stock_pass`.
- `EPISODE_STEP` records the masked observation before a command, legal moves, chosen action, `game_rules`, and `stock_pass`.
- `EPISODE_SUMMARY` records termination, progress, duration, `game_rules`, and the final `stock_pass`.

The `game_rules` object uses `drawCount`, `stockVisibility`, `maxStockPasses`, and `thoughtful`; unlimited passes serialize as JSON `null`. A game that exits before its first response still has a startup rules record and, when enabled, a summary. Episode steps include `known_stock_next_draw_first` and `known_tableau_bottom_to_top`, with `tableau_face_down` preserving physical face-down counts. These fields carry full positions in thoughtful mode. Episode observations mask unknown identities and include the unseen-card inventory; standard-mode observations do not expose the true hidden arrangement.

Set `-Dexperiment.run.id=<id>` to identify a run; otherwise the JVM generates an `auto-...` identifier. Set `SOLITAIRE_LOG_DIR` to a separate directory for concurrent runs. Preserve logs together with the code revision, game counts, deals/seeds, and player search budgets when reporting results.

Greedy search (`ai-greedy`) evaluates every legal immediate successor, including stock turns and foundation-to-tableau moves, using only player knowledge. Its deterministic score rewards actual tableau reveals, net new empty columns, waste deployment, and foundation progress with a conservative safety preference. It does not expand a second ply. Per-instance observed history prevents repeated-position cycles; create a fresh player per deal or call `resetGame()`. Stock termination follows knowledge-aware repeated positions and configured pass limits rather than a fixed draw cutoff.

Rule-based heuristics (`ai-rule`) uses deterministic ordered rules without successor simulation: opening draw, exposed foundation moves, whole-run tableau reveals, waste deployment, whole-run consolidation, then stock turns. It records attempted actions per knowledge-aware observation instead of guessing whether a stock pass made progress. Newly observed identities and changed positions permit further exploration; repeating unchanged stock cycles eventually exhausts the available rules. Create a fresh instance per deal or call `resetGame()`.

Beam search (`ai-beam`) expands all legal successors, including foundation-to-tableau moves, before retaining the best states at each layer (default depth 2, width 8). It keeps the best-scoring path to each duplicate within a layer; pruned states do not globally block later paths. Unknown observations and terminal positions remain leaves within the beam. The best final surviving path supplies the next command even when it scores below the root; seeded randomness breaks ties only. Ancestor checks and per-game observation/action history prevent repeated cycles. Scoring weights are unchanged, and a fresh instance or `resetGame()` starts a new deal.

Hill-climbing (`ai-hill`) evaluates each legal immediate successor once, chooses the best strict improvement, and uses seeded ties. At a plateau it prefers up to eight consecutive sideways moves, then permits one random eligible local perturbation; these are not restarts or backtracking. All moves share a 5,000-action budget. Stock and waste cards have the same undeployed-card cost, so draws and recycling alone are neutral. Actual observation and action history prevent repeated cycles without recording unknown predictions as facts. Create a fresh instance per deal or call `resetGame()`.
