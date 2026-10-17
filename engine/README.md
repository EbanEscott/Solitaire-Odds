# Engine

The engine is a Java/Spring Boot command-line implementation of Klondike Solitaire. It owns the core game rules, logging, and pluggable player system (human and AI). Search-based players such as greedy, hill-climbing, beam search, Monte Carlo, A*, and LLM-backed players (OpenAI and Ollama) all run here, and the AlphaSolitaire (MCTS + neural network) player integrates with the Python modeling stack in `../neural-network`.

## Prereqs

The engine is a Spring Boot command-line Solitaire (Klondike-style) app under the `ai.games` package.

- JDK 21+ (toolchain set to 21)
- Use the bundled Gradle wrapper (pinned to Gradle 8.7). Gradle 9.x is incompatible with Spring Boot 3.2.
  - If Gradle 9.x was cached: `rm -rf ~/.gradle/wrapper/dists/gradle-9.1.0-bin`

## Layout

### Source Code
- `src/main/java/ai/games/Game` — Spring Boot entry, constructor-injected `Player`.
- `src/main/java/ai/games/game/` — core model: `Solitaire`, `Deck`, `Card`, `Rank`, `Suit`.
- `src/main/java/ai/games/player/` — player base types:
  - `HumanPlayer` (default CLI)
  - `AIPlayer` base class
  - `LegalMovesHelper`
- `src/main/java/ai/games/player/ai/` — AI players (all `@Profile`-gated):
  - Search: `RuleBasedHeuristicsPlayer`, `GreedySearchPlayer`, `BeamSearchPlayer`, `HillClimberPlayer`, `MonteCarloPlayer`, `AStarPlayer`
  - Neural: `alpha/AlphaSolitairePlayer` (policy-value network)
  - LLM: `OpenAIPlayer`, `OllamaPlayer`, `CodexCliPlayer`, `CopilotCliPlayer`

### Build Files
`build.gradle`, `settings.gradle`, `gradlew*`, `gradle/wrapper/`.

### Simulation state copies

`Solitaire.copy()` allocates an empty state and copies the pile containers directly, sharing immutable cards. It avoids creating, shuffling, and dealing a throwaway deck on every search branch. Copies preserve the source's GAME or PLAN mode, revealed-card knowledge, and guesses without re-masking cards. They start with empty action history; guess containers are independent and share immutable values. Actions performed on a copy can be undone back to its starting position, independently of the source.

`Solitaire.planningCopy()` explicitly creates a PLAN board from a real game. It masks unseen tableau, stock, and covered waste cards, retains previously observed identities and their order, and leaves the real board unchanged. On an existing PLAN board it behaves like `copy()`. Mode is fixed at construction; `setMode()` has been removed. Both kinds of board continue to share the same movement and undo implementation inside `Solitaire`.

PLAN's possible unseen identities are derived once from the masked board and the standard deck. `getUnknownCards()` returns a retained immutable snapshot, shared safely by planning copies. `assumeCard(slot, identity)` replaces that board's pool with an ordered snapshot excluding the assigned identity and returns `1 / remainingCards`. The existing action record retains the previous pool, so `rollbackTo()` restores its exact ordering alongside the UNKNOWN slot, guess map, and state key. Ordinary moves and stock recycling do not change which identities remain available. The 52 immutable standard cards are also reused. GAME-mode unknown-card queries retain their existing behaviour.

All search players enter through `planningCopy()`. A* and Monte Carlo resolve only slots exposed by the selected visibility policy before continuing simulated play. Beam search and AlphaSolitaire evaluate unresolved reveal positions as leaves; greedy and hill-climbing evaluate masked one-step successors. An assigned card stays assigned when moved or recycled. UNKNOWN cards are no longer legal wildcards, and legal move generation never creates guesses as a side effect.

A* searches known cards first using `g + h`, deferring unknown reveals to a separate queue. Reveals are evaluated as complete outcomes: sets of up to eight are enumerated exactly; larger sets use eight uniform samples with no repeated identity within an outcome. Decision nodes choose the best continuation, while reveal nodes average outcome values using conditional probabilities or empirical sampling weights. Unexpanded branches retain their heuristic estimate. The search allows 1,024 expansions and 5,000 retained nodes, reserving half the initial expansion and node budgets for deferred work. This is bounded expected heuristic cost, not a calibrated win probability. Actual observations replace speculative descendants and reset probability to one. See [A* search policy](src/main/java/ai/games/player/ai/astar/README.md) for scheduling, sampling, pruning, and failure-cost details.

Monte Carlo samples a complete hidden arrangement from the PLAN unseen-card pool for each evaluation, including unknown stock and tableau slots. Its bounded depth-first solver treats those sampled identities as fixed full information, while preserving face-up counts and ordinary move legality. All assignments are undone before the next evaluation. It continues to use one board per root candidate, with no board copies inside a search. The previous random rollout evaluator remains selectable for comparison. Actual next-turn validation compares known cards and pile shapes against the predicted position, allowing predicted UNKNOWN slots to acquire real identities.

The root README results table is reserved for settled players. Interim game results and profiles remain in `engine/logs/profiles/` while the players are being improved; publish them only when requested.

Legal move generation in GAME and PLAN mode reads the visible tableau, face-up counts, foundations, and talon once per call and passes them to the move categories that need them. This avoids rebuilding identical snapshots and views while preserving move checks and ordering. These reads are local to the call; each subsequent call reads the current board again.

Tableau-to-tableau generation also computes each visible destination's required rank and colour once per call, then reads each source card's rank and colour once before checking destinations. Empty destinations accept either colour of king. Planning boards use the same placement rules after their revealed UNKNOWN cards have been assigned; unresolved UNKNOWN sources and destinations are excluded.

`LegalMovesHelper.listMoves()` generates `List<ai.games.game.Move>` directly. Monte Carlo uses these objects for root candidates, rollout pruning, and `TreeNode.applyMove(Move)`, avoiding command parsing in that path. The existing `listLegalMoves()` API remains a mutable list of strings for other players, guidance, logs, and service payloads, with the same order and card spelling. Human console input, `Player.nextCommand()`, and `TreeNode.applyMove(String)` remain text-based. Tree child keys remain strings, and the final call into `Solitaire.moveCard()` still uses its existing pile/card text API and validation.

### Monte Carlo evaluation

`MonteCarloPlayer` makes one root planning snapshot and one board copy per first legal move. Each candidate board is reused across evaluations. Every candidate is sampled once; rejected candidates are excluded. The remaining candidates share a total budget of `64 × eligible candidates` evaluations, including the initial samples. UCB (`mean + sqrt(2) × sqrt(log(total visits) / candidate visits)`) allocates the remaining samples. Individual candidates can receive more or fewer than 64 evaluations. The final move is selected by mean reward. Exact ties prefer an immediate win, then the higher existing board progress heuristic (foundation progress and tableau visibility). This uses the restored candidate planning board and does not change rollout rewards or UCB allocation.

The default evaluator is bounded depth-first search. `Solitaire.sampleUnknownCards(random)` assigns every UNKNOWN tableau, stock and waste slot uniformly without replacement from the PLAN pool. Known identities, existing assumptions and face-up counts are preserved. This operation uses the same assignment actions as `assumeCard()` and rejects GAME boards; ordinary `assumeCard()` still accepts only revealed slots, preserving A*'s reveal semantics.

The sampled arrangement stays fixed while DFS explores and backtracks. An explicit stack stores each position's remaining moves, incoming move, state key and history checkpoint. Moves follow legal-generation order after the existing inverse, useless-king and sibling-symmetry filters. Cycle detection rejects a repeated key on the active search path, with no global visited-state pruning across branches. Backtracking restores gameplay to a frame's checkpoint, retaining the sampled identities. The outer `finally` restores the candidate's original moves, assignments, unseen pool and cached key, including after exceptions. No board copies are made inside evaluation.

Each search permits 128 visits by default, counting its starting position and every attempted successor, including cycle rejections. A win on the final permitted visit earns `1.0`; exhausted permitted search earns `0.0`; remaining work at the visit limit earns `0.5`. Exhaustion is relative to the pruning rules. The cutoff score is a convention for unresolved search, not a calibrated win probability. Full-information solutions on sampled worlds can overestimate achievable play because different worlds may require different decisions before actual observations distinguish them.

Set `-Dmontecarlo.search.visits=1000` to experiment with a larger DFS budget. Set `-Dmontecarlo.evaluator=random` to retain the previous random policy (the default is `dfs`). Random rollouts allow 128 accepted moves, sample identities only as revealed, and score wins as `1.0`, exhausted continuations as `0.0`, and step-limit cutoffs using the normalised progress heuristic below `1.0`. Both evaluators reserve `-1.0` for rejecting a starting candidate; sampled losses remain eligible for further evaluations. The random evaluator retains its original cycle threshold of two prior occurrences.

Sampling, DFS backtracking, visit limits and rollback are covered by deterministic unit tests. The existing seeded copying-reference tests continue to cover the random comparator. Performance comparisons should use matching recorded deals and budgets; interim results stay in run logs.

`Solitaire.getStateKey()` computes the board hash lazily and reuses it until a successful card move or stock turn/recycle invalidates it. Undo restores the previous cached value and validity flag. Ordinary copies in either mode inherit a valid cached key. A GAME-to-PLAN `planningCopy()` calculates its own key because hidden cards have been masked. Callers, including cycle checks, keep using `getStateKey()` with the same API. The hash now includes pile sizes and face-up counts, so transfers of identical UNKNOWN placeholders and visibility changes are distinguishable.

`Solitaire` owns a single action history for card moves, stock turns, and planning assignments. Each successful action records the changes needed for `undoLastMove()` to restore pile order, face-up counts, revealed-card knowledge, and the hash cache directly, without re-dealing or replaying earlier moves. Failed moves and turns with both stock and talon empty add no history. Undo does not add actions. Save `getMoveHistorySize()` and call `rollbackTo(depth)` to undo a sequence using the same mechanism; depths refer to the current board and history branch. Training mode controls access to the player-facing `undo` command, which uses `undoTrainingMove()` to restore positions and pass allowance while retaining actual observations. Search uses `undoLastMove()` and `rollbackTo()` to restore exact simulated knowledge without retaining discarded guesses. Monte Carlo uses this rollback mechanism; other search players retain their existing copying behaviour.

## Running (from `engine/`)

Exactly one player profile must be active at a time. Run with `-Dspring.profiles.active=<profile>`.

### Quick Start

**Human player (CLI):**
```bash
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-human"

# Windows: Run 'chcp 65001' first for Unicode symbols (♥, ♦, ♣, ♠)
# Or use ASCII mode: add "-Dsuit.ascii=true" to display H, D, C, S instead
```

**Human player with training mode** (undo enabled):
```bash
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-human" "-Dtraining.mode=true"
```

**Codex CLI player using a ChatGPT subscription:**
```bash
codex login status # must report: Logged in using ChatGPT
./gradlew bootRun --console=plain \
  "-Dspring.profiles.active=ai-codex" \
  "-Dcodex.model=gpt-5.6-sol" \
  "-Dcodex.reasoning.effort=medium"
```

**AI player with guidance mode disabled** (suppress recommended moves and feedback):
```bash
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-astar" "-Dguidance.mode=false"
```

**Combined: training mode with guidance mode disabled:**
```bash
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-human" "-Dtraining.mode=true" "-Dguidance.mode=false"
```

### Configuration Properties

The engine supports the following runtime configuration options (passed via `-D<key>=<value>` or set in `src/main/resources/application.properties`):

#### Core Game Settings

Game rules are captured once per game using these JVM properties:

| Property | Values | Default |
| --- | --- | --- |
| `game.drawCount` | `1`, `3` | `3` |
| `game.stockVisibility` | `topOnly`, `allDrawn` | `topOnly` |
| `game.maxStockPasses` | `1`, `3`, `null` (unlimited) | `null` |

For example, add `-Dgame.drawCount=1 -Dgame.maxStockPasses=3` to `bootRun` or a selected test command. Invalid values fail at game construction. These properties are read directly from JVM system properties, not Spring's `application.properties`. Java callers can pass an explicit immutable `GameRules` to `new Solitaire(deck, rules)` to avoid runtime overrides.

All three settings are implemented. `topOnly` records only the final drawn card, while `allDrawn` records every drawn card. Playing the waste top observes its newly exposed successor; covering and recycling never erase observations. See [RULES.md](RULES.md) for the contract and [the checklist](../experiments/klondike_rules_checklist.md) for progress.

The initial stock traversal is pass 1. A permitted `turn` on empty stock recycles the waste and draws in one action. Exhausting the final pass disables stock turns, while other legal moves remain available. Copies and rollback preserve pass allowance; finite pass state participates in search keys and position comparisons. Unlimited traversal counts do not prevent cycle detection.

- **`training.mode`** (boolean, default: `false`)  
  Enable training mode to allow players to undo moves and explore alternative game paths without penalty.  
  Example: `-Dtraining.mode=true`

- **`guidance.mode`** (boolean, default: `true`)  
  Enable guidance mode to show recommended moves, feedback on illegal moves, and detection of unproductive patterns (ping-pong, repeated stock passes).  
  Useful for human players and LLM-based players; can be disabled for pure AI experiments.  
  Example: `-Dguidance.mode=false`

- **`log.episodes`** (boolean, default: `false`)  
  Enable episode logging to emit structured JSON logs for neural network training data generation.  
  Example: `-Dlog.episodes=true`

- **`max.moves.per.game`** (integer, default: `10000`)  
  Cap the maximum number of iterations (moves) per game. Games exceeding this limit are terminated to prevent runaway loops.  
  Example: `-Dmax.moves.per.game=1000`

### Player Profiles

#### Search-based Players

All search players use deterministic game-tree exploration with various strategies.

```bash
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-rule"          # Rule-based heuristics (deterministic rules)
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-greedy"        # Greedy search (one-step lookahead)
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-hill"          # Hill-climbing search (state-hash driven)
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-beam"          # Beam search (fixed-depth, fixed-width)
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-mcts"          # Monte Carlo Tree Search (MCTS)
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-astar"         # A* search (heuristic-guided tree exploration)
```

#### Neural Network Players

AlphaSolitaire combines Monte Carlo Tree Search with a learned policy-value network:

```bash
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-alpha-solitaire"  # MCTS + neural network (requires Python service)
```

See `../neural-network/README.md` for setup and training details.

#### LLM Players

Large language model-backed players via remote APIs or local inference:

```bash
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-ollama"        # Ollama via Spring AI (requires local Ollama)
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-openai"        # OpenAI via API (requires OPENAI_API_KEY or openai.apiKey)
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-codex"         # Codex CLI via ChatGPT subscription
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-copilot"       # GitHub Copilot CLI subscription
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-typesafe"      # TypeSafe AI Jev decision API
```

Ollama model selection:
- Default model is set in `src/main/resources/application.properties` (`ollama.model=llama3`).
- Override per run: `./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-ollama" "-Dollama.model=mistral-large:123b"`
- Or set env: `$env:OLLAMA_MODEL="mistral-large:123b"; ./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-ollama"`
- Each `OllamaPlayer` instance represents one game. It establishes and pins the selected profile's strategy before the deal, then retains a rolling window of recent board turns.
- Choose the shared game prompt profile with `-Dgame.prompt.profile=p0|detailed`. `p0` (the default, version P0) asks the model to supply its existing knowledge. `detailed` (version P1.5) supplies corrected engine rules, strategic priorities, progress requirements, and a more explicit game interface. A player captures one matched policy for the whole game. The older `-Dllm.prompt.profile` property remains accepted for reproducibility.
- Configure the rolling window with `-Dollama.memory.turns=8` and the Ollama context budget with `-Dollama.context.tokens=32768`. Input truncation is disabled so an unexpected overflow fails visibly.
- Thinking defaults to `off` so thinking-capable local models cannot spend an unbounded number of tokens choosing one move. Use `-Dollama.thinking=auto|on|low|medium|high` for controlled reasoning experiments and cap each response with `-Dollama.max.output.tokens=1024`.
- Recommended benchmark models (configured in Ollama and passed via `ollama.model` or `ollama.models`):
  - `gpt-oss:120b`
  - `llama4:scout`
  - `gemma3:27b`
  - `qwen3-coder:30b`
  - `mistral-large:123b`
  - `deepseek-r1:70b`

OpenAI setup:
- Add an OpenAI-compatible API key via either:
  - Spring property: `-Dopenai.apiKey=sk-...` (or in `application.properties`), or
  - Environment variable: `OPENAI_API_KEY=sk-...`.
- Configure the OpenAI chat model with `openai.model` (default is `gpt-4o`). Recommended models (see your OpenAI rate limit page for exact TPM/RPM):
  - `gpt-5.1`, `gpt-5-mini`, `gpt-5-nano`
  - `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
  - `o3`, `o4-mini`
  - `gpt-4o`, `gpt-4o-realtime-preview`
- Run the OpenAI-backed player with:  
  `./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-openai"`

Codex CLI setup (ChatGPT subscription rather than API billing):
- Install Codex CLI and sign in with ChatGPT; `codex login status` must report `Logged in using ChatGPT`.
- Configure the model with `-Dcodex.model` and reasoning effort with `-Dcodex.reasoning.effort`.
- Use `-Dcodex.models=gpt-5.6-luna,gpt-5.6-sol,gpt-6-astra` to compare several models in one benchmark.
- Each game uses a new persistent Codex session. Before the first board, the selected prompt profile establishes the strategy; every move then resumes that session so the model retains its strategy, earlier boards, commands, and feedback. Sessions are never reused across games.
- Codex result runs disable engine guidance. With the default P0 profile, the benchmark measures the model's self-authored strategy rather than engine recommendations.
- Transient capacity, timeout, and transport failures are retried up to five times with exponential backoff. Override this with `-Dcodex.retry.max.attempts` and `-Dcodex.retry.initial.delay.millis`; authentication and usage-limit failures are not retried.
- The player refuses non-ChatGPT authentication and removes API-key environment variables from CLI subprocesses.
- Run a small benchmark first:
  `./gradlew test --tests ai.games.results.CodexCliPlayerResultsTest --console=plain --rerun-tasks "-Dcodex.tests=true" "-Dcodex.models=gpt-5.6-luna,gpt-5.6-sol,gpt-6-astra" "-Dtest.games=1"`

GitHub Copilot CLI setup:
- Install `copilot`, authenticate it with the GitHub account that owns the Copilot subscription, and use `/model` interactively to confirm model availability.
- Configure one model with `-Dcopilot.model` or a sweep with `-Dcopilot.models`; the default is `claude-haiku-4.5`.
- Each game uses a fresh persistent Copilot session and the same selected prompt profile as the Codex player. Copilot runs in an empty temporary directory with custom instructions, MCP servers, and tools disabled.
- Copilot has no strict response-schema flag, so the player validates returned JSON against the engine's legal moves and makes up to three in-session formatting corrections. Configure this with `-Dcopilot.response.max.attempts`.
- Copilot result runs disable engine guidance and log the session ID plus Copilot's reported premium-request usage for every completed game.
- Set `SOLITAIRE_LOG_DIR` when running model sweeps concurrently so each process owns its rolling `game.log` and `episode.log`. For example, use `SOLITAIRE_LOG_DIR=logs/sonnet` for a Sonnet run while another process uses the default `logs` directory.
- Run a one-game smoke test with a 200-move cap:
  `./gradlew test --tests ai.games.results.CopilotCliPlayerResultsTest --console=plain --rerun-tasks "-Dcopilot.tests=true" "-Dcopilot.models=claude-haiku-4.5" "-Dtest.games=1" "-Dtest.max.moves.per.game=200" "-Dlog.episodes=true"`

TypeSafe AI setup:
- Set `TYPESAFE_API_KEY` in the environment or pass `-Dtypesafe.apiKey`. The repository's ignored `experiments/runtime/typesafe.env` file can be sourced before a run.
- The player pins `jev-1.13.0` by default for reproducibility. Override it with `-Dtypesafe.model` and override the endpoint with `-Dtypesafe.baseUrl` when needed.
- Each turn sends the current human-visible structured board plus compact observations, commands, and feedback from every previous turn to a System One Choice question. Historical observations retain foundation/tableau tops, hidden counts, and stock/waste state so card memory survives without exceeding Jev's context window. Jev returns an opaque option that maps to one engine-generated legal command.
- TypeSafe uses the shared `-Dgame.prompt.profile=p0|detailed` switch. P0 supplies no strategic policy; detailed/P1.5 adds corrected rules, eight decision priorities, and progress requirements shared with the conversational model players. Engine guidance remains disabled for both.
- TypeSafe supplies the detailed policy with one-step facts by simulating each legal command. It reports hidden-card and foundation progress, repeated-position signals, and stalls, and marks the strongest immediate-progress options without removing any legal choices. Episode metadata records the shared `policy_version`.
- Episode steps include Jev's model version, confidence, option probabilities, latency, and token usage when `-Dlog.episodes=true` is enabled.
- Transient rate limits and server or transport failures retry up to five times. Configure this with `-Dtypesafe.retry.max.attempts` and `-Dtypesafe.retry.initial.delay.millis`.
- Run a one-game smoke test with a 200-move cap:
  `source ../experiments/runtime/typesafe.env && SOLITAIRE_LOG_DIR=logs/typesafe ./gradlew test --tests ai.games.results.TypeSafePlayerResultsTest --console=plain --rerun-tasks "-Dtypesafe.tests=true" "-Dtest.games=1" "-Dtest.max.moves.per.game=200" "-Dlog.episodes=true"`

## Build & Test

Build:
```
./gradlew build
```

### Running Tests

All tests:
```
./gradlew test
```

#### By Category

Unit tests (game logic and training):
```
./gradlew test --tests "ai.games.unit.**"
./gradlew test --tests "ai.games.unit.game.**"                         # Legal/illegal moves, visibility, boundaries
./gradlew test --tests "ai.games.unit.training.**"                     # Undo and move history
```

AI player functional tests:
```
./gradlew test --tests "ai.games.player.ai.**"
./gradlew test --tests "ai.games.player.ai.GreedySearchPlayerTest"
./gradlew test --tests "ai.games.player.ai.MonteCarloPlayerTest"
./gradlew test --tests "ai.games.player.ai.AlphaSolitairePlayerTest"   # requires Python service
```

Performance benchmarks (500 games each; slow):
```
./gradlew test --tests "ai.games.results.**"
./gradlew test --tests ai.games.results.RuleBasedHeuristicsPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.GreedySearchPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.BeamSearchPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.HillClimberPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.MonteCarloPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.AStarPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.AlphaSolitairePlayerResultsTest --console=plain --rerun-tasks
```

LLM player benchmarks (enable with flags):
```
./gradlew test --tests ai.games.results.OpenAIPlayerResultsTest --console=plain --rerun-tasks "-Dopenai.tests=true"
./gradlew test --tests ai.games.results.OllamaPlayerResultsTest --console=plain --rerun-tasks "-Dollama.tests=true"
./gradlew test --tests ai.games.results.OllamaPlayerResultsTest --console=plain --rerun-tasks "-Dollama.tests=true" "-Dollama.models=gpt-oss:120b,llama4:scout,gemma3:27b,qwen3-coder:30b,mistral-large:123b,deepseek-r1:70b"
./gradlew test --tests ai.games.results.CopilotCliPlayerResultsTest --console=plain --rerun-tasks "-Dcopilot.tests=true" "-Dcopilot.models=claude-haiku-4.5" "-Dtest.games=1" "-Dtest.max.moves.per.game=200"
./gradlew test --tests ai.games.results.AlphaSolitairePlayerResultsTest --console=plain --rerun-tasks "-Dalphasolitaire.tests=true"
```

Game tree analysis (state space exploration):
```
./gradlew test --tests "ai.games.analysis.GameTreeAnalysisTest.testExhaustiveGameTreeAnalysis_100Games_Quick"      # ~2-3 min
./gradlew test --tests "ai.games.analysis.GameTreeAnalysisTest.testExhaustiveGameTreeAnalysis_1000Games"           # ~30-40 min
```

#### Generating Training Data (Episode Logging)

To generate clean episode logs for the neural network training pipeline, run any results test with `-Dlog.episodes=true`. Episodes are written as JSON lines to `logs/episode.log` (separate from debug logs in `game.log`):

```bash
# Generate episodes from A* player
./gradlew test --tests ai.games.results.AStarPlayerResultsTest "-Dlog.episodes=true"

# Generate episodes from Greedy player
./gradlew test --tests ai.games.results.GreedySearchPlayerResultsTest "-Dlog.episodes=true"

# Generate episodes from Rule-based player
./gradlew test --tests ai.games.results.RuleBasedHeuristicsPlayerResultsTest "-Dlog.episodes=true"

# Generate episodes from any player and verify the output
./gradlew test --tests "ai.games.results.**" "-Dlog.episodes=true"
wc -l logs/episode.log
head -1 logs/episode.log
```

Each episode log line contains:
- `EPISODE_GAME`: run ID plus model, session/conversation, prompt, and pre-game strategy metadata when the player supplies it
- `EPISODE_STEP`: per-move state, legal/recommended moves, chosen action
- `EPISODE_SUMMARY`: end-of-game statistics, termination reason, duration, and final board progress

Set `-Dexperiment.run.id=<descriptive-id>` to select one batch from the append-only log later. If
omitted, the engine assigns an `auto-...` run ID for that JVM.

These logs are consumed by the Python neural network training pipeline in `../neural-network`.

#### Single Test / Method

```
./gradlew test --tests ai.games.unit.game.LegalMovesTest
./gradlew test --tests ai.games.unit.game.LegalMovesTest.aceMovesToEmptyFoundation
```

Clean:
```
./gradlew clean
```

## Manual run without Gradle
Compile:
```
javac -cp "$(pwd)/src/main/java" $(find src/main/java -name "*.java")
```
Run:
```
java -cp src/main/java ai.games.Game
```

## Gameplay commands (CLI)
- `turn` — draw up to the configured count from stock to talon; recycle first when stock is empty and another pass is permitted.
- `move FROM TO` — e.g., `move W T3`, `move T7 F1`, `move T6 T1`.
- `undo` — revert the last move or turn directly from its recorded changes (training mode only).
- `quit` — exit.

Pile codes:
- Tableau: `T1`–`T7`
- Foundation: `F1`–`F4`
- Talon/Waste: `W`
- Stockpile: `S` (turned via `turn`, not `move`)

## Terminology

- **Tableau**: Seven main play piles. The active/top card is the nearest/last face-up card; covered cards beneath it are less visible.
- **Foundation**: Four suit piles built Ace → King.
- **Talon (Waste)**: Face-up cards flipped from stock.
- **Stockpile**: Face-down deck; `turn` flips up to three to the talon.
- **Top card**: The top most visible face-up card in a tableau pile (nearest to the player).
- **Bottom card**: The bottom least visible face-up card in a tableau pile (closet to the foundation).

## Notes
- ANSI suit symbols are used (hearts/diamonds in red).
- Tableau display shows face-up top rows with face-down counts next to headers.

## A* development experiments

A* exposes independent experimental JVM properties `astar.foundationWeight`
(default `1.0`), `astar.stockWeight` (default `0.5`), and `astar.knownFirstBudget`
(default `512`). Defaults preserve the existing policy; experimental scores apply to both
action and chance nodes. Default expansion and retained-node limits remain unchanged;
[the A* README](src/main/java/ai/games/player/ai/astar/README.md) documents optional
search-capacity, reveal-scoring, failure-cost and transposition experiments.

The opt-in `ai.games.experiments.AStarScreenTest` runs baseline and three single-change
variants on identical seeded deals and writes deck orders and per-game CSV results. Enable
with `-Dastar.screen=true -Dastar.output=/absolute/run/directory`; optionally set
`astar.games` (default 20) and `astar.seedStart` (default 0). Use a fresh output directory.
Research notes and analysis live in `experiments/notebooks/astar_improvement_research.ipynb`
at the repository root. Development runs do not update the canonical win-rate table.

Experimental A* duplicate-value reuse is enabled with both `astar.transpositions=true`
and `astar.reuseDuplicates=true` (screen preset `reuse`). Duplicate nodes retain a
link to their representative instead of disappearing from decision backups; values
include the difference in accumulated move cost. Recursive alias cycles receive the
finite failure cost. Backups cache completed node values separately for each root
action to bound work; cycle cutoffs can make these estimates traversal-dependent.
Links are cleared at each real observation and never cross chance
outcome contexts. Path-dependent pruning still makes this an approximate policy.
Defaults remain unchanged; development results are recorded in the A* notebook.

`astar.balancedSearch=true` tests round-robin expansion attempts across root actions,
then least-used chance contexts within an action, with ordinary f/h/state ordering
breaking ties. The `balanced` screen preset combines this with duplicate pruning.
Known/reveal phase scheduling and total expansion/node bounds remain unchanged.
This balances attempts among queued candidates, not retained nodes or guaranteed
successful expansions; blocked expansions still consume attempts. Root traces report
allocated attempts. The option is disabled by default.
