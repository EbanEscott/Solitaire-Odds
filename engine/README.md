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

`Solitaire.planningCopy()` explicitly creates a PLAN board from a real game. It masks unseen tableau and stock cards, retains previously observed stock identities and their order, and leaves the real board unchanged. On an existing PLAN board it behaves like `copy()`. Mode is fixed at construction; `setMode()` has been removed. Both kinds of board continue to share the same movement and undo implementation inside `Solitaire`.

PLAN's possible unseen identities are derived from the standard deck minus known or already assigned cards. `nextUnresolvedCard()` finds a revealed UNKNOWN slot; `assumeCard(slot, identity)` assigns it a concrete unseen identity and returns `1 / remainingCards`. Each assignment is recorded in the existing action history, so `rollbackTo()` restores the UNKNOWN slot, guess map, and state key. Copies keep committed identities while sibling branches have independent containers.

A* and Monte Carlo now enter search through `planningCopy()`. Other players retain their existing copying behaviour. Both resolve newly exposed tableau cards and all cards drawn to waste before continuing simulated play. An assigned card stays assigned when moved or recycled. UNKNOWN cards are no longer legal wildcards, and legal move generation never creates guesses as a side effect.

A* expands one revealed UNKNOWN at a time into separate card-assignment branches. Conditional probabilities multiply along the path; with three unseen cards, successive exact assignments have probabilities `1/3`, `1/2`, and `1`, giving `1/6` for that complete sequence. Assignments cost no game move. Ordinary moves inherit the cumulative probability. Ranking uses `(g + h + 1) / p - 1`, which retains a penalty even when the heuristic is zero. The existing 1,024-expansion and 5,000-open-node limits still apply, including chance branches. This is a bounded probability-weighted search heuristic, not an expected-win calculation or a guarantee of optimal play. Each actual turn refreshes the current state from observation, discards speculative descendants, and resets the probability to one.

Monte Carlo samples each newly revealed identity uniformly without replacement. The candidate move's reveals are sampled afresh inside each rollout, and all assignments are undone before the next rollout. It continues to use one board per root candidate, with no board copies inside a rollout. Actual next-turn validation compares known cards and pile shapes against the predicted position, allowing predicted UNKNOWN slots to acquire real identities.

Historical win rates and profiles predate this change and have not been remeasured.

Legal move generation in GAME and PLAN mode reads the visible tableau, face-up counts, foundations, and talon once per call and passes them to the move categories that need them. This avoids rebuilding identical snapshots and views while preserving move checks and ordering. These reads are local to the call; each subsequent call reads the current board again.

Tableau-to-tableau generation also computes each visible destination's required rank and colour once per call, then reads each source card's rank and colour once before checking destinations. Empty destinations accept either colour of king. Planning boards use the same placement rules after their revealed UNKNOWN cards have been assigned; unresolved UNKNOWN sources and destinations are excluded.

`LegalMovesHelper.listMoves()` generates `List<ai.games.game.Move>` directly. Monte Carlo uses these objects for root candidates, rollout pruning, and `TreeNode.applyMove(Move)`, avoiding command parsing in that path. The existing `listLegalMoves()` API remains a mutable list of strings for other players, guidance, logs, and service payloads, with the same order and card spelling. Human console input, `Player.nextCommand()`, and `TreeNode.applyMove(String)` remain text-based. Tree child keys remain strings, and the final call into `Solitaire.moveCard()` still uses its existing pile/card text API and validation.

### Monte Carlo rollouts

`MonteCarloPlayer` makes one root snapshot and one board copy per first legal move. Each candidate board is reused across up to 64 rollouts, with at most 128 accepted steps per rollout. There are no board copies inside `simulate()`. Each step generates legal moves once, then uses `TreeNode`'s inverse-move, useless-king, and symmetry checks before applying a candidate. Quit is excluded, and symmetric moves retain the first destination in legal-move order.

Candidates are sampled uniformly without replacement. A move that produces a state with two prior occurrences is rolled back before trying another candidate. The rollout keeps prior state keys and the preceding move, while the retained tree keeps only root candidates and their statistics. The original rollout-root occurrence is retained for cycle detection. On completion, exhaustion, or an exception, `rollbackTo()` restores the candidate board to its starting history depth before another rollout can begin.

PLAN rollouts restore guesses through the same action history as moves and turns. Seeded GAME-mode tests retain a comparison with the previous copying implementation; planning tests separately verify sampling without replacement and rollback after exceptions.

`Solitaire.getStateKey()` computes the board hash lazily and reuses it until a successful card move or stock turn/recycle invalidates it. Undo restores the previous cached value and validity flag. Ordinary copies in either mode inherit a valid cached key. A GAME-to-PLAN `planningCopy()` calculates its own key because hidden cards have been masked. Callers, including cycle checks, keep using `getStateKey()` with the same API. The hash now includes pile sizes and face-up counts, so transfers of identical UNKNOWN placeholders and visibility changes are distinguishable.

`Solitaire` owns a single action history for card moves, stock turns, and planning assignments. Each successful action records the changes needed for `undoLastMove()` to restore pile order, face-up counts, revealed-card knowledge, and the hash cache directly, without re-dealing or replaying earlier moves. Failed moves and turns with both stock and talon empty add no history. Undo does not add actions. Save `getMoveHistorySize()` and call `rollbackTo(depth)` to undo a sequence using the same mechanism; depths refer to the current board and history branch. Training mode still controls access to the player-facing `undo` command. Monte Carlo uses this rollback mechanism; other search players retain their existing copying behaviour.

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
- `turn` — flip up to three cards from stock to talon.
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
