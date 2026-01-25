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
  - LLM: `OpenAIPlayer`, `OllamaPlayer`

### Build Files
`build.gradle`, `settings.gradle`, `gradlew*`, `gradle/wrapper/`.

## Running (from `engine/`)

Exactly one player profile must be active at a time. Run with `-Dspring.profiles.active=<profile>`.

### Quick Start

**Human player (CLI):**
```bash
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-human"
```

**Human player with training mode** (undo enabled):
```bash
./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-human" "-Dtraining.mode=true"
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
```

Ollama model selection:
- Default model is set in `src/main/resources/application.properties` (`ollama.model=llama3`).
- Override per run: `./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-ollama" "-Dollama.model=mistral-large:123b"`
- Or set env: `$env:OLLAMA_MODEL="mistral-large:123b"; ./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-ollama"`
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
./gradlew test --tests ai.games.results.OpenAIPlayerResultsTest --console=plain --rerun-tasks -Dopenai.tests=true
./gradlew test --tests ai.games.results.OllamaPlayerResultsTest --console=plain --rerun-tasks -Dollama.tests=true
./gradlew test --tests ai.games.results.OllamaPlayerResultsTest --console=plain --rerun-tasks -Dollama.tests=true \
  -Dollama.models=gpt-oss:120b,llama4:scout,gemma3:27b,qwen3-coder:30b,mistral-large:123b,deepseek-r1:70b
./gradlew test --tests ai.games.results.AlphaSolitairePlayerResultsTest --console=plain --rerun-tasks -Dalphasolitaire.tests=true
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
- `EPISODE_STEP`: per-move state, legal/recommended moves, chosen action
- `EPISODE_SUMMARY`: end-of-game statistics (win/loss, move count, duration)

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
- `undo` — revert the last move or turn (training mode only; replay from start without that action).
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
