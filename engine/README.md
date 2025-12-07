# Engine

The engine is a Java/Spring Boot command-line implementation of Klondike Solitaire. It owns the core game rules, logging, and pluggable player system (human and AI). Search-based players such as greedy, hill-climbing, beam search, Monte Carlo, A*, and LLM-backed players (OpenAI and Ollama) all run here, and the AlphaSolitaire (MCTS + neural network) player integrates with the Python modeling stack in `../neural-network`.

## Prereqs

The engine is a Spring Boot command-line Solitaire (Klondike-style) app under the `ai.games` package.

- JDK 21+ (toolchain set to 21)
- Use the bundled Gradle wrapper (pinned to Gradle 8.7). Gradle 9.x is incompatible with Spring Boot 3.2.
  - If Gradle 9.x was cached: `rm -rf ~/.gradle/wrapper/dists/gradle-9.1.0-bin`

## Layout
- `src/main/java/ai/games/Game` — Spring Boot entry, constructor-injected `Player`.
- `src/main/java/ai/games/game/` — core model: `Solitaire`, `Deck`, `Card`, `Rank`, `Suit`.
- `src/main/java/ai/games/player/` — player base types:
  - `HumanPlayer` (default CLI)
  - `AIPlayer` base class
  - `LegalMovesHelper`
- `src/main/java/ai/games/player/ai/` — AI players (all `@Profile`-gated):
  - `OpenAIPlayer` (`ai-openai`)
  - `RuleBasedHeuristicsPlayer` (`ai-rule`)
  - `GreedySearchPlayer` (`ai-greedy`)
  - `BeamSearchPlayer` (`ai-beam`)
  - `HillClimberPlayer` (`ai-hill`)
  - `MonteCarloPlayer` (`ai-mcts`)
  - `AStarPlayer` (`ai-astar`)
  - `OllamaPlayer` (`ai-ollama`)
- `src/test/java/ai/games/` — JUnit 5 tests with seeded states:
  - `LegalMovesTest`, `IllegalMovesTest`, `BoundaryTest`, `SolitaireTestHelper`, AI player tests.
- Build files: `build.gradle`, `settings.gradle`, `gradlew*`, `gradle/wrapper/`.

## Running (from `engine/`)
Exactly one player profile must be active. The default profile is `ai-human` (set in `src/main/resources/application.properties`).

Human CLI (default):
```
./gradlew bootRun
```

AI profiles:
```
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-beam          # beam search (fixed-depth, fixed-width)
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-hill          # hill-climbing search (state-hash driven)
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-rule          # rule-based heuristics
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-greedy        # greedy search
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-mcts          # Monte Carlo (MCTS-style) search
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-astar         # A* search
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-ollama        # Ollama via Spring AI (requires local Ollama)
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-openai        # OpenAI via API (requires OPENAI_API_KEY or openai.apiKey)
```

Ollama model selection:
- Default model is set in `src/main/resources/application.properties` (`ollama.model=llama3`).
- Override per run: `./gradlew bootRun --console=plain -Dspring.profiles.active=ai-ollama -Dollama.model=mistral-large:123b`
- Or set env: `OLLAMA_MODEL=mistral-large:123b ./gradlew bootRun --console=plain -Dspring.profiles.active=ai-ollama`
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
  `./gradlew bootRun --console=plain -Dspring.profiles.active=ai-openai`

## Build & Test
Build:
```
./gradlew build
```

Tests:
```
./gradlew test
```
(Tests seed deterministic board states to verify legal/illegal moves, tableau flipping, foundation progression, and deck integrity.)

Single test / class:
```
./gradlew test --tests ai.games.LegalMovesTest
./gradlew test --tests ai.games.LegalMovesTest.aceMovesToEmptyFoundation
```

AI result sweeps (game counts set in `ResultsConfig`, default 500; use `--rerun-tasks` to force execution):
```
./gradlew test --tests ai.games.results.RuleBasedHeuristicsPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.GreedySearchPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.BeamSearchPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.HillClimberPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.MonteCarloPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.AStarPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.OpenAIPlayerResultsTest --console=plain --rerun-tasks           # enable with -Dopenai.tests=true
./gradlew test --tests ai.games.results.OllamaPlayerResultsTest --console=plain --rerun-tasks           # enable with -Dollama.tests=true
./gradlew test --tests ai.games.results.OllamaPlayerResultsTest --console=plain --rerun-tasks -Dollama.tests=true -Dollama.models=gpt-oss:120b,llama4:scout,gemma3:27b,qwen3-coder:30b,mistral-large:123b,deepseek-r1:70b
./gradlew test --tests ai.games.results.AlphaSolitairePlayerResultsTest --console=plain --rerun-tasks -Dalphasolitaire.tests=true   # requires Python AlphaSolitaire model service
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
- Tableau display shows face-up top rows with face-down counts next to headers.+
