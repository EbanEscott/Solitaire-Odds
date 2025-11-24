# Solitaire Odds

Have you ever wondered what the odds of winnng a game of Solitaire is? This project was built to demonstrate how AI (GPT-5.1-Codex-Max on Medium) and Human (Eban Escott on Coffee) can vibe to find the probability of winning a Solitaire game.

A well-shuffled 52-card deck has *52! permutations (about 8.1 × 10^67)*, so many that it dwarfs the *roughly 10^20 grains of sand on Earth*. In other words, almost every Solitaire deal you have ever seen is effectively a one-off in cosmic terms. Even at *one deal per second*, brute-forcing every deck order would take *around 2.6 × 10^60 years*, a timespan so huge the age of the universe does not even register on the same scale.

This means testing every deck permutation is impossible. Instead, we lean on AI and solid engineering to run repeatable regression test suites over large batches of randomly shuffled games, so we can measure performance statistically rather than brute-forcing every possible deal. The goal is not to “solve” all of Solitaire, but to apply a range of AI algorithms that reliably solve as many deals as possible and, in doing so, reveal the true probability of winning under real rules.

## Test Results

The last test run was performed at Nov 24, 2025 8:45:14 PM.

| Algorithm                     | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak |
|------------------------------|--------------|-----------|-------|---------------|------------|-----------|-----------------|
| Simple Rule-based Heuristics | 10000 | 462 | 4.62% | 0.001s | 5.799s | 735.87 | 3 |
| Greedy Search | 10000 | 651 | 6.51% | 0.003s | 31.046s | 242.42 | 3 |
| Hill-climbing Search | 10000 | 1317 | 13.17% | 0.002s | 16.514s | 97.82 | 4 |
| Beam Search | 10000 | 1154 | 11.54% | 0.039s | 391.002s | 902.94 | 4 |

* **Algorithm** Name of the decision or optimisation method being tested.
* **Games Played** Total number of solitaire games the algorithm attempted.
* **Games Won** Count of games successfully completed.
* **Win %** Percentage of games successfully completed (foundations fully built).
* **Avg Time/Game** Mean time taken to finish or fail a game.
* **Total Time** Sum of all time spent playing the batch of games.
* **Avg Moves** Average number of moves (legal actions) the algorithm performed per game.
* **Avg Score** Mean score based on whatever scoring system you’re using (e.g., Vegas, Microsoft, or custom).
* **Best Win Streak** Longest run of consecutive wins within the batch.

## Prereqs

A Spring Boot command-line Solitaire (Klondike-style) app under the `ai.games` package. The game supports pluggable players (human CLI by default; AI profile-ready).

- JDK 21+ (toolchain set to 21)
- Use the bundled Gradle wrapper (pinned to Gradle 8.7). Gradle 9.x is incompatible with Spring Boot 3.2.
  - If Gradle 9.x was cached: `rm -rf ~/.gradle/wrapper/dists/gradle-9.1.0-bin`

## Layout
- `src/main/java/ai/games/Game` — Spring Boot entry, constructor-injected `Player`.
- `src/main/java/ai/games/game/` — core model: `Solitaire`, `Deck`, `Card`, `Rank`, `Suit`.
- `src/main/java/ai/games/player/` — `Player` base plus:
  - `HumanPlayer` (default CLI)
  - `AIPlayer` base
  - `ai.games.player.ai.SimpleRuleBasedHeuristicsPlayer` (@Profile `ai-rule`)
  - `ai.games.player.ai.ComplexRuleBasedHeuristicsPlayer` (@Profile `ai-rule-complex`)
  - `ai.games.player.ai.GreedySearchPlayer` (@Profile `ai-greedy`)
  - `ai.games.player.ai.BeamSearchPlayer` (@Profile `ai-beam`)
  - `ai.games.player.HillClimberPlayer` (@Profile `ai-hill`)
- `src/test/java/ai/games/` — JUnit 5 tests with seeded states:
  - `LegalMovesTest`, `IllegalMovesTest`, `BoundaryTest`, `SolitaireTestHelper`, AI player tests.
- Build files: `build.gradle`, `settings.gradle`, `gradlew*`, `gradle/wrapper/`.

## Running (from `cards/`)
Exactly one player profile must be active. The default profile is `ai-human` (set in `src/main/resources/application.properties`).

Human CLI (default):
```
./gradlew bootRun
```

AI profiles:
```
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-beam          # beam search (fixed-depth, fixed-width)
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-hill          # hill-climbing search (state-hash driven)
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-rule          # simple rule-based heuristics
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-rule-complex  # complex rule-based heuristics (experimental)
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-greedy        # greedy search
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-ollama        # Ollama via Spring AI (requires local Ollama)
```

Ollama model selection:
- Default model is set in `src/main/resources/application.properties` (`ollama.model=llama3`).
- Override per run: `./gradlew bootRun --console=plain -Dspring.profiles.active=ai-ollama -Dollama.model=mistral:latest`
- Or set env: `OLLAMA_MODEL=mistral:latest ./gradlew bootRun --console=plain -Dspring.profiles.active=ai-ollama`

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
./gradlew test --tests ai.games.results.SimpleRuleBasedHeuristicsPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.ComplexRuleBasedHeuristicsPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.GreedySearchPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.BeamSearchPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.HillClimberPlayerResultsTest --console=plain --rerun-tasks
./gradlew test --tests ai.games.results.OllamaPlayerResultsTest --console=plain --rerun-tasks    # enable with -Dollama.tests=true
```

Hill-climbing player:
- Profile: `ai-hill`
- Uses a Zobrist-style `stateKey` from `Solitaire` as the primary handle for visited/duplicate detection and per-state scoring.
- Performs local hill-climbing with backtracking (rejecting moves that lead to strictly worse or known-bad states) and bounded random restarts to escape local maxima/plateaus.
- Tunable parameters (via constants in `HillClimberPlayer`):
  - `MAX_STEPS_PER_GAME` — hard cap on total decisions per game to avoid non-terminating runs.
  - `MAX_RESTARTS_PER_POSITION` — number of random restarts allowed from a local maximum before giving up.
  - Evaluation weights (foundation progress, visible vs facedown cards, stock size) that guide the climb.

The simple rule-based player encodes a strict, deterministic set of solitaire rules that are easy to reason about and provide a strong baseline.  
The complex rule-based player is an experimental variant intended for more advanced heuristics (e.g., deeper lookahead or richer safety checks) while still remaining mostly rule-driven.

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
