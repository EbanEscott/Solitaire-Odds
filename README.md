# Solitaire CLI (Spring Boot)

This project was built to demonstrate how AI (GPT-5.1-Codex-Max on Medium) and Human (Eban Escott on Coffee) can vibe to find the probability of winning a Solitaire game.

The total time it took to build this project was X.

## Test Results

The last test run was performed at X.

| Algorithm                     | Games Played | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak |
|------------------------------|--------------|-------|----------------|-------------|-----------|-----------|
| Rule-based Heuristics        | 500          | 12%   | 0.3s           | 150s        | 95        | 3                |
| Greedy Search                | 500          | 18%   | 0.5s           | 250s        | 105       | 4                |

* **Algorithm** Name of the decision or optimisation method being tested.
* **Games Played** Total number of solitaire games the algorithm attempted.
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
  - `ai.games.player.ai.RuleBasedHeuristicsPlayer` (@Profile `ai-rule`)
  - `ai.games.player.ai.GreedySearchPlayer` (@Profile `ai-greedy`)
- `src/test/java/ai/games/` — JUnit 5 tests with seeded states:
  - `LegalMovesTest`, `IllegalMovesTest`, `BoundaryTest`, `SolitaireTestHelper`, AI player tests.
- Build files: `build.gradle`, `settings.gradle`, `gradlew*`, `gradle/wrapper/`.

## Running (from `cards/`)
Human CLI (default):
```
./gradlew bootRun
```

AI profiles:
```
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-rule    # rule-based heuristics
./gradlew bootRun --console=plain -Dspring.profiles.active=ai-greedy  # greedy search
```

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

Terminology:
- Tableau: Seven main play piles. The active/top card is the nearest/last face-up card; covered cards beneath it are less visible.
- Foundation: Four suit piles built Ace → King.
- Talon (Waste): Face-up cards flipped from stock.
- Stockpile: Face-down deck; `turn` flips up to three to the talon.
- Top card: The top most visible face-up card in a tableau pile (nearest to the player).
- Bottom card: The bottom least visible face-up card in a tableau pile (closet to the foundation).

## Notes
- ANSI suit symbols are used (hearts/diamonds in red).
- Tableau display shows face-up top rows with face-down counts next to headers.+
