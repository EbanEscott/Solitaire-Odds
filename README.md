# Solitaire CLI (Spring Boot)

A Spring Boot command-line Solitaire (Klondike-style) app under the `ai.games` package. The game supports pluggable players (human CLI by default; AI profile-ready).

## Prereqs
- JDK 17+
- Use the bundled Gradle wrapper (pinned to Gradle 8.7). Gradle 9.x is incompatible with Spring Boot 3.2.
  - If Gradle 9.x was cached: `rm -rf ~/.gradle/wrapper/dists/gradle-9.1.0-bin`

## Layout
- `src/main/java/ai/games/Game` — Spring Boot entry, constructor-injected `Player`.
- `src/main/java/ai/games/game/` — core model: `Solitaire`, `Deck`, `Card`, `Rank`, `Suit`.
- `src/main/java/ai/games/player/` — `Player` plus implementations: `HumanPlayer` (default CLI), `AIPlayer` (stub, `@Profile("ai")`).
- `src/test/java/ai/games/` — JUnit 5 tests with seeded states:
  - `LegalMovesTest`, `IllegalMovesTest`, `BoundaryTest`, `SolitaireTestHelper`.
- Build files: `build.gradle`, `settings.gradle`, `gradlew*`, `gradle/wrapper/`.

## Running (from `cards/`)
Human CLI (default):
```
./gradlew bootRun
```

AI stub (loads `AIPlayer` bean):
```
./gradlew bootRun --args='' --console=plain -Dspring.profiles.active=ai
```
(AI currently returns a stub command; extend `AIPlayer` for real moves.)

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

## Notes
- ANSI suit symbols are used (hearts/diamonds in red).
- Tableau display shows face-up top rows with face-down counts next to headers.+
