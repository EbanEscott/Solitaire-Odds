# Solitaire CLI (Spring Boot)

Spring Boot command-line Solitaire (Klondike-style) demo under the `ai.games` package. The game deals a board, lets you turn three cards from the stock (`turn`) or move cards with pile codes (e.g., `move W T3`, `move T7 F1`, `quit`).

## Prerequisites
- JDK 17+
- Use the bundled Gradle wrapper (pinned to Gradle 8.7) to avoid Gradle 9.x incompatibility with Spring Boot 3.2.

If you inadvertently downloaded Gradle 9.x (wrapper caches under `~/.gradle/wrapper/dists`), delete that cached folder and rerun with the wrapper so it fetches 8.7:
```
rm -rf ~/.gradle/wrapper/dists/gradle-9.1.0-bin
./gradlew --version
```

## Common Gradle commands (run from `cards/`)
Run the app:
```
./gradlew bootRun
```

Build:
```
./gradlew build
```

Tests:
```
./gradlew test
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
java -cp src/main/java ai.games.HelloWorld
```

## Pile codes recap
- Tableau: `T1`–`T7` (move face-up top card)
- Foundation: `F1`–`F4`
- Talon/Waste: `W`
- Stockpile: `S` (turn via `turn`, not `move`)
