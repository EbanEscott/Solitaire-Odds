# Results Tests

Run player benchmarks with copy-paste commands.

## Quick Commands

```bash
# All tests
./gradlew test --tests "ai.games.results.**"

# A* Search
./gradlew test --tests "ai.games.results.AStarPlayerResultsTest" --rerun-tasks "-Dtest.max.moves.per.game=1000" "-Dtest.progress.log.interval=1" "-Dtest.games=100"

# Greedy Search
./gradlew test --tests "ai.games.results.GreedySearchPlayerResultsTest" --rerun-tasks "-Dtest.max.moves.per.game=1000" "-Dtest.progress.log.interval=1" "-Dtest.games=100"

# Beam Search
./gradlew test --tests "ai.games.results.BeamSearchPlayerResultsTest" --rerun-tasks "-Dtest.max.moves.per.game=1000" "-Dtest.progress.log.interval=1" "-Dtest.games=100"

# Hill Climber
./gradlew test --tests "ai.games.results.HillClimberPlayerResultsTest" --rerun-tasks "-Dtest.max.moves.per.game=1000" "-Dtest.progress.log.interval=1" "-Dtest.games=100"

# Monte Carlo
./gradlew test --tests "ai.games.results.MonteCarloPlayerResultsTest" --rerun-tasks "-Dtest.max.moves.per.game=1000" "-Dtest.progress.log.interval=1" "-Dtest.games=100"

# Rule-Based Heuristics
./gradlew test --tests "ai.games.results.RuleBasedHeuristicsPlayerResultsTest" --rerun-tasks "-Dtest.max.moves.per.game=1000" "-Dtest.progress.log.interval=1" "-Dtest.games=100"

# OpenAI GPT
./gradlew test --tests "ai.games.results.OpenAIPlayerResultsTest" --rerun-tasks "-Dtest.max.moves.per.game=1000" "-Dtest.progress.log.interval=1" "-Dtest.games=100"

# Ollama Local
./gradlew test --tests "ai.games.results.OllamaPlayerResultsTest" --rerun-tasks "-Dtest.max.moves.per.game=1000" "-Dtest.progress.log.interval=1" "-Dtest.games=100"

# AlphaSolitaire
./gradlew test --tests "ai.games.results.AlphaSolitairePlayerResultsTest" --rerun-tasks "-Dalphasolitaire.tests=true" "-Dtest.max.moves.per.game=1000" "-Dtest.progress.log.interval=1" "-Dtest.games=100"
```

## Configuration

All tests respect these system properties (via `-D` flags):

| Property | Default | Description |
|----------|---------|-------------|
| `-Dtest.games=<n>` | 10 | Number of games to play in the sweep |
| `-Dtest.progress.log.interval=<n>` | 1 | Log progress every N games (e.g., 50 = log at games 1, 50, 100, …) |
| `-Dtest.max.moves.per.game=<n>` | 1000 | Maximum moves allowed per game (prevents infinite loops) |

**Examples:**
```bash
# 100 games
./gradlew test --tests "ai.games.results.AStarPlayerResultsTest" "-Dtest.games=100"

# 500 games with less verbose logging
./gradlew test --tests "ai.games.results.AStarPlayerResultsTest" "-Dtest.games=500" "-Dtest.progress.log.interval=50"

# High move limit
./gradlew test --tests "ai.games.results.HillClimberPlayerResultsTest" "-Dtest.max.moves.per.game=1000"
```


