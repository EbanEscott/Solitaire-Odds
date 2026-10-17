# Results Tests

Run player benchmarks with copy-paste commands.

## Rules and reproducibility

Run commands from `engine/` and select an individual suite. The current rules default to draw three, `topOnly`, unlimited passes, and `thoughtful=false`. For recorded comparisons, pass `-Dgame.drawCount=3 -Dgame.stockVisibility=topOnly -Dgame.maxStockPasses=null -Dgame.thoughtful=false` explicitly. Record the revision, game count, move cap, guidance/training settings, search budgets, and log directory alongside results. `GAME_RULES` records the actual settings at each game start even with episode logging disabled; `-Dexperiment.run.id=<id>` joins these to optional episode records.

The configurable-rules refresh covers only the six search players below. Do not rerun model-backed suites for that refresh; preserve their historical results and dates. AlphaSolitaire is also outside that refresh. Benchmark execution alone does not authorize updating the root results table.

## Quick Commands

```bash
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

# TypeSafe AI Jev
source ../experiments/runtime/typesafe.env && SOLITAIRE_LOG_DIR=logs/typesafe ./gradlew test --tests "ai.games.results.TypeSafePlayerResultsTest" --rerun-tasks "-Dtypesafe.tests=true" "-Dgame.prompt.profile=p0" "-Dtest.max.moves.per.game=200" "-Dlog.episodes=true" "-Dtest.games=1"

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
