# Results Tests

Run player benchmarks with copy-paste commands.

## Rules and reproducibility

Run commands from `engine/` and select an individual suite. The current rules default to draw three, `topOnly`, unlimited passes, and `thoughtful=false`. For recorded comparisons, pass `-Dgame.drawCount=3 -Dgame.stockVisibility=topOnly -Dgame.maxStockPasses=null -Dgame.thoughtful=false` explicitly. Record the revision, game count, move cap, guidance/training settings, search budgets, and log directory alongside results. `GAME_RULES` records the actual settings at each game start even with episode logging disabled; `-Dexperiment.run.id=<id>` joins these to optional episode records.

The configurable-rules refresh covers only the six search players below. Do not rerun model-backed suites for that refresh; preserve their historical results and dates. AlphaSolitaire is also outside that refresh. Benchmark execution alone does not authorize updating the root results table.

## Reusable matched-deal search sweep

From `engine/`, run `python3 scripts/search_sweep.py --games 100`. It selects only rule-based, greedy, hill-climbing, beam, Monte Carlo, and A* players, using the same seeded deals for each. The default rules are draw-three, `topOnly`, unlimited passes, and `thoughtful=false`; guidance, training undo, and episode logging are disabled. The default game-loop cap is 1,000 commands. This is an opt-in `SearchSweepTest`, separate from legacy independent-random-deal result suites.

```bash
python3 scripts/search_sweep.py --games 100
python3 scripts/search_sweep.py --games 10000 --seed-start 0 --player-seed 20261017
python3 scripts/search_sweep.py --games 100 --players greedy astar --draw-count 1 --stock-visibility allDrawn --max-stock-passes 3 --thoughtful
python3 scripts/search_sweep.py --games 100 --players monteCarlo --property montecarlo.search.visits=256 --max-moves 1000
```

Use `--output /absolute/fresh/directory` to choose a destination; existing directories are rejected. Otherwise output goes to a new timestamped directory under `logs/profiles/`. The launcher selects only the opted-in test, forces execution, captures stdout/stderr in `console.log`, and preserves partial rows if a run fails. It never publishes the root table or runs model-backed players.

Artifacts include `manifest.json` (revision, actual command, UTC times, completion status), `engine-metadata.json` (rules, budgets, seeds, overrides), `deals.csv` (exact deck order, bottom-to-top), `games.csv` (per-game outcomes, seed, moves, termination, foundation count, seconds), and `summary.json` (wins, timings, moves, streaks, termination counts). `source/` preserves actual main/test sources, scripts, and build files, including uncommitted changes. Each completed game is flushed to disk. Do not interpret a quit or move cap as proof that a deal is unwinnable.

Deal `i` uses `Random(seedStart + i)` to shuffle the stable deck; hill/beam/Monte Carlo use a separate `Random(playerSeed + i)` stream. A* samples from state-derived seeds. Players are recreated for every deal and their execution order rotates. Beam uses its ordinary depth 2 and width 8. `--property KEY=VALUE` supports existing `astar.*` and `montecarlo.*` controls; source snapshots preserve other fixed budgets and defaults.

The harness validates every non-quit command on an independent board before the normal `Game` loop executes it. Invalid commands fail the run instead of silently consuming the cap. Reported time includes this validation overhead, so compare timings from the same harness. `GAME_RULES` names the checking wrapper; adjacent `SWEEP_START` records and CSV rows identify the actual player. Pilot runs remain development measurements and do not refresh the published table. Verify launcher aggregation with `python3 -m unittest discover -s src/test/python -v` from `engine/`.

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
