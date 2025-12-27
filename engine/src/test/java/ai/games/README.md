# Test Structure

This directory contains all tests for the Solitaire game engine and AI players, organized hierarchically:

- **`unit/`** — Unit tests for core game logic and training functionality
  - `game/` — Game rules, legal moves, visibility, and boundary conditions
  - `training/` — Training mode with undo functionality
  - `helpers/` — Helper classes for test support

- **`player/`** — AI player tests and evaluations
  - `ai/` — Functional tests for each AI player implementation

- **`results/`** — Performance sweep tests and win-rate benchmarks for all AI players

- **`analysis/`** — Advanced analysis tools (game tree exploration, exhaustive state space analysis)

## Quick Test Commands

```bash
# Run all tests
./gradlew test

# Run by category
./gradlew test --tests "ai.games.unit.**"                    # All unit tests
./gradlew test --tests "ai.games.player.ai.**"              # All AI player tests
./gradlew test --tests "ai.games.results.**"                # All performance benchmarks
./gradlew test --tests "ai.games.analysis.**"               # Game tree analysis

# Run specific test
./gradlew test --tests "ai.games.unit.game.LegalMovesTest"
./gradlew test --tests "ai.games.player.ai.GreedySearchPlayerTest"
```

See subdirectory READMEs for more detailed documentation.
