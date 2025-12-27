# AI Player Tests

Functional tests for each AI player implementation. These tests verify that players correctly implement their strategies and integrate properly with the game engine.

## AI Players

| Player | Class | Type | Test | Strategy |
|--------|-------|------|------|----------|
| Rule-based Heuristics | `RuleBasedHeuristicsPlayer` | Search | `RuleBasedHeuristicsPlayerTest` | Deterministic rules-based move selection with heuristic scoring |
| Greedy Search | `GreedySearchPlayer` | Search | `GreedySearchPlayerTest` | One-step lookahead with greedy scoring of immediate moves |
| Hill Climber | `HillClimberPlayer` | Search | `HillClimberPlayerTest` | Iterative improvement via hill climbing heuristic |
| Beam Search | `BeamSearchPlayer` | Search | `BeamSearchPlayerTest` | Beam search with limited width and depth |
| A* Search | `AStarPlayer` | Search | `AStarPlayerTest` | A* algorithm with game state heuristics |
| Monte Carlo | `MonteCarloPlayer` | Search | `MonteCarloPlayerTest` | Monte Carlo tree search with random playouts |
| AlphaSolitaire | `AlphaSolitairePlayer` | Neural | `AlphaSolitairePlayerTest` | Neural network policy-value player |

## Run Tests

```bash
# All AI player tests
./gradlew test --tests "ai.games.player.ai.**"

# Individual player
./gradlew test --tests "ai.games.player.ai.GreedySearchPlayerTest"
./gradlew test --tests "ai.games.player.ai.MonteCarloPlayerTest"
./gradlew test --tests "ai.games.player.ai.AlphaSolitairePlayerTest"
```

## Performance Benchmarks

For win-rate and performance metrics, see `ai.games.results.*` package.

```bash
# Run all performance sweeps
./gradlew test --tests "ai.games.results.**"
```
