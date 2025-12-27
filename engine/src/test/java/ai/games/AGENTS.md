# AI Players

Solitaire game features multiple AI players with different strategies, from deterministic rule-based approaches to neural network models. This document describes each player's implementation and testing approach.

## Search-Based Players

### Rule-Based Heuristics
- **Class**: [`RuleBasedHeuristicsPlayer`](../../main/java/ai/games/player/ai/RuleBasedHeuristicsPlayer.java)
- **Test**: [`RuleBasedHeuristicsPlayerTest`](ai/RuleBasedHeuristicsPlayerTest.java)
- **Profile**: `rule-based-heuristics`
- **Strategy**: Deterministic rule-based move selection using heuristic scoring
- **Configuration**: Hardcoded heuristic weights for card preferences
- **Use When**: You need predictable, deterministic behavior; baseline comparison; minimal computation
- **Performance**: Fast, consistent, baseline win-rate ~10-15%

### Greedy Search
- **Class**: [`GreedySearchPlayer`](../../main/java/ai/games/player/ai/GreedySearchPlayer.java)
- **Test**: [`GreedySearchPlayerTest`](ai/GreedySearchPlayerTest.java)
- **Profile**: `greedy-search`
- **Strategy**: One-step lookahead with greedy scoring of immediate moves
- **Configuration**: Heuristic scoring function for move evaluation
- **Use When**: You need simple but better-than-random behavior; fast decisions needed
- **Performance**: Faster than search-based approaches, better than rule-based ~15-20%

### Hill Climber
- **Class**: [`HillClimberPlayer`](../../main/java/ai/games/player/ai/HillClimberPlayer.java)
- **Test**: [`HillClimberPlayerTest`](ai/HillClimberPlayerTest.java)
- **Profile**: `hill-climber`
- **Strategy**: Iterative improvement via hill climbing with random restarts
- **Configuration**: Max iterations, restart frequency, search depth
- **Use When**: You need better than greedy but have limited computation; iterative improvement preferred
- **Performance**: Moderate computation, improved win-rate ~15-25%

### Beam Search
- **Class**: [`BeamSearchPlayer`](../../main/java/ai/games/player/ai/BeamSearchPlayerTest.java)
- **Test**: [`BeamSearchPlayerTest`](ai/BeamSearchPlayerTest.java)
- **Profile**: `beam-search`
- **Strategy**: Beam search with limited width and depth branching
- **Configuration**: Beam width (number of states kept), search depth, heuristic scorer
- **Use When**: You need better planning than greedy with controlled computation; need to explore multiple paths
- **Performance**: Moderate-to-high computation, better win-rate ~20-30%

### A* Search
- **Class**: [`AStarPlayer`](../../main/java/ai/games/player/ai/AStarPlayer.java)
- **Test**: [`AStarPlayerTest`](ai/AStarPlayerTest.java)
- **Profile**: `a-star`
- **Strategy**: A* algorithm with heuristic-guided search
- **Configuration**: Heuristic function, cost evaluation, pruning parameters
- **Use When**: You need optimal planning with heuristic guidance; willing to invest significant computation
- **Performance**: High computation, strong planning, win-rate ~25-35%

### Monte Carlo Tree Search
- **Class**: [`MonteCarloPlayer`](../../main/java/ai/games/player/ai/MonteCarloPlayer.java)
- **Test**: [`MonteCarloPlayerTest`](ai/MonteCarloPlayerTest.java)
- **Profile**: `monte-carlo`
- **Strategy**: Monte Carlo tree search with random playouts for evaluation
- **Configuration**: Number of iterations, UCB exploration constant, playout depth
- **Use When**: You need good exploration-exploitation balance; sampling-based approach preferred
- **Performance**: Moderate-to-high computation, strong results ~25-35%, depends on iteration count

## Neural Network Players

### AlphaSolitaire (Policy-Value Network)
- **Class**: [`AlphaSolitairePlayer`](../../main/java/ai/games/player/ai/alpha/AlphaSolitairePlayer.java)
- **Test**: [`AlphaSolitairePlayerTest`](ai/AlphaSolitairePlayerTest.java)
- **Profile**: `alpha-solitaire`
- **Strategy**: Neural network policy (move selection) and value (position evaluation) network
- **Configuration**: Model path, network architecture, policy/value weight mix
- **Infrastructure**: 
  - Server: [`AlphaSolitaireClient`](../../main/java/ai/games/player/ai/alpha/AlphaSolitaireClient.java) (REST calls to Python service)
  - Protocol: [`AlphaSolitaireRequest`](../../main/java/ai/games/player/ai/alpha/AlphaSolitaireRequest.java)/[`AlphaSolitaireResponse`](../../main/java/ai/games/player/ai/alpha/AlphaSolitaireResponse.java)
  - Training: See `neural-network/` module
- **Use When**: You want best performance from learned patterns; willing to run inference service; have trained model
- **Performance**: Depends on training; trained AlphaSolitaire achieves ~40-50%+ win-rate
- **Dependencies**: Python neural-network service running on port 8000 (configurable)

## LLM Players

### OpenAI
- **Class**: [`OpenAIPlayer`](../../main/java/ai/games/player/ai/OpenAIPlayer.java)
- **Profile**: `openai`
- **Strategy**: GPT-based LLM with game state reasoning
- **Configuration**: API key, model selection (e.g., `gpt-4-mini`), prompt engineering
- **Use When**: You want to evaluate LLM reasoning capabilities; have OpenAI API access
- **Performance**: Varies by model; gpt-4 typically ~20-30%, gpt-3.5 lower
- **Cost**: Per-token API charges; can be expensive for large test runs

### Ollama (Local LLM)
- **Class**: [`OllamaPlayer`](../../main/java/ai/games/player/ai/OllamaPlayer.java)
- **Profile**: `ollama`
- **Strategy**: Local LLM inference via Ollama service
- **Configuration**: Ollama server URL, model selection, prompt parameters
- **Use When**: You want LLM reasoning without API costs; have Ollama service running
- **Performance**: Depends on model; 7B-13B models typically ~15-25%, larger models better but slower
- **Dependencies**: Ollama service running (default: http://localhost:11434)

## Test Organization

All player tests follow a consistent pattern:

1. **Functional Tests** (`*PlayerTest.java`): Verify player implementation
   - Move selection correctness
   - Integration with game engine
   - Error handling

2. **Performance Tests** (`results/*PlayerResultsTest.java`): Benchmark win-rates
   - Games played: 500 (configurable in `ResultsConfig.java`)
   - Measures: win %, average moves, time per game
   - Repeated across multiple players for comparison

## Running Tests

```bash
# All AI player functional tests
./gradlew test --tests "ai.games.player.ai.**"

# Specific player
./gradlew test --tests "ai.games.player.ai.GreedySearchPlayerTest"
./gradlew test --tests "ai.games.player.ai.MonteCarloPlayerTest"

# All performance benchmarks (slow)
./gradlew test --tests "ai.games.results.**"

# Specific player performance
./gradlew test --tests "ai.games.results.GreedySearchPlayerResultsTest"
```

## Configuration

Player configuration uses profiles defined in application properties. See `engine/README.md` for profile setup and `engine/src/main/resources/application.properties` for detailed configuration.

### Performance Tuning

- **Beam Search**: Increase `beamWidth` for better exploration (slower)
- **A* Search**: Tune heuristic scorer to be more aggressive
- **Monte Carlo**: Increase iteration count for better exploration
- **AlphaSolitaire**: Improve training data and network architecture

### Testing Strategy

When evaluating a new player:
1. Run functional test to verify basic correctness
2. Run quick performance benchmark (50-100 games) to estimate win-rate
3. Compare against baselines (rule-based, greedy)
4. If promising, run full benchmark (500 games) for final metrics

## Adding New Players

To add a new AI player:
1. Create `NewPlayer` extending `AIPlayer`
2. Implement `selectMove()` method
3. Create `NewPlayerTest` in `ai.games.player.ai` package
4. Add profile in `application.properties`
5. Create performance test in `ai.games.results` package
6. Document here with strategy and configuration

See existing players for implementation patterns.
