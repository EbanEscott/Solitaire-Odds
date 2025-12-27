# Game Tree Analysis Test

Exhaustively explores Solitaire game trees to understand the scale of the search space. This helps inform AI training decisions like depth limits, branching factors, and win-rate expectations.

## Overview

The `GameTreeAnalysisTest` class explores complete game trees for Solitaire by:
1. Generating N distinct games (different random shuffles)
2. For each game, exhaustively exploring all possible moves and turns
3. Recording statistics: total nodes, max depth, branching factors, win/loss rates
4. Computing aggregate statistics across all games

## Quick Start

### Run Quick Test (100 games, ~2-3 minutes)

```bash
cd engine
./gradlew test --tests "GameTreeAnalysisTest.testExhaustiveGameTreeAnalysis_100Games_Quick"
```

### Run Full Test (1000 games, ~30-40 minutes)

```bash
cd engine
./gradlew test --tests "GameTreeAnalysisTest.testExhaustiveGameTreeAnalysis_1000Games"
```

### Run All Tests in the Class

```bash
cd engine
./gradlew test --tests GameTreeAnalysisTest
```

## Configuration

Edit the test method in `src/test/java/ai/games/GameTreeAnalysisTest.java`:

### Number of Games

```java
int numGames = 100;  // Change this to 10, 50, 100, 500, or 1000
```

### Depth Limit

```java
int maxDepthPerGame = 15;  // Increase for deeper exploration (but slower)
                           // Decrease for faster testing
```

**Depth limit guidance:**
- **15** (default): ~1-5 seconds per game, ~30-40 min for 1000 games
- **20**: ~5-15 seconds per game, ~2-4 hours for 1000 games
- **10**: ~0.5-1 second per game, ~10-20 min for 1000 games

## Understanding the Output

### Per-Game Statistics

```
Game    1: Nodes=6858, MaxDepth=15, Wins=0, Losses=0, AvgBranching=6.52, Time=1054ms
```

- **Nodes**: Total number of unique game states explored
- **MaxDepth**: Deepest level reached in the search
- **Wins**: Number of winning states found (all cards in foundation)
- **Losses**: Dead-end states (no legal moves available)
- **AvgBranching**: Average number of legal moves per state
- **Time**: Milliseconds to explore this game's tree

### Aggregate Statistics

```
Total Games Analyzed:       100
Total Nodes (all games):    1,234,567
Avg Nodes per Game:         12,345.7
Max Nodes in a Game:        45,000
Min Nodes in a Game:        2,500

Total Win States Found:     0 (0.00%)
Total Loss States Found:    12,345 (1.00%)
Total Intermediate States:  1,222,222 (99.00%)

Avg Max Depth per Game:     14.5
Avg Branching Factor:       5.25

Total Exploration Time:     24,826 ms (24.83 s)
Overall Elapsed Time:       24,830 ms (24.83 s)
```

**Key metrics:**
- **Avg Nodes**: Average game tree size - indicates complexity
- **Branching Factor**: Average moves available per state (typically 4-7)
- **Win/Loss Percentages**: How constrained the state space is

## Example Configurations

### Quick Development Test
```java
int numGames = 10;
int maxDepthPerGame = 10;
// Runtime: ~5-10 seconds
```

### Standard Analysis
```java
int numGames = 100;
int maxDepthPerGame = 15;
// Runtime: ~2-3 minutes
```

### Comprehensive Analysis
```java
int numGames = 1000;
int maxDepthPerGame = 15;
// Runtime: ~30-40 minutes
```

### Deep Exploration (for small sample)
```java
int numGames = 10;
int maxDepthPerGame = 25;
// Runtime: ~2-5 minutes (explores deeper trees)
```

## Implementation Details

### Algorithm
- **Search**: Depth-first search (DFS)
- **Cycle Detection**: Zobrist hashing (64-bit game state fingerprints)
- **Memory**: Game copies used instead of undo (cleaner, safer)
- **Branching**: Exhaustive enumeration of all legal moves

### Legal Moves Enumeration
For each state, the test checks:
1. Card moves: T1-T7 → T1-T7, F1-F4 (tableau to tableau/foundation)
2. Card moves: F1-F4 → T1-T7 (foundation to tableau)
3. Card moves: W (talon) → T1-T7, F1-F4
4. Card moves: S (stockpile) → T1-T7, F1-F4 (not standard, but checked)
5. Turn action: Draw 3 cards from stockpile to talon

## Interpreting Results

### What the stats tell you about AI training:

**Average nodes per game (~10K-20K):**
- Manageable for shallow search (depth 5-10)
- Challenging for exhaustive search (depth 15+)
- Good candidate for neural network value estimation

**Branching factor (~5-6 moves/state):**
- Relatively constrained search space
- Monte Carlo methods effective
- Beam search viable with reasonable width

**Win rates found (typically 0% at shallow depths):**
- Most shallow games are unwinnable without deep lookahead
- Explains why greedy/heuristic players have low win rates
- Confirms need for strategic planning

**Loss states (1-5% of explored):**
- Relatively few dead ends
- Suggests game allows meaningful exploration
- Good properties for game tree search

## Example Output from 10-Game Run

```
========================================
Exhaustive Game Tree Analysis: 10 Games
Max Depth Limit per Game: 20
========================================

Game    1: Nodes=2944, MaxDepth=20, Wins=0, Losses=72, AvgBranching=4.50, Time=469ms

========================================
AGGREGATE STATISTICS
========================================
Total Games Analyzed:       10
Total Nodes (all games):    184,775
Avg Nodes per Game:         18477.5
Max Nodes in a Game:        67,559
Min Nodes in a Game:        2,944
Std Dev (Nodes):            18306.9

Total Win States Found:     0 (0.00%)
Total Loss States Found:    2571 (1.39%)
Total Intermediate States:  182,204 (98.61%)

Avg Max Depth per Game:     20.0
Avg Branching Factor:       5.50

Total Exploration Time:     24,826 ms (24.83 s)
Overall Elapsed Time:       24,830 ms (24.83 s)
========================================
```

## Performance Tips

1. **Run on powerful hardware**: Multi-core systems see limited benefit (single-threaded DFS)
2. **Close other applications**: Java may compete for memory/CPU
3. **Use lower depth limits for quick feedback**: Iterate with depth 10, then depth 15
4. **Capture output**: Redirect to file for analysis
   ```bash
   ./gradlew test --tests GameTreeAnalysisTest 2>&1 | tee results.txt
   ```

## See Also

- `src/main/java/ai/games/game/Solitaire.java` - Game model with state hashing
- `src/test/java/ai/games/TrainingModeTest.java` - Move history and undo testing
- `engine/README.md` - Overall engine documentation
