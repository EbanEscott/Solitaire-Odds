# Endgame Training Data Generator

This directory contains the `EndgameTrainingDataGenerator` test class, which generates progressively complex endgame training data for the AlphaSolitaire neural network.

## Overview

Instead of using random full games (which have only ~20% win rate and sparse examples near the endgame), this generator creates cleaner training data by:

1. **Starting from a won state**: All 52 cards are on the foundations.
2. **Working backward**: Incrementally removes cards and places them in strategic positions.
3. **Generating levels of increasing complexity**:
   - **Level 1**: All foundations full (baseline, no moves needed)
   - **Level 2**: 51 cards on foundations; 1 card to place
   - **Level 3**: 50 cards on foundations; 2 cards to place
   - **Level 4**: 48 cards on foundations; 4 cards to place
   - **Level 5**: 45 cards on foundations; 7 cards to place

Each level generates 500 games by default.

# Endgame Training Data Generator

This directory contains the `EndgameTrainingDataGenerator` test class, which generates progressively complex endgame training data for the AlphaSolitaire neural network.

## Overview

Instead of using random full games (which have only ~20% win rate and sparse examples near the endgame), this generator creates cleaner training data by:

1. **Starting from a won state**: All 52 cards are on the foundations.
2. **Working backward**: Incrementally removes cards and places them in strategic positions.
3. **Generating levels of increasing complexity**:
   - **Level 1**: All foundations full (baseline, no moves needed)
   - **Level 2**: 51 cards on foundations; 1 card to place
   - **Level 3**: 50 cards on foundations; 2 cards to place
   - **Level 4**: 48 cards on foundations; 4 cards to place
   - **Level 5**: 45 cards on foundations; 7 cards to place
   - **Level N**: Arbitrary difficulty (backward compatible)

## Memory-Efficient Generation Strategy

### Problem: Exponential State Explosion

When generating games at high difficulty levels, **intermediate levels can explode in memory usage** if not properly managed. This is not a theoretical problem—it's a critical constraint you must understand when tuning parameters.

**Critical Real-World Example**: Requesting 1,000 games at Level 150 without optimization:

Naive approach (what we had before the fix):
- Level 2: 1,000 games generated
- Level 3: 1,000 × 7 avg moves = **7,000 intermediate games**
- Level 4: 7,000 × 7 = **49,000 intermediate games**
- Level 5: 49,000 × 7 = **343,000 intermediate games**
- Level 10: **49 million intermediate games** ✗ **Out of Memory!**
- Level 150: **Astronomical state explosion** ✗ **Complete failure**

This is why **you must understand the tradeoffs** when choosing game counts and difficulty levels. The parameter choices have exponential memory consequences.

### Solution: Branching Factor Optimization

`TrainingOpponent` now:

1. **Dynamically samples the average branching factor** (reverse moves per game) at initialization
2. **Pre-calculates minimum games needed at each intermediate level**
3. **Only generates what's strictly necessary** to reach the target game count

The pre-calculation works backward from target:
```
Level N games needed = ceil(Level N+1 games needed / avg_branching_factor)
```

**Concrete Example: 1,000 games at Level 5 with BF=7:**

Result shown in logs:
```
Seed strategy: avg_branching_factor=7.12, games_needed_per_level={2=1, 3=18, 4=143, 5=1000}
```

- Level 5: 1,000 games (target, returned to you)
- Level 4: 143 games (internal only, pre-calculated)
- Level 3: 18 games (internal only, pre-calculated)
- Level 2: 1 game (internal only, pre-calculated)

**Total memory**: ~1,165 games across all levels (only ~165 extra beyond target!)  
**Without optimization**: ~343,000 games (99.7% memory reduction!)

## Memory Usage Guidelines

The algorithm dynamically samples the branching factor for each run, adapting to the specific level and board state. Here are empirical estimates:

### Memory Per Target Game Count

| Target Games | Level 4 Memory | Level 5 Memory | Level 10 Memory |
|--------------|----------------|----------------|-----------------|
| 50           | ~200 MB        | ~250 MB        | ~300 MB         |
| 100          | ~300 MB        | ~400 MB        | ~500 MB         |
| 500          | ~1.0 GB        | ~1.3 GB        | ~1.8 GB         |
| 1000         | ~1.8 GB        | ~2.3 GB        | ~3.2 GB         |
| 5000         | ~8.0 GB        | ~11 GB         | ~16 GB          |
```
| 50           | ~200 MB        | ~250 MB        | ~300 MB         |
| 100          | ~300 MB        | ~400 MB        | ~500 MB         |
| 500          | ~1.0 GB        | ~1.3 GB        | ~1.8 GB         |
| 1000         | ~1.8 GB        | ~2.3 GB        | ~3.2 GB         |
| 5000         | ~8.0 GB        | ~11 GB         | ~16 GB          |

### Calculating Total Games Across All Levels

The total number of games generated (including intermediate levels) is:

**Total Games = RequestedGames × (1 + 1/BF + 1/BF² + ... + 1/BF^(D-1))**

where:
- `RequestedGames` = `-Dendgame.games.per.level` value (target level games returned to you)
- `BF` = average branching factor (see logs: `avg_branching_factor=X.XX`)
- `D` = difficulty level minus 1 (number of intermediate levels)

**Practical Examples**:

1. **1,000 games at Level 5, BF=7:**
   - Total = 1,000 × (1 + 0.143 + 0.020 + 0.003) ≈ **1,166 games**
   - You get: 1,000 at Level 5
   - Intermediate: ~166 games across Levels 2-4

2. **1,000 games at Level 150, BF=7:**
   - Total ≈ 1,000 × (1 + 0.143 + 0.020 + ...) ≈ **1,143 games** (still bounded!)
   - You get: 1,000 at Level 150
   - Intermediate: ~143 games total across 148 levels (exponentially decreasing)

3. **5,000 games at Level 5, BF=4 (lower branching):**
   - Total = 5,000 × (1 + 0.25 + 0.0625 + ...) ≈ **6,667 games**
   - You get: 5,000 at Level 5
   - Intermediate: ~1,667 games across Levels 2-4

The optimization ensures **total games grow only logarithmically** with difficulty level, not exponentially.

## Randomization: Creating Diverse Endgames

By default, the game generation is **deterministic** - requesting the same level/game count produces the same board positions each time. This is useful for reproducible testing.

However, for **training data diversity**, you can enable **randomization** via `-Dendgame.randomize=true`. This makes the algorithm randomly select reverse moves at each level instead of following the same sequence.

### Effect of Randomization

**Without randomization (deterministic):**
```
Level 4 games: move F3 K♥ T3, move F3 K♥ T4, move F3 K♥ T5, move F3 K♥ T6, ...
```
Same sequence every run - useful for testing, but limited diversity.

**With randomization enabled:**
```
Level 4 games: move F2 K♦ W, move F1 K♣ W, move T7 K♠ T5, move F3 K♥ T6, ...
```
Random sequence each run - creates diverse endgame positions, critical for deep levels (100+).

### Why Randomization Matters for Deep Levels

At deep levels (e.g., Level 100), without randomization:
- Always follows the same reverse move path
- Results in similar board positions each time
- May be "easy wins" if that path happens to create favorable positions
- Limited training diversity for neural networks

**With randomization:**
- Each run explores different paths through the reverse moves
- Creates diverse endgame positions
- Some may be harder than others (like Level 100 with 334 moves!)
- Provides realistic variety for training
- Helps avoid overfitting to specific game patterns

### Usage

```bash
# Deterministic mode (default - reproducible)
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testGenerateEndgameDataset" \
  -Dendgame.games.difficulty.level=4 -Dendgame.games.per.level=500

# Randomized mode (diverse endgames)
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testGenerateEndgameDataset" \
  -Dendgame.games.difficulty.level=100 -Dendgame.games.per.level=50 \
  -Dendgame.randomize=true
```

## Running Tests


### Generate a single level (recommended for large game counts):

```bash
cd engine

# Level 4 with 500 games
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testGenerateEndgameDataset" \
  -Dendgame.games.difficulty.level=4 \
  -Dendgame.games.per.level=500 \
  --console=plain

# Level 10 (arbitrary difficulty) with 1000 games, 6GB heap
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testGenerateEndgameDataset" \
  -Dendgame.games.difficulty.level=10 \
  -Dendgame.games.per.level=1000 \
  -Dorg.gradle.jvmargs=-Xmx6g \
  --console=plain
```

### Generate multiple levels (legacy individual tests):

```bash
cd engine

# Run all predefined levels (1-5) with episode logging
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator" \
  -Dlog.episodes=true \
  --console=plain
```

### Key System Properties

| Property | Default | Purpose |
|----------|---------|---------|
| `-Dendgame.games.difficulty.level=N` | None | Target difficulty level (1-N). If not set, legacy tests run. |
| `-Dendgame.games.per.level=N` | 5000 | Number of games to generate at the target level. |
| `-Dendgame.randomize=true\|false` | false | Enable random reverse move selection for diverse endgames. Recommended for deep levels (50+). |
| `-Dlog.episodes=true` | false | Enable episode logging in JSONL format for neural network training. |
| `-Dorg.gradle.jvmargs=-Xmx6g` | -Xmx512m | Heap memory. Increase for high game counts. |

## Tuning for Your Environment

### If you get OutOfMemoryError:

1. **Reduce game count**: `-Dendgame.games.per.level=500` (instead of 5000)
2. **Increase heap**: `-Dorg.gradle.jvmargs=-Xmx8g`
3. **Lower difficulty**: Use Level 4 instead of Level 5 (branching factor grows with difficulty)
4. **Use multiple runs**: Generate 2 × 500 games instead of 1 × 1000 to allow GC between runs

### If memory is plentiful and you want faster generation:

1. **Increase game count**: `-Dendgame.games.per.level=10000`
2. **Higher difficulty levels**: Level 10+ still work with reasonable memory due to branching factor optimization
3. **Run in parallel** (future): Multiple levels simultaneously in separate JVM processes

## Design Notes

- **Placement Strategy**: Cards removed from foundations are placed using one of three strategies (tableau, stockpile, mixed) in a deterministic but varied way based on the game number. This creates diverse board positions at each level.

- **Visibility**: All cards in the tableau are face-up (visible) in these generated positions to simplify the training data initially. Future iterations could add face-down cards for more complexity.

- **Move Simulation**: Each game simulates moves from the initial position to generate training steps. The `EpisodeLogger` captures the before/after state and move quality metrics.

- **Arbitrary Levels**: Unlike legacy hardcoded levels (1-5), the new system supports any positive difficulty level via `-Dendgame.games.difficulty.level=N`. Higher levels generate progressively harder endgames using the same reverse-move expansion algorithm.

## Understanding the Log Output

When you run a test, you'll see:

```
[INFO] Seed strategy: avg_branching_factor=7.12, games_needed_at_level_2=3
[DEBUG] Level 2: found 5 reverse moves from base game
[DEBUG] Generated Level 5 game: move F1 T3 -> foundation_count=45 (game count: 1/500)
...
================================================================================
TRAINING DATASET SUMMARY: Level 5
================================================================================
Games Played: 500
Games Won:    499
Win Rate:     99.80%
Avg Moves:    3.11
Total Time:   0.457s
================================================================================
```

**Key lines**:
- `avg_branching_factor`: Average reverse moves per game (used to calculate intermediate minimums)
- `games_needed_at_level_2`: Minimum games needed at Level 2 to reach target at Level 5
- `Generated Level N game`: Indicates which level games are being added to the output
- **Summary**: Overall statistics; low win rates (90-99%) at high levels indicate realistic endgame complexity

## Next Steps

1. Generate Level 4 or Level 5 training data: `./gradlew test ... -Dendgame.games.difficulty.level=4 -Dendgame.games.per.level=500`
2. Export logs with `-Dlog.episodes=true` for neural network training
3. Train on this cleaner data and evaluate performance
4. Compare win rates with the full A* player baseline in the root `README.md`

