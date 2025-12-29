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

## Running the Tests

### Run a specific level:
```bash
cd engine
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testEndgameLevel1" --console=plain
```

### Run all endgame levels:
```bash
cd engine
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator" --console=plain
```

## Generating Training Data Logs

To generate episode logs suitable for neural network training, enable episode logging:

```bash
cd engine
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testEndgameLevel2" --console=plain -Dlog.episodes=true
```

The logs will be written in JSONL format (one JSON object per line), suitable for ingestion by `neural-network/src/log_loader.py`.

## Design Notes

- **Placement Strategy**: Cards removed from foundations are placed using one of three strategies (tableau, stockpile, mixed) in a deterministic but varied way based on the game number. This creates diverse board positions at each level.

- **Visibility**: All cards in the tableau are face-up (visible) in these generated positions to simplify the training data initially. Future iterations could add face-down cards for more complexity.

- **Move Simulation**: Each game simulates up to 3 moves from the initial position to generate training steps. The `EpisodeLogger` captures the before/after state and move quality metrics.

## Next Steps

1. Run Level 1 and Level 2 to generate initial training data.
2. Train the model on this cleaner data and evaluate performance.
3. Gradually increase complexity (Level 3, 4, 5) as the model improves.
4. Compare win rates with the full A* player baseline in `README.md`.
