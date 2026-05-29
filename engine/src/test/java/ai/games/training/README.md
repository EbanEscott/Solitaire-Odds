# Training Tests: Endgame Generation and AlphaSolitaire Evaluation

Quick gradle commands for generating endgame training data and testing the AlphaSolitaire neural network player.

## 1. Generate Endgame Training Data

Use `EndgameTrainingDataGenerator` to create progressively harder endgame positions for neural network training:

```bash
cd engine

# Level 2: 1 card off foundations
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testEndgameLevel2" --rerun-tasks --console=plain "-Dlog.episodes=true"

# Level 4: 4 cards off foundations
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testEndgameLevel4" --rerun-tasks --console=plain "-Dlog.episodes=true"

# Level 5: 7 cards off foundations
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testEndgameLevel5" --rerun-tasks --console=plain "-Dlog.episodes=true"

# Custom level (e.g., Level 10)
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testGenerateEndgameDataset" --rerun-tasks --console=plain "-Dlog.episodes=true" "-Dendgame.games.per.level=100" "-Dendgame.games.difficulty.level=10"

# All levels 1-5
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator" --rerun-tasks --console=plain "-Dlog.episodes=true"
```

## 2. Test AlphaSolitaire Neural Player

Use `AlphaSolitaireLevelTest` to evaluate the neural network player on seeded endgame difficulty levels.

**Prerequisites:** Start the neural service in a separate terminal:
```bash
cd neural-network
source .venv/bin/activate
python -m src.service --checkpoint checkpoints/policy_value_latest.pt --host 127.0.0.1 --port 8000
```

Then run tests in the engine directory:

```bash
cd engine

# Level 2: 1 card off foundations
./gradlew test --tests "ai.games.training.AlphaSolitaireLevelTest.testOpponentLevel2" --rerun-tasks --console=plain "-Dendgame.games.per.level=10"

# Level 4: 4 cards off foundations
./gradlew test --tests "ai.games.training.AlphaSolitaireLevelTest.testOpponentLevel4" --rerun-tasks --console=plain "-Dendgame.games.per.level=10"

# Level 5: 7 cards off foundations
./gradlew test --tests "ai.games.training.AlphaSolitaireLevelTest.testOpponentLevel5" --rerun-tasks --console=plain "-Dendgame.games.per.level=10"

# Custom level (e.g., Level 10)
./gradlew test --tests "ai.games.training.AlphaSolitaireLevelTest.testOpponent" --rerun-tasks --console=plain "-Dendgame.games.per.level=10" "-Dendgame.games.difficulty.level=10"

# All levels 2-5
./gradlew test --tests "ai.games.training.AlphaSolitaireLevelTest" --rerun-tasks --console=plain "-Dendgame.games.per.level=5"
```

### Important: This is not the full random-game benchmark

`AlphaSolitaireLevelTest` seeds endgame positions using reverse moves from a solved board, then asks the neural player to finish those positions. The difficulty comes from the `-Dendgame.games.difficulty.level=<N>` property.

If you want the full random-deal benchmark sweep instead, use `AlphaSolitairePlayerResultsTest` from the `results` package:

```bash
cd engine
./gradlew test --tests ai.games.results.AlphaSolitairePlayerResultsTest --rerun-tasks --console=plain \
	"-Dalphasolitaire.tests=true" \
	"-Dtest.games=100"
```

That results test does not read `-Dendgame.games.difficulty.level`. It always creates fresh shuffled games and only uses the shared sweep properties such as `-Dtest.games` and `-Dtest.max.moves.per.game`.

## Difficulty Levels

| Level | Cards Off | Cards On Foundation | Typical Use |
|-------|-----------|---------------------|------------|
| 2     | 1         | 51                  | Quick validation |
| 3     | 2         | 50                  | Training start |
| 4     | 4         | 48                  | Main training |
| 5     | 7         | 45                  | Full training |
| N     | Varies    | 52-N                | Custom difficulty |
