# Neural Network

This directory contains the Python modeling stack for AlphaSolitaire. It turns logged Solitaire games from the Java engine into datasets, trains policy–value neural networks, and exposes an HTTP service that the `AlphaSolitairePlayer` in the engine can call to evaluate game states and choose moves.

## Prerequisites

- Python 3.9+ installed and available as `python3` on your `PATH` (typical on macOS).
- `pip` for installing Python packages.

## Setup

From the Python project root (`neural-network/`):

```bash
# (Optional but recommended) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run the Hello World script

From the same project root (and with the virtual environment activated, if you created one):

```bash
python -m src.hello
```

You should see a small training run for y = 2x + 1 and a prediction for a new x value.

## Extend the Endgame Curriculum

Use this sequence when you want to extend the current checkpoint to a new seeded level such as Level 7. The goal is to preserve the training lineage instead of treating `policy_value_latest.pt` and `engine/logs/episode*.log` as disposable working files.

### 1. Inspect the current checkpoint lineage

```bash
cd neural-network
source .venv/bin/activate

python - <<'PY'
import torch

checkpoint = torch.load("checkpoints/policy_value_latest.pt", map_location="cpu", weights_only=False)
metadata = checkpoint.get("metadata", {})
print("timestamp:", metadata.get("timestamp"))
print("training_samples:", metadata.get("training_samples"))
print("validation_samples:", metadata.get("validation_samples"))
print("data_sources:")
for path in metadata.get("data_sources", []):
    print("  -", path)
PY
```

This is the quickest way to confirm which archived logs produced `policy_value_latest.pt`. On the current checkpoint in this repo, the metadata shows archived endgame logs through Level 6.

### 2. Generate the next level in the Java engine

```bash
cd ../engine

./gradlew test \
  --tests "ai.games.training.EndgameTrainingDataGenerator.testGenerateEndgameDataset" \
  --rerun-tasks --console=plain \
  "-Dlog.episodes=true" \
  "-Dendgame.games.difficulty.level=7" \
  "-Dendgame.games.per.level=500"
```

Practical notes:
- The requested game count is a target, not a guarantee. With the current deterministic seed expansion, the validated Level 7 run requested 500 games and produced 404, so use the logged denominator in `[Level 7] Playing game 1/404` as the real count.
- If you want a different branch sample, add `"-Dendgame.randomise=true"`. That changes which reverse moves are selected, but it does not guarantee the full requested count.
- If you want the same randomised shard again, also add `"-Dendgame.random.seed=<seed>"`. This is useful when the experiments driver is collecting multiple reproducible shards for the same level.
- If this command fails with a state-validation error, stop there. The generator is supposed to reject broken seeded boards instead of silently logging bad data.

### 3. Archive the generated logs immediately

```bash
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive_dir="logs/archive/level7/$timestamp"
mkdir -p "$archive_dir"
find logs -maxdepth 1 -name 'episode*.log' -exec mv {} "$archive_dir"/ \;
find "$archive_dir" -maxdepth 1 -type f | sort
```

Do this before any other training or benchmark run. `engine/logs/episode.log` and its rotated siblings are working files and will be overwritten.

### 4. Train a new checkpoint from the archived curriculum

```bash
cd ../neural-network
source .venv/bin/activate

logs=(
  ../engine/logs/archive/level2/**/*.log(N)
  ../engine/logs/archive/level3/**/*.log(N)
  ../engine/logs/archive/level4/**/*.log(N)
  ../engine/logs/archive/level5/**/*.log(N)
  ../engine/logs/archive/level6/**/*.log(N)
  ../engine/logs/archive/level7/**/*.log(N)
)

printf '%s\n' "${logs[@]}"
python -m src.train_policy_value "${logs[@]}"
```

If older archived levels are missing on the current machine, copy them back before training. The checkpoint metadata from step 1 is the source of truth for which levels were previously used.

### 5. Promote the new checkpoint

```bash
python tools/promote_checkpoint.py \
  --name level7 \
  --description "Trained on archived endgame episodes through level 7"
```

This gives you a versioned checkpoint and companion metadata file so `policy_value_latest.pt` is not the only surviving copy.

### 6. Start the service and run evaluation

```bash
python -m src.service --checkpoint checkpoints/policy_value_latest.pt --host 127.0.0.1 --port 8000
```

Then, from `engine/`, run the seeded generalization test:

```bash
cd ../engine

./gradlew test --tests "ai.games.training.AlphaSolitaireLevelTest.testOpponent" \
  --rerun-tasks --console=plain \
  "-Dendgame.games.difficulty.level=20" \
  "-Dendgame.games.per.level=20"
```

Use the evaluation section below when you want the random full-game benchmark instead of seeded endgames.

## Generate Generic Training Data

If you are extending the endgame curriculum, use the workflow above. This section is for generic episode collection from the Java engine.

Before training, you need episode logs from the Java engine. From the `engine/` directory, run a results test with the `-Dlog.episodes=true` flag to generate clean episode JSON lines:

```bash
cd engine

# Generate episodes from any AI player (examples below)
./gradlew test --tests ai.games.results.AStarPlayerResultsTest "-Dlog.episodes=true"
./gradlew test --tests ai.games.results.GreedySearchPlayerResultsTest "-Dlog.episodes=true"
./gradlew test --tests ai.games.results.RuleBasedHeuristicsPlayerResultsTest "-Dlog.episodes=true"

# Or run all player benchmarks and generate episodes from each
./gradlew test --tests "ai.games.results.**" "-Dlog.episodes=true"

# Verify episodes were logged
wc -l logs/episode.log
head -1 logs/episode.log
```

Episodes are written to `engine/logs/episode.log`. Each line is a JSON object with type `EPISODE_STEP` (per-move state and action) or `EPISODE_SUMMARY` (game outcome).

## Use Java logs and run the training stub

Once you have Solitaire games logged from the Java engine (for example at `../engine/logs/episode.log` when you are in `neural-network/` with `-Dlog.episodes=true` enabled), you can load them and run the minimal training stub module:

```bash
cd neural-network
source .venv/bin/activate

# Single file
python -m src.train_stub ../engine/logs/episode.log

# Multiple files
python -m src.train_stub ../engine/logs/episode.1.log ../engine/logs/episode.2.log ../engine/logs/episode.3.log

# Glob pattern (quote to prevent shell expansion)
python -m src.train_stub "../engine/logs/episode*.log"
```

This will:
- Build a `SolitaireStateDataset` from the log file(s).
- Print the state dimension and action-space size.
- Run a small MLP over a few batches to verify shapes and loss behave sensibly.

## Train a policy–value network

To train a joint policy–value model with a validation split, run the full training script with configurable architecture:

```bash
cd neural-network
source .venv/bin/activate

# Default (256 hidden, 2 layers)
python -m src.train_policy_value ../engine/logs/episode.log

# Medium model (512 hidden, 3 layers) — recommended for full game tree training
python -m src.train_policy_value \
  --hidden-dim 512 \
  --num-layers 3 \
  ../engine/logs/episode.log

# Large model (1024 hidden, 3 layers) — for 200k+ samples
python -m src.train_policy_value \
  --hidden-dim 1024 \
  --num-layers 3 \
  ../engine/logs/episode.log

# Multiple files or glob patterns
python -m src.train_policy_value "../engine/logs/episode*.log"
```

**Configuration options** (see `ARCHITECTURE.md` for detailed explanation):
- `--hidden-dim` (default: 256): Width of hidden layers (128-2048+)
- `--num-layers` (default: 2): Depth of network (1-5+)
- `--batch-norm`: Enable batch normalization (experimental)
- `--residual`: Enable residual connections (experimental)
- `--epochs` (default: 5): Training epochs
- `--batch-size` (default: 64): Batch size
- `--learning-rate` (default: 1e-3): Adam learning rate

This will:
- Resolve all log files (supports glob patterns and multiple file arguments).
- Build train/validation splits from the logged games (90/10 split).
- **Train on full game trajectories**: Each step labeled with game outcome + MCTS-guided moves (ready for self-play RL).
- Train a `PolicyValueNet` to imitate the logged moves and predict win probability.
- Save a checkpoint to `checkpoints/policy_value_latest.pt`.

**Example output** (training on 346k+ samples from 1000 A* games):
```
Training on 39815 samples, validating on 4423 samples (state_dim=296, num_actions=2539, device=cpu)
Model Architecture: hidden_dim=512, num_layers=3, batch_norm=False, residual=False
Model Size: 731,241 total parameters, 731,241 trainable
Estimated checkpoint size: 2.79 MB
Training: 5 epochs, batch_size=64, lr=0.001
Epoch 1/5 - train_loss(p=2.065, v=0.101), train_acc(p=0.658, v=0.965) - val_loss(p=1.511, v=0.054), val_acc(p=0.658, v=0.979)
Epoch 2/5 - train_loss(p=1.354, v=0.050), train_acc(p=0.671, v=0.980) - val_loss(p=1.321, v=0.037), val_acc(p=0.698, v=0.988)
Epoch 3/5 - train_loss(p=1.115, v=0.037), train_acc(p=0.710, v=0.986) - val_loss(p=1.251, v=0.027), val_acc(p=0.742, v=0.991)
Epoch 4/5 - train_loss(p=0.956, v=0.027), train_acc(p=0.735, v=0.990) - val_loss(p=1.276, v=0.020), val_acc(p=0.764, v=0.993)
Epoch 5/5 - train_loss(p=0.842, v=0.021), train_acc(p=0.747, v=0.993) - val_loss(p=1.295, v=0.018), val_acc(p=0.757, v=0.994)
Saved model checkpoint to checkpoints/policy_value_latest.pt
```

**Metrics explained:**
- `train_loss(p=..., v=...)` — Policy (action prediction) and value (win probability) losses on training data
- `train_acc(p=..., v=...)` — Policy and value accuracy on training data
  - Policy accuracy: fraction of predicted actions matching the logged moves (~75% for A* player)
  - Value accuracy: binary accuracy of win/loss prediction (~99%)
- Validation metrics show the model generalizes well (val_acc ≈ train_acc)

**Key advancement:** The network now trains on **full game trajectories** where every step is labeled with the game outcome. This is critical for self-play: as MCTS improves and generates better move sequences, the network learns those sequences directly. This is how AlphaGo bootstraps from supervised learning into self-play RL.

### Architecture & Design

For detailed explanation of:
- How trajectory-aware training works
- Why it enables self-play
- How to choose model size for your data
- Bootstrapped values for self-play RL

See:
- `ARCHITECTURE.md` — Architecture configuration and capacity analysis
- `QUICK_START.md` — Quick commands for common scenarios
- `SELF_PLAY_RL.md` — How bootstrapped values will work for self-play loop

The checkpoint is now ready for use with the AlphaSolitaire service.



## Run the AlphaSolitaire model service

For integration with the Java engine (an `AlphaSolitairePlayer` that calls into Python), run the HTTP service module:

```bash
cd neural-network
source .venv/bin/activate
python -m src.service --checkpoint checkpoints/policy_value_latest.pt --host 127.0.0.1 --port 8000
```

## Evaluate the Current Checkpoint

There are two different Java-side AlphaSolitaire test modes, and only one of them accepts a difficulty level.

### Seeded Endgame Evaluation by Difficulty Level

Use this when you want to answer questions like "how far beyond the training levels does the checkpoint generalize?"

This path uses `AlphaSolitaireLevelTest` and supports:
- `-Dendgame.games.difficulty.level=<N>`
- `-Dendgame.games.per.level=<count>`

Example:

```bash
cd engine
./gradlew test --tests "ai.games.training.AlphaSolitaireLevelTest.testOpponent" \
  --rerun-tasks --console=plain \
  "-Dendgame.games.difficulty.level=20" \
  "-Dendgame.games.per.level=10"
```

These tests do not start from a random shuffled deal. They seed a game by applying reverse moves from a solved board until the requested difficulty level is reached.

### Random Full-Game Benchmark Sweep

Use this when you want the true end-to-end win rate from fresh shuffled games.

This path uses `AlphaSolitairePlayerResultsTest` and does not read `-Dendgame.games.difficulty.level`. It always runs full random games and instead supports the shared sweep properties:
- `-Dalphasolitaire.tests=true`
- `-Dtest.games=<count>`
- `-Dtest.progress.log.interval=<count>`
- `-Dtest.max.moves.per.game=<count>`

Example:

```bash
cd engine
./gradlew test --tests ai.games.results.AlphaSolitairePlayerResultsTest \
  --console=plain --rerun-tasks \
  "-Dalphasolitaire.tests=true" \
  "-Dtest.games=100"
```

Use the seeded level test for Phase 0 falloff work. Use the random full-game sweep for README-style benchmark numbers.

The service exposes a single endpoint:

- `POST /evaluate` with JSON body:

  ```json
  {
    "tableau_visible": [["3♦","4♠"], ["(etc)"]],
    "tableau_face_down": [3, 0, 0, 0, 0, 0, 0],
    "foundation": [["A♣"], [], [], []],
    "talon": ["7♣"],
    "stock_size": 24,
    "legal_moves": ["turn", "move W T1", "move T1 4♠ F1"]
  }
  ```

- The response JSON contains:

  ```json
  {
    "chosen_command": "move T1 4♠ F1",
    "win_probability": 0.73,
    "legal_moves": [
      {"command": "move T1 4♠ F1", "probability": 0.73},
      {"command": "turn", "probability": 0.20}
    ]
  }
  ```

On the Java side, an `AlphaSolitairePlayer` can mirror the existing logging structure to build this JSON, POST it to `/evaluate`, and use `chosen_command` as its move.
