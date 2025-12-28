# Neural Network

This directory contains the Python modeling stack for AlphaSolitaire. It turns logged Solitaire games from the Java engine into datasets, trains policy–value neural networks, and exposes an HTTP service that the `AlphaSolitairePlayer` in the engine can call to evaluate game states and choose moves.

## Prerequisites

- Python 3.9+ installed and available as `python3` on your `PATH` (typical on macOS).
- `pip` for installing Python packages.

## Setup

From the Python project root (`/Users/ebo/Code/solitaire/neural-network`):

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

## Generate Training Data (Step 1)

Before training, you need episode logs from the Java engine. From the `engine/` directory, run a results test with the `-Dlog.episodes=true` flag to generate clean episode JSON lines:

```bash
cd /Users/ebo/Code/solitaire/engine

# Generate episodes from any AI player (examples below)
./gradlew test --tests ai.games.results.AStarPlayerResultsTest -Dlog.episodes=true
./gradlew test --tests ai.games.results.GreedySearchPlayerResultsTest -Dlog.episodes=true
./gradlew test --tests ai.games.results.RuleBasedHeuristicsPlayerResultsTest -Dlog.episodes=true

# Or run all player benchmarks and generate episodes from each
./gradlew test --tests "ai.games.results.**" -Dlog.episodes=true

# Verify episodes were logged
wc -l logs/episode.log
head -1 logs/episode.log
```

Episodes are written to `engine/logs/episode.log`. Each line is a JSON object with type `EPISODE_STEP` (per-move state and action) or `EPISODE_SUMMARY` (game outcome).

## Use Java logs and run the training stub (Step 2)

Once you have Solitaire games logged from the Java engine (for example at `/Users/ebo/Code/solitaire/engine/logs/episode.log` with `-Dlog.episodes=true` enabled), you can load them and run the minimal training stub module:

```bash
cd /Users/ebo/Code/solitaire/neural-network
source .venv/bin/activate

# Single file
python -m src.train_stub /Users/ebo/Code/solitaire/engine/logs/episode.log

# Multiple files
python -m src.train_stub logs/episode.1.log logs/episode.2.log logs/episode.3.log

# Glob pattern (quote to prevent shell expansion)
python -m src.train_stub "logs/episode*.log"
```

This will:
- Build a `SolitaireStateDataset` from the log file(s).
- Print the state dimension and action-space size.
- Run a small MLP over a few batches to verify shapes and loss behave sensibly.

## Train a policy–value network (Step 3)

To train a joint policy–value model with a validation split, run the full training script:

```bash
cd /Users/ebo/Code/solitaire/neural-network
source .venv/bin/activate

# Single file
python -m src.train_policy_value /Users/ebo/Code/solitaire/engine/logs/episode.log

# Multiple files
python -m src.train_policy_value logs/episode.1.log logs/episode.2.log logs/episode.3.log

# Glob pattern (quote to prevent shell expansion)
python -m src.train_policy_value "logs/episode*.log"
```

This will:
- Resolve all log files (supports glob patterns and multiple file arguments).
- Build train/validation splits from the logged games (90/10 split).
- Train a `PolicyValueNet` to imitate the logged moves and predict win probability.
- Save a checkpoint to `checkpoints/policy_value_latest.pt`.

**Example output** (training on 346k+ samples from 1000 A* games):
```
Training on 39815 samples, validating on 4423 samples (state_dim=296, num_actions=2539, device=cpu)
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

The checkpoint is now ready for use with the AlphaSolitaire service.

## Run the AlphaSolitaire model service (Step 4)

For integration with the Java engine (an `AlphaSolitairePlayer` that calls into Python), run the HTTP service module:

```bash
cd /Users/ebo/Code/solitaire/neural-network
source .venv/bin/activate
python -m src.service --checkpoint checkpoints/policy_value_latest.pt --host 127.0.0.1 --port 8000
```

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
