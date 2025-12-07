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

## Use Java logs and run the training stub

Once you have Solitaire games logged from the Java engine (for example at `/Users/ebo/Code/solitaire/engine/logs/game.log` with `-Dlog.episodes=true` enabled), you can load them and run the minimal training stub module:

```bash
cd /Users/ebo/Code/solitaire/neural-network
source .venv/bin/activate
python -m src.train_stub /Users/ebo/Code/solitaire/engine/logs/game.log
```

This will:
- Build a `SolitaireStateDataset` from the log file.
- Print the state dimension and action-space size.
- Run a small MLP over a few batches to verify shapes and loss behave sensibly.

## Train a policy–value network

To train a joint policy–value model with a validation split:

```bash
cd /Users/ebo/Code/solitaire/neural-network
source .venv/bin/activate
python -m src.train_policy_value /Users/ebo/Code/solitaire/engine/logs/game.log
```

This will:
- Build train/validation splits from the logged games.
- Train a `PolicyValueNet` to imitate the logged moves and predict win probability.
- Save a checkpoint to `checkpoints/policy_value_latest.pt`.

## Run the AlphaSolitaire model service

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
