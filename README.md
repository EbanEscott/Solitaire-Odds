# Solitaire PyTorch Hello World

This folder contains a minimal PyTorch "Hello World" script for the solitaire project.

## Prerequisites

- Python 3.9+ installed and available as `python3` on your `PATH` (typical on macOS).
- `pip` for installing Python packages.

## Setup

From the Python project root (`/Users/ebo/Code/solitaire/solitaire`):

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
python3 hello.py
```

You should see the input tensor, the output tensor (`x * 2`), and some basic tensor metadata printed to the console.

## Use Java logs and run the training stub

Once you have Solitaire games logged from the Java engine (for example at `/Users/ebo/Code/cards/logs/game.log` with `-Dlog.episodes=true` enabled), you can load them and run the minimal training stub:

```bash
cd /Users/ebo/Code/solitaire/solitaire
source .venv/bin/activate
python3 train_stub.py /Users/ebo/Code/cards/logs/game.log
```

This will:
- Build a `SolitaireStateDataset` from the log file.
- Print the state dimension and action-space size.
- Run a small MLP over a few batches to verify shapes and loss behave sensibly.

## Train a policy–value network

To train a joint policy–value model with a validation split:

```bash
cd /Users/ebo/Code/solitaire/solitaire
source .venv/bin/activate
python3 train_policy_value.py /Users/ebo/Code/cards/logs/game.log
```

This will:
- Build train/validation splits from the logged games.
- Train a `PolicyValueNet` to imitate the logged moves and predict win probability.
- Save a checkpoint to `checkpoints/policy_value_latest.pt`.

## Run the AlphaSolitaire model service

For integration with the Java engine (an `AlphaSolitairePlayer` that calls into Python), run the HTTP service:

```bash
cd /Users/ebo/Code/solitaire/solitaire
source .venv/bin/activate
python3 service.py --checkpoint checkpoints/policy_value_latest.pt --host 127.0.0.1 --port 8000
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
