# Solitaire PyTorch Hello World

This folder contains a minimal PyTorch "Hello World" script for the solitaire project.

## Prerequisites

- Python 3.9+ installed and available as `python3` on your `PATH` (typical on macOS).
- `pip` for installing Python packages.

## Setup

From the repository root (`/Users/ebo/Code/solitaire`):

```bash
# (Optional but recommended) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run the Hello World script

From the same repository root (and with the virtual environment activated, if you created one):

```bash
python3 hello.py
```

You should see the input tensor, the output tensor (`x * 2`), and some basic tensor metadata printed to the console.

## Use Java logs and run the training stub

Once you have Solitaire games logged from the Java engine (for example at `/Users/ebo/Code/cards/logs/game.log` with `-Dlog.episodes=true` enabled), you can load them and run the minimal training stub:

```bash
cd /Users/ebo/Code/solitaire
source .venv/bin/activate
python3 train_stub.py /Users/ebo/Code/cards/logs/game.log
```

This will:
- Build a `SolitaireStateDataset` from the log file.
- Print the state dimension and action-space size.
- Run a small MLP over a few batches to verify shapes and loss behave sensibly.
