"""
Minimal training stub for AlphaSolitaire experiments.

This script:
  - Loads Solitaire episodes from a Java log file.
  - Builds a `SolitaireStateDataset`.
  - Runs a tiny neural network over a few batches to sanity-check shapes.

Usage (from the repo root):

    python3 train_stub.py /Users/ebo/Code/cards/logs/game.log
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import SolitaireStateDataset



class SimplePolicyNet(nn.Module):
    """Small MLP that maps state vectors to policy logits."""

    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        hidden_dim = 128
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main(args: list[str]) -> None:
    if not args:
        print("Usage: python3 train_stub.py /path/to/game.log")
        raise SystemExit(1)

    log_path = Path(args[0])
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        raise SystemExit(1)

    dataset = SolitaireStateDataset([log_path])
    if len(dataset) == 0:
        print("Dataset is empty; check that the Java engine was run with -Dlog.episodes=true.")
        raise SystemExit(1)

    # Peek at a single sample to determine dimensions.
    sample_state, sample_policy, _ = dataset[0]
    input_dim = sample_state.shape[0]
    num_actions = sample_policy.shape[0]

    print(f"Loaded {len(dataset)} samples from {log_path}")
    print(f"State dim: {input_dim}, action space size: {num_actions}")

    model = SimplePolicyNet(input_dim=input_dim, num_actions=num_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for step, (states, policies, _values) in enumerate(loader):
        # For now, use the one-hot policy's argmax as a class label.
        target_indices = policies.argmax(dim=-1)

        logits = model(states)
        loss = loss_fn(logits, target_indices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step:04d} - loss: {loss.item():.4f}")

        # Keep this stub short; stop after a few batches.
        if step >= 50:
            break


if __name__ == "__main__":
    main(sys.argv[1:])
