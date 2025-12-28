"""
Minimal training stub for AlphaSolitaire experiments.

This module can be run as:

    python -m src.train_stub /path/to/episode.log
    python -m src.train_stub /path/to/episode*.log
    python -m src.train_stub /path/to/episode.1.log /path/to/episode.2.log ...

It:
  - Loads Solitaire episodes from one or more Java log files (supports glob patterns).
  - Builds a `SolitaireStateDataset`.
  - Runs a tiny neural network over a few batches to sanity-check shapes.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import SolitaireStateDataset


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


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: python -m src.train_stub /path/to/episode.log [/path/to/episode.2.log ...]")
        print("       python -m src.train_stub /path/to/episode*.log")
        raise SystemExit(1)

    # Expand glob patterns and collect all log files
    log_paths = []
    for pattern in argv:
        expanded = glob.glob(pattern)
        if expanded:
            log_paths.extend(sorted(expanded))
        else:
            # If no glob match, treat as direct path
            log_paths.append(pattern)

    if not log_paths:
        print(f"Error: no log files found matching: {argv}")
        raise SystemExit(1)

    print(f"Loading {len(log_paths)} file(s)...")
    for p in log_paths:
        print(f"  - {p}")

    print("Parsing episodes...")
    dataset = SolitaireStateDataset([Path(p) for p in log_paths])
    if len(dataset) == 0:
        print("Dataset is empty; check that the Java engine was run with -Dlog.episodes=true.")
        raise SystemExit(1)

    # Peek at a single sample to determine dimensions.
    sample_state, sample_policy, _ = dataset[0]
    input_dim = sample_state.shape[0]
    num_actions = sample_policy.shape[0]

    print(f"Loaded {len(dataset)} samples ({input_dim}D state, {num_actions} actions)")
    print(f"State dim: {input_dim}, action space size: {num_actions}")

    model = SimplePolicyNet(input_dim=input_dim, num_actions=num_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    print(f"Running sanity check over {len(loader)} batches...")
    for step, (states, policies, _values) in enumerate(loader):
        if step % max(1, len(loader) // 10) == 0:
            print(f"  Step {step:04d} / {len(loader):04d}")
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
    main()
