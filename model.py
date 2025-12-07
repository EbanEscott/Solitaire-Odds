"""
Neural network architectures for AlphaSolitaire experiments.

Currently provides a simple shared MLP with:
  - A policy head (categorical over the action space).
  - A value head (scalar win-probability prediction).
"""

from __future__ import annotations

from torch import nn
import torch


class PolicyValueNet(nn.Module):
    """
    Simple policy–value network on top of 1D state features.

    Forward signature:
        logits, value_logits = model(state_batch)

    Where:
        - logits: unnormalised policy scores of shape (B, num_actions).
        - value_logits: raw scalar logits of shape (B, 1) for win probability.
          Use with `BCEWithLogitsLoss` during training.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        h = self.shared(x)
        logits = self.policy_head(h)
        value_logits = self.value_head(h)
        return logits, value_logits


__all__ = ["PolicyValueNet"]

