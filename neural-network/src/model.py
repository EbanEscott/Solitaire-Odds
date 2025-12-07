from torch import nn
import torch


class PolicyValueNet(nn.Module):
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

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        h = self.shared(x)
        logits = self.policy_head(h)
        value_logits = self.value_head(h)
        return logits, value_logits


__all__ = ["PolicyValueNet"]

