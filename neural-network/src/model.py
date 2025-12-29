from torch import nn
import torch
from typing import Dict


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
        
        # Tier 1 metric heads for multi-task learning
        self.foundation_move_head = nn.Linear(hidden_dim, 1)
        self.revealed_facedown_head = nn.Linear(hidden_dim, 1)
        self.talon_move_head = nn.Linear(hidden_dim, 1)
        self.cascading_move_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """
        Forward pass computing policy, value, and Tier 1 metrics.
        
        Args:
            x: Input state tensor of shape (batch_size, state_dim)
            
        Returns:
            Dict with keys:
            - 'policy': (batch_size, num_actions) policy logits
            - 'value': (batch_size, 1) value logits
            - 'foundation_move': (batch_size, 1) foundation move logits
            - 'revealed_facedown': (batch_size, 1) revealed facedown logits
            - 'talon_move': (batch_size, 1) talon move logits
            - 'cascading_move': (batch_size, 1) cascading move logits
        """
        h = self.shared(x)
        return {
            'policy': self.policy_head(h),
            'value': self.value_head(h),
            'foundation_move': self.foundation_move_head(h),
            'revealed_facedown': self.revealed_facedown_head(h),
            'talon_move': self.talon_move_head(h),
            'cascading_move': self.cascading_move_head(h),
        }


__all__ = ["PolicyValueNet"]

