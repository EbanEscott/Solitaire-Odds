from torch import nn
import torch
from typing import Dict, Optional


class PolicyValueNet(nn.Module):
    """
    Configurable policy-value network with flexible depth and width.
    
    Supports varying architectures for training on full game trajectories.
    Can be used as a bootstrapped value function for self-play scenarios.
    """
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        use_batch_norm: bool = False,
        use_residual: bool = False,
    ) -> None:
        """
        Args:
            state_dim: Input state dimension (e.g., 296 for Solitaire).
            num_actions: Number of possible actions.
            hidden_dim: Hidden layer dimension. Default 256 (small), can go up to 2048+ for larger models.
            num_layers: Number of hidden layers in shared backbone (1-5+). Default 2.
            use_batch_norm: If True, apply batch normalization after each layer.
            use_residual: If True, add residual connections (requires same in/out dims).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Build shared backbone
        layers = []
        in_dim = state_dim
        
        for layer_idx in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Output heads
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

