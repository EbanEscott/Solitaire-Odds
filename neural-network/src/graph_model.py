from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class GraphPolicyValueNet(nn.Module):
    """Policy-value network over a star graph of board state and legal move nodes.

    The encoded board state acts as the root node. Each action in the fixed action
    vocabulary has a learned embedding, and the current legal-move mask selects which
    action nodes participate in message passing for a given position.

    This keeps batching simple because the action vocabulary is fixed, while still giving
    the model relational structure that a flat MLP does not have. The network exchanges
    information between the root state node and the currently legal action nodes before
    producing policy logits and a scalar value estimate.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        action_embedding_dim: int = 128,
        message_passing_steps: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Create a graph-style policy-value model.

        Args:
            state_dim: Encoded board-state feature dimension.
            num_actions: Size of the fixed action vocabulary.
            hidden_dim: Shared hidden size for root and child node states.
            num_layers: Depth of the root-state encoder MLP.
            action_embedding_dim: Size of each action-node embedding.
            message_passing_steps: Number of root-child update rounds.
            dropout: Dropout probability applied after node updates.
        """

        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if message_passing_steps < 1:
            raise ValueError("message_passing_steps must be >= 1")
        if action_embedding_dim < 1:
            raise ValueError("action_embedding_dim must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")

        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.action_embedding_dim = action_embedding_dim
        self.message_passing_steps = message_passing_steps
        self.dropout_probability = dropout

        self.action_embedding = nn.Embedding(num_actions, action_embedding_dim)
        self.state_encoder = self._build_state_encoder(state_dim, hidden_dim, num_layers)
        self.child_init = nn.Linear(hidden_dim + action_embedding_dim, hidden_dim)
        self.root_updates = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(message_passing_steps)]
        )
        self.child_updates = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(message_passing_steps)]
        )
        self.dropout = nn.Dropout(dropout)
        self.policy_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim * 2, 1)

    def _build_state_encoder(self, state_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
        """Build the MLP that initializes the root board-state node."""

        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        return nn.Sequential(*layers)

    def _masked_mean(self, values: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Average legal action-node states while ignoring masked actions."""

        mask = legal_mask.unsqueeze(-1).to(dtype=values.dtype)
        summed = (values * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def _apply_legal_mask(self, values: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Zero masked action nodes so illegal actions do not influence message passing."""

        return values * legal_mask.unsqueeze(-1).to(dtype=values.dtype)

    def forward(  # type: ignore[override]
        self,
        state: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run message passing over the board-state root and current legal actions.

        Args:
            state: Tensor of shape ``(batch_size, state_dim)``.
            legal_mask: Binary tensor of shape ``(batch_size, num_actions)`` marking
                which action nodes are active in the current graph.

        Returns:
            Dict with keys:
            - ``policy``: masked logits over the full fixed action vocabulary.
            - ``value``: scalar value logits of shape ``(batch_size, 1)``.
        """

        if legal_mask.dim() != 2:
            raise ValueError("legal_mask must be a rank-2 tensor")
        if legal_mask.shape[1] != self.num_actions:
            raise ValueError(
                f"legal_mask width {legal_mask.shape[1]} does not match num_actions {self.num_actions}"
            )

        legal_mask_bool = legal_mask > 0
        root_hidden = self.state_encoder(state)

        action_indices = torch.arange(self.num_actions, device=state.device)
        action_embeddings = self.action_embedding(action_indices).unsqueeze(0).expand(
            state.shape[0],
            -1,
            -1,
        )
        root_context = root_hidden.unsqueeze(1).expand(-1, self.num_actions, -1)
        child_hidden = torch.relu(self.child_init(torch.cat([root_context, action_embeddings], dim=-1)))
        child_hidden = self._apply_legal_mask(child_hidden, legal_mask_bool)

        for root_update, child_update in zip(self.root_updates, self.child_updates):
            # Each round first lets the root aggregate from the currently legal action nodes,
            # then pushes the refreshed root context back down into those same action nodes.
            pooled_children = self._masked_mean(child_hidden, legal_mask_bool)
            root_hidden = torch.relu(root_update(torch.cat([root_hidden, pooled_children], dim=-1)))
            root_hidden = self.dropout(root_hidden)

            root_context = root_hidden.unsqueeze(1).expand(-1, self.num_actions, -1)
            child_hidden = torch.relu(child_update(torch.cat([child_hidden, root_context], dim=-1)))
            child_hidden = self.dropout(child_hidden)
            child_hidden = self._apply_legal_mask(child_hidden, legal_mask_bool)

        pooled_children = self._masked_mean(child_hidden, legal_mask_bool)
        raw_policy_logits = self.policy_head(child_hidden).squeeze(-1)
        # Illegal actions must stay impossible after softmax, so they are driven to the
        # minimum finite logit value instead of merely zeroing their pre-softmax scores.
        masked_policy_logits = raw_policy_logits.masked_fill(
            ~legal_mask_bool,
            torch.finfo(raw_policy_logits.dtype).min,
        )
        value_logits = self.value_head(torch.cat([root_hidden, pooled_children], dim=-1))
        return {
            "policy": masked_policy_logits,
            "value": value_logits,
        }


__all__ = ["GraphPolicyValueNet"]