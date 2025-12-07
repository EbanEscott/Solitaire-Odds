"""
PyTorch Dataset for Solitaire training data.

This module ties together:
    - Java logs parsed via `log_loader.load_episodes_from_log`.
    - State tensors from `state_encoding.encode_state`.
    - Action indices from `action_encoding.ActionSpace`.

Each sample corresponds to a single logged step and returns:
    (state_tensor, policy_target, value_target)

Where:
    - state_tensor: encoded board state (1D float tensor).
    - policy_target: one-hot vector over the fixed action space.
    - value_target: scalar float (1.0 if the episode was won, else 0.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from action_encoding import ActionSpace, encode_action
from log_loader import Episode, load_episodes_from_log
from state_encoding import encode_state


@dataclass
class SampleIndex:
    """Reference to a single step within an episode."""

    episode_idx: int
    step_idx: int


class SolitaireStateDataset(Dataset):
    """
    Dataset over Solitaire steps parsed from one or more Java log files.
    """

    def __init__(self, log_paths: Sequence[str | Path]) -> None:
        super().__init__()
        self._episodes: List[Episode] = []

        for path in log_paths:
            episodes = load_episodes_from_log(path)
            self._episodes.extend(episodes)

        self.action_space: ActionSpace = ActionSpace.from_episodes(self._episodes)

        # Flatten (episode, step) pairs into a single index space.
        self._indices: List[SampleIndex] = []
        for epi_idx, episode in enumerate(self._episodes):
            for step_idx, _ in enumerate(episode.steps):
                self._indices.append(SampleIndex(episode_idx=epi_idx, step_idx=step_idx))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        ref = self._indices[idx]
        episode = self._episodes[ref.episode_idx]
        step = episode.steps[ref.step_idx]

        # State tensor.
        state = encode_state(step)

        # Policy target: one-hot over the action space.
        action_idx = encode_action(self.action_space, step.chosen_command)
        policy = torch.zeros(self.action_space.size, dtype=torch.float32)
        if 0 <= action_idx < self.action_space.size:
            policy[action_idx] = 1.0

        # Value target: propagate final outcome back to all steps
        # in the episode (1.0 for win, 0.0 otherwise).
        won = False
        if episode.summary is not None:
            won = episode.summary.won
        value = torch.tensor(1.0 if won else 0.0, dtype=torch.float32)

        return state, policy, value


def build_dataset_from_logs(log_dir: str | Path, pattern: str = "*.log") -> SolitaireStateDataset:
    """
    Convenience helper to create a dataset from all matching log files
    in a directory.
    """
    root = Path(log_dir)
    paths = sorted(root.glob(pattern))
    return SolitaireStateDataset(paths)


__all__ = ["SolitaireStateDataset", "build_dataset_from_logs"]

