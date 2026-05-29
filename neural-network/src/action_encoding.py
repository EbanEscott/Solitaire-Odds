from typing import Iterable

import torch

from .log_loader import Episode

UNKNOWN_ACTION = "<UNK>"


class ActionSpace:
    def __init__(self, index_to_action, action_to_index):
        self.index_to_action = index_to_action
        self.action_to_index = action_to_index

    @property
    def size(self) -> int:
        return len(self.index_to_action)

    @classmethod
    def from_episodes(cls, episodes):
        actions = set()
        for episode in episodes:
            for step in episode.steps:
                cmd = step.chosen_command
                if cmd:
                    actions.add(cmd.strip())
                for move in step.legal_moves:
                    if move:
                        actions.add(move.strip())

        sorted_actions = sorted(a for a in actions if a)
        index_to_action = [UNKNOWN_ACTION] + sorted_actions
        action_to_index = {a: i for i, a in enumerate(index_to_action)}
        return cls(index_to_action=index_to_action, action_to_index=action_to_index)


def encode_action(space: ActionSpace, command: str) -> int:
    if not command:
        return space.action_to_index.get(UNKNOWN_ACTION, 0)
    normalized = command.strip()
    return space.action_to_index.get(normalized, space.action_to_index.get(UNKNOWN_ACTION, 0))


def encode_legal_mask(space: ActionSpace, commands: Iterable[str]) -> torch.Tensor:
    """Encode the currently legal moves as a fixed-width binary mask.

    The mask width matches the action vocabulary size. This representation keeps batching
    simple for graph-style models that need to know which action nodes are active in the
    current position.
    """

    mask = torch.zeros(space.size, dtype=torch.float32)
    for command in commands:
        action_index = encode_action(space, command)
        if 0 <= action_index < space.size:
            mask[action_index] = 1.0
    return mask


__all__ = ["ActionSpace", "encode_action", "encode_legal_mask", "UNKNOWN_ACTION"]

