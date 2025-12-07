"""
Action encoding for Solitaire moves.

This module defines a simple fixed action space built from the commands
seen in the Java engine logs (e.g., "turn", "quit", "move T7 Q♣ F1").

The mapping is string-based for now:

    action_index <-> exact command string

This keeps things straightforward while we are experimenting with models
and state encodings. We can later evolve this into a more structured
representation (e.g., from/to pile indices, card indices) without
changing the Episode logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from log_loader import Episode


UNKNOWN_ACTION = "<UNK>"


@dataclass
class ActionSpace:
    """Bidirectional mapping between action indices and command strings."""

    index_to_action: List[str]
    action_to_index: Dict[str, int]

    @property
    def size(self) -> int:
        return len(self.index_to_action)

    @classmethod
    def from_episodes(cls, episodes: Iterable[Episode]) -> "ActionSpace":
        """
        Build an action vocabulary from a sequence of episodes.

        We include:
        - All chosen commands.
        - All legal moves (so the policy head can, in principle, place
          probability mass on any legal action in the dataset).
        """
        actions: set[str] = set()
        for episode in episodes:
            for step in episode.steps:
                cmd = step.chosen_command
                if cmd:
                    actions.add(cmd.strip())
                for move in step.legal_moves:
                    if move:
                        actions.add(move.strip())

        # Ensure deterministic ordering:
        # 1) reserve index 0 for UNKNOWN_ACTION
        # 2) sort remaining actions lexicographically.
        sorted_actions = sorted(a for a in actions if a)
        index_to_action = [UNKNOWN_ACTION] + sorted_actions
        action_to_index = {a: i for i, a in enumerate(index_to_action)}
        return cls(index_to_action=index_to_action, action_to_index=action_to_index)


def encode_action(space: ActionSpace, command: str) -> int:
    """
    Map a command string to an action index.

    Unknown commands are mapped to the reserved UNKNOWN_ACTION index (0).
    """
    if not command:
        return space.action_to_index.get(UNKNOWN_ACTION, 0)
    normalized = command.strip()
    return space.action_to_index.get(normalized, space.action_to_index.get(UNKNOWN_ACTION, 0))


__all__ = ["ActionSpace", "encode_action", "UNKNOWN_ACTION"]

