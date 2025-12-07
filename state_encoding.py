"""
State encoding for AlphaSolitaire training.

This module converts a logged `EpisodeStep` (see `log_loader.py`) into a
fixed-size PyTorch tensor suitable as input to a neural network.

Design goals for this first encoder:

- Use only information exposed in the logs:
  - Full sequence of visible tableau cards + per-pile face-down counts.
  - Foundation piles.
  - Talon (waste) and stock size.
- Keep the representation simple and well-documented so it is easy to
  evolve later (e.g., to a richer convolutional layout).

Current encoding (1D feature vector):

Per tableau pile (7 piles):
    - face_down_count (float)
    - visible_count (float)
    - For each visible card, from bottom-most visible to top-most visible,
      a pair:
        - rank (float, 0 if padding, otherwise 1–13)
        - is_red (0.0 or 1.0)

  The visible sequence is padded with zeros up to MAX_VISIBLE_PER_PILE cards
  so that every pile contributes a fixed number of features.

Per foundation pile (4 piles):
    - pile_size (float)
    - top_rank (float, 0 if empty, otherwise 1–13)
    - top_is_red (0.0 or 1.0)

Talon (waste):
    - talon_size (float)
    - top_rank (float, 0 if empty, otherwise 1–13)
    - top_is_red (0.0 or 1.0)

Stock:
    - stock_size (float)

Total feature dimension currently depends on MAX_VISIBLE_PER_PILE; see
the code comments for details.
"""

from __future__ import annotations

from typing import Tuple

import torch

from log_loader import EpisodeStep


RANK_LABEL_TO_VALUE = {
    "A": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
}

SUIT_SYMBOL_TO_IS_RED = {
    "♣": 0.0,
    "♠": 0.0,
    "♦": 1.0,
    "♥": 1.0,
}

# Maximum number of visible cards we encode per tableau pile.
# If a pile has fewer, the remaining slots are zero-padded; if a pile
# ever had more, the extra cards beyond this limit would be truncated
# from the *bottom* of the visible segment.
#
# In standard Klondike, a pile can have at most 19 visible cards
# (when all 24 non-tableau cards have been moved away and over time
# flipped), so 19 is a safe upper bound.
MAX_VISIBLE_PER_PILE = 19


def _parse_short_name(short_name: str) -> Tuple[float, float]:
    """
    Parse a Java `Card.shortName()` string into (rank_value, is_red).

    Examples of short names:
        "Q♠", "10♦", "A♥"

    Returns:
        rank_value: 1–13 for A–K, or 0.0 on parse failure.
        is_red: 1.0 for hearts/diamonds, 0.0 otherwise.
    """
    if not short_name:
        return 0.0, 0.0

    # Suit is always the final character.
    suit_symbol = short_name[-1]
    rank_label = short_name[:-1]

    rank_value = float(RANK_LABEL_TO_VALUE.get(rank_label, 0))
    is_red = SUIT_SYMBOL_TO_IS_RED.get(suit_symbol, 0.0)
    return rank_value, is_red


def encode_state(step: EpisodeStep) -> torch.Tensor:
    """
    Encode a single logged Solitaire state into a 1D tensor.

    The encoding is deliberately simple and documented in the module docstring.
    It can be evolved later without changing the EpisodeStep interface.
    """
    features = []

    tableau_visible = step.tableau_visible
    tableau_face_down = step.tableau_face_down

    # 7 tableau piles: for each pile we encode:
    #   - face_down_count
    #   - visible_count
    #   - MAX_VISIBLE_PER_PILE * (rank, is_red) from bottom-visible to top.
    for pile_index in range(7):
        # Visible cards for this pile (may be missing if logs or game variant change).
        if pile_index < len(tableau_visible):
            full_pile = tableau_visible[pile_index]
        else:
            full_pile = []

        face_down = float(tableau_face_down[pile_index]) if pile_index < len(tableau_face_down) else 0.0
        visible_count = float(len(full_pile))
        features.extend([face_down, visible_count])

        # If there are more visible cards than MAX_VISIBLE_PER_PILE, drop
        # the oldest visible cards (from the bottom) and keep the most recent
        # ones; moves tend to operate near the top.
        if len(full_pile) > MAX_VISIBLE_PER_PILE:
            pile = full_pile[-MAX_VISIBLE_PER_PILE :]
        else:
            pile = list(full_pile)

        # Encode from bottom-visible to top-visible, padding with zeros.
        for idx in range(MAX_VISIBLE_PER_PILE):
            if idx < len(pile):
                rank, is_red = _parse_short_name(pile[idx])
            else:
                rank, is_red = 0.0, 0.0
            features.extend([rank, is_red])

    # 4 foundation piles: pile_size, top_rank, top_is_red.
    foundation = step.foundation
    for pile_index in range(4):
        if pile_index < len(foundation):
            pile = foundation[pile_index]
        else:
            pile = []

        pile_size = float(len(pile))
        if pile:
            top_card = pile[-1]
            top_rank, top_is_red = _parse_short_name(top_card)
        else:
            top_rank, top_is_red = 0.0, 0.0

        features.extend([pile_size, top_rank, top_is_red])

    # Talon (waste): size, top_rank, top_is_red.
    talon = step.talon
    talon_size = float(len(talon))
    if talon:
        top_card = talon[-1]
        talon_top_rank, talon_top_is_red = _parse_short_name(top_card)
    else:
        talon_top_rank, talon_top_is_red = 0.0, 0.0

    features.extend([talon_size, talon_top_rank, talon_top_is_red])

    # Stock size (hidden cards; order unknown to the model).
    stock_size = float(step.stock_size)
    features.append(stock_size)

    # Convert to a 1D float tensor.
    return torch.tensor(features, dtype=torch.float32)


__all__ = ["encode_state"]
