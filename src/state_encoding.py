from typing import Tuple

import torch

from .log_loader import EpisodeStep


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

MAX_VISIBLE_PER_PILE = 19


def _parse_short_name(short_name: str) -> Tuple[float, float]:
    if not short_name:
        return 0.0, 0.0

    suit_symbol = short_name[-1]
    rank_label = short_name[:-1]

    rank_value = float(RANK_LABEL_TO_VALUE.get(rank_label, 0))
    is_red = SUIT_SYMBOL_TO_IS_RED.get(suit_symbol, 0.0)
    return rank_value, is_red


def encode_state(step: EpisodeStep) -> torch.Tensor:
    features = []

    tableau_visible = step.tableau_visible
    tableau_face_down = step.tableau_face_down

    for pile_index in range(7):
        if pile_index < len(tableau_visible):
            full_pile = tableau_visible[pile_index]
        else:
            full_pile = []

        face_down = float(tableau_face_down[pile_index]) if pile_index < len(tableau_face_down) else 0.0
        visible_count = float(len(full_pile))
        features.extend([face_down, visible_count])

        if len(full_pile) > MAX_VISIBLE_PER_PILE:
            pile = full_pile[-MAX_VISIBLE_PER_PILE :]
        else:
            pile = list(full_pile)

        for idx in range(MAX_VISIBLE_PER_PILE):
            if idx < len(pile):
                rank, is_red = _parse_short_name(pile[idx])
            else:
                rank, is_red = 0.0, 0.0
            features.extend([rank, is_red])

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

    talon = step.talon
    talon_size = float(len(talon))
    if talon:
        top_card = talon[-1]
        talon_top_rank, talon_top_is_red = _parse_short_name(top_card)
    else:
        talon_top_rank, talon_top_is_red = 0.0, 0.0

    features.extend([talon_size, talon_top_rank, talon_top_is_red])

    stock_size = float(step.stock_size)
    features.append(stock_size)

    return torch.tensor(features, dtype=torch.float32)


__all__ = ["encode_state"]

