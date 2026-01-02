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

# Card ordering for unknown card inventory (52 cards)
CARD_ORDER = []
for suit in ["♣", "♠", "♦", "♥"]:
    for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]:
        CARD_ORDER.append(f"{rank}{suit}")

# Maximum unknown positions: 7 tableau piles × 22 max depth + 24 stockpile = 46
MAX_UNKNOWN_POSITIONS = 46


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

    # ===== OPTION C: Unknown Card Constraints (236 features) =====
    
    # Part 1: Card Inventory (52 features)
    # Binary flag for each real card: 1.0 if unknown, 0.0 if known/revealed
    unknown_card_set = set(step.unknown_cards)
    for card_name in CARD_ORDER:
        is_unknown = 1.0 if card_name in unknown_card_set else 0.0
        features.append(is_unknown)
    
    # Part 2: Guess Constraints (184 features = 46 positions × 4 features)
    # For each position, encode: rank_low, rank_high, has_red, has_black
    guesses_ordered = step.unknown_guesses_ordered
    
    for pos_idx in range(MAX_UNKNOWN_POSITIONS):
        if pos_idx < len(guesses_ordered) and guesses_ordered[pos_idx]:
            possibilities = guesses_ordered[pos_idx]
            
            # Extract ranks from possibilities
            ranks = []
            has_red = False
            has_black = False
            
            for card_name in possibilities:
                if not card_name:
                    continue
                # Parse rank
                suit_symbol = card_name[-1]
                rank_label = card_name[:-1]
                rank_value = RANK_LABEL_TO_VALUE.get(rank_label, 0)
                
                if rank_value > 0:
                    ranks.append(rank_value)
                
                # Check color
                if suit_symbol in "♦♥":
                    has_red = True
                elif suit_symbol in "♣♠":
                    has_black = True
            
            # Normalize rank bounds to [0, 1]
            if ranks:
                rank_low = float(min(ranks)) / 13.0
                rank_high = float(max(ranks)) / 13.0
            else:
                rank_low = 0.0
                rank_high = 0.0
            
            features.extend([rank_low, rank_high, float(has_red), float(has_black)])
        else:
            # No guess at this position
            features.extend([0.0, 0.0, 0.0, 0.0])

    return torch.tensor(features, dtype=torch.float32)


__all__ = ["encode_state"]

