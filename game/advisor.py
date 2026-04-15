import numpy as np
from sb3_contrib import MaskablePPO

model = None

ACTION_NAMES = {0: "Stand", 1: "Hit", 2: "Double", 3: "Split"}

PAIR_BUCKETS = {
    2: 0.1, 3: 0.2, 4: 0.3, 5: 0.0,
    6: 0.5, 7: 0.6, 8: 1.0, 9: 0.8,
    10: 0.0, 11: 0.9,
}


def load_model(path: str = "blackjack_infinite_shoe_model_35m.zip") -> None:
    global model
    model = MaskablePPO.load(path)


def calculate_hand(hand: list) -> int:
    """Calculate hand value. Accepts list of Card dicts or list of ints (ranks)."""
    total, aces = 0, 0
    for card in hand:
        # Handle both Card dict and raw int
        value = card["rank"] if isinstance(card, dict) else card
        if value == 11:
            aces += 1
            total += 11
        else:
            total += value
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total


def has_usable_ace(hand: list) -> bool:
    """Check for usable ace. Accepts list of Card dicts or list of ints (ranks)."""
    total_low = 0
    num_aces = 0
    for card in hand:
        value = card["rank"] if isinstance(card, dict) else card
        total_low += 1 if value == 11 else value
        if value == 11:
            num_aces += 1
    return num_aces > 0 and total_low + 10 <= 21


def _build_obs(player_hand: list, dealer_upcard: int, split_in_progress: bool) -> np.ndarray:
    """Build observation for model. player_hand can be Card dicts or ints."""
    player_sum = calculate_hand(player_hand)
    usable_ace = 1.0 if has_usable_ace(player_hand) else 0.0

    is_pair        = 0.0
    pair_card_norm = 0.0
    if len(player_hand) == 2 and not split_in_progress:
        # Extract ranks for comparison
        rank1 = player_hand[0]["rank"] if isinstance(player_hand[0], dict) else player_hand[0]
        rank2 = player_hand[1]["rank"] if isinstance(player_hand[1], dict) else player_hand[1]
        if rank1 == rank2:
            is_pair        = 1.0
            pair_card_norm = PAIR_BUCKETS.get(rank1, 0.0)

    hand_size_norm = 0.0 if len(player_hand) == 2 else 1.0

    return np.array([
        (player_sum    - 4) / 17.0,
        (dealer_upcard - 2) /  9.0,
        usable_ace,
        is_pair,
        pair_card_norm,
        hand_size_norm,
    ], dtype=np.float32)


def _build_mask(player_hand: list, split_in_progress: bool) -> np.ndarray:
    """Build action mask. player_hand can be Card dicts or ints."""
    mask = [True, True, False, False]
    if len(player_hand) == 2:
        mask[2] = True
        if not split_in_progress:
            # Extract ranks for comparison
            rank1 = player_hand[0]["rank"] if isinstance(player_hand[0], dict) else player_hand[0]
            rank2 = player_hand[1]["rank"] if isinstance(player_hand[1], dict) else player_hand[1]
            if rank1 == rank2:
                mask[3] = True
    return np.array(mask, dtype=bool)


def get_advice(player_hand: list, dealer_upcard: int, split_in_progress: bool = False) -> str | None:
    """Get AI advice. player_hand can be list of Card dicts or list of ints (ranks)."""
    if model is None:
        return None
    obs    = _build_obs(player_hand, dealer_upcard, split_in_progress)
    mask   = _build_mask(player_hand, split_in_progress)
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return ACTION_NAMES[int(action)]
