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


def _split_rank(card) -> int:
    """
    Return the identifier used to decide whether two cards form a valid split pair.

    - Card dict  → display_rank (1=Ace, 2-10, 11=J, 12=Q, 13=K).
                   This mirrors model.py's rank_id logic: K and Q have different
                   display_ranks, so they are NOT a pair despite sharing value 10.
    - Plain int  → the int itself (used by ai_test.py / evaluation scripts).
    """
    if isinstance(card, dict):
        return card["display_rank"]
    return card


def _build_obs(player_hand: list, dealer_upcard: int, split_in_progress: bool) -> np.ndarray:
    """Build observation for model. player_hand can be Card dicts or ints."""
    player_sum = calculate_hand(player_hand)
    usable_ace = 1.0 if has_usable_ace(player_hand) else 0.0

    player_sum_norm    = (player_sum    - 4) / (21 - 4)
    dealer_upcard_norm = (dealer_upcard - 2) / (11 - 2)

    # Extract card values for normalization
    if len(player_hand) == 2:
        card1_value = player_hand[0]["rank"] if isinstance(player_hand[0], dict) else player_hand[0]
        card2_value = player_hand[1]["rank"] if isinstance(player_hand[1], dict) else player_hand[1]
        card1_norm = (card1_value - 2) / (11 - 2)
        card2_norm = (card2_value - 2) / (11 - 2)
    else:
        card1_norm = 0.0
        card2_norm = 0.0

    hand_size_norm = 0.0 if len(player_hand) == 2 else 1.0

    return np.array([
        player_sum_norm,
        dealer_upcard_norm,
        usable_ace,
        card1_norm,
        card2_norm,
        hand_size_norm,
    ], dtype=np.float32)


def _build_mask(player_hand: list, split_in_progress: bool) -> np.ndarray:
    """Build action mask. player_hand can be Card dicts or ints."""
    mask = [True, True, False, False]
    if len(player_hand) == 2:
        mask[2] = True                                        # Double Down
        if not split_in_progress:
            # Use display_rank so K+Q is NOT masked as a valid split
            if _split_rank(player_hand[0]) == _split_rank(player_hand[1]):
                mask[3] = True                                # Split
    return np.array(mask, dtype=bool)


def get_advice(player_hand: list, dealer_upcard: int, split_in_progress: bool = False) -> str | None:
    """Get AI advice. player_hand can be list of Card dicts or list of ints (ranks)."""
    if model is None:
        return None
    obs    = _build_obs(player_hand, dealer_upcard, split_in_progress)
    mask   = _build_mask(player_hand, split_in_progress)
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return ACTION_NAMES[int(action)]