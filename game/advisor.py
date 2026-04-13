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


def calculate_hand(hand: list[int]) -> int:
    total, aces = 0, 0
    for card in hand:
        if card == 11:
            aces += 1
            total += 11
        else:
            total += card
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total


def has_usable_ace(hand: list[int]) -> bool:
    total_low = sum(1 if c == 11 else c for c in hand)
    num_aces  = sum(1 for c in hand if c == 11)
    return num_aces > 0 and total_low + 10 <= 21


def _build_obs(player_hand: list[int], dealer_upcard: int, split_in_progress: bool) -> np.ndarray:
    player_sum = calculate_hand(player_hand)
    usable_ace = 1.0 if has_usable_ace(player_hand) else 0.0

    is_pair        = 0.0
    pair_card_norm = 0.0
    if len(player_hand) == 2 and player_hand[0] == player_hand[1] and not split_in_progress:
        is_pair        = 1.0
        pair_card_norm = PAIR_BUCKETS.get(player_hand[0], 0.0)

    hand_size_norm = 0.0 if len(player_hand) == 2 else 1.0

    return np.array([
        (player_sum    - 4) / 17.0,
        (dealer_upcard - 2) /  9.0,
        usable_ace,
        is_pair,
        pair_card_norm,
        hand_size_norm,
    ], dtype=np.float32)


def _build_mask(player_hand: list[int], split_in_progress: bool) -> np.ndarray:
    mask = [True, True, False, False]
    if len(player_hand) == 2:
        mask[2] = True
    if len(player_hand) == 2 and player_hand[0] == player_hand[1] and not split_in_progress:
        mask[3] = True
    return np.array(mask, dtype=bool)


def get_advice(player_hand: list[int], dealer_upcard: int, split_in_progress: bool = False) -> str | None:
    if model is None:
        return None
    obs    = _build_obs(player_hand, dealer_upcard, split_in_progress)
    mask   = _build_mask(player_hand, split_in_progress)
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return ACTION_NAMES[int(action)]
