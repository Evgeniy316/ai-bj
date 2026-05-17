import numpy as np
from sb3_contrib import MaskablePPO

# ── load once at startup ──────────────────────────────────────
model = MaskablePPO.load("blackjack_infinite_shoe_model_35m.zip")

ACTION_NAMES = {0: "Stand", 1: "Hit", 2: "Double", 3: "Split"}

# ── helpers copied from model.py ──────────────────────────────
def calculate_hand(hand):
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

def has_usable_ace(hand):
    total_low = sum(1 if c == 11 else c for c in hand)
    num_aces  = sum(1 for c in hand if c == 11)
    return num_aces > 0 and total_low + 10 <= 21

PAIR_BUCKETS = {
            2: 0.1, 3: 0.2,    # split vs 2-7
            4: 0.3,             # split only vs 5-6
            5: 0.0,             # NEVER split (treat as 10)
            6: 0.5, 7: 0.6,    # split vs 2-6 / 2-7
            8: 1.0,             # ALWAYS split
            9: 0.8,             # split vs 2-6, 8-9
            10: 0.0,            # NEVER split
            11: 0.9,            # ALWAYS split
}

# ── state + mask builders ─────────────────────────────────────
def build_obs(player_hand, dealer_upcard, split_in_progress=False):
    player_sum  = calculate_hand(player_hand)
    usable_ace  = 1.0 if has_usable_ace(player_hand) else 0.0

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

def build_mask(player_hand, split_in_progress=False):
    mask = [True, True, False, False]   # stand, hit always on

    if len(player_hand) == 2:
        mask[2] = True                  # double down

    if (len(player_hand) == 2
            and player_hand[0] == player_hand[1]
            and not split_in_progress):
        mask[3] = True                  # split

    return np.array(mask, dtype=bool)

# ── the one function you call from your game ──────────────────
def get_advice(player_hand, dealer_upcard, split_in_progress=False):
    obs  = build_obs(player_hand, dealer_upcard, split_in_progress)
    mask = build_mask(player_hand, split_in_progress)

    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return ACTION_NAMES[int(action)]


# ── example usage ─────────────────────────────────────────────
if __name__ == "__main__":
    # Ace + 8  vs dealer 6
    print(get_advice([11, 8], dealer_upcard=6))   # → "Stand"

    # 8 + 8  vs dealer 9
    print(get_advice([8, 8], dealer_upcard=10))    # → "Split"

    # 8 + 8  vs dealer 9, but we're already in a split
    print(get_advice([8, 8], dealer_upcard=9, split_in_progress=True))  # → "Hit"