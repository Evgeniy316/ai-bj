"""
Basic strategy for blackjack.
Rules: infinite shoe · dealer stands on soft 17 (S17) · double after split allowed (DAS).
"""

from advisor import calculate_hand, has_usable_ace


# ── helpers ───────────────────────────────────────────────────

def _ranks(hand: list) -> list[int]:
    """Return blackjack VALUES (2-11) for hand calculation."""
    return [c["rank"] if isinstance(c, dict) else c for c in hand]


def _split_rank(card) -> int:
    """
    Return the identifier used to decide whether two cards form a valid split pair.

    - Card dict  → display_rank (1=Ace, 2-10, 11=J, 12=Q, 13=K).
                   This mirrors model.py's rank_id logic: K and Q have different
                   display_ranks, so they are NOT a pair despite sharing value 10.
    - Plain int  → the int itself.
    """
    if isinstance(card, dict):
        return card["display_rank"]
    return card


def _can_double(hand_size: int) -> bool:
    return hand_size == 2


# ── pair strategy ─────────────────────────────────────────────

def _pair_strategy(pair_value: int, dealer: int) -> str:
    """pair_value is the blackjack VALUE (2-11) of the pair card."""
    if pair_value == 11:                        # European: no split vs A (lose to BJ)
        return "Split" if dealer <= 10 else "Hit"

    if pair_value == 10:                        # 10,10 → always stand
        return "Stand"

    if pair_value == 9:                         # 9,9 → split vs 2-6, 8-9
        return "Stand" if dealer in (7, 10, 11) else "Split"

    if pair_value == 8:                         # European: no split vs 10 or A (lose to BJ)
        return "Split" if dealer <= 9 else "Hit"

    if pair_value == 7:                         # 7,7 → split vs 2-7
        return "Split" if 2 <= dealer <= 7 else "Hit"

    if pair_value == 6:                         # 6,6 → split vs 2-6
        return "Split" if 2 <= dealer <= 6 else "Hit"

    if pair_value == 5:                         # 5,5 → never split; treat as hard 10
        return "Double" if 2 <= dealer <= 9 else "Hit"

    if pair_value == 4:                         # 4,4 → split vs 5-6 (DAS)
        return "Split" if dealer in (5, 6) else "Hit"

    # 2,2 / 3,3 → split vs 2-7
    return "Split" if 2 <= dealer <= 7 else "Hit"


# ── soft-hand strategy ────────────────────────────────────────

def _soft_strategy(total: int, dealer: int, hand_size: int) -> str:
    dbl = _can_double(hand_size)

    if total >= 19:                             # soft 19-21 → stand
        return "Stand"

    if total == 18:                             # soft 18 (A+7)
        if dealer in (2, 7, 8):
            return "Stand"
        if 3 <= dealer <= 6:
            return "Double" if dbl else "Stand"
        return "Hit"                            # vs 9, 10, A

    if total == 17:                             # soft 17 (A+6)
        return ("Double" if dbl else "Hit") if 3 <= dealer <= 6 else "Hit"

    if total in (15, 16):                       # soft 15-16 (A+4, A+5)
        return ("Double" if dbl else "Hit") if 4 <= dealer <= 6 else "Hit"

    if total in (13, 14):                       # soft 13-14 (A+2, A+3)
        return ("Double" if dbl else "Hit") if 5 <= dealer <= 6 else "Hit"

    return "Hit"


# ── hard-hand strategy ────────────────────────────────────────

def _hard_strategy(total: int, dealer: int, hand_size: int) -> str:
    dbl = _can_double(hand_size)

    if total >= 17:
        return "Stand"

    if 13 <= total <= 16:                       # hard 13-16
        return "Stand" if 2 <= dealer <= 6 else "Hit"

    if total == 12:
        return "Stand" if 4 <= dealer <= 6 else "Hit"

    if total == 11:                             # European: no double vs 10 or A (lose to BJ)
        return ("Double" if dbl else "Hit") if 2 <= dealer <= 9 else "Hit"

    if total == 10:                             # double vs 2-9
        return ("Double" if dbl else "Hit") if 2 <= dealer <= 9 else "Hit"

    if total == 9:                              # double vs 3-6
        return ("Double" if dbl else "Hit") if 3 <= dealer <= 6 else "Hit"

    return "Hit"                                # hard 4-8


# ── public API ────────────────────────────────────────────────

def get_strategy(player_hand: list, dealer_upcard: int, split_in_progress: bool = False) -> str:
    """
    Return the basic-strategy action for the current hand.

    Args:
        player_hand:       list of Card dicts or ints (ranks 2-10, 11=Ace).
        dealer_upcard:     int (2-10, 11=Ace).
        split_in_progress: True when this hand was created by a split
                           (suppresses re-split and adjusts pair detection).

    Returns:
        One of: "Stand" | "Hit" | "Double" | "Split"
    """
    ranks = _ranks(player_hand)   # blackjack values, used for hand calculation

    # Pairs: use display_rank for equality so K+Q is NOT treated as a pair.
    # _pair_strategy receives the blackjack VALUE (ranks[0]) for its lookup table.
    if (
        len(player_hand) == 2
        and _split_rank(player_hand[0]) == _split_rank(player_hand[1])
        and not split_in_progress
    ):
        return _pair_strategy(ranks[0], dealer_upcard)

    total = calculate_hand(ranks)
    soft  = has_usable_ace(ranks)

    if soft:
        return _soft_strategy(total, dealer_upcard, len(ranks))

    return _hard_strategy(total, dealer_upcard, len(ranks))