import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


# ---------------------------------------------------------------------------
# Card distribution for an infinite shoe.
# Ranks: 2-9 (one denomination each = 1/13), 10-value (4 denominations = 4/13),
# Ace (one denomination = 1/13).
# ---------------------------------------------------------------------------
_RANKS       = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_WEIGHTS     = [1, 1, 1, 1, 1, 1, 1, 1,  4,  1]   # 10-value cards are 4× as likely
_TOTAL_WEIGHT = sum(_WEIGHTS)
_PROBS       = [w / _TOTAL_WEIGHT for w in _WEIGHTS]


def _draw_infinite() -> int:
    """Draw one card from an infinite shoe (uniform over 13 denominations)."""
    return random.choices(_RANKS, weights=_WEIGHTS, k=1)[0]


class AdvancedBlackjackEnv(gym.Env):
    """
    Blackjack environment with an **infinite shoe**.

    Because every draw is i.i.d. from a fixed distribution, card-counting
    features carry no information and are omitted.

    State (3 elements, all in [0, 1]):
      [0]  player_sum_norm    - normalised player total        (4-21 → 0-1)
      [1]  dealer_upcard_norm - normalised dealer up-card      (2-11 → 0-1)
      [2]  usable_ace         - 1.0 if player has a usable ace, else 0.0

    Actions: 0=Stand  1=Hit  2=Double Down  3=Split
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.action_space      = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Hand / split bookkeeping
        self.player_hand: list[int] = []
        self.dealer_hand: list[int] = []

        self.split_in_progress  = False
        self.hands: list | None  = None
        self.hand_bets: list | None = None
        self.current_hand_idx   = 0

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        player_sum    = self._calculate_hand(self.player_hand)
        dealer_upcard = self.dealer_hand[0]
        usable_ace    = 1.0 if self._has_usable_ace(self.player_hand) else 0.0

        player_sum_norm    = (player_sum    - 4)  / (21 - 4)
        dealer_upcard_norm = (dealer_upcard - 2)  / (11 - 2)

        if len(self.player_hand) == 2:
            card1_norm = (self.player_hand[0] - 2) / (11 - 2)
            card2_norm = (self.player_hand[1] - 2) / (11 - 2)
        else:
            card1_norm = 0.0
            card2_norm = 0.


        hand_size_norm = 0.0 if len(self.player_hand) == 2 else 1.0

        return np.array(
            [player_sum_norm, dealer_upcard_norm, usable_ace,
            card1_norm, card2_norm, hand_size_norm],
            dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------

    def _get_valid_actions(self) -> list[int]:
        """Boolean mask for [Stand, Hit, Double, Split]."""
        valid = [1, 1, 0, 0]
        if len(self.player_hand) == 2:
            valid[2] = 1                                      # Double Down
        if (
            len(self.player_hand) == 2
            and self.player_hand[0] == self.player_hand[1]
            and not self.split_in_progress
        ):
            valid[3] = 1                                      # Split (no re-split)
        return valid

    def action_masks(self) -> np.ndarray:
        """Required by sb3-contrib ActionMasker wrapper."""
        return np.array(self._get_valid_actions(), dtype=bool)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.split_in_progress = False
        self.hands             = None
        self.hand_bets         = None
        self.current_hand_idx  = 0

        self.player_hand = [_draw_infinite(), _draw_infinite()]
        dealer_upcard    = _draw_infinite()
        dealer_hole      = _draw_infinite()
        self.dealer_hand = [dealer_upcard, dealer_hole]

        info = {"valid_actions": self._get_valid_actions()}
        return self._get_state(), info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int):
        valid_actions = self._get_valid_actions()

        # Safety fallback — action masking should prevent this
        if not valid_actions[action]:
            return (
                self._get_state(),
                -100.0,
                True,
                False,
                {"valid_actions": valid_actions},
            )

        terminated = truncated = False
        reward     = 0.0

        # ----------------------------------------------------------------
        # Split-hand flow
        # ----------------------------------------------------------------
        if self.split_in_progress:
            active_hand = self.hands[self.current_hand_idx]

            if action == 0:                                   # Stand
                self.current_hand_idx += 1
                if self.current_hand_idx < len(self.hands):
                    self.player_hand = self.hands[self.current_hand_idx]
                else:
                    reward     = self._resolve_split_hands()
                    terminated = True

            elif action == 1:                                 # Hit
                active_hand.append(_draw_infinite())
                if self._calculate_hand(active_hand) > 21:
                    self.current_hand_idx += 1
                    if self.current_hand_idx < len(self.hands):
                        self.player_hand = self.hands[self.current_hand_idx]
                    else:
                        reward     = self._resolve_split_hands()
                        terminated = True
                else:
                    self.player_hand = active_hand

            elif action == 2:                                 # Double Down on split
                active_hand.append(_draw_infinite())
                self.hand_bets[self.current_hand_idx] *= 2
                self.current_hand_idx += 1
                if self.current_hand_idx < len(self.hands):
                    self.player_hand = self.hands[self.current_hand_idx]
                else:
                    reward     = self._resolve_split_hands()
                    terminated = True

            elif action == 3:                                 # Re-split — blocked
                return (
                    self._get_state(), -100.0, True, False,
                    {"valid_actions": valid_actions},
                )

            return (
                self._get_state(),
                float(reward),
                bool(terminated),
                bool(truncated),
                {"valid_actions": self._get_valid_actions()},
            )

        # ----------------------------------------------------------------
        # Regular (non-split) flow
        # ----------------------------------------------------------------
        if action == 0:                                       # Stand
            dealer_total = self._dealer_play()
            reward       = self._compare_hand_against_dealer(
                self.player_hand, dealer_total, is_split=False
            )
            terminated = True

        elif action == 1:                                     # Hit
            self.player_hand.append(_draw_infinite())
            if self._calculate_hand(self.player_hand) > 21:
                reward     = -1.0
                terminated = True

        elif action == 2:                                     # Double Down
            self.player_hand.append(_draw_infinite())
            if self._calculate_hand(self.player_hand) > 21:
                reward = -2.0
            else:
                dealer_total = self._dealer_play()
                reward = self._compare_hand_against_dealer(
                    self.player_hand, dealer_total, is_split=False
                ) * 2.0
            terminated = True

        elif action == 3:                                     # Split
            c1, c2       = self.player_hand
            h1           = [c1, _draw_infinite()]
            h2           = [c2, _draw_infinite()]
            self.hands   = [h1, h2]
            self.hand_bets          = [1.0, 1.0]
            self.split_in_progress  = True
            self.current_hand_idx   = 0
            self.player_hand        = self.hands[0]

        return (
            self._get_state(),
            float(reward),
            bool(terminated),
            bool(truncated),
            {"valid_actions": self._get_valid_actions()},
        )

    # ------------------------------------------------------------------
    # Split resolution
    # ------------------------------------------------------------------

    def _resolve_split_hands(self) -> float:
        all_busted   = all(self._calculate_hand(h) > 21 for h in self.hands)
        dealer_total = 22 if all_busted else self._dealer_play()

        total_reward = 0.0
        for idx, h in enumerate(self.hands):
            r             = self._compare_hand_against_dealer(h, dealer_total, is_split=True)
            total_reward += r * self.hand_bets[idx]
        return total_reward

    # ------------------------------------------------------------------
    # Hand helpers
    # ------------------------------------------------------------------

    def _calculate_hand(self, hand: list[int]) -> int:
        total = aces = 0
        for c in hand:
            if c == 11:
                aces  += 1
                total += 11
            else:
                total += c
        while total > 21 and aces > 0:
            total -= 10
            aces  -= 1
        return total

    def _has_usable_ace(self, hand: list[int]) -> bool:
        total_low = sum(1 if c == 11 else c for c in hand)
        num_aces  = sum(1 for c in hand if c == 11)
        return num_aces > 0 and total_low + 10 <= 21

    def _dealer_play(self) -> int:
        """Dealer hits on soft 16 and below, stands on hard/soft 17+."""
        while self._calculate_hand(self.dealer_hand) < 17:
            self.dealer_hand.append(_draw_infinite())
        return self._calculate_hand(self.dealer_hand)

    def _compare_hand_against_dealer(
        self,
        player_hand: list[int],
        dealer_total: int,
        is_split: bool = False,
    ) -> float:
        """Returns reward: +1.5 natural BJ, +1 win, 0 push, -1 loss."""
        player_total = self._calculate_hand(player_hand)

        if not is_split and player_total == 21 and len(player_hand) == 2:
            if dealer_total == 21 and len(self.dealer_hand) == 2:
                return 0.0   # Both natural — push
            return 1.5       # 3:2 payout

        if player_total > 21:   return -1.0
        if dealer_total > 21:   return  1.0
        if player_total > dealer_total: return  1.0
        if player_total == dealer_total: return  0.0
        return -1.0
