import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class AdvancedBlackjackEnv(gym.Env):
    """
    Custom Blackjack Environment supporting:
    - Multiple decks (e.g., 8)
    - Variable deck penetration (e.g., 80% to 90%)
    - Actions: Stand, Hit, Double Down, Split
    - State: Includes remaining card composition for AI card counting
    """

    def __init__(self, num_decks=8, min_pen=0.8, max_pen=0.9):
        super(AdvancedBlackjackEnv, self).__init__()

        self.num_decks = num_decks
        self.min_pen = min_pen
        self.max_pen = max_pen
        self.shoe_counts = {}

        # Actions: 0=Stand, 1=Hit, 2=Double Down, 3=Split
        self.action_space = spaces.Discrete(4)

        # Step 3: State Space Normalization
        # Now all 13 variables are strictly bounded between 0.0 and 1.0
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float32)

        # Internal shoe management
        self.shoe = []
        self.penetration_cutoff = 0
        self.shoe_done = True 

        # Hand / split bookkeeping
        self.player_hand = []
        self.dealer_hand = []
        self.split_in_progress = False
        self.hands = None
        self.hand_bets = None
        self.current_hand_idx = 0

    def _shuffle_shoe(self):
        single_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
        self.shoe = single_deck * self.num_decks
        random.shuffle(self.shoe)

         # Initialize running counts once at shuffle time
        self.shoe_counts = {}
        for card in self.shoe:
            self.shoe_counts[card] = self.shoe_counts.get(card, 0) + 1


        total_cards = len(self.shoe)
        pen_percentage = random.uniform(self.min_pen, self.max_pen)
        self.penetration_cutoff = int(total_cards * (1.0 - pen_percentage))
        self.shoe_done = False

    def _count_remaining_ratios(self):
        max_of_rank = 4 * self.num_decks
        max_of_tens = 16 * self.num_decks
        ratios = [
            self.shoe_counts.get(2,  0) / max_of_rank,
            self.shoe_counts.get(3,  0) / max_of_rank,
            self.shoe_counts.get(4,  0) / max_of_rank,
            self.shoe_counts.get(5,  0) / max_of_rank,
            self.shoe_counts.get(6,  0) / max_of_rank,
            self.shoe_counts.get(7,  0) / max_of_rank,
            self.shoe_counts.get(8,  0) / max_of_rank,
            self.shoe_counts.get(9,  0) / max_of_rank,
            self.shoe_counts.get(10, 0) / max_of_tens,
            self.shoe_counts.get(11, 0) / max_of_rank,
        ]
        return ratios

    def _get_state(self):
        """Builds the 13-element state array, fully normalized."""
        player_sum = self._calculate_hand(self.player_hand)
        dealer_upcard = self.dealer_hand[0]
        usable_ace = 1.0 if self._has_usable_ace(self.player_hand) else 0.0
        ratios = self._count_remaining_ratios()
        
        # Step 3: Normalize the sums to keep neural network inputs between 0 and 1
        # player_sum_norm = player_sum / 30.0
        # dealer_upcard_norm = dealer_upcard / 11.0
        player_sum_norm = (player_sum - 4) / (21 - 4)
        dealer_upcard_norm = (dealer_upcard - 2) / (11 - 2)

        state = [player_sum_norm, dealer_upcard_norm, usable_ace] + ratios
        return np.array(state, dtype=np.float32)

    def _get_valid_actions(self):
        """Returns a boolean mask of valid actions [Stand, Hit, Double, Split]."""
        valid = [1, 1, 0, 0]
        if len(self.player_hand) == 2:
            valid[2] = 1
        
        # FIX: Prevent the agent from resplitting, which triggers the -100 penalty
        if len(self.player_hand) == 2 and self.player_hand[0] == self.player_hand[1] and not self.split_in_progress:
            valid[3] = 1
            
        return valid

    # Step 2 Helper: Required by sb3-contrib for ActionMasker wrapper
    def action_masks(self):
        return np.array(self._get_valid_actions(), dtype=bool)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if len(self.shoe) <= self.penetration_cutoff or self.shoe_done:
            self._shuffle_shoe()

        self.split_in_progress = False
        self.hands = None
        self.hand_bets = None
        self.current_hand_idx = 0

        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]

        info = {
            "valid_actions": self._get_valid_actions(),
            "shoe_done": len(self.shoe) <= self.penetration_cutoff
        }
        return self._get_state(), info
    
    
    def _resolve_split_hands(self):
        """
        Play out the dealer and score all split hands.
        Skips dealer draw entirely if all hands already busted.
        Returns total reward across all hands.
        """
        all_busted = all(self._calculate_hand(h) > 21 for h in self.hands)
        
        if all_busted:
            # No point drawing dealer cards — all hands lose regardless
            # Use 22 as a sentinel value meaning "dealer doesn't need to play"
            dealer_total = 22
        else:
            dealer_total = self._dealer_play()
        
        total_reward = 0.0
        for idx, h in enumerate(self.hands):
            r = self._compare_hand_against_dealer(h, dealer_total, is_split=True)
            total_reward += r * self.hand_bets[idx]
        
        return total_reward
    

    def step(self, action):
        valid_actions = self._get_valid_actions()
        
        # With Action Masking, the agent should never hit this block, 
        # but it's kept as a safe fallback just in case.
        if not valid_actions[action]:
            return self._get_state(), -100.0, True, False, {"valid_actions": valid_actions, 
                                                            "shoe_done": len(self.shoe) <= self.penetration_cutoff}

        terminated = False
        truncated = False
        reward = 0.0

        if self.split_in_progress:
            active_hand = self.hands[self.current_hand_idx]

            if action == 0:  # Stand
                # Player is done with this hand, move to next
                self.current_hand_idx += 1

                if self.current_hand_idx < len(self.hands):
                    # Still more hands to play
                    self.player_hand = self.hands[self.current_hand_idx]

                else:
                    # ✅ FIXED: was calling _dealer_play() unconditionally
                    # Now skips dealer if all hands busted (impossible after Stand
                    # since standing can't bust, but kept for consistency and safety)
                    reward = self._resolve_split_hands()
                    terminated = True

            elif action == 1:  # Hit
                active_hand.append(self._draw_card())

                if self._calculate_hand(active_hand) > 21:
                    # This hand busted, move to next
                    self.current_hand_idx += 1

                    if self.current_hand_idx < len(self.hands):
                        # Still more hands to play
                        self.player_hand = self.hands[self.current_hand_idx]

                    else:
                        # ✅ FIXED: this is the critical case
                        # Both hands could have busted via hitting — old code still
                        # called _dealer_play() and wasted shoe cards
                        reward = self._resolve_split_hands()
                        terminated = True

                else:
                    # Hand still alive, update reference
                    self.player_hand = active_hand

            elif action == 2:  # Double Down on split hand
                active_hand.append(self._draw_card())
                self.hand_bets[self.current_hand_idx] *= 2
                self.current_hand_idx += 1

                if self.current_hand_idx < len(self.hands):
                    # Still more hands to play
                    self.player_hand = self.hands[self.current_hand_idx]

                else:
                    # ✅ FIXED: if first hand busted on double and second hand
                    # busts on double, old code still played the dealer
                    reward = self._resolve_split_hands()
                    terminated = True

            elif action == 3:  # Re-split attempt — blocked by action masking
                return self._get_state(), -100.0, True, False, {"valid_actions": valid_actions}

            info = {
                "valid_actions": self._get_valid_actions(),
                "shoe_done": len(self.shoe) <= self.penetration_cutoff
            }
            return self._get_state(), float(reward), bool(terminated), bool(truncated), info

        # --- regular (non-split) flow ---
        if action == 0:  # Stand
            dealer_total = self._dealer_play()
            reward = self._compare_hand_against_dealer(self.player_hand, dealer_total, is_split=False)
            terminated = True

        elif action == 1:  # Hit
            self.player_hand.append(self._draw_card())
            if self._calculate_hand(self.player_hand) > 21:
                reward = -1.0
                terminated = True

        elif action == 2:  # Double Down
            self.player_hand.append(self._draw_card())
            if self._calculate_hand(self.player_hand) > 21:
                reward = -2.0
            else:
                dealer_total = self._dealer_play()
                base_reward = self._compare_hand_against_dealer(self.player_hand, dealer_total, is_split=False)
                reward = base_reward * 2.0
            terminated = True

        elif action == 3:  # Split
            card1, card2 = self.player_hand[0], self.player_hand[1]
            h1 = [card1, self._draw_card()]
            h2 = [card2, self._draw_card()]
            self.hands = [h1, h2]
            self.hand_bets = [1.0, 1.0]
            self.split_in_progress = True
            self.current_hand_idx = 0
            self.player_hand = self.hands[0]

        info = {"valid_actions": self._get_valid_actions(), "shoe_done": len(self.shoe) <= self.penetration_cutoff}
        return self._get_state(), float(reward), bool(terminated), bool(truncated), info

    def _draw_card(self):
        if not self.shoe:
            self._shuffle_shoe()
        card = self.shoe.pop()
        self.shoe_counts[card] -= 1   # O(1) update instead of O(n) rescan
        return card

    def _calculate_hand(self, hand):
        total = 0
        aces = 0
        for c in hand:
            if c == 11:
                aces += 1
                total += 11
            else:
                total += c
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total

    # def _has_usable_ace(self, hand):
    #     total = 0
    #     aces = 0
    #     for c in hand:
    #         if c == 11:
    #             aces += 1
    #             total += 11
    #         else:
    #             total += c
    #     return aces > 0 and total <= 21

    def _has_usable_ace(self, hand):
        total_low = sum(1 if c == 11 else c for c in hand)  # all aces = 1
        num_aces = sum(1 for c in hand if c == 11)
        return num_aces > 0 and total_low + 10 <= 21  # can we upgrade one ace to 11?


    def _dealer_play(self):
        while True:
            total = self._calculate_hand(self.dealer_hand)
            if total < 17:
                self.dealer_hand.append(self._draw_card())
                continue
            break
        return self._calculate_hand(self.dealer_hand)

    # Step 1: Implemented 3:2 Natural Blackjack Payout
    def _compare_hand_against_dealer(self, player_hand, dealer_total, is_split=False):
        """Compare hand to dealer_total. +1.5 for Natural, +1 win, -1 loss, 0 push."""
        player_total = self._calculate_hand(player_hand)
        
        # Check for Natural Blackjack (only on initial 2 cards, never on a split)
        if not is_split and player_total == 21 and len(player_hand) == 2:
            if dealer_total == 21 and len(self.dealer_hand) == 2:
                return 0.0  # Push
            return 1.5      # 3:2 Payout!

        if player_total > 21:
            return -1.0
        if dealer_total > 21:
            return 1.0
        if player_total > dealer_total:
            return 1.0
        if player_total == dealer_total:
            return 0.0
        return -1.0