import random
from dataclasses import dataclass
from typing import TypedDict
from advisor import calculate_hand, get_advice

# ── card type ─────────────────────────────────────────────────
class Card(TypedDict):
    rank:         int   # blackjack value: 2-10, 11 (Ace)
    display_rank: int   # 1 (Ace), 2-10, 11 (J), 12 (Q), 13 (K)
    suit:         str

# ── infinite shoe ─────────────────────────────────────────────
_RANKS   = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_WEIGHTS = [1, 1, 1, 1, 1, 1, 1, 1,  4,  1]
_SUITS   = ["♠", "♥", "♦", "♣"]

# 10-value cards: 10, J=11, Q=12, K=13 — each equally likely
_TEN_DISPLAY_RANKS = [10, 11, 12, 13]

def _draw() -> Card:
    rank = random.choices(_RANKS, weights=_WEIGHTS, k=1)[0]
    suit = random.choice(_SUITS)

    if rank == 11:          # Ace
        display_rank = 1
    elif rank == 10:        # 10 / J / Q / K
        display_rank = random.choice(_TEN_DISPLAY_RANKS)
    else:
        display_rank = rank

    return {"rank": rank, "display_rank": display_rank, "suit": suit}


# ── single round ──────────────────────────────────────────────
class BlackjackGame:
    def __init__(self) -> None:
        self.player_hands:      list[list[Card]] = [[_draw(), _draw()]]
        self.dealer_hand:       list[Card]       = [_draw(), _draw()]
        self.current_hand_idx:  int              = 0
        self.split_in_progress: bool             = False
        self.round_over:        bool             = False
        self.results:           list[str]        = []

    def action(self, action_name: str) -> None:
        if self.round_over:
            raise ValueError("Round is already over. Start a new round.")

        name  = action_name.lower()
        valid = self._valid_actions()
        if name not in valid:
            raise ValueError(f"'{name}' is not valid here. Valid: {valid}")

        hand = self.player_hands[self.current_hand_idx]

        if name == "hit":
            hand.append(_draw())
            if calculate_hand(hand) > 21:
                self._advance()

        elif name == "stand":
            self._advance()

        elif name == "double":
            hand.append(_draw())
            self._advance()

        elif name == "split":
            c  = hand[0]["rank"]
            h1 = [{"rank": c, "display_rank": hand[0]["display_rank"], "suit": hand[0]["suit"]}, _draw()]
            h2 = [{"rank": c, "display_rank": hand[1]["display_rank"], "suit": hand[1]["suit"]}, _draw()]
            self.player_hands[self.current_hand_idx] = h1
            self.player_hands.insert(self.current_hand_idx + 1, h2)
            self.split_in_progress = True

    def _advance(self) -> None:
        self.current_hand_idx += 1
        if self.current_hand_idx >= len(self.player_hands):
            self._resolve()

    def _resolve(self) -> None:
        all_bust = all(calculate_hand(h) > 21 for h in self.player_hands)
        if not all_bust:
            while calculate_hand(self.dealer_hand) < 17:
                self.dealer_hand.append(_draw())

        dealer_total = calculate_hand(self.dealer_hand)

        for hand in self.player_hands:
            player_total = calculate_hand(hand)

            if (not self.split_in_progress
                    and len(self.player_hands) == 1
                    and len(hand) == 2
                    and player_total == 21):
                if dealer_total == 21 and len(self.dealer_hand) == 2:
                    self.results.append("push")
                else:
                    self.results.append("blackjack")

            elif player_total > 21:
                self.results.append("bust")

            elif dealer_total > 21 or player_total > dealer_total:
                self.results.append("win")

            elif player_total == dealer_total:
                self.results.append("push")

            else:
                self.results.append("loss")

        self.round_over = True

    def _valid_actions(self) -> list[str]:
        if self.round_over:
            return []
        hand    = self.player_hands[self.current_hand_idx]
        actions = ["hit", "stand"]
        if len(hand) == 2:
            actions.append("double")
        if len(hand) == 2 and hand[0]["rank"] == hand[1]["rank"] and not self.split_in_progress:
            actions.append("split")
        return actions

    def state(self) -> dict:
        if self.round_over:
            suggestion    = None
            valid_actions = []
        else:
            active_hand   = self.player_hands[self.current_hand_idx]
            active_ranks  = [card["rank"] for card in active_hand]
            dealer_upcard = self.dealer_hand[0]["rank"]
            suggestion    = get_advice(active_ranks, dealer_upcard, self.split_in_progress)
            valid_actions = self._valid_actions()

        return {
            "player_hands":      self.player_hands,
            "player_totals":     [calculate_hand(h) for h in self.player_hands],
            "current_hand_idx":  self.current_hand_idx if not self.round_over else None,
            "dealer_upcard":     self.dealer_hand[0],
            "dealer_hand":       self.dealer_hand if self.round_over else None,
            "dealer_total":      calculate_hand(self.dealer_hand) if self.round_over else None,
            "split_in_progress": self.split_in_progress,
            "valid_actions":     valid_actions,
            "suggestion":        suggestion,
            "round_over":        self.round_over,
            "results":           self.results if self.round_over else None,
        }


# ── persistent player session ─────────────────────────────────
@dataclass
class RoundSummary:
    results: list[str]
    hands:   list[list[Card]]
    dealer:  list[Card]


class PlayerSession:
    def __init__(self) -> None:
        self.history:       list[RoundSummary]   = []
        self.current_round: BlackjackGame | None = None

    # ── public API ────────────────────────────────────────────

    def start_round(self) -> dict:
        if self.current_round and not self.current_round.round_over:
            raise ValueError("Finish the current round before starting a new one.")

        self.current_round = BlackjackGame()
        return self._response()

    def action(self, action_name: str) -> dict:
        if self.current_round is None:
            raise ValueError("No active round. Start a round first.")

        self.current_round.action(action_name)

        if self.current_round.round_over:
            self.history.append(RoundSummary(
                results = self.current_round.results,
                hands   = [list(h) for h in self.current_round.player_hands],
                dealer  = list(self.current_round.dealer_hand),
            ))

        return self._response()

    def get_state(self) -> dict:
        return self._response()

    def get_history(self) -> dict:
        return {
            "rounds_played": len(self.history),
            "history": [
                {
                    "results": summary.results,
                    "hands": summary.hands,
                    "dealer": summary.dealer,
                }
                for summary in self.history
            ]
        }

    # ── internals ─────────────────────────────────────────────

    def _response(self) -> dict:
        return {
            "rounds_played": len(self.history),
            "round": self.current_round.state() if self.current_round else None,
        }