import random
from dataclasses import dataclass
from advisor import calculate_hand, get_advice

# ── infinite shoe ─────────────────────────────────────────────
_RANKS   = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_WEIGHTS = [1, 1, 1, 1, 1, 1, 1, 1,  4,  1]

def _draw() -> int:
    return random.choices(_RANKS, weights=_WEIGHTS, k=1)[0]


# ── single round ──────────────────────────────────────────────
class BlackjackGame:
    def __init__(self, bet: float) -> None:
        self.base_bet          = bet
        self.player_hands:     list[list[int]] = [[_draw(), _draw()]]
        self.dealer_hand:      list[int]       = [_draw(), _draw()]
        self.bets:             list[float]     = [bet]
        self.current_hand_idx: int             = 0
        self.split_in_progress: bool           = False
        self.round_over:        bool           = False
        self.results:          list[str]       = []
        self.payouts:          list[float]     = []

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
            self.bets[self.current_hand_idx] *= 2
            self._advance()

        elif name == "split":
            c  = hand[0]
            h1 = [c, _draw()]
            h2 = [c, _draw()]
            self.player_hands[self.current_hand_idx] = h1
            self.player_hands.insert(self.current_hand_idx + 1, h2)
            self.bets.insert(self.current_hand_idx + 1, self.base_bet)
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

        for idx, hand in enumerate(self.player_hands):
            player_total = calculate_hand(hand)
            bet          = self.bets[idx]

            if (not self.split_in_progress
                    and len(self.player_hands) == 1
                    and len(hand) == 2
                    and player_total == 21):
                if dealer_total == 21 and len(self.dealer_hand) == 2:
                    self.results.append("push")
                    self.payouts.append(0.0)
                else:
                    self.results.append("blackjack")
                    self.payouts.append(round(bet * 1.5, 2))

            elif player_total > 21:
                self.results.append("bust")
                self.payouts.append(-bet)

            elif dealer_total > 21 or player_total > dealer_total:
                self.results.append("win")
                self.payouts.append(bet)

            elif player_total == dealer_total:
                self.results.append("push")
                self.payouts.append(0.0)

            else:
                self.results.append("loss")
                self.payouts.append(-bet)

        self.round_over = True

    def _valid_actions(self) -> list[str]:
        if self.round_over:
            return []
        hand    = self.player_hands[self.current_hand_idx]
        actions = ["hit", "stand"]
        if len(hand) == 2:
            actions.append("double")
        if len(hand) == 2 and hand[0] == hand[1] and not self.split_in_progress:
            actions.append("split")
        return actions

    def state(self) -> dict:
        if self.round_over:
            suggestion    = None
            valid_actions = []
        else:
            active_hand   = self.player_hands[self.current_hand_idx]
            suggestion    = get_advice(active_hand, self.dealer_hand[0], self.split_in_progress)
            valid_actions = self._valid_actions()

        return {
            "player_hands":      self.player_hands,
            "player_totals":     [calculate_hand(h) for h in self.player_hands],
            "current_hand_idx":  self.current_hand_idx if not self.round_over else None,
            "dealer_upcard":     self.dealer_hand[0],
            "dealer_hand":       self.dealer_hand if self.round_over else None,
            "dealer_total":      calculate_hand(self.dealer_hand) if self.round_over else None,
            "bets":              self.bets,
            "split_in_progress": self.split_in_progress,
            "valid_actions":     valid_actions,
            "suggestion":        suggestion,
            "round_over":        self.round_over,
            "results":           self.results  if self.round_over else None,
            "payouts":           self.payouts  if self.round_over else None,
            "net":               round(sum(self.payouts), 2) if self.round_over else None,
        }


# ── persistent player session ─────────────────────────────────
@dataclass
class RoundSummary:
    results: list[str]
    net:     float
    bet:     float
    hands:   list[list[int]]
    dealer:  list[int]


class PlayerSession:
    def __init__(self, starting_balance: float = 1000.0) -> None:
        self.balance:       float                    = starting_balance
        self.history:       list[RoundSummary]       = []
        self.current_round: BlackjackGame | None     = None

    # ── public API ────────────────────────────────────────────

    def start_round(self, bet: float) -> dict:
        if bet <= 0:
            raise ValueError("Bet must be positive.")
        if bet > self.balance:
            raise ValueError(f"Bet {bet} exceeds balance {self.balance}.")
        if self.current_round and not self.current_round.round_over:
            raise ValueError("Finish the current round before starting a new one.")

        self.balance       = round(self.balance - bet, 2)
        self.current_round = BlackjackGame(bet=bet)
        return self._response()

    def action(self, action_name: str) -> dict:
        if self.current_round is None:
            raise ValueError("No active round. Start a round first.")

        self.current_round.action(action_name)

        # Settle balance and record history when round ends
        if self.current_round.round_over:
            net          = sum(self.current_round.payouts)
            # Return the original bet + winnings (bet was deducted on start_round)
            self.balance = round(self.balance + self.current_round.base_bet + net, 2)
            self.history.append(RoundSummary(
                results = self.current_round.results,
                net     = round(net, 2),
                bet     = self.current_round.base_bet,
                hands   = [list(h) for h in self.current_round.player_hands],
                dealer  = list(self.current_round.dealer_hand),
            ))

        return self._response()

    def get_state(self) -> dict:
        return self._response()

    # ── internals ─────────────────────────────────────────────

    def _response(self) -> dict:
        return {
            "balance":       self.balance,
            "rounds_played": len(self.history),
            "history":       [
                {
                    "results": s.results,
                    "net":     s.net,
                    "bet":     s.bet,
                    "hands":   s.hands,
                    "dealer":  s.dealer,
                }
                for s in self.history[-10:]     # send only last 10 rounds
            ],
            "round": self.current_round.state() if self.current_round else None,
        }
