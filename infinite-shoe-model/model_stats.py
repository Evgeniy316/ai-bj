"""
collect_stats.py
────────────────
Runs a trained MaskablePPO blackjack model for N games and collects
per-strategy action statistics.

Usage:
    python collect_stats.py <model_path> [n_games] [output_csv]

    model_path  : path to the saved .zip (without extension is fine)
    n_games     : number of full episodes  (default: 500_000)
    output_csv  : output file name         (default: blackjack_stats.csv)

Example:
    python collect_stats.py blackjack_infinite_shoe_model_test_run 500000
"""

import sys
import csv
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from model import AdvancedBlackjackEnv


# ── constants ─────────────────────────────────────────────────────────────────

ACTION_NAMES = {0: "stand", 1: "hit", 2: "double", 3: "split"}

STRATEGIES   = ("pair", "soft", "hard")


# ── environment factory ───────────────────────────────────────────────────────

def _make_env() -> ActionMasker:
    env = AdvancedBlackjackEnv()
    return ActionMasker(env, lambda e: e.action_masks())


# ── hand classifier ───────────────────────────────────────────────────────────

def _classify(inner: AdvancedBlackjackEnv) -> str:
    """
    Classify the current player hand as 'pair', 'soft', or 'hard'.

    Pair  : exactly 2 cards, same rank_id, no split already in progress.
            (rank_id check mirrors model.py — K+Q is NOT a pair even though
            both have value 10.)
    Soft  : hand contains a usable ace.
    Hard  : everything else.
    """
    hand = inner.player_hand

    if (
        len(hand) == 2
        and hand[0][0] == hand[1][0]      # same rank_id (not just same value)
        and not inner.split_in_progress
    ):
        return "pair"

    if inner._has_usable_ace(hand):
        return "soft"

    return "hard"


# ── main collection loop ──────────────────────────────────────────────────────

def collect(model_path: str, n_games: int) -> dict:
    """
    Run `n_games` episodes and return a nested stats dict:

        stats[strategy][(player_sum, dealer_upcard)][action_name] = count

    where strategy     ∈ {"pair", "soft", "hard"}
          player_sum   is the integer hand total (4-21)
          dealer_upcard is the dealer's visible card value (2-11)
          action_name  ∈ {"stand", "hit", "double", "split"}
    """
    # Build stats structure
    def _action_counter():
        return {"stand": 0, "hit": 0, "double": 0, "split": 0}

    stats = {s: defaultdict(_action_counter) for s in STRATEGIES}

    # Environment + model
    env   = _make_env()
    model = MaskablePPO.load(model_path, env=env)
    inner: AdvancedBlackjackEnv = env.unwrapped   # direct access to game state

    obs, _ = env.reset()

    log_every = max(1, n_games // 20)   # progress every 5 %

    for game_idx in range(1, n_games + 1):

        obs, _ = env.reset()
        done   = False

        while not done:
            # ── classify BEFORE the action is taken ──────────────────────
            strategy      = _classify(inner)
            player_sum    = inner._calculate_hand(inner.player_hand)
            dealer_upcard = inner.dealer_hand[0][1]   # (rank_id, value)[1]

            # ── model picks action ────────────────────────────────────────
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=env.action_masks(),
            )
            action = int(action)

            # ── record ───────────────────────────────────────────────────
            stats[strategy][(player_sum, dealer_upcard)][ACTION_NAMES[action]] += 1

            # ── step ─────────────────────────────────────────────────────
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if game_idx % log_every == 0:
            pct = 100 * game_idx / n_games
            print(f"  {game_idx:>9,} / {n_games:,}  ({pct:.0f}%)")

    env.close()
    return stats


# ── CSV writer ────────────────────────────────────────────────────────────────

def save_csv(stats: dict, path: str) -> None:
    """
    Write one row per (strategy, player_sum, dealer_upcard) combination.

    Columns: strategy | player_sum | dealer_upcard | stand | hit | double | split | total
    """
    fieldnames = ["strategy", "player_sum", "dealer_upcard", "stand", "hit", "double", "split", "total"]

    rows = []
    for strategy in STRATEGIES:
        for (player_sum, dealer_upcard) in sorted(stats[strategy].keys()):
            counts = stats[strategy][(player_sum, dealer_upcard)]
            total  = sum(counts.values())
            rows.append({
                "strategy"    : strategy,
                "player_sum"  : player_sum,
                "dealer_upcard": dealer_upcard,
                "stand"       : counts["stand"],
                "hit"         : counts["hit"],
                "double"      : counts["double"],
                "split"       : counts["split"],
                "total"       : total,
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nStats written to: {path}")
    print(f"Rows: {len(rows)}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "logs/checkpoints/blackjack_model_28000000_steps.zip"
    n_games    = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    out_csv    = sys.argv[3] if len(sys.argv) > 3 else "blackjack_stats.csv"

    print(f"Model  : {model_path}")
    print(f"Games  : {n_games:,}")
    print(f"Output : {out_csv}")
    print()

    stats = collect(model_path, n_games)
    save_csv(stats, out_csv)

    # ── quick sanity summary ──────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────")
    for strategy in STRATEGIES:
        sums        = stats[strategy]
        total_decs  = sum(sum(v.values()) for v in sums.values())
        n_situations = len(sums)
        print(f"  {strategy:4s}  {n_situations:3d} unique (sum, dealer) pairs  |  {total_decs:>12,} total decisions")
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()