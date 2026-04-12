import numpy as np
from multiprocessing import Pool
from sb3_contrib import MaskablePPO
from model import AdvancedBlackjackEnv
from tqdm import tqdm


# ─────────────────────────────────────────────
#  CONFIG  (edit these)
# ─────────────────────────────────────────────
MODEL_PATH       = "blackjack_infinite_shoe_model.zip"
NUM_RUNS         = 5         # concurrent runs (one run = one core)
EPISODES_PER_RUN = 100_000    # episodes each core plays
# ─────────────────────────────────────────────


def calculate_hand_value(hand):
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


def safe_int_action(action):
    if isinstance(action, np.ndarray):
        return int(action.item())
    try:
        return int(action)
    except Exception:
        return int(np.squeeze(action).item())


# ─────────────────────────────────────────────
#  Worker – one full run on one core
# ─────────────────────────────────────────────
def run_episodes(args):
    model_path, num_episodes, run_id = args

    env   = AdvancedBlackjackEnv()
    model = MaskablePPO.load(model_path)

    action_map = {0: "Stand", 1: "Hit", 2: "Double", 3: "Split"}

    stats = dict(
        run_id              = run_id,
        count               = 0,
        total_reward        = 0.0,
        wins                = 0,
        natural_bj          = 0,
        losses              = 0,
        pushes              = 0,
        total_splits        = 0,
        ace_splits          = 0,
        non_ace_split_pairs = {},
    )

    for _ in tqdm(range(num_episodes),
                  desc=f"  run {run_id + 1}",
                  position=run_id,
                  leave=True):

        obs, _       = env.reset()
        done         = False
        initial_hand = list(env.player_hand)

        while not done:
            masks  = env.action_masks()
            masks  = masks[0] if masks.ndim == 2 else masks
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = safe_int_action(action)

            if action_map.get(action) == "Split":
                stats["total_splits"] += 1
                if len(initial_hand) == 2 and initial_hand[0] == initial_hand[1]:
                    if initial_hand[0] == 11:
                        stats["ace_splits"] += 1
                    else:
                        key = f"{initial_hand[0]}-{initial_hand[1]}"
                        stats["non_ace_split_pairs"][key] = (
                            stats["non_ace_split_pairs"].get(key, 0) + 1
                        )

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        stats["count"]        += 1
        stats["total_reward"] += float(reward)
        if   reward == 1.5: stats["natural_bj"] += 1
        elif reward  >  0:  stats["wins"]       += 1
        elif reward  <  0:  stats["losses"]     += 1
        else:               stats["pushes"]     += 1

    return stats


# ─────────────────────────────────────────────
#  Merge all runs into one combined view
# ─────────────────────────────────────────────
def merge_stats(run_list):
    merged = dict(
        count=0, total_reward=0.0,
        wins=0, natural_bj=0, losses=0, pushes=0,
        total_splits=0, ace_splits=0, non_ace_split_pairs={},
    )
    for s in run_list:
        merged["count"]        += s["count"]
        merged["total_reward"] += s["total_reward"]
        merged["wins"]         += s["wins"]
        merged["natural_bj"]   += s["natural_bj"]
        merged["losses"]       += s["losses"]
        merged["pushes"]       += s["pushes"]
        merged["total_splits"] += s["total_splits"]
        merged["ace_splits"]   += s["ace_splits"]
        for k, v in s["non_ace_split_pairs"].items():
            merged["non_ace_split_pairs"][k] = (
                merged["non_ace_split_pairs"].get(k, 0) + v
            )
    return merged


# ─────────────────────────────────────────────
#  Pretty printer
# ─────────────────────────────────────────────
def print_report(stats, label=""):
    N              = stats["count"]
    avg_reward     = stats["total_reward"] / N
    wins           = stats["wins"]
    natural_bj     = stats["natural_bj"]
    losses         = stats["losses"]
    pushes         = stats["pushes"]
    total_splits   = stats["total_splits"]
    ace_splits     = stats["ace_splits"]
    non_ace_splits = total_splits - ace_splits
    pairs          = stats["non_ace_split_pairs"]

    header = f"BLACKJACK MODEL PERFORMANCE ANALYSIS"
    if label:
        header += f"  [{label}]"

    print()
    print("=" * 60)
    print(header)
    print("=" * 60)
    print(f"Episodes analyzed          : {N:,}")
    print(f"Average reward per hand    : {avg_reward:+.5f}  ({avg_reward*100:+.3f}% house edge)")
    print(f"Player advantage hands     : {(wins+natural_bj)/N*100:5.2f}%  (Win + Natural BJ)")
    print(f"Regular win rate           : {wins/N*100:5.2f}%")
    print(f"Natural Blackjack rate     : {natural_bj/N*100:5.2f}%")
    print(f"Loss rate                  : {losses/N*100:5.2f}%")
    print(f"Push rate                  : {pushes/N*100:5.2f}%")
    print("-" * 60)
    print(f"\nSPLIT BEHAVIOR")
    print(f"Total splits performed     : {total_splits} ({total_splits/N*100:.2f}% of all hands)")

    if total_splits > 0:
        print(f"→ Aces only                : {ace_splits} ({ace_splits/total_splits*100:.1f}% of splits)")
        print(f"→ Non-ace pairs            : {non_ace_splits} ({non_ace_splits/total_splits*100:.1f}% of splits)")
    else:
        print("→ No splits detected")

    if pairs:
        print("\nNON-ACE PAIRS THE MODEL SPLITS (sorted by frequency):")
        for pair, count in sorted(pairs.items(), key=lambda x: x[1], reverse=True):
            print(f"   {pair:8} → {count:3d} times")
    else:
        print("\n   (No non-ace splits detected)")

    print("=" * 60)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    print(f"Launching {NUM_RUNS} concurrent runs "
          f"({EPISODES_PER_RUN:,} episodes each) ...")

    worker_args = [
        (MODEL_PATH, EPISODES_PER_RUN, run_id)
        for run_id in range(NUM_RUNS)
    ]

    # NUM_RUNS workers run fully in parallel; results arrive when ALL finish
    with Pool(processes=NUM_RUNS) as pool:
        all_stats = pool.map(run_episodes, worker_args)

    # ── Print individual run results ──────────────────────────
    for stats in all_stats:
        print_report(stats, label=f"Run {stats['run_id'] + 1}")

    # ── Print combined summary ────────────────────────────────
    if NUM_RUNS > 1:
        combined = merge_stats(all_stats)
        print_report(combined, label=f"All {NUM_RUNS} runs combined  —  {combined['count']:,} episodes total")


if __name__ == "__main__":
    main()