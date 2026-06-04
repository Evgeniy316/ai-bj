import csv
import pandas as pd
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from model import AdvancedBlackjackEnv
from tqdm import tqdm


def safe_int_action(action):
    """Convert action (possibly np.ndarray) to plain int safely."""
    if isinstance(action, np.ndarray):
        return int(action.item())
    try:
        return int(action)
    except Exception:
        return int(np.squeeze(action).item())


def _card_value(card) -> int:
    """Extract blackjack value from a (rank_id, value) tuple or a plain int."""
    return card[1] if isinstance(card, tuple) else card


def calculate_hand_value(hand) -> int:
    """Calculate hand value. Accepts lists of (rank_id, value) tuples or plain ints."""
    total, aces = 0, 0
    for card in hand:
        value = _card_value(card)
        if value == 11:
            aces += 1
            total += 11
        else:
            total += value
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total


def _hand_str(hand) -> str:
    """Human-readable hand string showing only card values."""
    return str([_card_value(c) for c in hand])


def test_and_save_logs(model_path, num_episodes=50000, output_file="blackjack_logs.csv"):
    env = AdvancedBlackjackEnv()
    model = MaskablePPO.load(model_path)

    print(f"Testing {num_episodes} hands and saving to {output_file}...")

    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Episode",
            "Player_Initial_Hand",
            "Player_Initial_Total",
            "Player_Total",
            "Dealer_Upcard",
            "Usable_Ace",
            "Card1_Norm",
            "Card2_Norm",
            "Hand_Size_Norm",
            "Action_Taken",
            "Reward",
            "Result",
            "Dealer_Full_Hand",
            "Dealer_Total",
            "Dealer_Hand_Size",
            "Player_Final_Hand(s)",
        ])

        for ep in tqdm(range(num_episodes)):
            obs, info = env.reset()
            done = False

            initial_hand_str = _hand_str(env.player_hand)
            initial_total = calculate_hand_value(env.player_hand)

            dealer_card = round(obs[1] * 9) + 2
            usable_ace = int(obs[2])

            card1_norm = float(obs[3])
            card2_norm = float(obs[4])
            hand_size_norm = float(obs[5])

            episode_actions = []

            while not done:
                action_masks = env.action_masks()
                action_masks_to_pass = action_masks[0] if action_masks.ndim == 2 else action_masks

                action, _ = model.predict(obs, action_masks=action_masks_to_pass, deterministic=True)
                action = safe_int_action(action)

                action_map = {0: "Stand", 1: "Hit", 2: "Double", 3: "Split"}
                episode_actions.append(action_map.get(action, f"Action_{action}"))

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            if env.split_in_progress:
                final_hands_str = " | ".join(_hand_str(h) for h in env.hands)
                final_totals = [calculate_hand_value(h) for h in env.hands]
                player_total_str = " | ".join(str(t) for t in final_totals)
            else:
                final_hands_str = _hand_str(env.player_hand)
                player_total_str = str(calculate_hand_value(env.player_hand))

            dealer_full_hand = env.dealer_hand.copy()
            dealer_total = calculate_hand_value(dealer_full_hand)
            dealer_hand_size = len(dealer_full_hand)
            dealer_hand_str = _hand_str(dealer_full_hand)

            if reward == 1.5:
                result = "Natural BJ"
            elif reward > 0:
                result = "Win"
            elif reward < 0:
                result = "Loss"
            else:
                result = "Push"

            writer.writerow([
                ep,
                initial_hand_str,
                initial_total,
                player_total_str,
                dealer_card,
                usable_ace,
                card1_norm,
                card2_norm,
                hand_size_norm,
                " -> ".join(episode_actions),
                reward,
                result,
                dealer_hand_str,
                dealer_total,
                dealer_hand_size,
                final_hands_str,
            ])


if __name__ == "__main__":
    # test_and_save_logs("blackjack_infinite_shoe_model.zip")
    test_and_save_logs("../game/blackjack_model_22400000_steps.zip")