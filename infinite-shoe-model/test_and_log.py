import csv
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

def calculate_hand_value(hand):
    """Calculate the value of a blackjack hand."""
    total = 0
    aces = 0
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


def test_and_save_logs(model_path, num_episodes=50000, output_file="blackjack_logs.csv"):
    env = AdvancedBlackjackEnv()
    model = MaskablePPO.load(model_path)

    print(f"Testing {num_episodes} hands and saving to {output_file}...")

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Episode", "Player_Initial_Hand", "Player_Initial_Total",
            "Player_Total", "Dealer_Upcard", "Usable_Ace",
            "Action_Taken", "Reward", "Result",
            "Dealer_Full_Hand", "Dealer_Total", "Dealer_Hand_Size",
            "Player_Final_Hand(s)",
            "Count_2", "Count_3", "Count_4", "Count_5", "Count_6",
            "Count_7", "Count_8", "Count_9", "Count_10", "Count_A"
        ])

        for ep in tqdm(range(num_episodes)):
            obs, info = env.reset()
            done = False

            # ---- Capture initial hand ----
            initial_hand_str = str(env.player_hand)
            initial_total = calculate_hand_value(env.player_hand)
            # ------------------------------

            # For logging, we'll compute the correct initial total
            # (this column will be overwritten with final total later)
            dealer_card = round(obs[1] * 11)
            usable_ace = int(obs[2])
            counts = obs[3:13]

            episode_actions = []

            # ---- MAIN GAME LOOP (plays entire hand, including splits) ----
            while not done:
                action_masks = env.action_masks()
                if action_masks.ndim == 2:
                    action_masks_to_pass = action_masks[0]
                else:
                    action_masks_to_pass = action_masks

                action, _ = model.predict(obs, action_masks=action_masks_to_pass, deterministic=True)
                action = safe_int_action(action)

                action_map = {0: "Stand", 1: "Hit", 2: "Double", 3: "Split"}
                action_str = action_map.get(action, f"Action_{action}")
                episode_actions.append(action_str)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            # ----------------------------------------------------------------

            # ---- After episode finishes, capture final hand(s) and total ----
            if env.split_in_progress:
                # Multiple split hands
                final_hands = env.hands
                final_hands_str = " | ".join(str(hand) for hand in final_hands)
                # For Player_Total, show total of each hand separated by " | "
                final_totals = [calculate_hand_value(h) for h in final_hands]
                player_total_str = " | ".join(str(t) for t in final_totals)
            else:
                final_hands_str = str(env.player_hand)
                player_total_str = str(calculate_hand_value(env.player_hand))
            # -----------------------------------------------------------------

            # Dealer info
            dealer_full_hand = env.dealer_hand.copy()
            dealer_total = calculate_hand_value(dealer_full_hand)
            dealer_hand_size = len(dealer_full_hand)
            dealer_hand_str = str(dealer_full_hand)

            # Determine result string
            if reward == 1.5:
                result = "Natural BJ"
            elif reward > 0:
                result = "Win"
            elif reward < 0:
                result = "Loss"
            else:
                result = "Push"

            actions_taken = " -> ".join(episode_actions)
            counts_list = [float(x) for x in counts]

            # Write row
            writer.writerow([
                ep,
                initial_hand_str,
                initial_total,
                player_total_str,          # now final total(s)
                dealer_card,
                usable_ace,
                actions_taken,
                reward,
                result,
                dealer_hand_str,
                dealer_total,
                dealer_hand_size,
                final_hands_str,
                *counts_list
            ])

    print(f"\nDone! File saved as {output_file}")
    
    # Print some statistics about dealer hands
    print("\nDealer Hand Statistics:")
    df = pd.read_csv(output_file)
    print(f"Average dealer total: {df['Dealer_Total'].mean():.2f}")
    print(f"Average dealer hand size: {df['Dealer_Hand_Size'].mean():.2f}")
    print(f"Dealer bust rate: {(df['Dealer_Total'] > 21).mean():.1%}")
    print(f"Dealer blackjack rate: {((df['Dealer_Total'] == 21) & (df['Dealer_Hand_Size'] == 2)).mean():.1%}")

if __name__ == "__main__":
    import pandas as pd  # Add this import at the top
    test_and_save_logs("blackjack_infinite_shoe_model.zip")