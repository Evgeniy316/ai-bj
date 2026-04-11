import csv
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from advanced_model_1 import AdvancedBlackjackEnv
from tqdm import tqdm

def safe_int_action(action):
    """Convert action (possibly np.ndarray) to plain int safely."""
    if isinstance(action, np.ndarray):
        return int(action.item())
    try:
        return int(action)
    except Exception:
        # fallback: squeeze and take first element
        return int(np.squeeze(action).item())

def test_and_save_logs(model_path, num_episodes=5000, output_file="blackjack_logs.csv"):
    env = AdvancedBlackjackEnv()
    model = MaskablePPO.load(model_path)

    print(f"Testing {num_episodes} hands and saving to {output_file}...")

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header for the CSV
        writer.writerow([
            "Episode", "Player_Total", "Dealer_Upcard", "Usable_Ace", 
            "Action_Taken", "Reward", "Result", 
            "Count_2", "Count_3", "Count_4", "Count_5", "Count_6", 
            "Count_7", "Count_8", "Count_9", "Count_10", "Count_A"
        ])

        for ep in tqdm(range(num_episodes)):
            obs, info = env.reset()
            done = False

            # Correct indices based on AdvancedBlackjackEnv._get_state:
            # state = [player_sum_norm, dealer_upcard_norm, usable_ace] + ratios
            player_sum = round(obs[0] * 30)       # env uses player_sum/30.0
            dealer_card = round(obs[1] * 11)      # env uses dealer_upcard/11.0
            usable_ace = int(obs[2])
            counts = obs[3:13]                    # 10 normalized counts

            # Get action mask and make sure shape is what predict expects.
            action_masks = env.action_masks()
            # If model was trained with vec env, predict may expect a 1D mask for single env.
            # If mask is 2D (n_envs x n_actions), take the first row.
            if action_masks.ndim == 2:
                action_masks_to_pass = action_masks[0]
            else:
                action_masks_to_pass = action_masks

            # Predict and ensure action is an integer
            action, _ = model.predict(obs, action_masks=action_masks_to_pass, deterministic=True)
            action = safe_int_action(action)

            # Map action number to string
            action_map = {0: "Stand", 1: "Hit", 2: "Double", 3: "Split"}

            # Step the environment with the scalar action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Determine Result string
            result = "Push"
            if reward > 0: result = "Win"
            if reward < 0: result = "Loss"
            if reward == 1.5: result = "Natural BJ"

            # Convert counts to native floats for CSV readability
            counts_list = [float(x) for x in counts]

            # Write row to CSV
            writer.writerow([
                ep, player_sum, dealer_card, usable_ace,
                action_map.get(action, f"Action_{action}"), reward, result,
                *counts_list
            ])

    print(f"\nDone! File saved as {output_file}")

if __name__ == "__main__":
    # Ensure this matches your saved model name
    test_and_save_logs("blackjack_advanced_parallel_model.zip")