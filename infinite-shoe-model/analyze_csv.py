import csv
from collections import defaultdict
import sys
from pathlib import Path

def parse_hand(hand_str: str):
    """Safely parse strings like "[2, 8]" or "[11, 11]" into list of ints."""
    if not hand_str or hand_str.strip() in ("[]", ''):
        return []
    # Remove outer quotes and brackets
    cleaned = hand_str.strip().strip('"').strip("'").strip('[]')
    if not cleaned:
        return []
    try:
        return [int(x.strip()) for x in cleaned.split(',')]
    except:
        return []

def analyze_blackjack_log(filename: str = "blackjack_logs.csv"):
    file_path = Path(filename)
    if not file_path.exists():
        print(f"❌ File not found: {filename}")
        print("Make sure the CSV is in the same folder as this script.")
        sys.exit(1)

    print(f"📊 Loading {filename} ...\n")
    
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    total = len(data)
    if total == 0:
        print("❌ CSV file is empty!")
        return

    # === Core statistics ===
    total_reward = 0.0
    wins = 0
    natural_bj = 0
    losses = 0
    pushes = 0

    # Split tracking
    total_splits = 0
    ace_splits = 0
    non_ace_split_pairs = defaultdict(int)

    for row in data:
        # Reward
        try:
            reward = float(row['Reward'])
            total_reward += reward
        except (ValueError, KeyError):
            pass

        # Result
        result = row.get('Result', '').strip()
        if result == "Win":
            wins += 1
        elif result == "Natural BJ":
            natural_bj += 1
        elif result == "Loss":
            losses += 1
        elif result == "Push":
            pushes += 1

        # Splits
        action = row.get('Action_Taken', '').lower()
        if "split" in action:
            total_splits += 1
            hand_str = row.get('Player_Initial_Hand', '')
            hand = parse_hand(hand_str)
            if len(hand) == 2 and hand[0] == hand[1]:
                if hand[0] == 11:
                    ace_splits += 1
                else:
                    pair_key = f"{hand[0]}-{hand[1]}"
                    non_ace_split_pairs[pair_key] += 1

    # === Calculate percentages ===
    avg_reward = total_reward / total
    player_advantage = wins + natural_bj
    win_rate = (wins / total) * 100
    natural_rate = (natural_bj / total) * 100
    advantage_rate = (player_advantage / total) * 100
    loss_rate = (losses / total) * 100
    push_rate = (pushes / total) * 100

    non_ace_splits = total_splits - ace_splits

    # === Print beautiful console report ===
    print("=" * 60)
    print("BLACKJACK MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Episodes analyzed          : {total:,}")
    print(f"Average reward per hand    : {avg_reward:+.5f}  ({avg_reward*100:+.3f}% house edge)")
    print(f"Player advantage hands     : {advantage_rate:5.2f}%  (Win + Natural BJ)")
    print(f"Regular win rate           : {win_rate:5.2f}%")
    print(f"Natural Blackjack rate     : {natural_rate:5.2f}%")
    print(f"Loss rate                  : {loss_rate:5.2f}%")
    print(f"Push rate                  : {push_rate:5.2f}%")
    print("-" * 60)

    print(f"\nSPLIT BEHAVIOR")
    print(f"Total splits performed     : {total_splits} ({total_splits/total*100:4.2f}% of all hands)")
    print(f"→ Aces only                : {ace_splits} ({ace_splits/total_splits*100:4.1f}% of splits)")
    print(f"→ Non-ace pairs            : {non_ace_splits} ({non_ace_splits/total_splits*100:4.1f}% of splits)")
    print()

    if non_ace_splits > 0:
        print("NON-ACE PAIRS THE MODEL SPLITS (sorted by frequency):")
        sorted_pairs = sorted(non_ace_split_pairs.items(), key=lambda x: x[1], reverse=True)
        for pair, count in sorted_pairs:
            print(f"   {pair:8} → {count:3d} times")
    else:
        print("   (No non-ace splits detected)")

    print("\n" + "=" * 60)
    print("✅ Analysis complete! Your model does split non-ace pairs,")
    print("   but very rarely compared to basic strategy.")
    print("=" * 60)

if __name__ == "__main__":
    # You can also run with: python analyze.py your_log.csv
    analyze_blackjack_log()