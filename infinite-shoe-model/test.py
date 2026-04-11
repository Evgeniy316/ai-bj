import numpy as np

# Use 'with' to automatically close the file after loading
with np.load("infinite-shoe-model/logs/results/evaluations.npz") as data:
    # 1. List all array names in the file
    timesteps = data['timesteps']
    results = data['results']
    ep_lengths = data['ep_lengths']

    # Summary statistics across episodes (axis=1)
    mean_results = np.mean(results, axis=1)        # shape (12,)
    std_results = np.std(results, axis=1)
    mean_lengths = np.mean(ep_lengths, axis=1)

    for t in range(len(timesteps)):
        print(f"t={timesteps[t]}: result={mean_results[t]:.3f} ± {std_results[t]:.3f}, "
            f"mean ep_len={mean_lengths[t]:.1f}")