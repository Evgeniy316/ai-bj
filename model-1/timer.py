import time
from stable_baselines3.common.callbacks import BaseCallback


class EpochTimerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None

    def _on_rollout_start(self) -> None:
        # Called at start of each rollout (epoch)
        self.epoch_start_time = time.time()

    def _on_rollout_end(self) -> None:
        # Called at end of rollout (before training step)
        duration = time.time() - self.epoch_start_time
        print(f"Epoch {self.num_timesteps} | Duration: {duration:.3f} seconds")

    def _on_step(self) -> bool:
        # REQUIRED by BaseCallback
        # Must return True to continue training
        return True