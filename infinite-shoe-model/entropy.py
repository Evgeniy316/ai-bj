import math
from stable_baselines3.common.callbacks import BaseCallback


class EntropyScheduleCallback(BaseCallback):
    def __init__(self, ent_start: float, ent_end: float):
        super().__init__()
        self.ent_start = ent_start
        self.ent_end = ent_end

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        progress = self.model.num_timesteps / self.model._total_timesteps
        # cosine ease-out: decays fast early, flattens at the end
        t = 0.5 * (1 - math.cos(math.pi * progress))
        new_ent = self.ent_start + (self.ent_end - self.ent_start) * t
        self.model.ent_coef = new_ent
        self.logger.record("train/ent_coef", new_ent)