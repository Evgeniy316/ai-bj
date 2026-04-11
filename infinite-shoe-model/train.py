import gymnasium as gym
import torch
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from timer import EpochTimerCallback

torch.set_num_threads(1)


# -------- ENV FACTORY --------
def make_env():
    def _init():
        from model import AdvancedBlackjackEnv

        env = AdvancedBlackjackEnv()

        def mask_fn(env: gym.Env):
            return env.action_masks()

        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def make_eval_env():
    from model import AdvancedBlackjackEnv

    env = AdvancedBlackjackEnv()

    def mask_fn(env: gym.Env):
        return env.action_masks()

    env = ActionMasker(env, mask_fn)
    env = Monitor(env)
    return env


if __name__ == "__main__":
    num_cpu = 8

    vec_env  = SubprocVecEnv([make_env() for _ in range(num_cpu)])
    eval_env = DummyVecEnv([make_eval_env])

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=4,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[512, 512],
            )
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    timer_callback = EpochTimerCallback()

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/results/",
        eval_freq=50_000,
        n_eval_episodes=2000,
        deterministic=True,
        use_masking=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./logs/checkpoints/",
        name_prefix="blackjack_model",
    )

    print("Starting training (infinite-shoe)...")

    model.learn(
        total_timesteps=5_000_000,
        callback=[timer_callback, eval_callback, checkpoint_callback],
    )

    model.save("blackjack_infinite_shoe_model")
    print("Training complete.")
