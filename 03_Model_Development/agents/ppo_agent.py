"""
PPO Agent for Risk-Aware Continuous Portfolio Optimization
Project: Antigravity - Risk-Aware Reinforcement Learning Trading Bot

Proximal Policy Optimization (PPO) is the primary recommendation for continuous action
portfolio allocation owing to its clipped surrogate objective which avoids destructively
large policy shifts in noisy financial markets.
"""

from typing import Optional, Dict, Any
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class AntigravityPPOAgent:
    """Wrapper around Stable-Baselines3 PPO tailored for quantitative trading."""

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 512,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        tensorboard_log: Optional[str] = "./runs/ppo"
    ):
        self.env = env
        self.lr = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.tensorboard_log = tensorboard_log
        self.model = None

        if SB3_AVAILABLE:
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=self.lr,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                verbose=1,
                tensorboard_log=self.tensorboard_log
            )

    def train(self, total_timesteps: int = 20_000, callback=None):
        if not SB3_AVAILABLE:
            print("[!] stable-baselines3 is not installed. Please install via: pip install stable-baselines3")
            return None
        return self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.model is not None:
            action, _states = self.model.predict(observation, deterministic=deterministic)
            return action
        # Heuristic fallback if model not loaded
        return np.array([0.5], dtype=np.float32)

    def save(self, filepath: str):
        if self.model is not None:
            self.model.save(filepath)
            print(f"[+] PPO model checkpoint saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str, env=None):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required to load PPO models.")
        instance = cls(env=env)
        instance.model = PPO.load(filepath, env=env)
        return instance
