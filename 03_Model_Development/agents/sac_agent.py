"""
SAC Agent for Maximum Entropy Continuous Portfolio Optimization
Project: Antigravity - Risk-Aware Reinforcement Learning Trading Bot

Soft Actor-Critic (SAC) optimizes return while maximizing policy entropy,
promoting robust exploration and preventing premature convergence in financial markets.
"""

from typing import Optional
import numpy as np

try:
    from stable_baselines3 import SAC
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class AntigravitySACAgent:
    """Wrapper around Stable-Baselines3 SAC tailored for financial portfolio optimization."""

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
        ent_coef: str = "auto",
        tensorboard_log: Optional[str] = "./runs/sac"
    ):
        self.env = env
        self.lr = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.ent_coef = ent_coef
        self.tensorboard_log = tensorboard_log
        self.model = None

        if SB3_AVAILABLE:
            self.model = SAC(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=self.lr,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                gamma=self.gamma,
                tau=self.tau,
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
        return np.array([0.5], dtype=np.float32)

    def save(self, filepath: str):
        if self.model is not None:
            self.model.save(filepath)
            print(f"[+] SAC model checkpoint saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str, env=None):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required to load SAC models.")
        instance = cls(env=env)
        instance.model = SAC.load(filepath, env=env)
        return instance
