"""Shared Stable-Baselines3 wrapper for portfolio agents.

Concrete agents (``PPOAgent``, ``SACAgent``) only declare which SB3 algorithm they wrap.
Hyper-parameters always come from the YAML config's ``hyperparams`` block (AGENTS.md §4.4).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)

POLICY = "MlpPolicy"


class BasePortfolioAgent:
    """Thin wrapper exposing ``train`` / ``predict`` / ``save`` / ``load``."""

    algorithm: ClassVar[type[BaseAlgorithm]]

    def __init__(
        self,
        env: VecEnv,
        hyperparams: dict[str, Any],
        seed: int | None = None,
        tensorboard_log: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        """Build the SB3 model.

        Args:
            env: Vectorised training environment.
            hyperparams: SB3 constructor kwargs from YAML; ``net_arch`` is moved into
                ``policy_kwargs``.
            seed: Random seed for the algorithm.
            tensorboard_log: Directory for TensorBoard event files.
            device: Torch device (CPU is fastest for small MLP policies).
        """
        params = dict(hyperparams)
        policy_kwargs: dict[str, Any] = dict(params.pop("policy_kwargs", {}))
        if "net_arch" in params:
            policy_kwargs["net_arch"] = params.pop("net_arch")
        self.model: BaseAlgorithm = self.algorithm(
            POLICY,
            env,
            seed=seed,
            tensorboard_log=str(tensorboard_log) if tensorboard_log is not None else None,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
            **params,
        )

    def train(
        self, total_timesteps: int, callback: BaseCallback | None = None, tb_log_name: str = "run"
    ) -> BasePortfolioAgent:
        """Train for ``total_timesteps`` environment steps."""
        logger.info("Training %s for %d timesteps", type(self).__name__, total_timesteps)
        self.model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name=tb_log_name)
        return self

    def predict(self, obs: NDArray[np.float32], deterministic: bool = True) -> NDArray[np.float32]:
        """Return the action (allocation logits) for ``obs``."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def save(self, path: str | Path) -> Path:
        """Save the SB3 checkpoint (``.zip``) and return its path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(out)
        logger.info("Saved checkpoint to %s", out)
        return out.with_suffix(".zip")

    @classmethod
    def load(cls, path: str | Path, env: VecEnv | None = None) -> BasePortfolioAgent:
        """Load a checkpoint saved by ``save``."""
        agent = cls.__new__(cls)
        agent.model = cls.algorithm.load(str(path), env=env, device="cpu")
        return agent
