"""Training callbacks.

``SharpeCheckpointCallback`` periodically rolls the current policy out over the full
validation period and keeps the checkpoint with the best annualised Sharpe ratio. Unlike
SB3's ``EvalCallback`` (which ranks by episode reward), this criterion is comparable across
reward schemes A0–A3, whose rewards live on different scales.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from env.vn30_portfolio_env import VN30PortfolioEnv
from rollout import rollout

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
MIN_RETURN_STD = 1e-12


def annualised_sharpe(daily_returns: NDArray[np.float64]) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate (0.0 for flat series)."""
    std = float(np.std(daily_returns, ddof=1)) if daily_returns.size > 1 else 0.0
    if std < MIN_RETURN_STD:
        return 0.0
    return float(np.mean(daily_returns) / std * np.sqrt(TRADING_DAYS_PER_YEAR))


class _ModelPolicy:
    """Adapts an SB3 model (``predict`` returns ``(action, state)``) to ``rollout.Policy``."""

    def __init__(self, model: BaseAlgorithm) -> None:
        self.model = model

    def predict(self, obs: NDArray[np.float32], deterministic: bool = True) -> NDArray[np.float32]:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)


class SharpeCheckpointCallback(BaseCallback):
    """Save the model whenever its validation Sharpe ratio improves."""

    def __init__(self, val_env: VN30PortfolioEnv, eval_freq: int, save_path: Path) -> None:
        """Create the callback.

        Args:
            val_env: Full-period validation environment (``episode_length=None``).
            eval_freq: Evaluate every ``eval_freq`` calls to ``_on_step`` (i.e. vec-env steps).
            save_path: Where the best checkpoint is written (``.zip`` appended by SB3).
        """
        super().__init__(verbose=0)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_sharpe = -np.inf

    def evaluate(self) -> float:
        """Roll out the current policy on validation and return its Sharpe ratio."""
        ledger = rollout(_ModelPolicy(self.model), self.val_env)
        returns = ledger["daily_return"].to_numpy(dtype=np.float64)[1:]
        return annualised_sharpe(returns)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        sharpe = self.evaluate()
        self.logger.record("eval/val_sharpe", sharpe)
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.model.save(self.save_path)
            logger.info("New best validation Sharpe %.3f at %d steps", sharpe, self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        # Guarantee a best checkpoint exists even if no evaluation fired.
        if not np.isfinite(self.best_sharpe):
            self.best_sharpe = self.evaluate()
            self.model.save(self.save_path)
