"""Deterministic policy rollouts over a full period of ``VN30PortfolioEnv``.

Used by ``train.py`` for validation-based checkpoint selection and, in week 6, by the
backtester. Returns a daily ledger DataFrame that ``metrics.evaluate_portfolio`` consumes.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from env.vn30_portfolio_env import VN30PortfolioEnv


class Policy(Protocol):
    """Anything with an SB3-style ``predict`` returning allocation logits."""

    def predict(self, obs: NDArray[np.float32], deterministic: bool = True) -> NDArray[np.float32]:
        """Return an action for ``obs``."""
        ...


def rollout(policy: Policy, env: VN30PortfolioEnv) -> pd.DataFrame:
    """Run ``policy`` deterministically from the first to the last day of ``env``.

    Args:
        policy: Trained agent (or any ``Policy``).
        env: Environment built with ``episode_length=None`` (full period).

    Returns:
        DataFrame indexed by date with columns ``nav, daily_return, cash_weight,
        turnover, transaction_cost, drawdown, cvar_95``; first row is the initial NAV.
    """
    obs, info = env.reset(seed=0)
    rows = [
        {
            "date": pd.Timestamp(info["date"]),
            "nav": info["portfolio_value"],
            "daily_return": 0.0,
            "cash_weight": 1.0,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "drawdown": 0.0,
            "cvar_95": 0.0,
        }
    ]
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy.predict(obs, deterministic=True))
        rows.append(
            {
                "date": pd.Timestamp(info["date"]),
                "nav": info["portfolio_value"],
                "daily_return": info["daily_return"],
                "cash_weight": info["cash_weight"],
                "turnover": info["turnover"],
                "transaction_cost": info["transaction_cost"],
                "drawdown": info["drawdown"],
                "cvar_95": info["cvar_95"],
            }
        )
    return pd.DataFrame(rows).set_index("date")
