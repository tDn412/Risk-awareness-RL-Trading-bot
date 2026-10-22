"""PPO agent (primary algorithm) for VN30 portfolio allocation.

PPO's clipped surrogate objective limits destructive policy updates in noisy markets.
Hyper-parameters are loaded from ``configs/ppo_baseline.yaml``.
"""

from __future__ import annotations

from stable_baselines3 import PPO

from agents.base_agent import BasePortfolioAgent


class PPOAgent(BasePortfolioAgent):
    """Stable-Baselines3 PPO wrapper."""

    algorithm = PPO
