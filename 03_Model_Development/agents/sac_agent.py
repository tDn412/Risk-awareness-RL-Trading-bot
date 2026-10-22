"""SAC agent (secondary algorithm) for VN30 portfolio allocation.

Soft Actor-Critic maximises return plus policy entropy (off-policy, replay buffer).
Hyper-parameters are loaded from ``configs/sac_baseline.yaml``.
"""

from __future__ import annotations

from stable_baselines3 import SAC

from agents.base_agent import BasePortfolioAgent


class SACAgent(BasePortfolioAgent):
    """Stable-Baselines3 SAC wrapper."""

    algorithm = SAC
