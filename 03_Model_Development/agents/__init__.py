"""Stable-Baselines3 agent wrappers (PPO primary, SAC secondary)."""

from agents.base_agent import BasePortfolioAgent
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent

AGENTS: dict[str, type[BasePortfolioAgent]] = {"ppo": PPOAgent, "sac": SACAgent}

__all__ = ["AGENTS", "BasePortfolioAgent", "PPOAgent", "SACAgent"]
