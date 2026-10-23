"""Gymnasium environments for VN30 portfolio allocation."""

import gymnasium as gym

from env.market_data import FeatureScaler, MarketData, load_market_data, make_synthetic_market_data
from env.rewards import REWARD_MODES, RewardConfig
from env.vn30_portfolio_env import VN30PortfolioEnv

ENV_ID = "VN30Portfolio-v0"
if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="env.vn30_portfolio_env:VN30PortfolioEnv")

__all__ = [
    "ENV_ID",
    "FeatureScaler",
    "MarketData",
    "REWARD_MODES",
    "RewardConfig",
    "VN30PortfolioEnv",
    "load_market_data",
    "make_synthetic_market_data",
]
