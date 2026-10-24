"""DoD tests for ``VN30PortfolioEnv`` (AGENTS.md §6.3)."""

from __future__ import annotations

import warnings
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env import ENV_ID, RewardConfig, VN30PortfolioEnv, make_synthetic_market_data
from env.vn30_portfolio_env import (
    BROKERAGE_FEE_RATE,
    LOT_SIZE,
    PRICE_LIMIT_PCT,
    SELL_TAX_RATE,
    clip_to_price_band,
)

PRICE = 50_000.0
CAPITAL = 1_000_000_000.0


def _all_in(n_assets: int, asset: int) -> np.ndarray:
    """Logits putting (almost) all weight on ``asset`` (1-based; 0 = cash)."""
    action = -np.ones(n_assets + 1, dtype=np.float32)
    action[asset] = 1.0
    return action


def _const_env(n_assets: int = 3, n_days: int = 20) -> VN30PortfolioEnv:
    data = make_synthetic_market_data(n_days=n_days, n_assets=n_assets, constant_prices=True)
    return VN30PortfolioEnv(data, initial_cash=CAPITAL)


def test_check_env() -> None:
    env = gym.make(ENV_ID, data=make_synthetic_market_data(n_days=120), episode_length=60)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env.unwrapped)


def test_t2_settlement() -> None:
    env = _const_env()
    env.reset(seed=0)
    cash_action = _all_in(3, 0)

    env.step(_all_in(3, 1))  # buy on day T
    bought = env.shares[0]
    assert bought > 0
    # Day T+1: still unsettled -> cannot be sold.
    assert env.sellable_shares()[0] == 0
    env.step(cash_action)
    assert env.shares[0] == bought
    # Day T+2: settled -> sellable and sold.
    assert env.sellable_shares()[0] == bought
    env.step(cash_action)
    assert env.shares[0] == 0


def test_price_limit_clip() -> None:
    ref = np.array([10_000.0, 10_000.0, 10_000.0])
    order = np.array([11_000.0, 9_000.0, 10_200.0])  # +10%, -10%, +2%
    clipped = clip_to_price_band(order, ref)
    np.testing.assert_allclose(clipped, [10_000.0 * (1 + PRICE_LIMIT_PCT), 10_000.0 * (1 - PRICE_LIMIT_PCT), 10_200.0])

    # A stock closing at the ceiling has no sellers: buy orders are not filled.
    base = make_synthetic_market_data(n_days=5, n_assets=2, constant_prices=True)
    close = base.close.copy()
    close[1:, 0] = PRICE * (1 + PRICE_LIMIT_PCT)
    env = VN30PortfolioEnv(replace(base, close=close), initial_cash=CAPITAL)
    env.reset(seed=0)
    env.step(_all_in(2, 0))  # day 0: stay in cash
    env.step(_all_in(2, 1))  # day 1: asset 1 is limit-up
    assert env.shares[0] == 0


def test_fee_deduction() -> None:
    env = _const_env(n_assets=1)
    env.reset(seed=0)

    _, _, _, _, info = env.step(_all_in(1, 1))
    lot_cost = PRICE * LOT_SIZE * (1 + BROKERAGE_FEE_RATE)
    lots = int(CAPITAL // lot_cost)
    buy_value = lots * LOT_SIZE * PRICE
    expected_fee = buy_value * BROKERAGE_FEE_RATE
    assert env.shares[0] == lots * LOT_SIZE
    assert info["transaction_cost"] == pytest.approx(expected_fee)
    assert info["portfolio_value"] == pytest.approx(CAPITAL - expected_fee)

    env.step(_all_in(1, 0))  # T+1: cannot sell yet
    _, _, _, _, info = env.step(_all_in(1, 0))  # T+2: sell everything
    expected_sell_cost = buy_value * (BROKERAGE_FEE_RATE + SELL_TAX_RATE)
    assert info["transaction_cost"] == pytest.approx(expected_sell_cost)
    assert info["portfolio_value"] == pytest.approx(CAPITAL - expected_fee - expected_sell_cost)


def test_episode_deterministic() -> None:
    def rollout(seed: int) -> list[tuple[float, float]]:
        env = VN30PortfolioEnv(
            make_synthetic_market_data(n_days=200, seed=3),
            reward_config=RewardConfig(mode="A3"),
            episode_length=50,
        )
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        trace = [(float(obs.sum()), 0.0)]
        done = False
        while not done:
            obs, reward, done, _, info = env.step(env.action_space.sample())
            trace.append((float(obs.sum()), reward + info["portfolio_value"]))
        return trace

    assert rollout(7) == rollout(7)
    assert rollout(7) != rollout(8)


def test_info_keys_and_lot_size() -> None:
    env = VN30PortfolioEnv(make_synthetic_market_data(n_days=30))
    env.reset(seed=1)
    _, _, _, _, info = env.step(env.action_space.sample())
    for key in ("portfolio_value", "daily_return", "cvar_95", "turnover", "transaction_cost"):
        assert key in info
    assert np.all(env.shares % LOT_SIZE == 0)
    assert env.cash >= 0.0
    assert set(env.obs_schema) == set(VN30PortfolioEnv.OBS_SCHEMA)


def test_unlisted_assets_get_zero_weight() -> None:
    base = make_synthetic_market_data(n_days=10, n_assets=3)
    listed = base.is_listed.copy()
    listed[:, 2] = False
    close = np.where(listed, base.close, np.nan)
    env = VN30PortfolioEnv(replace(base, is_listed=listed, close=close))
    env.reset(seed=0)
    weights = env.action_to_weights(np.ones(4, dtype=np.float32))
    assert weights[3] == 0.0
    assert weights.sum() == pytest.approx(1.0)
    env.step(np.ones(4, dtype=np.float32))
    assert env.shares[2] == 0


def test_no_trade_band_suppresses_small_rebalances() -> None:
    data = make_synthetic_market_data(n_days=30, n_assets=5, seed=4)
    env = VN30PortfolioEnv(data, no_trade_band=0.05)
    env.reset(seed=0)
    env.step(np.zeros(6, dtype=np.float32))  # equal split: ~16.7% per asset, all > band
    held = env.shares
    assert held.sum() > 0
    _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))  # same target: only drift-sized moves
    np.testing.assert_array_equal(env.shares, held)
    assert info["transaction_cost"] == 0.0


def test_action_smoothing_moves_partially_toward_target() -> None:
    env = _const_env(n_assets=1)
    env = VN30PortfolioEnv(env.data, initial_cash=CAPITAL, action_smoothing=0.5)
    env.reset(seed=0)
    env.step(_all_in(1, 1))
    assert env.weights()[1] == pytest.approx(0.5, abs=0.01)


def test_invalid_turnover_controls_rejected() -> None:
    data = make_synthetic_market_data(n_days=10, n_assets=2)
    with pytest.raises(ValueError):
        VN30PortfolioEnv(data, no_trade_band=1.5)
    with pytest.raises(ValueError):
        VN30PortfolioEnv(data, action_smoothing=0.0)
