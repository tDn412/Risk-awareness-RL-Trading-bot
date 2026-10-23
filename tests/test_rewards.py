"""Tests for reward schemes A0–A3 in ``env/rewards.py``."""

from __future__ import annotations

import numpy as np
import pytest

from env.rewards import RewardConfig, compute_reward, historical_cvar, prospect_utility

WINDOW = np.array([0.01, -0.02, 0.005, -0.04, 0.03])


def test_a0_is_scaled_return() -> None:
    assert compute_reward(RewardConfig("A0", reward_scale=1.0), 0.01, 0.3, WINDOW) == pytest.approx(0.01)


def test_a1_volatility_penalty() -> None:
    cfg = RewardConfig("A1", reward_scale=1.0, lambda_vol=5.0)
    assert compute_reward(cfg, 0.01, 0.0, WINDOW) == pytest.approx(0.01 - 5.0 * np.var(WINDOW))


def test_a2_downside_and_drawdown() -> None:
    cfg = RewardConfig("A2", reward_scale=1.0, lambda_down=10.0, lambda_dd=0.5, dd_threshold=0.15)
    expected = -0.02 - 10.0 * 0.02**2 - 0.5 * (0.25 - 0.15)
    assert compute_reward(cfg, -0.02, 0.25, WINDOW) == pytest.approx(expected)
    # No penalty for gains below the drawdown threshold.
    assert compute_reward(cfg, 0.02, 0.1, WINDOW) == pytest.approx(0.02)


def test_a3_prospect_and_cvar() -> None:
    cfg = RewardConfig("A3", reward_scale=1.0, lambda_cvar=1.0, cvar_target=0.01, lambda_dd=0.0)
    expected = prospect_utility(-0.02, 0.88, 0.88, 2.25) - (historical_cvar(WINDOW) - 0.01)
    assert compute_reward(cfg, -0.02, 0.0, WINDOW) == pytest.approx(expected)


def test_prospect_loss_aversion() -> None:
    gain = prospect_utility(0.01, 0.88, 0.88, 2.25)
    loss = prospect_utility(-0.01, 0.88, 0.88, 2.25)
    assert loss == pytest.approx(-2.25 * gain)


def test_historical_cvar() -> None:
    assert historical_cvar(WINDOW) == pytest.approx(0.04)
    assert historical_cvar(np.array([])) == 0.0


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError):
        RewardConfig("A9")
