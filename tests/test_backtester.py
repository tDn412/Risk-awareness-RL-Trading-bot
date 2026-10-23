"""Tests for baselines and helpers in ``04_Backtesting_Evaluation/backtester.py``."""

from __future__ import annotations

import torch  # noqa: F401
import numpy as np
import pandas as pd
import pytest

from backtester import (
    cash_ledger,
    compute_metrics,
    equal_weight_target,
    mean_curves,
    simulate_baseline,
    summarize,
)
from env import make_synthetic_market_data

ENV_KWARGS = {"initial_cash": 1_000_000_000.0, "softmax_temperature": 0.2, "risk_window": 20}


def test_equal_weight_target_sums_to_one_and_masks_unlisted() -> None:
    listed = np.array([True, True, False, True])
    weights = equal_weight_target(0.6)(0, listed)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.4)
    assert weights[3] == 0.0  # unlisted asset 2 (index shifted by cash)
    assert weights[1] == pytest.approx(0.2)


def test_equal_weight_baseline_is_mostly_invested_with_low_turnover() -> None:
    data = make_synthetic_market_data(n_days=80, n_assets=10, seed=1)
    ledger = simulate_baseline("ew", data, ENV_KWARGS, equal_weight_target(1.0))
    assert ledger.cash_weight is not None and ledger.turnover is not None
    assert ledger.cash_weight.iloc[5:].mean() < 0.05
    assert ledger.turnover.iloc[5:].mean() < 0.05  # holds between monthly rebalances


def test_cash_ledger_and_metrics() -> None:
    dates = pd.bdate_range("2022-01-03", periods=30)
    metrics = compute_metrics(cash_ledger(dates, 1e9))
    assert metrics["Total Return (%)"] == 0.0
    assert metrics["Max Drawdown (%)"] == 0.0
    assert metrics["Avg Cash Weight"] == 1.0


def test_summarize_and_curves() -> None:
    data = make_synthetic_market_data(n_days=60, n_assets=5, seed=2)
    ledgers = {f"r{i}": simulate_baseline(f"r{i}", data, ENV_KWARGS, equal_weight_target(0.8)) for i in range(2)}
    rows = [
        {"key": key, "group": "PPO A0", "kind": "agent", "seed": i, **compute_metrics(led)}
        for i, (key, led) in enumerate(ledgers.items())
    ]
    results = pd.DataFrame(rows)
    table = summarize(results)
    assert table.loc[0, "n"] == 2
    curve = mean_curves(ledgers, results)["PPO A0"]
    assert curve["nav"].iloc[0] == pytest.approx(100.0)
    assert (curve["drawdown"] <= 0).all()
