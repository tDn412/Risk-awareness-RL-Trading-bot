"""
Unit tests for Performance and Risk Metrics Evaluation Engine
Project: Risk-Aware RL for VN30 Portfolio Allocation (INT3011E)
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04_Backtesting_Evaluation")))
from metrics import compute_drawdown, evaluate_portfolio, sortino_ratio


def test_compute_drawdown():
    """Verify drawdown calculation against known peak-to-trough series."""
    equity = pd.Series([100.0, 120.0, 90.0, 110.0, 60.0])
    dd = compute_drawdown(equity)
    
    # Peak at step 1: 120 -> 90 is -25%
    # Peak at step 1: 120 -> 60 is -50%
    assert dd.iloc[0] == pytest.approx(0.0, abs=1e-5)
    assert dd.iloc[1] == pytest.approx(0.0, abs=1e-5)
    assert dd.iloc[2] == pytest.approx(-0.25, abs=1e-4)
    assert dd.iloc[4] == pytest.approx(-0.50, abs=1e-4)


def test_evaluate_portfolio_standard():
    """Verify portfolio metrics on synthetic deterministic series."""
    # 5-day series with constant positive return
    returns = np.array([0.01, 0.02, -0.01, 0.03, -0.005])
    equity = 10000.0 * np.cumprod(1.0 + returns)
    
    metrics = evaluate_portfolio(returns, equity, annual_trading_days=252)
    
    assert "CAGR (%)" in metrics
    assert "Sharpe Ratio" in metrics
    assert "Sortino Ratio" in metrics
    assert "Max Drawdown (%)" in metrics
    assert "VaR (95%)" in metrics
    assert "CVaR (95%)" in metrics
    assert "Win Rate (%)" in metrics
    assert metrics["Win Rate (%)"] == pytest.approx(60.0, abs=0.1)


def test_evaluate_portfolio_insufficient_data():
    """Verify ValueError is raised when series length < 2."""
    with pytest.raises(ValueError, match="Insufficient data"):
        evaluate_portfolio([0.01], [100.0])


def test_evaluate_portfolio_all_positive():
    """Verify profit factor and Sortino when no losses occur."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.03])
    equity = 1000.0 * (1.0 + returns).cumprod()
    
    metrics = evaluate_portfolio(returns, equity)
    assert metrics["Profit Factor"] == 999.0
    assert metrics["Max Drawdown (%)"] == 0.0


def test_sortino_ratio():
    """Known input/output pair (4 d.p.) and the no-downside edge case."""
    returns = pd.Series([0.02, -0.01, 0.03, -0.02])
    # mean = 0.005; downside dev = sqrt((0.01^2 + 0.02^2) / 4) = sqrt(0.000125)
    expected = 0.005 / np.sqrt(0.000125) * np.sqrt(252)
    assert sortino_ratio(returns) == pytest.approx(expected, abs=1e-4)
    assert round(sortino_ratio(returns), 4) == 7.0993
    assert sortino_ratio(pd.Series([0.01, 0.0, 0.02])) == float("inf")
