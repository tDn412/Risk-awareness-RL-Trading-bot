"""
Quantitative Performance and Risk Metrics Evaluation Engine
Project: Antigravity - Risk-Aware Reinforcement Learning Trading Bot

Provides standard and risk-adjusted metrics:
- CAGR, Annualized Volatility
- Sharpe Ratio, Sortino Ratio
- Maximum Drawdown (MDD), Calmar Ratio
- Value-at-Risk (VaR 95%), Conditional VaR (CVaR 95%)
- Win Rate, Profit Factor
"""

from typing import Any, Dict, Union
import numpy as np
import pandas as pd


def compute_drawdown(equity_series: pd.Series) -> pd.Series:
    """Compute drawdown series as percentage decline from peak."""
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / (peak + 1e-9)
    return drawdown


def evaluate_portfolio(
    returns_series: Union[pd.Series, np.ndarray[Any, Any]],
    equity_series: Union[pd.Series, np.ndarray[Any, Any]],
    risk_free_rate: float = 0.02,
    annual_trading_days: int = 252
) -> Dict[str, float]:
    """
    Compute full suite of quantitative metrics for portfolio evaluation.

    Args:
        returns_series: Daily or periodic portfolio percentage returns.
        equity_series: Portfolio cumulative equity curve (starts at initial capital).
        risk_free_rate: Annualized risk-free benchmark rate (e.g. 2% = 0.02).
        annual_trading_days: 252 for stocks/VN-Index, 365 for Crypto.

    Returns:
        Dictionary of formatted quantitative performance metrics.
    """
    if not isinstance(returns_series, pd.Series):
        returns_series = pd.Series(returns_series)
    if not isinstance(equity_series, pd.Series):
        equity_series = pd.Series(equity_series)

    # Clean data
    returns_series = returns_series.fillna(0.0)
    equity_series = equity_series.ffill().dropna()

    total_periods = len(returns_series)
    if total_periods < 2:
        raise ValueError("Insufficient data: at least 2 periods required")

    initial_value = float(equity_series.iloc[0])
    final_value = float(equity_series.iloc[-1])
    total_return = (final_value - initial_value) / (initial_value + 1e-9)

    # 1. CAGR (Compound Annual Growth Rate)
    years = max(total_periods / annual_trading_days, 0.01)
    cagr = (final_value / initial_value) ** (1.0 / years) - 1.0 if final_value > 0 else -1.0

    # 2. Volatility
    mean_return = float(returns_series.mean())
    std_return = float(returns_series.std())
    ann_volatility = std_return * np.sqrt(annual_trading_days)

    # 3. Sharpe Ratio
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / annual_trading_days) - 1.0
    excess_returns = returns_series - rf_daily
    excess_std = returns_series.std()
    sharpe_ratio = (
        (excess_returns.mean() / (excess_std + 1e-9)) * np.sqrt(annual_trading_days)
        if excess_std > 0 else 0.0
    )

    # 4. Sortino Ratio (Downside deviation)
    downside_returns = returns_series[returns_series < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 1 else 1e-9
    sortino_ratio = (
        (excess_returns.mean() / (downside_std + 1e-9)) * np.sqrt(annual_trading_days)
        if downside_std > 0 else 0.0
    )

    # 5. Maximum Drawdown (MDD)
    dd_series = compute_drawdown(equity_series)
    max_drawdown = float(abs(dd_series.min()))

    # 6. Calmar Ratio
    calmar_ratio = float(cagr / (max_drawdown + 1e-9)) if max_drawdown > 0 else 0.0

    # 7. Value at Risk (VaR) & CVaR (Expected Shortfall) at 95%
    var_95 = float(np.percentile(returns_series, 5))
    tail_losses = returns_series[returns_series <= var_95]
    cvar_95 = float(tail_losses.mean()) if len(tail_losses) > 0 else var_95

    # 8. Win Rate & Profit Factor
    gains = returns_series[returns_series > 0]
    losses = returns_series[returns_series < 0]
    win_rate = len(gains) / total_periods if total_periods > 0 else 0.0
    profit_factor = (
        float(gains.sum() / (abs(losses.sum()) + 1e-9))
        if len(losses) > 0 else 999.0
    )

    return {
        "Total Return (%)": round(total_return * 100, 2),
        "CAGR (%)": round(cagr * 100, 2),
        "Annualized Volatility (%)": round(ann_volatility * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 3),
        "Sortino Ratio": round(sortino_ratio, 3),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "Calmar Ratio": round(calmar_ratio, 3),
        "VaR (95%)": round(var_95 * 100, 2),
        "CVaR (95%)": round(cvar_95 * 100, 2),
        "Win Rate (%)": round(win_rate * 100, 2),
        "Profit Factor": round(profit_factor, 2),
    }


def sortino_ratio(returns: pd.Series, target: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised Sortino ratio.

    Formula: ``mean(r - target) / DD * sqrt(periods_per_year)`` with downside deviation
    ``DD = sqrt(mean(min(r - target, 0)^2))`` taken over *all* observations.

    Args:
        returns: Periodic (daily) simple returns.
        target: Per-period minimum acceptable return.
        periods_per_year: Annualisation factor.

    Returns:
        The ratio; ``float('inf')`` if no return falls below ``target``.
    """
    excess = returns.astype(float) - target
    downside = np.minimum(excess, 0.0)
    if not (downside < 0.0).any():
        return float("inf")
    downside_dev = float(np.sqrt((downside**2).mean()))
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def print_metrics_summary(strategy_name: str, metrics: Dict[str, float]) -> None:
    """Pretty prints metrics dictionary in terminal."""
    print("=" * 48)
    print(f" Performance Summary: {strategy_name}")
    print("=" * 48)
    for k, v in metrics.items():
        print(f" {k:<28}: {v}")
    print("=" * 48)


if __name__ == "__main__":
    # Self-test on random walk series
    np.random.seed(42)
    fake_returns = np.random.normal(0.0006, 0.015, 500)
    fake_equity = 10000.0 * np.cumprod(1.0 + fake_returns)
    results = evaluate_portfolio(fake_returns, fake_equity)
    print_metrics_summary("Synthetic Test Portfolio", results)
