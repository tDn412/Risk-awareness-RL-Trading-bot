"""
Backtesting and Evaluation Engine
Project: Antigravity - Risk-Aware Reinforcement Learning Trading Bot

Simulates out-of-sample execution, benchmark comparisons (Buy & Hold, Moving Average Crossover),
and computes quantitative metrics (Sharpe, Sortino, Drawdown, Calmar).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add subdirectories to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ["02_Data_Pipeline", "03_Model_Development", "04_Backtesting_Evaluation"]:
    sub_path = os.path.join(PROJECT_ROOT, sub)
    if sub_path not in sys.path:
        sys.path.insert(0, sub_path)

from env.trading_env import RiskAwareTradingEnv
from metrics import evaluate_portfolio, print_metrics_summary
from data_collector import fetch_market_data, add_features


def backtest_buy_and_hold(df: pd.DataFrame, initial_balance: float = 10_000.0):
    """Calculates passive Buy & Hold benchmark equity curve."""
    prices = df["Close"].values
    returns = np.diff(prices) / prices[:-1]
    equity = initial_balance * np.cumprod(np.insert(1.0 + returns, 0, 1.0))
    return returns, equity


def backtest_sma_rule(df: pd.DataFrame, initial_balance: float = 10_000.0, fee: float = 0.001):
    """Calculates SMA Trend-Following (SMA 10 > SMA 50) benchmark."""
    close = df["Close"]
    sma10 = close.rolling(10).mean()
    sma50 = close.rolling(50).mean()
    signal = (sma10 > sma50).astype(float).shift(1).fillna(0.0).values

    prices = close.values
    returns = np.diff(prices) / prices[:-1]
    signals = signal[1:]
    
    # Position changes incur fee
    pos_change = np.abs(np.diff(np.insert(signals, 0, 0.0)))
    fee_cost = pos_change * fee

    strat_returns = signals * returns - fee_cost
    equity = initial_balance * np.cumprod(np.insert(1.0 + strat_returns, 0, 1.0))
    return strat_returns, equity


def run_backtest_comparison(symbol: str = "BTC-USD", start: str = "2021-01-01", end: str = None):
    print("=" * 60)
    print(f" Antigravity Out-of-Sample Backtesting Suite | Asset: {symbol}")
    print("=" * 60)

    # 1. Fetch data
    raw = fetch_market_data(symbol, start=start, end=end)
    df = add_features(raw)

    # 2. Chronological Out-of-Sample Split (Last 30% for test)
    split_idx = int(len(df) * 0.7)
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    print(f"[+] Out-of-sample Test Set: {len(df_test)} days ({df_test.index[0]} to {df_test.index[-1]})")

    # 3. Benchmark 1: Buy & Hold
    bh_returns, bh_equity = backtest_buy_and_hold(df_test)
    bh_metrics = evaluate_portfolio(bh_returns, bh_equity)
    print_metrics_summary("Benchmark: Buy & Hold", bh_metrics)

    # 4. Benchmark 2: SMA Trend Following
    sma_returns, sma_equity = backtest_sma_rule(df_test)
    sma_metrics = evaluate_portfolio(sma_returns, sma_equity)
    print_metrics_summary("Benchmark: SMA Trend Following", sma_metrics)

    # 5. Risk-Aware Environment Simulation (Heuristic / Checkpoint model)
    env = RiskAwareTradingEnv(df_test, reward_mode="volatility_penalty")
    obs, info = env.reset()
    terminated = truncated = False
    
    equity_curve = [10_000.0]
    periodic_returns = []

    while not (terminated or truncated):
        # Heuristic Risk-Aware action: Scale exposure down if drawdown spikes or sentiment drops
        curr_dd = info["drawdown"]
        sentiment = obs[-3] if len(obs) > 2 else 0.0  # approximate index
        
        # De-risk during high drawdown
        if curr_dd > 0.05:
            action = np.array([0.2], dtype=np.float32)
        elif sentiment < -0.3:
            action = np.array([0.1], dtype=np.float32)
        else:
            action = np.array([0.8], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        equity_curve.append(info["portfolio_value"])
        periodic_returns.append(info["net_return"])

    ra_metrics = evaluate_portfolio(np.array(periodic_returns), np.array(equity_curve))
    print_metrics_summary("Antigravity: Risk-Aware Strategy", ra_metrics)

    print("\n[+] Backtest comparison completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest and Benchmark Antigravity RL Trading Agent")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Start date")
    parser.add_argument("--end", type=str, default=None, help="End date")

    args = parser.parse_args()
    run_backtest_comparison(args.symbol, args.start, args.end)
