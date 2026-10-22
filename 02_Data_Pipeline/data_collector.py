"""
Data Collector and Feature Engineering Pipeline
Project: Antigravity - Risk-Aware Reinforcement Learning Trading Bot

Fetches market data (Stocks, Crypto, VN-Index), computes technical indicators,
integrates sentiment features, and produces normalized feature matrices for RL training.
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

try:
    import yfinance as yf
except ImportError:
    yf = None


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) without external dependencies."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD line, signal line, and histogram."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands and %B position."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    pct_b = (series - lower) / ((upper - lower) + 1e-9)
    bandwidth = (upper - lower) / (sma + 1e-9)
    return upper, lower, pct_b.fillna(0.5), bandwidth.fillna(0.0)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate normalized Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    norm_atr = atr / (close + 1e-9)
    return norm_atr.fillna(0.0)


def generate_synthetic_data(days: int = 1000, start_price: float = 100.0) -> pd.DataFrame:
    """
    Generate synthetic Geometric Brownian Motion market data with realistic regimes
    for testing when internet/yfinance is offline.
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=days, freq="B")
    
    # Simulate regime shifts (bull, bear, sideways)
    regimes = np.array([0.0005] * (days // 3) + [-0.0008] * (days // 3) + [0.0001] * (days - 2 * (days // 3)))
    volatilities = np.array([0.015] * (days // 3) + [0.030] * (days // 3) + [0.012] * (days - 2 * (days // 3)))
    
    returns = regimes + volatilities * np.random.randn(days)
    price_series = start_price * np.exp(np.cumsum(returns))
    
    high = price_series * (1 + np.abs(np.random.normal(0, 0.008, days)))
    low = price_series * (1 - np.abs(np.random.normal(0, 0.008, days)))
    open_p = price_series * (1 + np.random.normal(0, 0.004, days))
    volume = np.random.lognormal(mean=14, sigma=0.5, size=days)

    df = pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": price_series,
        "Volume": volume
    }, index=dates)
    return df


def fetch_market_data(symbol: str, start: Optional[str] = None, end: Optional[str] = None, fallback_on_fail: bool = True) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance, falling back gracefully if needed."""
    print(f"[*] Fetching market data for: {symbol} (from {start} to {end})...")
    df = None
    if yf is not None:
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if df is not None and not df.empty:
                # Flatten multi-level columns if returned by newer yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                print(f"[+] Successfully fetched {len(df)} candles from Yahoo Finance.")
        except Exception as e:
            print(f"[-] yfinance download failed: {e}")

    if (df is None or df.empty) and fallback_on_fail:
        print("[!] Generating synthetic market data for offline/standalone execution...")
        df = generate_synthetic_data(days=800, start_price=100.0)

    if df is None or df.empty:
        raise ValueError(f"Unable to retrieve data for symbol {symbol}")

    df = df.dropna().copy()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich raw OHLCV dataframe with quantitative features:
    - Log Returns & Momentum
    - Moving Average Ratios
    - Technical Indicators (RSI, MACD, Bollinger Bands, ATR)
    - Volatility & Simulated Sentiment
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # 1. Price Returns
    df["return_1d"] = np.log(close / close.shift(1))
    df["return_5d"] = np.log(close / close.shift(5))
    df["return_20d"] = np.log(close / close.shift(20))

    # 2. Moving Average Ratios (Trend)
    sma_10 = close.rolling(10).mean()
    sma_50 = close.rolling(50).mean()
    df["sma_ratio_10"] = (close - sma_10) / (sma_10 + 1e-9)
    df["sma_ratio_50"] = (close - sma_50) / (sma_50 + 1e-9)

    # 3. Oscillators & Volatility
    df["rsi_14"] = compute_rsi(close, 14) / 100.0  # Normalized to [0, 1]
    macd, signal, hist = compute_macd(close)
    df["macd_norm"] = hist / (close + 1e-9)
    
    _, _, pct_b, bandwidth = compute_bollinger_bands(close)
    df["bb_pct_b"] = pct_b
    df["bb_bandwidth"] = bandwidth

    df["atr_norm"] = compute_atr(high, low, close, 14)
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # 4. Multimodal Sentiment Score Feature (-1.0 Bearish to +1.0 Bullish)
    # Modeled as a momentum-correlated signal with structural noise
    # (can be replaced with live FinBERT/VnBERT embeddings)
    noise = np.random.normal(0, 0.2, len(df))
    sentiment = np.tanh(df["return_5d"] * 10 + noise)
    df["sentiment_score"] = sentiment

    # Drop warm-up NaN rows
    df = df.dropna().copy()
    return df


def export_pipeline(symbol: str, output_dir: str = "data/processed", start: str = "2020-01-01", end: Optional[str] = None) -> str:
    """End-to-end data pipeline execution and export."""
    os.makedirs(output_dir, exist_ok=True)
    clean_sym = symbol.replace("^", "").replace("-", "_").replace(".", "_")
    output_path = os.path.join(output_dir, f"{clean_sym}_features.csv")

    df_raw = fetch_market_data(symbol, start=start, end=end)
    df_features = add_features(df_raw)

    df_features.to_csv(output_path)
    print(f"[+] Processed feature matrix exported to: {output_path}")
    print(f"[+] Total samples: {len(df_features)} | Features: {list(df_features.columns)}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antigravity Data Collector and Feature Engineering Pipeline")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol (e.g. BTC-USD, ^VNINDEX, HPG.VN, SPY)")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--outdir", type=str, default="data/processed", help="Output directory")

    args = parser.parse_args()
    export_pipeline(args.symbol, output_dir=args.outdir, start=args.start, end=args.end)
