"""Feature Engineering Processor for VN30 Quantitative Equities.

Project: Risk-Aware RL for VN30 Portfolio Allocation (INT3011E)
Calculates multi-horizon returns, momentum (RSI), trend (MACD), volatility (BB, ATR),
downside risk, and liquidity features for all 30 VN30 tickers.
Enforces strict anti-look-ahead rules using rolling statistics and unlisted masks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_ticker_features(ticker_df: pd.DataFrame, warmup_period: int = 20) -> pd.DataFrame:
    """Compute 11 technical and risk features for a single ticker.

    Args:
        ticker_df: DataFrame containing [date, ticker, close, volume, is_listed].
        warmup_period: Lookback window for rolling metrics.

    Returns:
        DataFrame augmented with technical features.
    """
    df = ticker_df.copy().sort_values("date").reset_index(drop=True)

    close = df["close"]
    volume = df["volume"]
    is_listed = df["is_listed"]

    # 1. Multi-horizon returns (log returns)
    ret_1d = np.log(close / close.shift(1))
    ret_5d = np.log(close / close.shift(5))
    ret_20d = np.log(close / close.shift(20))

    # 2. RSI (14 periods) - Wilder's Smoothing
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))

    # 3. MACD Histogram (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # 4. Bollinger Bands (20 periods, 2 std)
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std(ddof=0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_pctb = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-9)

    # 5. Normalized ATR (approx from daily close diffs when H/L not present)
    hl_range = close.diff().abs()
    atr_14 = hl_range.rolling(window=14).mean()
    atr_norm = atr_14 / (close + 1e-9)

    # 6. Rolling Annualized Volatility (20 days)
    roll_vol_20 = ret_1d.rolling(window=20).std() * np.sqrt(252)

    # 7. Downside Semi-Variance Volatility (20 days)
    neg_ret = ret_1d.clip(upper=0)
    downside_vol_20 = neg_ret.rolling(window=20).std() * np.sqrt(252)

    # 8. Normalized Turnover Ratio
    traded_val = volume * close
    turnover_ratio = traded_val / (traded_val.rolling(window=20).mean() + 1e-9)

    # Assign features
    df["ret_1d"] = ret_1d
    df["ret_5d"] = ret_5d
    df["ret_20d"] = ret_20d
    df["rsi_14"] = rsi_14
    df["macd_hist"] = macd_hist
    df["bb_pctb"] = bb_pctb
    df["bb_width"] = bb_width
    df["atr_norm"] = atr_norm
    df["roll_vol_20"] = roll_vol_20
    df["downside_vol_20"] = downside_vol_20
    df["turnover_ratio"] = turnover_ratio

    # For unlisted periods, force all features to 0.0 to prevent look-ahead bias
    feature_cols = [
        "ret_1d", "ret_5d", "ret_20d", "rsi_14", "macd_hist",
        "bb_pctb", "bb_width", "atr_norm", "roll_vol_20",
        "downside_vol_20", "turnover_ratio"
    ]
    unlisted_mask = ~is_listed
    df.loc[unlisted_mask, feature_cols] = 0.0

    return df


def process_build_features(
    aligned_parquet_path: Path | str = "data/processed/aligned_prices.parquet",
    output_feature_path: Path | str = "data/processed/feature_matrix.parquet",
) -> pd.DataFrame:
    """Build multi-ticker feature matrix and save to Parquet.

    Args:
        aligned_parquet_path: Path to input aligned prices Parquet.
        output_feature_path: Path to output feature matrix Parquet.

    Returns:
        Complete feature matrix DataFrame.
    """
    input_path = Path(aligned_parquet_path)
    output_path = Path(output_feature_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Aligned prices file not found: {input_path}")

    aligned_df = pd.read_parquet(input_path)
    tickers = aligned_df["ticker"].unique()

    processed_list: List[pd.DataFrame] = []
    for ticker in tickers:
        ticker_data = aligned_df[aligned_df["ticker"] == ticker]
        features_df = compute_ticker_features(ticker_data)
        processed_list.append(features_df)

    feature_matrix = pd.concat(processed_list, ignore_index=True)
    feature_matrix.to_parquet(output_path, index=False)
    logger.info(
        f"Built feature matrix for {len(tickers)} tickers. "
        f"Saved to {output_path} (shape: {feature_matrix.shape})."
    )
    return feature_matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    process_build_features()
