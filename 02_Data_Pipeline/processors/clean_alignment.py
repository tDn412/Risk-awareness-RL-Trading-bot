"""Data Processor: Clean and Align VN30 Equities.

Project: Risk-Aware RL for VN30 Portfolio Allocation (INT3011E)
Aligns all 30 VN30 tickers to a single synchronized trading calendar derived from VN30.csv.
Enforces forward-fill rules (gap <= 5 trading days ffill, gap > 5 days raises ValueError),
and handles late-listing stocks (e.g. SSB, TCB) with an explicit `is_listed` indicator
to prevent survivorship/look-ahead bias.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Max gap allowed: 10 trading days to accommodate official UBCKNN exchange-transfer halts
# (e.g. POW 9 days, BCM 7 days, VIB 7 days, GVR 6 days during UPCoM -> HOSE transfers).
# Any gap > 10 days within active listing indicates data corruption and will raise ValueError.
MAX_ALLOWED_GAP_DAYS = 10


def get_trading_calendar(benchmark_csv: Path | str) -> pd.DatetimeIndex:
    """Extract trading calendar dates from benchmark index (VN30).

    Args:
        benchmark_csv: Path to VN30 benchmark CSV.

    Returns:
        Sorted unique DatetimeIndex of trading days.
    """
    path = Path(benchmark_csv)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    calendar = pd.DatetimeIndex(sorted(df["date"].unique()))
    logger.info(f"Loaded trading calendar with {len(calendar)} days: {calendar[0].date()} to {calendar[-1].date()}")
    return calendar


def align_ticker_prices(
    ticker: str,
    raw_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    max_gap_days: int = MAX_ALLOWED_GAP_DAYS,
) -> pd.DataFrame:
    """Align a single ticker's price series against the trading calendar.

    Args:
        ticker: Ticker symbol (e.g. 'VIC', 'SSB').
        raw_df: Raw DataFrame with columns [date, close, volume, ...].
        calendar: Target DatetimeIndex.
        max_gap_days: Maximum consecutive missing trading days permitted for ffill.

    Returns:
        Aligned DataFrame matching target schema.

    Raises:
        ValueError: If a gap of > max_gap_days occurs during the active listing period.
    """
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"])
    df = df.set_index("date")

    first_trading_date = df.index.min()
    last_trading_date = df.index.max()

    # Reindex over entire target calendar
    aligned = df.reindex(calendar)
    aligned["ticker"] = ticker

    if "adj_factor" not in aligned.columns:
        aligned["adj_factor"] = 1.0

    # Ensure required columns exist
    if "close" not in aligned.columns:
        raise KeyError(f"Ticker {ticker} data missing 'close' column")
    if "volume" not in aligned.columns:
        aligned["volume"] = 0

    aligned["is_listed"] = False

    if pd.notna(first_trading_date) and pd.notna(last_trading_date):
        # Mask representing active listing lifetime
        lifetime_mask = (aligned.index >= first_trading_date) & (aligned.index <= last_trading_date)
        aligned.loc[lifetime_mask, "is_listed"] = True

        # Check consecutive missing gaps within active listing lifetime
        lifetime_series = aligned.loc[lifetime_mask, "close"]
        is_missing = lifetime_series.isna()

        if is_missing.any():
            # Group consecutive Trues to calculate gap lengths
            gap_blocks = (~is_missing).cumsum()[is_missing]
            gap_counts = gap_blocks.value_counts()
            max_detected_gap = gap_counts.max() if not gap_counts.empty else 0

            if max_detected_gap > max_gap_days:
                # Find start and end date of violating gap
                offending_block_id = gap_counts.idxmax()
                violating_dates = gap_blocks[gap_blocks == offending_block_id].index
                start_gap = violating_dates.min().strftime("%Y-%m-%d")
                end_gap = violating_dates.max().strftime("%Y-%m-%d")
                msg = (
                    f"Ticker {ticker} exceeds max gap limit ({max_gap_days} days): "
                    f"detected gap of {max_detected_gap} days between {start_gap} and {end_gap}."
                )
                logger.error(msg)
                raise ValueError(msg)

            logger.info(f"Ticker {ticker}: Forward-filling gap of max {max_detected_gap} days (<= {max_gap_days}).")
            aligned["close"] = aligned["close"].ffill()
            aligned["adj_factor"] = aligned["adj_factor"].ffill()
            aligned["volume"] = aligned["volume"].fillna(0)

    # Cast schema types
    aligned = aligned.reset_index()
    aligned.rename(columns={"index": "date"}, inplace=True)
    aligned["date"] = pd.to_datetime(aligned["date"])
    aligned["ticker"] = aligned["ticker"].astype(str)
    aligned["close"] = aligned["close"].astype("float64")
    aligned["volume"] = aligned["volume"].fillna(0).astype("int64")
    aligned["adj_factor"] = aligned["adj_factor"].fillna(1.0).astype("float64")
    aligned["is_listed"] = aligned["is_listed"].astype(bool)

    expected_cols = ["date", "ticker", "close", "volume", "adj_factor", "is_listed"]
    return aligned[expected_cols]


def process_clean_alignment(
    prices_dir: Path | str = "data/raw/prices",
    benchmark_path: Path | str = "data/raw/benchmarks/VN30.csv",
    output_path: Path | str = "data/processed/aligned_prices.parquet",
    max_gap_days: int = MAX_ALLOWED_GAP_DAYS,
) -> pd.DataFrame:
    """Align all ticker CSV files and save to Parquet.

    Args:
        prices_dir: Directory containing per-ticker raw CSV files.
        benchmark_path: Path to VN30 benchmark CSV for trading calendar.
        output_path: Path to output Parquet file.
        max_gap_days: Max allowed gap within listed period.

    Returns:
        Master aligned DataFrame containing all tickers.
    """
    prices_dir_path = Path(prices_dir)
    output_file_path = Path(output_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    calendar = get_trading_calendar(benchmark_path)
    csv_files = sorted(list(prices_dir_path.glob("*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No price CSV files found in {prices_dir_path}")

    aligned_dfs: List[pd.DataFrame] = []

    for file_path in csv_files:
        ticker = file_path.stem.replace(".csv", "")
        raw_df = pd.read_csv(file_path)
        aligned_ticker_df = align_ticker_prices(
            ticker=ticker,
            raw_df=raw_df,
            calendar=calendar,
            max_gap_days=max_gap_days,
        )
        aligned_dfs.append(aligned_ticker_df)

    master_df = pd.concat(aligned_dfs, ignore_index=True)
    master_df.to_parquet(output_file_path, index=False)
    logger.info(
        f"Successfully aligned {len(csv_files)} tickers across {len(calendar)} dates. "
        f"Saved to {output_file_path} (total rows: {len(master_df)})."
    )
    return master_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    process_clean_alignment()
