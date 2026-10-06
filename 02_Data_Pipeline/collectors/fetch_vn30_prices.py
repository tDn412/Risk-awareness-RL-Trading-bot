"""
VN30 Historical Prices Collector
Project: Risk-Aware RL for VN30 Portfolio Allocation (INT3011E)

Fetches daily OHLCV historical price data (2018–2025) for all 30 tickers in the VN30 basket
using the vnstock API, standardizes column formats, and stores individual CSV files.
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict, Optional
import pandas as pd

# Turn off vnstock telemetry to avoid unnecessary background network calls
os.environ["VNSTOCK_TELEMETRY"] = "off"

try:
    from vnstock.api.quote import Quote
except ImportError:
    Quote = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VN30PriceCollector")

# 30 Representative Equities in VN30 Universe (2018–2025)
DEFAULT_VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
]


def fetch_single_ticker_price(
    ticker: str,
    start_date: str = "2018-01-01",
    end_date: str = "2025-12-31",
    source: str = "VCI",
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV prices for a single ticker with automatic retries.

    Args:
        ticker: Stock symbol (e.g., 'HPG').
        start_date: Start date string 'YYYY-MM-DD'.
        end_date: End date string 'YYYY-MM-DD'.
        source: Data source ('VCI' recommended).
        max_retries: Number of retry attempts on failure.
        retry_delay: Seconds to wait between retries.

    Returns:
        Cleaned pandas DataFrame or None if failed.
    """
    if Quote is None:
        raise ImportError("vnstock library is not installed. Run: pip install vnstock")

    for attempt in range(1, max_retries + 1):
        try:
            q = Quote(symbol=ticker, source=source)
            df = q.history(start=start_date, end=end_date, interval="1D")

            if df is None or df.empty:
                logger.warning(f"[{ticker}] No data returned on attempt {attempt}.")
                time.sleep(retry_delay)
                continue

            # Standardize column naming
            rename_map = {
                "time": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            }
            df = df.rename(columns=rename_map)

            # Ensure expected columns exist
            required_cols = ["date", "open", "high", "low", "close", "volume"]
            available_cols = [c for c in required_cols if c in df.columns]
            df = df[available_cols].copy()

            # Format date and sort chronologically
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # Convert numeric columns
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["ticker"] = ticker
            return df

        except Exception as e:
            logger.warning(f"[{ticker}] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)

    logger.error(f"[{ticker}] Failed to fetch data after {max_retries} attempts.")
    return None


def fetch_all_vn30_prices(
    tickers: Optional[List[str]] = None,
    start_date: str = "2018-01-01",
    end_date: str = "2025-12-31",
    output_dir: str = "data/raw/prices",
    delay_between_requests: float = 3.5,
    skip_existing: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily prices for all VN30 tickers and save each to individual CSV files.

    Args:
        tickers: List of ticker symbols. Defaults to DEFAULT_VN30_TICKERS.
        start_date: Start date string.
        end_date: End date string.
        output_dir: Output directory path.
        delay_between_requests: Sleep duration between API calls (>=3.5s recommended for 20 calls/min limit).
        skip_existing: If True, skips tickers whose CSV already exists with valid data (>1000 rows).

    Returns:
        Dictionary mapping ticker to DataFrame.
    """
    if tickers is None:
        tickers = DEFAULT_VN30_TICKERS

    os.makedirs(output_dir, exist_ok=True)
    results = {}
    summary = {}

    total = len(tickers)
    logger.info(f"Starting fetch for {total} VN30 tickers from {start_date} to {end_date}...")

    for idx, ticker in enumerate(tickers, 1):
        out_path = os.path.join(output_dir, f"{ticker}.csv")

        # Check if already downloaded
        if skip_existing and os.path.exists(out_path):
            try:
                existing_df = pd.read_csv(out_path)
                if len(existing_df) >= 1000:
                    logger.info(f"[{idx}/{total}] [{ticker}] Already downloaded ({len(existing_df)} rows). Skipping.")
                    results[ticker] = existing_df
                    summary[ticker] = {
                        "status": "cached",
                        "rows": len(existing_df),
                        "path": out_path
                    }
                    continue
            except Exception:
                pass

        logger.info(f"[{idx}/{total}] Fetching OHLCV for {ticker}...")
        df = fetch_single_ticker_price(ticker, start_date=start_date, end_date=end_date)

        if df is not None and not df.empty:
            df.to_csv(out_path, index=False)
            results[ticker] = df

            summary[ticker] = {
                "status": "success",
                "rows": len(df),
                "start_date": str(df["date"].min().date()),
                "end_date": str(df["date"].max().date()),
                "path": out_path
            }
            logger.info(f"[{ticker}] Saved {len(df)} rows ({summary[ticker]['start_date']} -> {summary[ticker]['end_date']})")
        else:
            summary[ticker] = {
                "status": "failed",
                "rows": 0
            }
            logger.error(f"[{ticker}] Failed to download data.")

        # Respect API rate limits (Guest tier: max 20 requests per minute -> ~3.5s delay)
        time.sleep(delay_between_requests)

    # Save summary report
    summary_path = os.path.join(output_dir, "prices_fetch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Completed fetching {len(results)}/{total} tickers. Summary saved to {summary_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch VN30 Daily Prices")
    parser.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--outdir", default="data/raw/prices", help="Output directory")
    parser.add_argument("--delay", type=float, default=3.5, help="Delay seconds between calls (>=3.5s)")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip existing files")
    args = parser.parse_args()

    fetch_all_vn30_prices(
        start_date=args.start,
        end_date=args.end,
        output_dir=args.outdir,
        delay_between_requests=args.delay,
        skip_existing=not args.no_skip
    )
