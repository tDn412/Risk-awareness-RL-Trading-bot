"""
Master Data Pipeline Runner for VN30 Quantitative Trading Framework
Project: Risk-Aware RL for VN30 Portfolio Allocation (INT3011E)

Orchestrates data collection for:
1. Daily OHLCV Prices for 30 VN30 tickers (2018–2025).
2. Corporate Events (cash dividends, bonus shares, splits).
3. Market Benchmark Indices (VN30, VNINDEX).
"""

import os
import sys
import argparse
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from collectors import (
    fetch_all_vn30_prices,
    fetch_all_corporate_events,
    fetch_benchmark_indices,
    DEFAULT_VN30_TICKERS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DataPipeline")


def run_full_pipeline(
    start_date: str = "2018-01-01",
    end_date: str = "2025-12-31",
    prices_dir: str = "data/raw/prices",
    events_dir: str = "data/raw/corporate_actions",
    benchmarks_dir: str = "data/raw/benchmarks",
    delay_seconds: float = 1.0,
    skip_prices: bool = False,
    skip_events: bool = False,
    skip_benchmarks: bool = False
):
    """Run all data collectors sequentially."""
    logger.info("=" * 60)
    logger.info("🚀 STARTING VN30 RAW DATA INGESTION PIPELINE")
    logger.info(f"Target Period: {start_date} -> {end_date}")
    logger.info(f"Number of Equities: {len(DEFAULT_VN30_TICKERS)}")
    logger.info("=" * 60)

    # 1. Fetch Benchmarks
    if not skip_benchmarks:
        logger.info("\n[STEP 1/3] Fetching Benchmark Indices (VN30, VNINDEX)...")
        fetch_benchmark_indices(
            start_date=start_date,
            end_date=end_date,
            output_dir=benchmarks_dir,
            delay_between_requests=delay_seconds
        )
    else:
        logger.info("\n[STEP 1/3] Skipping Benchmark Indices.")

    # 2. Fetch Corporate Events
    if not skip_events:
        logger.info("\n[STEP 2/3] Fetching Corporate Events (Dividends & Splits)...")
        fetch_all_corporate_events(
            tickers=DEFAULT_VN30_TICKERS,
            output_dir=events_dir,
            delay_between_requests=delay_seconds
        )
    else:
        logger.info("\n[STEP 2/3] Skipping Corporate Events.")

    # 3. Fetch Prices
    if not skip_prices:
        logger.info("\n[STEP 3/3] Fetching Daily OHLCV Prices for 30 Equities...")
        fetch_all_vn30_prices(
            tickers=DEFAULT_VN30_TICKERS,
            start_date=start_date,
            end_date=end_date,
            output_dir=prices_dir,
            delay_between_requests=delay_seconds
        )
    else:
        logger.info("\n[STEP 3/3] Skipping Prices.")

    logger.info("\n" + "=" * 60)
    logger.info("✅ RAW DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full VN30 Data Collection Pipeline")
    parser.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay seconds between API calls")
    parser.add_argument("--skip-prices", action="store_true", help="Skip price fetching")
    parser.add_argument("--skip-events", action="store_true", help="Skip corporate events fetching")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmarks fetching")
    args = parser.parse_args()

    run_full_pipeline(
        start_date=args.start,
        end_date=args.end,
        delay_seconds=args.delay,
        skip_prices=args.skip_prices,
        skip_events=args.skip_events,
        skip_benchmarks=args.skip_benchmarks
    )
