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
from processors import (
    process_clean_alignment,
    process_build_features
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
    aligned_path: str = "data/processed/aligned_prices.parquet",
    features_path: str = "data/processed/feature_matrix.parquet",
    delay_seconds: float = 1.0,
    skip_prices: bool = False,
    skip_events: bool = False,
    skip_benchmarks: bool = False,
    skip_processors: bool = False,
    dry_run: bool = False
) -> None:
    """Run all data collectors and processors sequentially."""
    logger.info("=" * 60)
    logger.info("🚀 STARTING VN30 END-TO-END DATA PIPELINE")
    logger.info(f"Target Period: {start_date} -> {end_date}")
    logger.info(f"Number of Equities: {len(DEFAULT_VN30_TICKERS)}")
    if dry_run:
        logger.info("[DRY RUN MODE ENABLED] Validating configuration without making API requests.")
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY RUN] Target directories and files:")
        logger.info(f"  - Prices: {prices_dir} (skip={skip_prices})")
        logger.info(f"  - Corporate Events: {events_dir} (skip={skip_events})")
        logger.info(f"  - Benchmarks: {benchmarks_dir} (skip={skip_benchmarks})")
        logger.info(f"  - Aligned Parquet: {aligned_path} (skip={skip_processors})")
        logger.info(f"  - Feature Matrix: {features_path} (skip={skip_processors})")
        logger.info(f"  - Equities ({len(DEFAULT_VN30_TICKERS)}): {', '.join(DEFAULT_VN30_TICKERS[:5])}...")
        logger.info("\n" + "=" * 60)
        logger.info("✅ DRY RUN VALIDATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        return

    # 1. Fetch Benchmarks
    if not skip_benchmarks:
        logger.info("\n[STEP 1/5] Fetching Benchmark Indices (VN30, VNINDEX)...")
        fetch_benchmark_indices(
            start_date=start_date,
            end_date=end_date,
            output_dir=benchmarks_dir,
            delay_between_requests=delay_seconds
        )
    else:
        logger.info("\n[STEP 1/5] Skipping Benchmark Indices.")

    # 2. Fetch Corporate Events
    if not skip_events:
        logger.info("\n[STEP 2/5] Fetching Corporate Events (Dividends & Splits)...")
        fetch_all_corporate_events(
            tickers=DEFAULT_VN30_TICKERS,
            output_dir=events_dir,
            delay_between_requests=delay_seconds
        )
    else:
        logger.info("\n[STEP 2/5] Skipping Corporate Events.")

    # 3. Fetch Prices
    if not skip_prices:
        logger.info("\n[STEP 3/5] Fetching Daily OHLCV Prices for 30 Equities...")
        fetch_all_vn30_prices(
            tickers=DEFAULT_VN30_TICKERS,
            start_date=start_date,
            end_date=end_date,
            output_dir=prices_dir,
            delay_between_requests=delay_seconds
        )
    else:
        logger.info("\n[STEP 3/5] Skipping Prices.")

    # 4. Clean & Align Prices
    if not skip_processors:
        benchmark_file = os.path.join(benchmarks_dir, "VN30.csv")
        logger.info("\n[STEP 4/5] Cleaning and Aligning Prices to Trading Calendar...")
        process_clean_alignment(
            prices_dir=prices_dir,
            benchmark_path=benchmark_file,
            output_path=aligned_path
        )

        # 5. Build Technical & Risk Feature Matrix
        logger.info("\n[STEP 5/5] Generating Multi-Factor Feature Matrix...")
        process_build_features(
            aligned_parquet_path=aligned_path,
            output_feature_path=features_path
        )
    else:
        logger.info("\n[STEPS 4-5/5] Skipping Processors.")

    logger.info("\n" + "=" * 60)
    logger.info("✅ END-TO-END DATA PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full VN30 Data Collection Pipeline")
    parser.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay seconds between API calls")
    parser.add_argument("--skip-prices", action="store_true", help="Skip price fetching")
    parser.add_argument("--skip-events", action="store_true", help="Skip corporate events fetching")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmarks fetching")
    parser.add_argument("--skip-processors", action="store_true", help="Skip data processing steps")
    parser.add_argument("--dry-run", action="store_true", help="Perform smoke validation without network calls")
    args = parser.parse_args()

    run_full_pipeline(
        start_date=args.start,
        end_date=args.end,
        delay_seconds=args.delay,
        skip_prices=args.skip_prices,
        skip_events=args.skip_events,
        skip_benchmarks=args.skip_benchmarks,
        skip_processors=args.skip_processors,
        dry_run=args.dry_run
    )
