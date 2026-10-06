"""
VN30 Corporate Events & Actions Collector
Project: Risk-Aware RL for VN30 Portfolio Allocation (INT3011E)

Fetches corporate actions (cash dividends, stock dividends, bonus shares, splits)
for all 30 tickers in VN30 basket using vnstock. Filtered and structured to support
realistic portfolio NAV accounting and share adjustments (O1, O2).
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict, Optional
import pandas as pd

# Turn off vnstock telemetry
os.environ["VNSTOCK_TELEMETRY"] = "off"

try:
    from vnstock.api.company import Company
except ImportError:
    Company = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CorporateEventsCollector")

# 30 Representative Equities in VN30 Universe (2018–2025)
DEFAULT_VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
]


def fetch_single_ticker_events(
    ticker: str,
    source: str = "VCI",
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> Optional[pd.DataFrame]:
    """
    Fetch corporate actions and events for a single stock ticker.

    Args:
        ticker: Stock symbol (e.g., 'HPG').
        source: Data source ('VCI' recommended).
        max_retries: Number of retry attempts.
        retry_delay: Delay between retries in seconds.

    Returns:
        DataFrame with filtered dividend and split events or None.
    """
    if Company is None:
        raise ImportError("vnstock library is not installed. Run: pip install vnstock")

    for attempt in range(1, max_retries + 1):
        try:
            c = Company(symbol=ticker, source=source)
            df = c.events()

            if df is None or df.empty:
                logger.warning(f"[{ticker}] No events found on attempt {attempt}.")
                return pd.DataFrame()

            # Select and rename relevant columns
            keep_cols = [
                "ticker", "event_name_vi", "event_code", "category",
                "exright_date", "record_date", "payout_date",
                "exercise_ratio", "value_per_share"
            ]
            available_cols = [col for col in keep_cols if col in df.columns]
            df_filtered = df[available_cols].copy()

            # Keep only events with dividend cash value or share ratio
            has_cash = df_filtered["value_per_share"].notna() if "value_per_share" in df_filtered.columns else False
            has_ratio = df_filtered["exercise_ratio"].notna() if "exercise_ratio" in df_filtered.columns else False
            df_actionable = df_filtered[has_cash | has_ratio].copy()

            if not df_actionable.empty and "exright_date" in df_actionable.columns:
                df_actionable["exright_date"] = pd.to_datetime(df_actionable["exright_date"])
                df_actionable = df_actionable.sort_values("exright_date").reset_index(drop=True)

            return df_actionable

        except Exception as e:
            logger.warning(f"[{ticker}] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)

    logger.error(f"[{ticker}] Failed to fetch corporate events after {max_retries} attempts.")
    return None


def fetch_all_corporate_events(
    tickers: Optional[List[str]] = None,
    output_dir: str = "data/raw/corporate_actions",
    delay_between_requests: float = 1.0
) -> pd.DataFrame:
    """
    Fetch corporate events for all VN30 tickers, save individual and consolidated CSVs.

    Args:
        tickers: List of ticker symbols. Defaults to DEFAULT_VN30_TICKERS.
        output_dir: Output directory path.
        delay_between_requests: Delay between calls to prevent rate limiting.

    Returns:
        Consolidated DataFrame of all corporate events.
    """
    if tickers is None:
        tickers = DEFAULT_VN30_TICKERS

    os.makedirs(output_dir, exist_ok=True)
    all_events_list = []
    summary = {}

    total = len(tickers)
    logger.info(f"Starting corporate events fetch for {total} VN30 tickers...")

    for idx, ticker in enumerate(tickers, 1):
        logger.info(f"[{idx}/{total}] Fetching corporate events for {ticker}...")
        df = fetch_single_ticker_events(ticker)

        if df is not None and not df.empty:
            out_path = os.path.join(output_dir, f"{ticker}_events.csv")
            df.to_csv(out_path, index=False)
            all_events_list.append(df)

            summary[ticker] = {
                "status": "success",
                "events_count": len(df),
                "path": out_path
            }
            logger.info(f"[{ticker}] Found {len(df)} actionable events (cash dividends/splits).")
        else:
            summary[ticker] = {
                "status": "no_events" if df is not None else "failed",
                "events_count": 0
            }

        time.sleep(delay_between_requests)

    # Consolidate into single DataFrame
    if all_events_list:
        consolidated_df = pd.concat(all_events_list, ignore_index=True)
        if "exright_date" in consolidated_df.columns:
            consolidated_df = consolidated_df.sort_values(["exright_date", "ticker"]).reset_index(drop=True)

        consolidated_path = os.path.join(output_dir, "vn30_all_corporate_events.csv")
        consolidated_df.to_csv(consolidated_path, index=False)
        logger.info(f"Saved consolidated corporate actions ({len(consolidated_df)} total events) -> {consolidated_path}")
    else:
        consolidated_df = pd.DataFrame()
        logger.warning("No corporate events were retrieved.")

    # Save summary report
    summary_path = os.path.join(output_dir, "corporate_events_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return consolidated_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch VN30 Corporate Events")
    parser.add_argument("--outdir", default="data/raw/corporate_actions", help="Output directory")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay seconds between calls")
    args = parser.parse_args()

    fetch_all_corporate_events(
        output_dir=args.outdir,
        delay_between_requests=args.delay
    )
