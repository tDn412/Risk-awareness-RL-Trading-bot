"""
Benchmark Indices Collector (VN30 & VN-Index)
Project: Risk-Aware RL for VN30 Portfolio Allocation (INT3011E)

Fetches daily historical prices for market benchmark indices (VN30-Index, VNINDEX)
from 2018 to 2025 using vnstock API. Used as baseline references for CAGR, MDD, and Sharpe.
"""

import os
import sys
import time
import logging
from typing import List, Optional
import pandas as pd

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
logger = logging.getLogger("BenchmarkCollector")

DEFAULT_INDICES = ["VN30", "VNINDEX"]


def fetch_benchmark_indices(
    indices: Optional[List[str]] = None,
    start_date: str = "2018-01-01",
    end_date: str = "2025-12-31",
    output_dir: str = "data/raw/benchmarks",
    source: str = "VCI",
    delay_between_requests: float = 1.0
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily benchmark index history and save each as CSV.

    Args:
        indices: List of index symbols (e.g. ['VN30', 'VNINDEX']).
        start_date: Start date string.
        end_date: End date string.
        output_dir: Directory to save CSVs.
        source: Data source.
        delay_between_requests: Rate limit delay.

    Returns:
        Dict mapping index symbol to DataFrame.
    """
    if indices is None:
        indices = DEFAULT_INDICES

    if Quote is None:
        raise ImportError("vnstock library is not installed.")

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for idx, symbol in enumerate(indices, 1):
        logger.info(f"[{idx}/{len(indices)}] Fetching benchmark index: {symbol}...")
        try:
            q = Quote(symbol=symbol, source=source)
            df = q.history(start=start_date, end=end_date, interval="1D")

            if df is not None and not df.empty:
                rename_map = {
                    "time": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume"
                }
                df = df.rename(columns=rename_map)
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                df["ticker"] = symbol

                out_path = os.path.join(output_dir, f"{symbol}.csv")
                df.to_csv(out_path, index=False)
                results[symbol] = df
                logger.info(f"[{symbol}] Saved {len(df)} rows -> {out_path}")
            else:
                logger.warning(f"[{symbol}] No data returned.")

        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch benchmark: {e}")

        time.sleep(delay_between_requests)

    return results


if __name__ == "__main__":
    fetch_benchmark_indices()
