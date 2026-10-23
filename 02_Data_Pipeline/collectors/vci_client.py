"""Minimal direct client for the Vietcap (VCI) daily OHLC chart endpoint.

Fallback used by the price and benchmark collectors when the ``vnstock`` package
cannot be imported (as of 2026-10, ``vnstock`` and its dependency ``vnai`` are no
longer published on PyPI). It calls the same public endpoint that
``vnstock.explorer.vci.quote.Quote.history`` uses and returns the same schema:
``[date, open, high, low, close, volume]`` with split/dividend-adjusted prices.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

VCI_CHART_URL = "https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart"
VCI_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://trading.vietcap.com.vn/",
    "Origin": "https://trading.vietcap.com.vn",
}
REQUEST_TIMEOUT_SECONDS = 30
# Upper bound on daily bars requested; filtered back to [start, end] afterwards.
COUNT_BACK_SAFETY_MARGIN = 10
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def fetch_vci_daily_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV bars for a stock or index symbol from the VCI chart API.

    Args:
        symbol: Ticker (e.g. ``"FPT"``) or index (``"VN30"``, ``"VNINDEX"``).
        start_date: Inclusive start date ``YYYY-MM-DD``.
        end_date: Inclusive end date ``YYYY-MM-DD``.

    Returns:
        DataFrame with columns ``[date, open, high, low, close, volume]``, sorted by date.
        Empty DataFrame (same columns) if the API returned no bars.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    end_stamp = int(datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc).timestamp())
    count_back = int(np.busday_count(start.date(), end.date())) + COUNT_BACK_SAFETY_MARGIN

    payload = {"timeFrame": "ONE_DAY", "symbols": [symbol], "to": end_stamp, "countBack": count_back}
    response = requests.post(VCI_CHART_URL, json=payload, headers=VCI_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    body = response.json()

    if not isinstance(body, list) or not body or not body[0].get("t"):
        logger.warning("[%s] VCI returned no bars.", symbol)
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    bars = body[0]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(bars["t"]).astype("int64"), unit="s").dt.normalize(),
            "open": pd.to_numeric(pd.Series(bars["o"]), errors="coerce"),
            "high": pd.to_numeric(pd.Series(bars["h"]), errors="coerce"),
            "low": pd.to_numeric(pd.Series(bars["l"]), errors="coerce"),
            "close": pd.to_numeric(pd.Series(bars["c"]), errors="coerce"),
            "volume": pd.to_numeric(pd.Series(bars["v"]), errors="coerce"),
        }
    )
    df["date"] = df["date"].astype("datetime64[ns]")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
