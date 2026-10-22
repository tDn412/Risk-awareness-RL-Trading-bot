"""Unit tests for clean_alignment processor.

Validates schema, forward-fill rules (gap <= 5 days), gap exceptions (> 5 days),
and pre-listing mask isolation (no look-ahead).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from processors.clean_alignment import (
    align_ticker_prices,
    get_trading_calendar,
    process_clean_alignment,
)


@pytest.fixture
def sample_calendar() -> pd.DatetimeIndex:
    """Fixture providing a 10-day consecutive trading calendar."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    return pd.DatetimeIndex(dates)


def test_gap_ffill_short(sample_calendar: pd.DatetimeIndex) -> None:
    """Test that gaps <= 5 trading days are forward filled properly."""
    # Data has day 0, day 4 (gap of 3 days: day 1, 2, 3 missing)
    dates = [sample_calendar[0], sample_calendar[4]]
    raw_df = pd.DataFrame({
        "date": dates,
        "close": [10.0, 12.0],
        "volume": [1000, 2000],
        "ticker": ["TEST", "TEST"],
    })

    aligned = align_ticker_prices(
        ticker="TEST",
        raw_df=raw_df,
        calendar=sample_calendar,
        max_gap_days=5,
    )

    # Days 1, 2, 3 should be ffilled with 10.0
    for i in range(1, 4):
        assert aligned.loc[aligned["date"] == sample_calendar[i], "close"].values[0] == 10.0
        assert aligned.loc[aligned["date"] == sample_calendar[i], "is_listed"].values[0] is True or aligned.loc[aligned["date"] == sample_calendar[i], "is_listed"].values[0] == True


def test_gap_raises_long(sample_calendar: pd.DatetimeIndex) -> None:
    """Test that gaps > 5 trading days raise a ValueError."""
    # Day 0 and day 7 (gap of 6 days: days 1, 2, 3, 4, 5, 6 missing)
    dates = [sample_calendar[0], sample_calendar[7]]
    raw_df = pd.DataFrame({
        "date": dates,
        "close": [10.0, 15.0],
        "volume": [1000, 2000],
        "ticker": ["GAP_TEST", "GAP_TEST"],
    })

    with pytest.raises(ValueError, match="exceeds max gap limit"):
        align_ticker_prices(
            ticker="GAP_TEST",
            raw_df=raw_df,
            calendar=sample_calendar,
            max_gap_days=5,
        )


def test_output_schema(sample_calendar: pd.DatetimeIndex) -> None:
    """Validate schema columns and data types match AGENTS.md specification."""
    raw_df = pd.DataFrame({
        "date": [sample_calendar[0], sample_calendar[1]],
        "close": [25.5, 26.0],
        "volume": [10000, 12000],
        "adj_factor": [1.0, 1.0],
        "ticker": ["SCHEMA_TEST", "SCHEMA_TEST"],
    })

    aligned = align_ticker_prices(
        ticker="SCHEMA_TEST",
        raw_df=raw_df,
        calendar=sample_calendar,
        max_gap_days=5,
    )

    expected_cols = ["date", "ticker", "close", "volume", "adj_factor", "is_listed"]
    assert list(aligned.columns) == expected_cols

    assert pd.api.types.is_datetime64_ns_dtype(aligned["date"]) or pd.api.types.is_datetime64_any_dtype(aligned["date"])
    assert aligned["ticker"].dtype == object or str(aligned["ticker"].dtype) == "string"
    assert aligned["close"].dtype == np.float64
    assert aligned["volume"].dtype == np.int64
    assert aligned["adj_factor"].dtype == np.float64
    assert aligned["is_listed"].dtype == bool


def test_pre_listing_not_ffilled(sample_calendar: pd.DatetimeIndex) -> None:
    """Ensure pre-listing dates (e.g. SSB) remain is_listed=False and are not ffilled."""
    # Listed starting at day 5
    raw_df = pd.DataFrame({
        "date": [sample_calendar[5], sample_calendar[6]],
        "close": [50.0, 52.0],
        "volume": [5000, 6000],
        "ticker": ["LATE_LIST", "LATE_LIST"],
    })

    aligned = align_ticker_prices(
        ticker="LATE_LIST",
        raw_df=raw_df,
        calendar=sample_calendar,
        max_gap_days=5,
    )

    # Days 0 to 4 must be is_listed = False
    for i in range(5):
        row = aligned.loc[aligned["date"] == sample_calendar[i]]
        assert bool(row["is_listed"].values[0]) is False
        assert pd.isna(row["close"].values[0])


def test_process_clean_alignment_flow(tmp_path: Path) -> None:
    """Test full process_clean_alignment pipeline with temporary directory."""
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()

    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    vn30_df = pd.DataFrame({"date": dates, "close": [1000.0] * 5, "volume": [100000] * 5, "ticker": ["VN30"] * 5})
    vn30_path = benchmarks_dir / "VN30.csv"
    vn30_df.to_csv(vn30_path, index=False)

    stock_df = pd.DataFrame({"date": dates, "close": [50.0] * 5, "volume": [10000] * 5, "ticker": ["AAA"] * 5})
    (prices_dir / "AAA.csv").write_text(stock_df.to_csv(index=False))

    output_parquet = tmp_path / "aligned.parquet"
    result_df = process_clean_alignment(
        prices_dir=prices_dir,
        benchmark_path=vn30_path,
        output_path=output_parquet
    )

    assert output_parquet.exists()
    assert len(result_df) == 5
    assert set(result_df["ticker"].unique()) == {"AAA"}

