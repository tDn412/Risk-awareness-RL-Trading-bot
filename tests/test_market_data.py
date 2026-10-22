"""Tests for ``env/market_data.py`` loaders and scaler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from env.market_data import (
    FEATURE_COLUMNS,
    FeatureScaler,
    load_market_data,
    make_synthetic_market_data,
    market_data_from_frame,
)


def _long_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-01", periods=4)
    rows = []
    for ticker, listed_from in (("AAA", 0), ("BBB", 2)):
        for i, date in enumerate(dates):
            listed = i >= listed_from
            row = {"date": date, "ticker": ticker, "close": 100.0 + i if listed else np.nan, "is_listed": listed}
            row.update({col: float(i) for col in FEATURE_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_market_data_from_frame_shapes_and_mask() -> None:
    data = market_data_from_frame(_long_frame())
    assert data.tickers == ("AAA", "BBB")
    assert data.close.shape == (4, 2)
    assert data.features.shape == (4, 2, len(FEATURE_COLUMNS))
    assert data.is_listed[:, 1].tolist() == [False, False, True, True]
    assert np.isnan(data.close[0, 1])


def test_load_and_slice(tmp_path: Path) -> None:
    path = tmp_path / "fm.parquet"
    _long_frame().to_parquet(path)
    data = load_market_data(path, start="2021-01-05", end="2021-01-06")
    assert data.n_days == 2
    assert load_market_data(path).n_days == 4
    with pytest.raises(ValueError):
        data.slice_dates("2030-01-01")


def test_scaler_bounds_and_empty() -> None:
    data = make_synthetic_market_data(n_days=50, n_assets=4)
    scaled = data.fit_scaler().transform(data.features)
    assert np.all(np.abs(scaled) <= 1.0)
    empty = FeatureScaler.fit(data.features, np.zeros_like(data.is_listed))
    np.testing.assert_array_equal(empty.std, np.ones(data.n_features))
