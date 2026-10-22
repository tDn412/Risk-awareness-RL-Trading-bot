from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from processors.build_features import compute_ticker_features


@pytest.fixture
def sample_ticker_df() -> pd.DataFrame:
    """Fixture providing a 100-day synthetic ticker DataFrame."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    # Simulated geometric random walk
    returns = np.random.normal(0.0005, 0.015, size=100)
    prices = 100.0 * np.exp(np.cumsum(returns))
    volumes = np.random.randint(50000, 200000, size=100)

    df = pd.DataFrame({
        "date": dates,
        "ticker": ["TEST_FEAT"] * 100,
        "close": prices,
        "volume": volumes,
        "is_listed": [True] * 100,
    })
    return df


def test_output_schema(sample_ticker_df: pd.DataFrame) -> None:
    """Verify that all 11 required features are generated."""
    features_df = compute_ticker_features(sample_ticker_df)

    expected_features = [
        "ret_1d", "ret_5d", "ret_20d", "rsi_14", "macd_hist",
        "bb_pctb", "bb_width", "atr_norm", "roll_vol_20",
        "downside_vol_20", "turnover_ratio"
    ]

    for col in expected_features:
        assert col in features_df.columns
        assert np.issubdtype(features_df[col].dtype, np.floating)


def test_no_lookahead(sample_ticker_df: pd.DataFrame) -> None:
    """Verify that features computed at time t do not depend on data after t."""
    full_features = compute_ticker_features(sample_ticker_df)

    # Cut dataset at t=50 and recompute
    truncated_df = sample_ticker_df.iloc[:51].copy()
    truncated_features = compute_ticker_features(truncated_df)

    # Features at t=50 must match exactly between full and truncated
    for col in ["ret_1d", "ret_5d", "rsi_14", "macd_hist", "bb_pctb"]:
        val_full = full_features.loc[50, col]
        val_trunc = truncated_features.loc[50, col]
        np.testing.assert_allclose(val_full, val_trunc, rtol=1e-5, atol=1e-5)


def test_unlisted_isolation() -> None:
    """Ensure that rows where is_listed=False have zeroed features."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["LATE"] * 30,
        "close": [10.0] * 30,
        "volume": [1000] * 30,
        "is_listed": [False] * 10 + [True] * 20,
    })

    features_df = compute_ticker_features(df)
    feature_cols = [
        "ret_1d", "ret_5d", "ret_20d", "rsi_14", "macd_hist",
        "bb_pctb", "bb_width", "atr_norm", "roll_vol_20",
        "downside_vol_20", "turnover_ratio"
    ]

    for i in range(10):
        for col in feature_cols:
            assert features_df.loc[i, col] == 0.0


def test_process_build_features_flow(tmp_path: Path) -> None:
    """Test full process_build_features execution on sample parquet."""
    from processors.build_features import process_build_features

    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    aligned_df = pd.DataFrame({
        "date": list(dates) * 2,
        "ticker": ["T1"] * 25 + ["T2"] * 25,
        "close": [10.0 + i * 0.1 for i in range(25)] + [20.0 - i * 0.1 for i in range(25)],
        "volume": [1000] * 50,
        "adj_factor": [1.0] * 50,
        "is_listed": [True] * 50,
    })

    in_parquet = tmp_path / "aligned.parquet"
    out_parquet = tmp_path / "features.parquet"
    aligned_df.to_parquet(in_parquet, index=False)

    res_df = process_build_features(
        aligned_parquet_path=in_parquet,
        output_feature_path=out_parquet
    )

    assert out_parquet.exists()
    assert len(res_df) == 50
    assert "ret_1d" in res_df.columns

