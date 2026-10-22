"""Market data container and loaders for ``VN30PortfolioEnv``.

Converts the long-format ``data/processed/feature_matrix.parquet`` (one row per
``date x ticker``) into dense ``[T, N]`` / ``[T, N, F]`` arrays that the environment
indexes by trading-day position. Also provides a deterministic synthetic market
generator used by unit tests and as a fallback when processed data is absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_MATRIX_PATH = REPO_ROOT / "data" / "processed" / "feature_matrix.parquet"

# Columns produced by 02_Data_Pipeline/processors/build_features.py
FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "rsi_14",
    "macd_hist",
    "bb_pctb",
    "bb_width",
    "atr_norm",
    "roll_vol_20",
    "downside_vol_20",
    "turnover_ratio",
)

MIN_FEATURE_STD = 1e-8
SYNTHETIC_DAILY_VOL = 0.02
SYNTHETIC_DAILY_DRIFT = 0.0003
SYNTHETIC_START_PRICE = 50_000.0
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class FeatureScaler:
    """Per-feature z-score statistics (fit on training data only to avoid leakage)."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]

    @classmethod
    def fit(cls, features: NDArray[np.float64], is_listed: NDArray[np.bool_]) -> FeatureScaler:
        """Fit mean/std over listed ``(day, asset)`` cells, ignoring NaNs.

        Args:
            features: ``[T, N, F]`` raw feature tensor.
            is_listed: ``[T, N]`` listing mask.

        Returns:
            Fitted scaler.
        """
        cells = features[is_listed]
        if cells.size == 0:
            n_features = features.shape[-1]
            return cls(mean=np.zeros(n_features), std=np.ones(n_features))
        mean = np.nanmean(cells, axis=0)
        std = np.nanstd(cells, axis=0)
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.where(np.isfinite(std) & (std > MIN_FEATURE_STD), std, 1.0)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    def transform(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Z-score then squash with ``tanh`` into ``[-1, 1]``; NaN becomes 0."""
        z = (features - self.mean) / self.std
        out: NDArray[np.float64] = np.tanh(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0))
        return out


@dataclass(frozen=True)
class MarketData:
    """Dense, date-indexed market arrays for a fixed universe.

    Attributes:
        dates: Trading calendar, length ``T``.
        tickers: Asset symbols, length ``N``.
        close: ``[T, N]`` adjusted close; NaN where the asset is not listed.
        features: ``[T, N, F]`` raw (unscaled) features.
        is_listed: ``[T, N]`` True when the asset trades on that day.
        feature_names: Names of the ``F`` feature channels.
    """

    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    close: NDArray[np.float64]
    features: NDArray[np.float64]
    is_listed: NDArray[np.bool_]
    feature_names: tuple[str, ...]

    @property
    def n_days(self) -> int:
        """Number of trading days ``T``."""
        return len(self.dates)

    @property
    def n_assets(self) -> int:
        """Number of assets ``N``."""
        return len(self.tickers)

    @property
    def n_features(self) -> int:
        """Number of feature channels ``F``."""
        return len(self.feature_names)

    def slice_dates(self, start: str | None = None, end: str | None = None) -> MarketData:
        """Return a copy restricted to ``start <= date <= end`` (inclusive)."""
        mask = np.ones(self.n_days, dtype=bool)
        if start is not None:
            mask &= self.dates >= pd.Timestamp(start)
        if end is not None:
            mask &= self.dates <= pd.Timestamp(end)
        if not mask.any():
            raise ValueError(f"No trading days between {start} and {end}.")
        return replace(
            self,
            dates=self.dates[mask],
            close=self.close[mask],
            features=self.features[mask],
            is_listed=self.is_listed[mask],
        )

    def fit_scaler(self) -> FeatureScaler:
        """Fit a ``FeatureScaler`` on this data (call on the training slice only)."""
        return FeatureScaler.fit(self.features, self.is_listed)


def market_data_from_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> MarketData:
    """Build ``MarketData`` from a long-format feature matrix.

    Args:
        frame: Columns ``date, ticker, close, is_listed`` plus ``feature_columns``.
        feature_columns: Feature channels to extract.

    Returns:
        Dense ``MarketData`` with tickers sorted alphabetically.
    """
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    tickers = tuple(sorted(df["ticker"].astype(str).unique()))

    def pivot(column: str) -> pd.DataFrame:
        return df.pivot(index="date", columns="ticker", values=column).reindex(index=dates, columns=list(tickers))

    is_listed = pivot("is_listed").fillna(False).to_numpy(dtype=bool)
    close = pivot("close").to_numpy(dtype=np.float64)
    close = np.where(is_listed, close, np.nan)
    features = np.stack([pivot(col).to_numpy(dtype=np.float64) for col in feature_columns], axis=-1)

    return MarketData(
        dates=dates,
        tickers=tickers,
        close=close,
        features=features,
        is_listed=is_listed,
        feature_names=feature_columns,
    )


def load_market_data(
    path: Path | str = DEFAULT_FEATURE_MATRIX_PATH,
    start: str | None = None,
    end: str | None = None,
) -> MarketData:
    """Load the processed feature matrix Parquet into ``MarketData``.

    Args:
        path: Path to ``feature_matrix.parquet``.
        start: Optional inclusive start date.
        end: Optional inclusive end date.

    Returns:
        ``MarketData`` restricted to the requested period.
    """
    data = market_data_from_frame(pd.read_parquet(path))
    logger.info("Loaded market data: %d days x %d assets from %s", data.n_days, data.n_assets, path)
    if start is None and end is None:
        return data
    return data.slice_dates(start, end)


def make_synthetic_market_data(
    n_days: int = 300,
    n_assets: int = 30,
    seed: int = 0,
    constant_prices: bool = False,
) -> MarketData:
    """Generate a deterministic synthetic market (geometric Brownian motion).

    Args:
        n_days: Number of trading days.
        n_assets: Number of assets.
        seed: RNG seed.
        constant_prices: If True, every close equals ``SYNTHETIC_START_PRICE`` (for fee tests).

    Returns:
        Synthetic ``MarketData`` with all assets listed on all days.
    """
    rng = np.random.default_rng(seed)
    if constant_prices:
        close = np.full((n_days, n_assets), SYNTHETIC_START_PRICE)
    else:
        log_ret = rng.normal(SYNTHETIC_DAILY_DRIFT, SYNTHETIC_DAILY_VOL, size=(n_days, n_assets))
        log_ret[0] = 0.0
        close = SYNTHETIC_START_PRICE * np.exp(np.cumsum(log_ret, axis=0))

    returns = np.vstack([np.zeros((1, n_assets)), np.diff(np.log(close), axis=0)])
    noise = rng.normal(0.0, 0.01, size=(n_days, n_assets, len(FEATURE_COLUMNS)))
    features = noise + returns[:, :, None]

    return MarketData(
        dates=pd.bdate_range("2020-01-01", periods=n_days),
        tickers=tuple(f"S{i:02d}" for i in range(n_assets)),
        close=close.astype(np.float64),
        features=features.astype(np.float64),
        is_listed=np.ones((n_days, n_assets), dtype=bool),
        feature_names=FEATURE_COLUMNS,
    )
