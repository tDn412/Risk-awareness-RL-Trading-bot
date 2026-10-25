"""Gymnasium environment for VN30 portfolio allocation under HOSE execution rules.

Timing convention (one step = one trading day ``t``):
    1. The agent observes features computed up to and including the close of day ``t``.
    2. The action (31 logits: cash + 30 stocks) is mapped to target weights with a
       temperature softmax; unlisted stocks are masked to zero weight. With
       ``rebalance_speed`` κ < 1 the portfolio only moves a fraction κ of the way from its
       current weights to that target (partial rebalancing), which damps day-to-day churn.
    3. Orders are executed at the closing auction (ATC) of day ``t`` under HOSE rules:
       * price band ±7% around the reference price (previous close): order price is
         clipped into the band; a stock closing at the ceiling cannot be bought and one
         closing at the floor cannot be sold (no counterparty);
       * round lots of 100 shares;
       * T+2 settlement: shares bought on day ``T`` cannot be sold on ``T`` or ``T+1``;
       * brokerage fee 0.15% on both sides and 0.1% sell tax.
       Sale proceeds are usable immediately (cash-advance, standard at VN brokers).
    4. The portfolio is marked to market at the close of ``t+1``; the net return of the
       step includes all transaction costs.

The reward scheme (A0–A3) is defined in ``env/rewards.py``.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from env.market_data import (
    DEFAULT_FEATURE_MATRIX_PATH,
    FeatureScaler,
    MarketData,
    load_market_data,
    make_synthetic_market_data,
)
from env.rewards import RewardConfig, compute_reward, historical_cvar

logger = logging.getLogger(__name__)

# HOSE market rules (AGENTS.md §3)
PRICE_LIMIT_PCT = 0.07
LOT_SIZE = 100
BROKERAGE_FEE_RATE = 0.0015
SELL_TAX_RATE = 0.001
SETTLEMENT_DAYS = 2
# A close within this relative distance of the band edge is treated as limit-locked.
LIMIT_LOCK_TOLERANCE = 0.002

DEFAULT_INITIAL_CASH = 1_000_000_000.0  # 1 billion VND
DEFAULT_SOFTMAX_TEMPERATURE = 0.2
DEFAULT_RISK_WINDOW = 60
DEFAULT_REBALANCE_SPEED = 1.0  # κ = 1: trade fully to the target every day
CASH_INDEX = 0


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one rebalance at the closing auction."""

    traded_value: float
    transaction_cost: float
    turnover: float


def clip_to_price_band(
    order_price: NDArray[np.float64],
    reference_price: NDArray[np.float64],
    limit_pct: float = PRICE_LIMIT_PCT,
) -> NDArray[np.float64]:
    """Clip order prices into the HOSE daily band ``[ref·(1-limit), ref·(1+limit)]``.

    Args:
        order_price: Desired execution prices.
        reference_price: Reference (previous close) prices.
        limit_pct: Daily band half-width (0.07 on HOSE).

    Returns:
        Prices guaranteed to lie inside the band.
    """
    clipped: NDArray[np.float64] = np.clip(
        order_price, reference_price * (1.0 - limit_pct), reference_price * (1.0 + limit_pct)
    )
    return clipped


class VN30PortfolioEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Long-only VN30 portfolio allocation environment with HOSE microstructure.

    Observation slices are documented in ``OBS_SCHEMA`` (filled per instance because the
    widths depend on ``N`` assets and ``F`` features); see ``obs_schema``.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    OBS_SCHEMA: dict[str, str] = {
        "features": "[N*F] tanh(z-score) of per-asset technical/risk features at day t, asset-major",
        "weights": "[N+1] current portfolio weights after drift; index 0 = cash",
        "drawdown": "[1] drawdown of NAV from running peak, in [0, 1]",
        "pending": "[N] fraction of each holding still unsettled (T+2), in [0, 1]",
        "listed": "[N] 1.0 if the asset is listed/tradable on day t else 0.0",
    }

    def __init__(
        self,
        data: MarketData | None = None,
        reward_config: RewardConfig | None = None,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
        episode_length: int | None = None,
        risk_window: int = DEFAULT_RISK_WINDOW,
        feature_scaler: FeatureScaler | None = None,
        rebalance_speed: float = DEFAULT_REBALANCE_SPEED,
    ) -> None:
        """Create the environment.

        Args:
            data: Market data; defaults to ``data/processed/feature_matrix.parquet`` or, if
                absent, a synthetic 30-asset market.
            reward_config: Reward scheme; defaults to A0.
            initial_cash: Starting capital in VND.
            softmax_temperature: Temperature ``τ`` of the logits → weights softmax.
            episode_length: Steps per episode with a random start; ``None`` runs the whole
                period from the first day.
            risk_window: Rolling window (days) for variance / CVaR in rewards and info.
            feature_scaler: Scaler fitted on training data; fitted on ``data`` if omitted.
            rebalance_speed: Fraction κ in ``(0, 1]`` of the gap between current and target
                weights traded each day; 1.0 rebalances fully.
        """
        super().__init__()
        if data is None:
            if DEFAULT_FEATURE_MATRIX_PATH.exists():
                data = load_market_data(DEFAULT_FEATURE_MATRIX_PATH)
            else:
                logger.warning("No processed data at %s; using synthetic market.", DEFAULT_FEATURE_MATRIX_PATH)
                data = make_synthetic_market_data()
        if data.n_days < 2:
            raise ValueError("Need at least 2 trading days.")
        if not 0.0 < rebalance_speed <= 1.0:
            raise ValueError("rebalance_speed must be in (0, 1].")
        if episode_length is not None and not 1 <= episode_length < data.n_days:
            raise ValueError(f"episode_length must be in [1, {data.n_days - 1}].")

        self.data = data
        self.reward_config = reward_config or RewardConfig()
        self.initial_cash = float(initial_cash)
        self.temperature = float(softmax_temperature)
        self.rebalance_speed = float(rebalance_speed)
        self.episode_length = episode_length
        self.risk_window = int(risk_window)

        scaler = feature_scaler or data.fit_scaler()
        n, f = data.n_assets, data.n_features
        self._scaled_features = scaler.transform(data.features).reshape(data.n_days, n * f).astype(np.float32)
        # Mark-to-market price: last valid close (0 before first listing).
        self._mark_prices = np.nan_to_num(_ffill_2d(data.close), nan=0.0)

        self.obs_schema: dict[str, slice] = {}
        offset = 0
        for name, width in (("features", n * f), ("weights", n + 1), ("drawdown", 1), ("pending", n), ("listed", n)):
            self.obs_schema[name] = slice(offset, offset + width)
            offset += width

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(offset,), dtype=np.float32)

        # Ledger state (initialised in reset)
        self._t = 0
        self._end = data.n_days - 1
        self._cash = self.initial_cash
        self._shares = np.zeros(n, dtype=np.int64)
        # _unsettled[k] = shares bought k trading days ago (k = 0 today, 1 yesterday).
        self._unsettled = np.zeros((SETTLEMENT_DAYS, n), dtype=np.int64)
        self._peak_value = self.initial_cash
        self._returns: deque[float] = deque(maxlen=self.risk_window)

    # ------------------------------------------------------------------ helpers
    @property
    def current_day(self) -> int:
        """Index of the current trading day in ``data``."""
        return self._t

    @property
    def cash(self) -> float:
        """Current cash balance (VND)."""
        return self._cash

    @property
    def shares(self) -> NDArray[np.int64]:
        """Copy of current share holdings per asset."""
        return self._shares.copy()

    def sellable_shares(self) -> NDArray[np.int64]:
        """Shares that have settled (T+2) and may be sold today."""
        sellable: NDArray[np.int64] = self._shares - self._unsettled.sum(axis=0)
        return sellable

    def portfolio_value(self) -> float:
        """NAV at the current day's mark prices."""
        return float(self._cash + self._shares @ self._mark_prices[self._t])

    def weights(self) -> NDArray[np.float64]:
        """Current weights ``[cash, asset_1..asset_N]`` at mark prices."""
        value = self.portfolio_value()
        holdings = self._shares * self._mark_prices[self._t]
        result: NDArray[np.float64] = np.concatenate([[self._cash], holdings]) / value
        return result

    def action_to_weights(self, action: NDArray[np.float32]) -> NDArray[np.float64]:
        """Temperature softmax over logits with unlisted assets masked to zero weight."""
        logits = np.asarray(action, dtype=np.float64) / self.temperature
        tradable = np.concatenate([[True], self.data.is_listed[self._t]])
        logits = np.where(tradable, logits, -np.inf)
        logits -= logits.max()
        exp = np.exp(logits)
        result: NDArray[np.float64] = exp / exp.sum()
        return result

    def _reference_prices(self) -> NDArray[np.float64]:
        prev = max(self._t - 1, 0)
        ref = self.data.close[prev]
        close = self.data.close[self._t]
        # First trading day of a newly listed stock has no previous close.
        out: NDArray[np.float64] = np.where(np.isnan(ref), close, ref)
        return out

    def _get_obs(self) -> NDArray[np.float32]:
        n = self.data.n_assets
        drawdown = 1.0 - self.portfolio_value() / self._peak_value
        pending_frac = np.divide(
            self._unsettled.sum(axis=0), self._shares, out=np.zeros(n), where=self._shares > 0
        )
        obs = np.concatenate(
            [
                self._scaled_features[self._t],
                self.weights(),
                [drawdown],
                pending_frac,
                self.data.is_listed[self._t].astype(np.float64),
            ]
        ).astype(np.float32)
        out: NDArray[np.float32] = np.clip(obs, -1.0, 1.0)
        return out

    # ------------------------------------------------------------------ execution
    def _execute(self, target_weights: NDArray[np.float64]) -> ExecutionResult:
        """Rebalance toward ``target_weights`` at today's close under HOSE rules."""
        t = self._t
        listed = self.data.is_listed[t]
        close = np.nan_to_num(self.data.close[t], nan=0.0)
        ref = np.nan_to_num(self._reference_prices(), nan=0.0)
        price = clip_to_price_band(close, ref)
        ceiling_locked = close >= ref * (1.0 + PRICE_LIMIT_PCT) * (1.0 - LIMIT_LOCK_TOLERANCE)
        floor_locked = close <= ref * (1.0 - PRICE_LIMIT_PCT) * (1.0 + LIMIT_LOCK_TOLERANCE)

        weights_before = self.weights()
        value = self.portfolio_value()
        safe_price = np.where(price > 0, price, 1.0)
        target_shares = np.where(
            listed, np.round(target_weights[1:] * value / (safe_price * LOT_SIZE)) * LOT_SIZE, self._shares
        ).astype(np.int64)
        delta = target_shares - self._shares

        # Sells first (frees cash). Holdings are always whole lots, so is sellable.
        sell_qty = np.where(listed & ~floor_locked & (delta < 0), np.minimum(-delta, self.sellable_shares()), 0)
        sell_value = float(sell_qty @ price)
        sell_cost = sell_value * (BROKERAGE_FEE_RATE + SELL_TAX_RATE)
        self._cash += sell_value - sell_cost
        self._shares -= sell_qty

        # Buys, scaled down (in whole lots) if cash is insufficient.
        buy_lots = np.where(listed & ~ceiling_locked & (delta > 0), delta // LOT_SIZE, 0)
        lot_cost = price * LOT_SIZE * (1.0 + BROKERAGE_FEE_RATE)
        required = float(buy_lots @ lot_cost)
        if required > self._cash and required > 0:
            buy_lots = np.floor(buy_lots * (self._cash / required)).astype(np.int64)
            while float(buy_lots @ lot_cost) > self._cash and buy_lots.any():
                buy_lots[np.argmax(buy_lots * lot_cost)] -= 1
        buy_qty = buy_lots * LOT_SIZE
        buy_value = float(buy_qty @ price)
        buy_cost = buy_value * BROKERAGE_FEE_RATE
        self._cash -= buy_value + buy_cost
        self._shares += buy_qty
        self._unsettled[0] += buy_qty

        weights_after = self.weights()
        turnover = 0.5 * float(np.abs(weights_after - weights_before).sum())
        return ExecutionResult(
            traded_value=sell_value + buy_value,
            transaction_cost=sell_cost + buy_cost,
            turnover=turnover,
        )

    def _advance_day(self) -> None:
        self._t += 1
        # Shares bought SETTLEMENT_DAYS ago settle; age the rest by one day.
        self._unsettled = np.roll(self._unsettled, 1, axis=0)
        self._unsettled[0] = 0

    # ------------------------------------------------------------------ gym API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the ledger to all-cash at the episode start day."""
        super().reset(seed=seed)
        if self.episode_length is None:
            start = 0
            self._end = self.data.n_days - 1
        else:
            start = int(self.np_random.integers(0, self.data.n_days - self.episode_length))
            self._end = start + self.episode_length
        self._t = start
        self._cash = self.initial_cash
        self._shares = np.zeros(self.data.n_assets, dtype=np.int64)
        self._unsettled = np.zeros((SETTLEMENT_DAYS, self.data.n_assets), dtype=np.int64)
        self._peak_value = self.initial_cash
        self._returns.clear()
        info = {"portfolio_value": self.initial_cash, "date": str(self.data.dates[self._t].date())}
        return self._get_obs(), info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Execute one trading day (see module docstring for timing)."""
        value_before = self.portfolio_value()
        target = self.action_to_weights(action)
        if self.rebalance_speed < 1.0:
            current = self.weights()
            target = current + self.rebalance_speed * (target - current)
        execution = self._execute(target)

        self._advance_day()
        value_after = self.portfolio_value()
        net_return = value_after / value_before - 1.0
        self._peak_value = max(self._peak_value, value_after)
        drawdown = 1.0 - value_after / self._peak_value
        self._returns.append(net_return)
        window = np.fromiter(self._returns, dtype=np.float64)

        reward = compute_reward(self.reward_config, net_return, drawdown, window, execution.turnover)
        terminated = self._t >= self._end
        info: dict[str, Any] = {
            "portfolio_value": value_after,
            "daily_return": net_return,
            "cvar_95": historical_cvar(window),
            "turnover": execution.turnover,
            "transaction_cost": execution.transaction_cost,
            "drawdown": drawdown,
            "cash_weight": float(self.weights()[CASH_INDEX]),
            "date": str(self.data.dates[self._t].date()),
        }
        return self._get_obs(), reward, terminated, False, info


def _ffill_2d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Forward-fill NaNs along axis 0 (time) of a ``[T, N]`` array."""
    out = values.copy()
    for i in range(1, out.shape[0]):
        missing = np.isnan(out[i])
        out[i, missing] = out[i - 1, missing]
    return out
