"""Out-of-sample backtester: trained agents vs the five SSOT baselines (SOURCE_OF_TRUTH §7).

Every strategy except the index buy-and-hold is simulated through ``VN30PortfolioEnv``, so
agents and baselines face identical HOSE rules (T+2, ±7% band, 100-share lots, fees, tax).

Baselines:
    1. Buy & Hold VN30-Index (frictionless index returns)
    2. Equal Weight 1/N, monthly rebalance, 0% cash
    3. 100% Cash
    4. Static 60/40 stock-cash, monthly rebalance
    5. Equivalent Equity Benchmark (EEB): per run, equal-weight stocks scaled to that run's
       daily equity exposure (1 - agent cash weight); isolates stock-selection/timing skill
       from simple de-risking (hypothesis H3).

Example:
    python 04_Backtesting_Evaluation/backtester.py --window window_1
"""

from __future__ import annotations

import torch  # noqa: F401  (must load before numpy/pandas DLLs on Windows)
import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("03_Model_Development", "04_Backtesting_Evaluation"):
    if str(REPO_ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _sub))

from agents import AGENTS  # noqa: E402
from env import RewardConfig, VN30PortfolioEnv, load_market_data  # noqa: E402
from env.market_data import FeatureScaler, MarketData  # noqa: E402
from metrics import evaluate_portfolio, sortino_ratio  # noqa: E402
from rollout import rollout  # noqa: E402

logger = logging.getLogger(__name__)

MODELS_DIR = REPO_ROOT / "outputs" / "models"
RESULTS_DIR = REPO_ROOT / "outputs" / "backtest_results"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
BENCHMARK_PATH = REPO_ROOT / "data" / "raw" / "benchmarks" / "VN30.csv"
CONFIG_DIR = REPO_ROOT / "03_Model_Development" / "configs"

REBALANCE_DAYS = 21  # ~monthly
TRADING_DAYS_PER_YEAR = 252
MIN_WEIGHT = 1e-12
STOCK_FRACTION_60_40 = 0.6
INDEX_NAME = "VN30 Index (Buy&Hold)"
EQUAL_WEIGHT_NAME = "Equal Weight 1/N"
CASH_NAME = "100% Cash"
STATIC_NAME = "Static 60/40"
REWARD_COLORS = {"A0": "#2a78d6", "A1": "#eb6834", "A2": "#1baf7a", "A3": "#e87ba4"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


@dataclass(frozen=True)
class Ledger:
    """Daily results of one strategy run."""

    name: str
    nav: pd.Series
    daily_return: pd.Series
    cash_weight: pd.Series | None = None
    turnover: pd.Series | None = None
    transaction_cost: pd.Series | None = None


class WeightPolicy:
    """Policy that rebalances to a target-weight function every ``rebalance_days`` steps.

    Between rebalances it *holds*: the logits reproduce the current weights, so the env
    performs no trades.
    """

    def __init__(
        self,
        env: VN30PortfolioEnv,
        target_fn: Callable[[int, NDArray[np.bool_]], NDArray[np.float64]],
        rebalance_days: int = REBALANCE_DAYS,
    ) -> None:
        self.env = env
        self.target_fn = target_fn
        self.rebalance_days = rebalance_days
        self._step = 0

    def predict(self, obs: NDArray[np.float32], deterministic: bool = True) -> NDArray[np.float32]:
        """Return logits for the target weights (or current weights when holding)."""
        schema = self.env.obs_schema
        listed = obs[schema["listed"]] > 0.5
        if self._step % self.rebalance_days == 0:
            weights = self.target_fn(self._step, listed)
        else:
            weights = np.asarray(obs[schema["weights"]], dtype=np.float64)
        self._step += 1
        logits: NDArray[np.float32] = (self.env.temperature * np.log(np.maximum(weights, MIN_WEIGHT))).astype(
            np.float32
        )
        return logits


def equal_weight_target(equity_fraction: float) -> Callable[[int, NDArray[np.bool_]], NDArray[np.float64]]:
    """Target: ``equity_fraction`` spread equally over listed stocks, remainder in cash."""

    def target(_: int, listed: NDArray[np.bool_]) -> NDArray[np.float64]:
        weights = np.zeros(len(listed) + 1)
        n_listed = int(listed.sum())
        if n_listed == 0:
            weights[0] = 1.0
            return weights
        weights[1:][listed] = equity_fraction / n_listed
        weights[0] = 1.0 - equity_fraction
        return weights

    return target


def eeb_target(equity_fractions: NDArray[np.float64]) -> Callable[[int, NDArray[np.bool_]], NDArray[np.float64]]:
    """Target for the Equivalent Equity Benchmark: follows a per-day equity fraction schedule."""

    def target(step: int, listed: NDArray[np.bool_]) -> NDArray[np.float64]:
        fraction = float(equity_fractions[min(step, len(equity_fractions) - 1)])
        return equal_weight_target(fraction)(step, listed)

    return target


def _ledger_from_frame(name: str, frame: pd.DataFrame) -> Ledger:
    return Ledger(
        name=name,
        nav=frame["nav"],
        daily_return=frame["daily_return"],
        cash_weight=frame["cash_weight"],
        turnover=frame["turnover"],
        transaction_cost=frame["transaction_cost"],
    )


def simulate_baseline(
    name: str,
    data: MarketData,
    env_kwargs: dict[str, Any],
    target_fn: Callable[[int, NDArray[np.bool_]], NDArray[np.float64]],
    rebalance_days: int = REBALANCE_DAYS,
) -> Ledger:
    """Run a target-weight baseline through the HOSE environment."""
    env = VN30PortfolioEnv(data, **env_kwargs)
    return _ledger_from_frame(name, rollout(WeightPolicy(env, target_fn, rebalance_days), env))


def index_ledger(dates: pd.DatetimeIndex, initial_cash: float, benchmark_path: Path = BENCHMARK_PATH) -> Ledger:
    """Frictionless buy-and-hold of the VN30 index on the given dates."""
    bench = pd.read_csv(benchmark_path, parse_dates=["date"]).set_index("date")["close"].reindex(dates)
    nav = initial_cash * bench / bench.iloc[0]
    return Ledger(name=INDEX_NAME, nav=nav, daily_return=nav.pct_change().fillna(0.0))


def cash_ledger(dates: pd.DatetimeIndex, initial_cash: float) -> Ledger:
    """Constant NAV (0% interest)."""
    nav = pd.Series(initial_cash, index=dates)
    return Ledger(
        name=CASH_NAME,
        nav=nav,
        daily_return=pd.Series(0.0, index=dates),
        cash_weight=pd.Series(1.0, index=dates),
    )


def compute_metrics(ledger: Ledger) -> dict[str, float]:
    """SSOT §7.2 metrics plus cash weight, turnover and annual friction."""
    metrics: dict[str, float] = dict(evaluate_portfolio(ledger.daily_return, ledger.nav, risk_free_rate=0.0))
    metrics["Sortino Ratio"] = round(sortino_ratio(ledger.daily_return), 3)
    metrics["Avg Cash Weight"] = round(float(ledger.cash_weight.mean()), 3) if ledger.cash_weight is not None else np.nan
    metrics["Avg Turnover"] = round(float(ledger.turnover.mean()), 4) if ledger.turnover is not None else np.nan
    if ledger.transaction_cost is not None:
        friction = (ledger.transaction_cost / ledger.nav.shift(1)).fillna(0.0).sum()
        metrics["Friction (%/yr)"] = round(float(friction) * TRADING_DAYS_PER_YEAR / len(ledger.nav) * 100, 2)
    else:
        metrics["Friction (%/yr)"] = np.nan
    return metrics


@dataclass(frozen=True)
class RunSpec:
    """A finished training run to evaluate."""

    run_id: str
    agent: str
    reward_mode: str
    seed: int
    window: str
    checkpoint: Path
    config: dict[str, Any]


def discover_runs(window: str, checkpoint: str = "best", models_dir: Path = MODELS_DIR) -> list[RunSpec]:
    """Collect non-dry-run training summaries for ``window``."""
    runs = []
    for path in sorted(models_dir.glob("*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("dry_run") or summary["window"] != window:
            continue
        runs.append(
            RunSpec(
                run_id=summary["run_id"],
                agent=summary["agent"],
                reward_mode=summary["reward_mode"],
                seed=int(summary["seed"]),
                window=window,
                checkpoint=Path(summary[f"{checkpoint}_checkpoint"]),
                config=summary["config"],
            )
        )
    return runs


def run_backtest(
    window: str, checkpoint: str = "best", models_dir: Path = MODELS_DIR
) -> tuple[dict[str, Ledger], pd.DataFrame]:
    """Backtest all runs of ``window`` on its held-out test period.

    Returns:
        ``(ledgers, results)`` where ledgers maps ``"<run_id>"`` / baseline names to ledgers
        and ``results`` has one metrics row per strategy run (with ``group`` and ``kind``).
    """
    runs = discover_runs(window, checkpoint, models_dir)
    if not runs:
        raise FileNotFoundError(f"No finished runs for {window} in {models_dir}")
    config = runs[0].config
    bounds = config["data"]["windows"][window]
    full = load_market_data(REPO_ROOT / config["data"]["feature_matrix"])
    scaler: FeatureScaler = full.slice_dates(*bounds["train"]).fit_scaler()
    test = full.slice_dates(*bounds["test"])
    env_cfg = config["env"]
    env_kwargs: dict[str, Any] = {
        "initial_cash": float(env_cfg["initial_cash"]),
        "softmax_temperature": float(env_cfg["softmax_temperature"]),
        "risk_window": int(env_cfg["risk_window"]),
        "feature_scaler": scaler,
        "no_trade_band": float(env_cfg.get("no_trade_band", 0.0)),
        "action_smoothing": float(env_cfg.get("action_smoothing", 1.0)),
    }
    logger.info("Test period %s..%s (%d days)", test.dates[0].date(), test.dates[-1].date(), test.n_days)

    ledgers: dict[str, Ledger] = {}
    rows: list[dict[str, Any]] = []

    def add(key: str, ledger: Ledger, group: str, kind: str, seed: int | None = None) -> None:
        ledgers[key] = ledger
        rows.append({"key": key, "group": group, "kind": kind, "seed": seed, **compute_metrics(ledger)})

    initial = env_kwargs["initial_cash"]
    dates: pd.DatetimeIndex | None = None
    for run in runs:
        reward = RewardConfig(**{**run.config["reward"], "mode": run.reward_mode})
        env = VN30PortfolioEnv(test, reward_config=reward, **env_kwargs)
        agent = AGENTS[run.agent].load(run.checkpoint)
        ledger = _ledger_from_frame(run.run_id, rollout(agent, env))
        dates = pd.DatetimeIndex(ledger.nav.index)
        group = f"{run.agent.upper()} {run.reward_mode}"
        add(run.run_id, ledger, group, "agent", run.seed)

        # Equivalent Equity Benchmark: same daily equity exposure, equal-weight stocks, daily rebalance.
        assert ledger.cash_weight is not None
        equity = 1.0 - ledger.cash_weight.to_numpy(dtype=np.float64)[1:]
        eeb = simulate_baseline(f"EEB {run.run_id}", test, env_kwargs, eeb_target(equity), rebalance_days=1)
        add(f"EEB {run.run_id}", eeb, f"EEB of {group}", "eeb", run.seed)
        p_value = float(stats.ttest_rel(ledger.daily_return.iloc[1:], eeb.daily_return.iloc[1:]).pvalue)
        rows[-2]["p_vs_EEB"] = p_value
        rows[-2]["mean_daily_excess_vs_EEB (%)"] = float(
            (ledger.daily_return.iloc[1:].to_numpy() - eeb.daily_return.iloc[1:].to_numpy()).mean() * 100
        )

    assert dates is not None
    add(INDEX_NAME, index_ledger(dates, initial), INDEX_NAME, "baseline")
    add(EQUAL_WEIGHT_NAME, simulate_baseline(EQUAL_WEIGHT_NAME, test, env_kwargs, equal_weight_target(1.0)),
        EQUAL_WEIGHT_NAME, "baseline")
    add(CASH_NAME, cash_ledger(dates, initial), CASH_NAME, "baseline")
    add(STATIC_NAME, simulate_baseline(STATIC_NAME, test, env_kwargs, equal_weight_target(STOCK_FRACTION_60_40)),
        STATIC_NAME, "baseline")
    return ledgers, pd.DataFrame(rows)


SUMMARY_COLUMNS = [
    "Total Return (%)", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)", "Calmar Ratio",
    "CVaR (95%)", "Annualized Volatility (%)", "Avg Cash Weight", "Avg Turnover", "Friction (%/yr)",
]


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std across seeds per strategy group (single-run groups show std 0)."""
    grouped = results.groupby("group", sort=False)
    mean = grouped[SUMMARY_COLUMNS].mean()
    std = grouped[SUMMARY_COLUMNS].std().fillna(0.0)
    table = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
    table.insert(0, "n", grouped.size())
    if "p_vs_EEB" in results:
        table["p vs EEB (mean)"] = grouped["p_vs_EEB"].mean().round(3)
    return table.reset_index()


def mean_curves(ledgers: dict[str, Ledger], results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Seed-averaged NAV (indexed to 100), drawdown and cash weight per group."""
    curves: dict[str, pd.DataFrame] = {}
    for group, rows in results.groupby("group", sort=False):
        if rows["kind"].iloc[0] == "eeb":
            continue
        navs = pd.concat([ledgers[k].nav / ledgers[k].nav.iloc[0] * 100 for k in rows["key"]], axis=1)
        nav = navs.mean(axis=1)
        frame = pd.DataFrame({"nav": nav, "drawdown": (nav / nav.cummax() - 1.0) * 100})
        cash = [ledgers[k].cash_weight for k in rows["key"] if ledgers[k].cash_weight is not None]
        frame["cash_weight"] = pd.concat(cash, axis=1).mean(axis=1) * 100 if cash else np.nan
        curves[str(group)] = frame
    return curves


def plot_curves(curves: dict[str, pd.DataFrame], window: str, path: Path) -> None:
    """NAV, drawdown and cash-weight panels (agents coloured by reward, baselines neutral)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 2, 1.5]})
    styles: dict[str, tuple[str, str]] = {INDEX_NAME: (INK_PRIMARY, "-"), EQUAL_WEIGHT_NAME: (INK_SECONDARY, "--")}
    for name, frame in curves.items():
        if name in (CASH_NAME, STATIC_NAME):
            continue
        reward = name.split()[-1]
        color, style = styles.get(name, (REWARD_COLORS.get(reward, INK_SECONDARY), ":" if name.startswith("SAC") else "-"))
        for ax, column in zip(axes, ("nav", "drawdown", "cash_weight")):
            if column == "cash_weight" and name not in {n for n in curves if n.split()[-1] in REWARD_COLORS}:
                continue
            series = frame[column].iloc[1:] if column == "cash_weight" else frame[column]  # day 0 is all-cash
            ax.plot(series.index, series, color=color, linestyle=style, linewidth=2 if style == "-" else 1.5,
                    label=name)
    for ax, title in zip(axes, ("NAV (start = 100)", "Drawdown (%)", "Agent cash weight (%)")):
        ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=11)
        ax.set_facecolor(SURFACE)
        ax.grid(color="#e4e3df", linewidth=0.6)
        ax.tick_params(colors=INK_SECONDARY)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_PRIMARY, ncol=2)
    fig.suptitle(f"Out-of-sample test, {window} (seed-averaged)", color=INK_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", default="window_1")
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ledgers, results = run_backtest(args.window, args.checkpoint)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.window}_test"
    results.drop(columns=["key"]).to_csv(RESULTS_DIR / f"{stem}_runs.csv", index=False)
    table = summarize(results)
    markdown = table.to_markdown(index=False)
    (RESULTS_DIR / f"{stem}_summary.md").write_text(markdown + "\n", encoding="utf-8")
    curves = mean_curves(ledgers, results)
    pd.concat(curves, axis=1).to_csv(RESULTS_DIR / f"{stem}_curves.csv")
    plot_curves(curves, args.window, FIGURES_DIR / f"{stem}_nav.png")
    logger.info("\n%s", markdown)
    logger.info("Saved results to %s and figure to %s", RESULTS_DIR, FIGURES_DIR / f"{stem}_nav.png")


if __name__ == "__main__":
    main()
