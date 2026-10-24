"""Training entrypoint for PPO / SAC agents on ``VN30PortfolioEnv``.

Example:
    python 03_Model_Development/train.py --config ppo_baseline.yaml --reward A2 --seed 42
    python 03_Model_Development/train.py --config ppo_baseline.yaml --dry-run

Per run (``run_id = <agent>_<reward>_<window>_s<seed>_<timestamp>``) it writes:
    outputs/models/<run_id>.zip       final checkpoint
    outputs/models/<run_id>_best.zip  checkpoint with best validation Sharpe ratio
    outputs/models/<run_id>.json      config snapshot + validation metrics
    outputs/logs/<run_id>/            TensorBoard logs (incl. rollout/ep_rew_mean)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

MODEL_DIR_NAME = "03_Model_Development"
REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in (MODEL_DIR_NAME, "04_Backtesting_Evaluation"):
    if str(REPO_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / sub))

from agents import AGENTS  # noqa: E402
from callbacks import SharpeCheckpointCallback  # noqa: E402
from env import RewardConfig, VN30PortfolioEnv, load_market_data, make_synthetic_market_data  # noqa: E402
from env.market_data import FeatureScaler, MarketData  # noqa: E402
from metrics import evaluate_portfolio  # noqa: E402
from rollout import rollout  # noqa: E402

logger = logging.getLogger(__name__)

CONFIG_DIR = REPO_ROOT / MODEL_DIR_NAME / "configs"
MODELS_DIR = REPO_ROOT / "outputs" / "models"
LOGS_DIR = REPO_ROOT / "outputs" / "logs"
DRY_RUN_EVAL_FREQ = 500
DRY_RUN_N_STEPS = 128
DRY_RUN_SYNTHETIC_DAYS = 400


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config by file name (looked up in ``configs/``) or path."""
    path = Path(name)
    if not path.exists():
        path = CONFIG_DIR / name
    with path.open(encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)
    return config


def build_env(
    data: MarketData,
    config: dict[str, Any],
    reward: RewardConfig,
    scaler: FeatureScaler,
    full_period: bool,
) -> VN30PortfolioEnv:
    """Instantiate the env from the ``env`` config block."""
    env_cfg = config["env"]
    return VN30PortfolioEnv(
        data,
        reward_config=reward,
        initial_cash=float(env_cfg["initial_cash"]),
        softmax_temperature=float(env_cfg["softmax_temperature"]),
        episode_length=None if full_period else int(env_cfg["episode_length"]),
        risk_window=int(env_cfg["risk_window"]),
        feature_scaler=scaler,
        no_trade_band=float(env_cfg.get("no_trade_band", 0.0)),
        action_smoothing=float(env_cfg.get("action_smoothing", 1.0)),
    )


def train(
    config: dict[str, Any],
    reward_mode: str,
    seed: int,
    window: str,
    total_timesteps: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train one agent and return the run summary (also written to JSON).

    Args:
        config: Parsed YAML config.
        reward_mode: One of ``A0``–``A3``.
        seed: Random seed.
        window: Walk-forward window key, e.g. ``window_1``.
        total_timesteps: Environment steps to train for.
        dry_run: Use synthetic data and tiny rollouts; nothing is kept on disk.

    Returns:
        Run summary with paths and validation metrics.
    """
    train_cfg = config["training"]
    torch.set_num_threads(int(train_cfg["torch_threads"]))
    reward = RewardConfig(**{**config["reward"], "mode": reward_mode})
    agent_name = str(config["agent"])

    if dry_run:
        full = make_synthetic_market_data(n_days=DRY_RUN_SYNTHETIC_DAYS, seed=seed)
        split = full.dates[DRY_RUN_SYNTHETIC_DAYS * 3 // 4]
        train_data = full.slice_dates(end=str(split.date()))
        val_data = full.slice_dates(start=str(split.date()))
    else:
        bounds = config["data"]["windows"][window]
        full = load_market_data(REPO_ROOT / config["data"]["feature_matrix"])
        train_data = full.slice_dates(*bounds["train"])
        val_data = full.slice_dates(*bounds["val"])
    scaler = train_data.fit_scaler()  # fitted on train only: no look-ahead into val/test

    n_envs = int(train_cfg["n_envs"])
    vec_env = make_vec_env(
        lambda: Monitor(build_env(train_data, config, reward, scaler, full_period=False)),
        n_envs=n_envs,
        seed=seed,
    )
    eval_env = build_env(val_data, config, reward, scaler, full_period=True)

    run_id = f"{agent_name}_{reward_mode}_{window}_s{seed}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = (Path(config.get("_tmp_dir", MODELS_DIR)) if dry_run else MODELS_DIR)
    log_dir = out_dir / "logs" if dry_run else LOGS_DIR
    best_path = out_dir / f"{run_id}_best.zip"

    hyperparams = dict(config["hyperparams"])
    eval_freq = int(train_cfg["eval_freq"])
    if dry_run:
        eval_freq = DRY_RUN_EVAL_FREQ
        if "n_steps" in hyperparams:
            hyperparams["n_steps"] = DRY_RUN_N_STEPS
        if "learning_starts" in hyperparams:
            hyperparams["learning_starts"] = DRY_RUN_N_STEPS

    agent = AGENTS[agent_name](vec_env, hyperparams, seed=seed, tensorboard_log=log_dir)
    callback = SharpeCheckpointCallback(eval_env, eval_freq=max(eval_freq // n_envs, 1), save_path=best_path)
    agent.train(total_timesteps, callback=callback, tb_log_name=run_id)

    final_path = agent.save(out_dir / run_id)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "agent": agent_name,
        "reward_mode": reward_mode,
        "window": window,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "dry_run": dry_run,
        "final_checkpoint": str(final_path),
        "best_checkpoint": str(best_path),
        "best_val_sharpe": float(callback.best_sharpe),
        "config": config,
        "validation": {},
    }
    val_env = build_env(val_data, config, reward, scaler, full_period=True)
    for label, path in (("final", final_path), ("best", best_path)):
        ledger = rollout(AGENTS[agent_name].load(path), val_env)
        metrics = evaluate_portfolio(ledger["daily_return"], ledger["nav"], risk_free_rate=0.0)
        metrics["Avg Cash Weight"] = round(float(ledger["cash_weight"].mean()), 3)
        metrics["Avg Turnover"] = round(float(ledger["turnover"].mean()), 4)
        summary["validation"][label] = metrics
        logger.info("[%s] validation (%s): %s", run_id, label, metrics)

    with (out_dir / f"{run_id}.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train a PPO/SAC agent on VN30PortfolioEnv")
    parser.add_argument("--config", default="ppo_baseline.yaml", help="YAML in configs/ or a path")
    parser.add_argument("--reward", choices=["A0", "A1", "A2", "A3"], help="Reward scheme (default: config)")
    parser.add_argument("--seed", type=int, help="Seed (default: first of training.seeds)")
    parser.add_argument("--window", help="Walk-forward window (default: training.window)")
    parser.add_argument("--timesteps", type=int, help="Override training.total_timesteps")
    parser.add_argument("--dry-run", action="store_true", help="Synthetic data, ~1000 steps, temp outputs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_config(args.config)
    train_cfg = config["training"]
    reward_mode = args.reward or config["reward"]["mode"]
    seed = args.seed if args.seed is not None else int(train_cfg["seeds"][0])
    window = args.window or str(train_cfg["window"])

    if args.dry_run:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config["_tmp_dir"] = tmp
            steps = args.timesteps or int(train_cfg["dry_run_timesteps"])
            summary = train(config, reward_mode, seed, window, steps, dry_run=True)
        logger.info("DRY RUN OK: %s trained %d steps", summary["run_id"], steps)
        return

    steps = args.timesteps or int(train_cfg["total_timesteps"])
    summary = train(config, reward_mode, seed, window, steps)
    logger.info("Finished %s -> %s", summary["run_id"], summary["final_checkpoint"])


if __name__ == "__main__":
    main()
