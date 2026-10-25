"""Launch the week-5 experiment matrix as parallel ``train.py`` processes.

Example:
    python 03_Model_Development/run_experiments.py --config ppo_baseline.yaml \
        --rewards A0 A1 A2 A3 --windows window_2 window_3 window_4 --parallel 4
Each run logs to ``outputs/logs/train_<agent>_<reward>_<window>_s<seed>.log``.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "03_Model_Development" / "train.py"
CONFIG_DIR = REPO_ROOT / "03_Model_Development" / "configs"
LOG_DIR = REPO_ROOT / "outputs" / "logs"


def run_one(
    config: str,
    agent: str,
    reward: str,
    window: str,
    seed: int,
    timesteps: int | None,
    lambda_turnover: float | None = None,
    rebalance_speed: float | None = None,
) -> tuple[str, int]:
    """Run a single training process; return ``(label, exit_code)``."""
    turnover_tag = f"_T{lambda_turnover:g}".replace(".", "p") if lambda_turnover else ""
    turnover_tag += f"_K{rebalance_speed:g}".replace(".", "p") if rebalance_speed is not None else ""
    label = f"{agent}_{reward}{turnover_tag}_{window}_s{seed}"
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--config", config, "--reward", reward, "--window", window,
           "--seed", str(seed)]
    if timesteps is not None:
        cmd += ["--timesteps", str(timesteps)]
    if lambda_turnover is not None:
        cmd += ["--lambda-turnover", str(lambda_turnover)]
    if rebalance_speed is not None:
        cmd += ["--rebalance-speed", str(rebalance_speed)]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / f"train_{label}.log").open("w", encoding="utf-8") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    logger.info("%s finished with exit code %d", label, code)
    return label, code


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ppo_baseline.yaml")
    parser.add_argument("--rewards", nargs="+", default=["A0", "A1", "A2", "A3"])
    parser.add_argument("--windows", nargs="+", help="Default: training.window from config")
    parser.add_argument("--seeds", nargs="+", type=int, help="Default: training.seeds from config")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--rebalance-speed", type=float, help="Default: env.rebalance_speed from config")
    parser.add_argument("--lambda-turnovers", nargs="+", type=float, help="Default: reward.lambda_turnover from config")
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with (CONFIG_DIR / args.config).open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    seeds = args.seeds or config["training"]["seeds"]
    windows = args.windows or [config["training"]["window"]]
    turnovers: list[float | None] = list(args.lambda_turnovers) if args.lambda_turnovers else [None]
    jobs = list(itertools.product(windows, args.rewards, turnovers, seeds))
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(
            pool.map(lambda job: run_one(args.config, config["agent"], job[1], job[0], job[3], args.timesteps, job[2], args.rebalance_speed), jobs)
        )
    failed = [label for label, code in results if code != 0]
    logger.info("Done: %d runs, %d failed %s", len(results), len(failed), failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
