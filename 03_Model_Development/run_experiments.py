"""Launch the week-5 experiment matrix as parallel ``train.py`` processes.

Example:
    python 03_Model_Development/run_experiments.py --config ppo_baseline.yaml \
        --rewards A0 A1 A2 A3 --parallel 4
Each run logs to ``outputs/logs/train_<agent>_<reward>_s<seed>.log``.
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


def run_one(config: str, agent: str, reward: str, seed: int, timesteps: int | None) -> tuple[str, int]:
    """Run a single training process; return ``(label, exit_code)``."""
    label = f"{agent}_{reward}_s{seed}"
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--config", config, "--reward", reward, "--seed", str(seed)]
    if timesteps is not None:
        cmd += ["--timesteps", str(timesteps)]
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
    parser.add_argument("--seeds", nargs="+", type=int, help="Default: training.seeds from config")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with (CONFIG_DIR / args.config).open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    seeds = args.seeds or config["training"]["seeds"]
    jobs = list(itertools.product(args.rewards, seeds))
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(
            pool.map(lambda job: run_one(args.config, config["agent"], job[0], job[1], args.timesteps), jobs)
        )
    failed = [label for label, code in results if code != 0]
    logger.info("Done: %d runs, %d failed %s", len(results), len(failed), failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
