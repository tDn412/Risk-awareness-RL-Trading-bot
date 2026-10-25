"""Aggregate validation metrics of finished training runs.

Reads ``outputs/models/*.json`` (written by ``train.py``) and prints / saves a table
with mean ± std across seeds per ``(agent, reward, lambda_turnover, window)`` for the best-on-validation
checkpoint.

Example:
    python 03_Model_Development/summarize_runs.py --out outputs/validation_summary.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "outputs" / "models"
GROUP_KEYS = ["agent", "reward_mode", "lambda_turnover", "window"]
REPORT_METRICS = [
    "Total Return (%)",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Max Drawdown (%)",
    "CVaR (95%)",
    "Avg Cash Weight",
    "Avg Turnover",
]


def load_runs(models_dir: Path = MODELS_DIR, checkpoint: str = "best") -> pd.DataFrame:
    """Load one row per run with validation metrics of ``checkpoint`` (best|final)."""
    rows = []
    for path in sorted(models_dir.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("dry_run"):
            continue
        run.setdefault("lambda_turnover", run["config"]["reward"].get("lambda_turnover", 0.0))
        row = {key: run[key] for key in GROUP_KEYS + ["seed", "run_id"]}
        row.update(run["validation"][checkpoint])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per group, formatted as strings."""
    grouped = runs.groupby(GROUP_KEYS)[REPORT_METRICS]
    mean, std = grouped.mean(), grouped.std().fillna(0.0)
    table = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
    table.insert(0, "n_seeds", grouped.size())
    return table.reset_index()


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    parser.add_argument("--out", type=Path, help="Optional Markdown output path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    runs = load_runs(checkpoint=args.checkpoint)
    if runs.empty:
        logger.info("No finished runs in %s", MODELS_DIR)
        return
    table = summarize(runs)
    markdown = table.to_markdown(index=False)
    logger.info("\n%s", markdown)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
