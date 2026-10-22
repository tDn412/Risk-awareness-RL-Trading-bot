"""Smoke tests for the training entrypoint (dry-run on synthetic data)."""

from __future__ import annotations

from pathlib import Path

import pytest

import train
from agents import AGENTS


@pytest.mark.parametrize("config_name", ["ppo_baseline.yaml", "sac_baseline.yaml"])
def test_dry_run_trains_and_reports(config_name: str, tmp_path: Path) -> None:
    config = train.load_config(config_name)
    config["_tmp_dir"] = str(tmp_path)
    summary = train.train(config, "A2", seed=0, window="window_1", total_timesteps=300, dry_run=True)

    assert Path(summary["final_checkpoint"]).exists()
    assert Path(summary["best_checkpoint"]).exists()
    assert (tmp_path / f"{summary['run_id']}.json").exists()
    assert "Sharpe Ratio" in summary["validation"]["best"]
    loaded = AGENTS[config["agent"]].load(summary["final_checkpoint"])
    assert loaded.model is not None


def test_cli_dry_run() -> None:
    train.main(["--config", "ppo_baseline.yaml", "--reward", "A0", "--dry-run", "--timesteps", "300"])
