"""Tests for the validation-Sharpe checkpoint callback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from callbacks import annualised_sharpe


def test_annualised_sharpe_known_value() -> None:
    returns = np.array([0.01, -0.005, 0.02, 0.0])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert annualised_sharpe(returns) == pytest.approx(expected)


def test_annualised_sharpe_flat_series() -> None:
    assert annualised_sharpe(np.zeros(10)) == 0.0
    assert annualised_sharpe(np.array([0.01])) == 0.0


def test_best_checkpoint_written_by_training(tmp_path: Path) -> None:
    import train

    config = train.load_config("ppo_baseline.yaml")
    config["_tmp_dir"] = str(tmp_path)
    summary = train.train(config, "A0", seed=0, window="window_1", total_timesteps=600, dry_run=True)
    assert Path(summary["best_checkpoint"]).exists()
    assert np.isfinite(summary["best_val_sharpe"])
