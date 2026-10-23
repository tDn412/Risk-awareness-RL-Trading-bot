"""Smoke tests for the Streamlit demo (``05_Deployment_Demo/app.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app

APP_PATH = Path(app.__file__)
HAS_RESULTS = all(p.exists() for p in app.result_paths(app.DEFAULT_WINDOW))


@pytest.mark.skipif(not HAS_RESULTS, reason="backtest results not generated")
def test_loaders_and_tables() -> None:
    curves_path, runs_path = app.result_paths(app.DEFAULT_WINDOW)
    curves = app.load_curves(curves_path)
    runs = app.load_runs(runs_path)
    assert "VN30 Index (Buy&Hold)" in curves
    assert curves["PPO A0"]["nav"].iloc[0] == pytest.approx(100.0)
    assert "PPO A2" in app.metric_table(runs).index
    assert (app.eeb_table(runs)["Agent − EEB (pp)"] < 0).any()


@pytest.mark.skipif(not HAS_RESULTS, reason="backtest results not generated")
def test_app_renders_without_exception() -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception
    assert len(at.multiselect) == 2
    at.multiselect[0].set_value(["PPO A2", "SAC A3"]).run()
    assert not at.exception
