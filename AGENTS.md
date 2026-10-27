# AGENTS.md — Risk-Aware RL Trading Bot

> **For every AI agent (and human contributor) working on this codebase.**  
> Read this file along with [SOURCE_OF_TRUTH.md](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/SOURCE_OF_TRUTH.md). Together they constitute the authoritative Single Source of Truth for architecture, conventions, and completion criteria.

---

## 0. Mission Statement

Build a **Risk-Aware Reinforcement Learning Portfolio Allocation Agent** for the VN30 index.  
The agent optimises risk-adjusted returns (CVaR, Sharpe) under realistic HOSE execution constraints using **PPO and SAC**.  
Final deliverable: a reproducible experiment + Streamlit demo that outperforms a VN30 Index passive benchmark.

---

## 1. Tech Stack

| Layer | Technology | Version Pinned |
|---|---|---|
| Language | Python | 3.12.x |
| Deep Learning | PyTorch | ≥ 2.0.0 |
| RL Framework | Stable-Baselines3 | ≥ 2.3.0 |
| Environment API | Gymnasium | ≥ 0.29.1 |
| Data Source | `vnstock` (primary), `yfinance` (benchmark) | latest stable |
| Data Processing | pandas ≥ 2.0, numpy ≥ 1.24,< 2.0 | see requirements.txt |
| Technical Indicators | `ta` (Technical Analysis library) | ≥ 0.11.0 |
| Visualisation | matplotlib, seaborn, plotly | see requirements.txt |
| Dashboard | Streamlit | ≥ 1.30.0 |
| Testing | pytest + pytest-cov | latest |
| Type Checking | mypy | latest |

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest pytest-cov mypy # add to requirements-dev if missing
```

### Known vnstock Gotchas

```python
import os
os.environ["VNSTOCK_TELEMETRY"] = "off"   # MUST be set before any vnstock import
from vnstock.api.quote import Quote         # use vnstock.api.* not legacy top-level API
```

Rate limit: **20 requests / minute**. Use `time.sleep(3.5)` between tickers.  
Always use `skip_existing=True` pattern when downloading to avoid redundant calls.

---

## 2. Project Layout

```
.
├── 01_Research_Proposal/        # Literature review, proposal docs
├── 02_Data_Pipeline/
│   ├── collectors/              # Raw data downloaders (vnstock, yfinance)
│   ├── processors/              # Cleaning, alignment, feature engineering
│   └── run_pipeline.py          # Entrypoint: full pipeline orchestration
├── 03_Model_Development/
│   ├── env/                     # Gymnasium environments (VN30PortfolioEnv)
│   ├── agents/                  # PPO & SAC wrappers around SB3
│   ├── configs/                 # Hyperparameter YAML files
│   └── train.py                 # Training entrypoint
├── 04_Backtesting_Evaluation/
│   ├── backtester.py            # Walk-forward + hold-out evaluation
│   └── metrics.py               # Sharpe, Sortino, CVaR, Max Drawdown
├── 05_Deployment_Demo/
│   └── streamlit_app.py         # Interactive Streamlit dashboard
├── data/
│   ├── raw/                     # Immutable raw downloads — NEVER edit manually
│   ├── processed/               # Output of processors/
│   └── external/                # VN30 constituent lists, index weights
├── outputs/
│   ├── checkpoints/             # Saved SB3 .zip checkpoints
│   ├── logs/                    # TensorBoard logs
│   └── figures/                 # Generated plots
├── tests/                       # Mirrors source layout (test_*.py)
├── requirements.txt
├── AGENTS.md                    # Operational AI Agent Harness Guide
├── SOURCE_OF_TRUTH.md           # Authoritative Single Source of Truth
└── README.md
```

**Architectural data-flow:**

```
vnstock API
   │
   ▼
02_Data_Pipeline/collectors/    →  data/raw/
   │
   ▼
02_Data_Pipeline/processors/    →  data/processed/
   │
   ▼
03_Model_Development/env/       (VN30PortfolioEnv reads data/processed/)
   │
   ▼
03_Model_Development/agents/    (PPO / SAC wrappers)
   │
   ▼
04_Backtesting_Evaluation/      →  outputs/figures/, outputs/models/
   │
   ▼
05_Deployment_Demo/             (reads outputs/ for demo)
```

---

## 3. Market Constraints (HOSE Rules — MUST be encoded in the Env)

These are hard constraints. Any environment or backtester that ignores them is **invalid**.

| Constraint | Value | Implementation location |
|---|---|---|
| Settlement | T+2 (shares available 2 days after buy) | `env/vn30_portfolio_env.py` |
| Daily price limit | ±7% from reference price | `env/vn30_portfolio_env.py` |
| Lot size | 100 shares minimum | `env/vn30_portfolio_env.py` |
| Transaction tax (sell) | 0.1% of sell value | `env/vn30_portfolio_env.py` |
| Brokerage fee (both sides) | 0.15% of trade value | `env/vn30_portfolio_env.py` |
| Short selling | Not allowed (weight ≥ 0) | Action space simplex constraint |
| Cash as asset | Weight[0] = cash (no-op asset) | Simplex: sum(weights) = 1.0 |

**Action space:** N+1 simplex where N = 30 stocks + 1 cash position. Actions must be normalised to sum to 1.

---

## 4. Coding Conventions

### 4.1 General

- **All modules must have a module-level docstring** describing what it does.
- **All public functions/classes must have type annotations** and a docstring.
- **No magic numbers** — define named constants at module top or in a config file.
- Use `logging` (not `print`) for all informational output. Logger name = `__name__`.
- Config values go in `03_Model_Development/configs/*.yaml`, loaded via `PyYAML`.

### 4.2 Data Conventions

- Raw files in `data/raw/` are **read-only**. Processors write to `data/processed/`.
- Date columns are always `datetime64[ns]` — never strings.
- Price data uses `close` (adjusted for splits/dividends where available).
- Missing values strategy: forward-fill ≤ 5 consecutive days, else flag and raise.

### 4.3 Environment Conventions

- Env class name: `VN30PortfolioEnv` in `03_Model_Development/env/vn30_portfolio_env.py`.
- Must pass `gymnasium.utils.env_checker.check_env(env)` without warnings.
- `observation_space` must document every feature slice in a class-level `OBS_SCHEMA` dict.
- `info` dict returned by `step()` must include: `portfolio_value`, `daily_return`, `cvar_95`, `turnover`.

### 4.4 Agent Conventions

- SB3 wrappers live in `03_Model_Development/agents/`.
- Each agent class exposes: `train(total_timesteps)`, `predict(obs)`, `save(path)`, `load(path)`.
- Hyperparameters are **not hardcoded** — always loaded from a YAML config file.

### 4.5 Imports

```python
# Standard order: stdlib → third-party → local
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
```

---

## 5. Verification Commands

Run these **in order** before declaring any task complete.

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Type check (must produce zero errors)
mypy --strict --ignore-missing-imports 02_Data_Pipeline/ 03_Model_Development/ 04_Backtesting_Evaluation/

# 3. Unit tests with coverage
pytest tests/ -v \
  --cov=02_Data_Pipeline \
  --cov=03_Model_Development \
  --cov=04_Backtesting_Evaluation \
  --cov-report=term-missing

# 4. Environment sanity check (when env code changes)
python -c "
import sys, importlib
sys.path.insert(0, '03_Model_Development')
from gymnasium.utils.env_checker import check_env
mod = importlib.import_module('env.vn30_portfolio_env')
env = mod.VN30PortfolioEnv()
check_env(env)
print('OK: Env check passed')
"

# 5. Data pipeline smoke test
python 02_Data_Pipeline/run_pipeline.py --dry-run
```

**Minimum bar for any task completion:**
- `mypy --strict` → 0 errors
- `pytest` → all tests pass, coverage ≥ 80% on changed modules
- No regressions in existing tests

---

## 6. Definition of Done (DoD) — Per Feature Category

Every task must satisfy **all** criteria listed for its category.  
"It runs locally" is **not** a DoD. Use the exact criteria below.

---

### 6.1 Data Collector

**Example task:** "Add a collector for VN30 constituent changes over time"

Completion Criteria:
- [ ] New file `02_Data_Pipeline/collectors/<name>.py` with module docstring and typed functions.
- [ ] Writes output to `data/raw/<name>/` as Parquet files named `<ticker>_<start>_<end>.parquet`.
- [ ] `skip_existing=True` logic implemented — re-running does not re-download existing files.
- [ ] Rate limit guard: `time.sleep(3.5)` between ticker requests.
- [ ] Test file `tests/test_<name>.py` with:
  - `test_output_schema()` — validates columns and dtypes.
  - `test_skip_existing()` — mock the API and verify no second call is made.
- [ ] `mypy --strict` passes on the new file.
- [ ] `pytest tests/test_<name>.py` passes.
- [ ] Module added to `02_Data_Pipeline/run_pipeline.py` collector registry.

---

### 6.2 Data Processor

**Example task:** "Implement `clean_alignment.py` to align all tickers to common trading dates"

Completion Criteria:
- [ ] New file `02_Data_Pipeline/processors/clean_alignment.py`.
- [ ] Input: `data/raw/prices/*.parquet`. Output: `data/processed/aligned_prices.parquet`.
- [ ] Forward-fill rule: gap ≤ 5 trading days → ffill; gap > 5 days → `ValueError` with stock name and dates logged.
- [ ] Output DataFrame schema exactly: `{date: datetime64[ns], ticker: str, close: float64, volume: int64, adj_factor: float64}`.
- [ ] Test `tests/test_clean_alignment.py`:
  - `test_gap_ffill_short()` — gaps ≤ 5 days are filled correctly.
  - `test_gap_raises_long()` — gaps > 5 days raise `ValueError`.
  - `test_output_schema()` — output columns and dtypes match spec exactly.
- [ ] `mypy --strict` passes.
- [ ] `pytest tests/test_clean_alignment.py -v` — all tests pass.

---

### 6.3 Environment

**Example task:** "Implement `VN30PortfolioEnv` with T+2 settlement"

Completion Criteria:
- [ ] Class `VN30PortfolioEnv` in `03_Model_Development/env/vn30_portfolio_env.py`.
- [ ] `gymnasium.utils.env_checker.check_env(env)` passes with zero warnings.
- [ ] `OBS_SCHEMA` class attribute documents every feature slice.
- [ ] `step()` returns `info` dict containing: `portfolio_value`, `daily_return`, `cvar_95`, `turnover`, `transaction_cost`.
- [ ] T+2 settlement: shares bought on day T unavailable until day T+2.
- [ ] ±7% price limit: action clipped so order price never exceeds limit band.
- [ ] Lot size: allocations rounded to nearest 100-share lots.
- [ ] Fees applied: 0.15% buy+sell brokerage, 0.1% sell tax.
- [ ] Tests in `tests/test_vn30_env.py`:
  - `test_check_env()` — wraps `check_env` assertion.
  - `test_t2_settlement()` — buys shares, verifies unavailable for T, T+1, available at T+2.
  - `test_price_limit_clip()` — action exceeding ±7% is clipped correctly.
  - `test_fee_deduction()` — known trade, verified fee amount deducted from portfolio value.
  - `test_episode_deterministic()` — same seed → same episode trajectory.
- [ ] `mypy --strict` passes.
- [ ] `pytest tests/test_vn30_env.py -v` — all 5 tests pass.

---

### 6.4 Training Run

**Example task:** "Train PPO baseline for 1M steps"

Completion Criteria:
- [ ] Config file `03_Model_Development/configs/ppo_baseline.yaml` with all hyperparameters documented.
- [ ] Training run completes without crash (exit code 0).
- [ ] Model checkpoint saved to `outputs/models/ppo_baseline_<timestamp>.zip`.
- [ ] TensorBoard log saved to `outputs/logs/ppo_baseline_<timestamp>/`.
- [ ] Final `ep_rew_mean` logged and non-negative (sanity — not convergence requirement).
- [ ] `train.py --config ppo_baseline.yaml --dry-run` runs 1000 steps without error.

---

### 6.5 Backtest / Metric

**Example task:** "Add Sortino ratio to `metrics.py`"

Completion Criteria:
- [ ] Function `sortino_ratio(returns: pd.Series, target: float = 0.0) -> float` in `04_Backtesting_Evaluation/metrics.py`.
- [ ] Docstring explains formula and parameters.
- [ ] Edge case handled: all returns ≥ target → return `float('inf')`.
- [ ] Test `tests/test_metrics.py::test_sortino_ratio`:
  - Known input/output pair verified to 4 decimal places.
  - Edge case (no downside) verified.
- [ ] `mypy --strict` passes on `metrics.py`.
- [ ] `pytest tests/test_metrics.py::test_sortino_ratio -v` passes.

---

## 7. Diagnostic Loop

When a verification step fails, **do not retry blindly**. Follow this protocol:

### Failure Triage Template

```
DATE: YYYY-MM-DD
TASK: <short task name>
LAYER FAILED: [Data Collector | Processor | Environment | Agent | Backtest | Infrastructure]
SYMPTOM: <exact error message or failed assertion>
ROOT CAUSE: <identified cause, e.g. "vnstock API shape changed">
FIX APPLIED: <what was changed>
TEST ADDED: <new test that would have caught this>
PREVENTION: <what was updated in AGENTS.md / harness to prevent recurrence>
```

Write every incident up in the PR description (or commit message) of the fix.

### Known Historical Failures

| Date | Layer | Symptom | Fix |
|---|---|---|---|
| 2025-10 | Data Collector | `vnstock` top-level API deprecated | Migrated to `vnstock.api.quote.Quote` |
| 2025-10 | Data Collector | Rate limit hit (429) on full VN30 download | Added `time.sleep(3.5)` + `skip_existing` guard |

---

## 8. Weekly Milestones

| Week | Goal | Key Deliverable |
|---|---|---|
| 1 | Data infrastructure | All 30 tickers downloaded, `aligned_prices.parquet` created |
| 2 | Feature engineering | `feature_matrix.parquet` with TA indicators + risk features |
| 3 | Environment | `VN30PortfolioEnv` passing `check_env`, all 5 DoD tests green |
| 4 | Agent baseline | PPO trained 1M steps, benchmark comparison logged |
| 5 | Risk-aware reward | CVaR penalty integrated, SAC trained, both agents benchmarked |
| 6 | Demo + Report | Streamlit app live, final report written |

---

## 9. File Naming Convention

| Asset | Convention | Example |
|---|---|---|
| Raw price data | `data/raw/prices/<TICKER>.parquet` | `data/raw/prices/VIC.parquet` |
| Processed aligned | `data/processed/aligned_prices.parquet` | — |
| Feature matrix | `data/processed/feature_matrix.parquet` | — |
| Model checkpoint | `outputs/models/<agent>_<run_id>.zip` | `outputs/models/ppo_v1_20251015.zip` |
| TensorBoard logs | `outputs/logs/<agent>_<run_id>/` | `outputs/logs/ppo_v1_20251015/` |
| Test file | `tests/test_<module_name>.py` | `tests/test_clean_alignment.py` |
| Config | `03_Model_Development/configs/<agent>_<variant>.yaml` | `configs/ppo_baseline.yaml` |

---

## 10. Git Workflow

```bash
# Feature branch per task
git checkout -b feat/<short-description>

# Commit message format
git commit -m "<type>(<scope>): <summary>"
# Types: feat | fix | test | refactor | docs | chore
# Example: feat(env): implement T+2 settlement in VN30PortfolioEnv

# Before push: run full verification suite (Section 5)
# PR description must list which DoD checkboxes (Section 6) were satisfied
```

`main` branch is protected — no direct push. All changes via PR.

---

*Last updated: 2025-10 | Maintained by: Risk-Aware RL Trading Bot Team*
