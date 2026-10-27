# Risk-Aware Reinforcement Learning for VN30 Portfolio Allocation

A course project (INT3011E) that trains PPO/SAC agents to allocate a long-only portfolio across the 30 stocks of the
VN30 index plus cash, under the real trading rules of the Ho Chi Minh Stock Exchange (HOSE). The question is whether
risk-aware reward shaping gives a better risk/return trade-off than a plain return-maximising agent, once fees, taxes
and settlement rules are taken into account.

Full problem definition, hypotheses (H1–H3) and maths: [SOURCE_OF_TRUTH.md](SOURCE_OF_TRUTH.md).
Coding conventions and Definition of Done: [AGENTS.md](AGENTS.md).

## What is in the project

**Market environment** (`03_Model_Development/env/vn30_portfolio_env.py`), a Gymnasium env that enforces:

- T+2 settlement (shares bought today can only be sold two sessions later)
- ±7% daily price band
- 100-share lot sizes
- 0.15% brokerage fee on both sides, 0.1% sell tax
- long-only weights over 30 stocks + cash, produced by a softmax over the agent's action
- optional partial rebalancing `env.rebalance_speed` (κ): each day the portfolio only moves a fraction κ of the way to
  the agent's target weights (κ = 1 is a full rebalance)

**Reward schemes** (`03_Model_Development/env/rewards.py`):

| Mode | Reward |
|---|---|
| A0 | risk-neutral: net daily return |
| A1 | net return − λ_vol · rolling variance |
| A2 | net return − downside semi-variance penalty − drawdown barrier (above d* = 15%) |
| A3 | prospect-theory utility (λ_loss = 2.25) − CVaR95 barrier − drawdown barrier |

Every mode can also subtract a turnover penalty `reward.lambda_turnover` (λ). Both κ and λ are off by default.

**Evaluation** (`04_Backtesting_Evaluation/backtester.py`): walk-forward over four windows, with model selection on the
validation year only and the test year kept untouched:

| Window | Train | Validation | Test |
|---|---|---|---|
| 1 | 2018–2020 | 2021 | 2022 (bear market) |
| 2 | 2018–2021 | 2022 | 2023 |
| 3 | 2018–2022 | 2023 | 2024 |
| 4 | 2018–2023 | 2024 | 2025 |

Agents are compared with VN30 buy & hold, Equal Weight 1/N, Static 60/40, 100% cash, and an Equivalent Equity
Benchmark (EEB: an equal-weight portfolio holding the same average cash weight as the agent, used for a paired t-test).
Metrics: total return, Sharpe, Sortino, max drawdown, Calmar, CVaR95, volatility, cash weight, turnover, and friction
(fees + tax as % of NAV per year).

## Results

All PPO runs: 500k steps, best-on-validation checkpoint. Window 1 has 3 seeds (42, 101, 2024); windows 2–4 have one
seed (42). Full tables are in `outputs/backtest_results/window_*_test_summary.md` and NAV plots in `outputs/figures/`.

**Window 1, test 2022** (mean of 3 seeds, total return % / max drawdown %):

| Strategy | Return | MDD | Friction %/yr |
|---|---|---|---|
| PPO A0 | −47.8 | 53.0 | 30.2 |
| PPO A1 | −40.2 | 47.7 | 20.8 |
| PPO A2 | −34.8 | 44.7 | 17.3 |
| PPO A3 | −32.0 | 40.4 | 18.4 |
| VN30 buy & hold | −35.5 | 42.0 | – |
| Equal Weight 1/N | −26.3 | 36.8 | 0.3 |
| Static 60/40 | −16.3 | 24.0 | 0.2 |

**Windows 2–4, before and after cost control** (total return %, Sharpe in brackets; cost-aware = κ 0.1, λ 0.03, tuned on
the window-2 validation year):

| Strategy | 2023 | 2024 | 2025 |
|---|---|---|---|
| PPO A0 | −12.1 (−0.63) | −14.9 (−1.02) | −8.2 (−0.26) |
| PPO A3 | −3.2 (−0.11) | 7.6 (0.61) | 22.1 (1.07) |
| PPO A2, cost-aware | −1.4 (0.00) | 9.7 (0.76) | 38.7 (1.65) |
| PPO A3, cost-aware | −1.1 (0.01) | **16.6 (1.15)** | 36.1 (1.58) |
| Equal Weight 1/N | 8.7 (0.58) | 15.7 (1.11) | 39.6 (1.68) |
| VN30 buy & hold | 8.0 (0.53) | 18.8 (1.29) | 51.2 (1.99) |

Main takeaways:

1. The risk-aware rewards help consistently: A2 and A3 beat A0 in every window, mostly because they trade less.
2. Without cost control the agents churn 10–34% of NAV a day, and fees + tax (9–33% of NAV a year) explain most of the
   gap to the baselines.
3. Partial rebalancing plus a turnover penalty cut friction to about 1–4% a year and raised returns by 2–17 points.
   Cost-aware A3 in 2024 is the only run that beat an investable baseline (Equal Weight) out of sample, but its edge
   over its EEB is not significant (p = 0.28).
4. No agent beats the VN30 index in any year. Windows 2–4 and the cost-aware setting still need more seeds, and
   window 1 has not been rerun with cost control.

## Repository layout

```text
01_Research_Proposal/      proposal, literature review, project charter
02_Data_Pipeline/          vnstock collectors, cleaning/alignment, feature building (run_pipeline.py)
03_Model_Development/
  env/                     VN30PortfolioEnv, rewards A0–A3, market data loader
  agents/                  PPO and SAC wrappers (Stable-Baselines3)
  configs/                 ppo_baseline.yaml, sac_baseline.yaml (windows, env, reward, hyperparams)
  train.py                 train one run
  run_experiments.py       grid over rewards × windows × seeds
  summarize_runs.py        validation summaries
04_Backtesting_Evaluation/
  backtester.py            out-of-sample walk-forward backtest, baselines, EEB, t-tests
  metrics.py               Sharpe, Sortino, MDD, Calmar, CVaR, ...
  backtest_engine.py       early single-asset prototype, not used for the VN30 results
05_Deployment_Demo/app.py  Streamlit dashboard over the saved backtest results
outputs/                   backtest tables/curves (CSV, MD) and figures; checkpoints are not committed
tests/                     pytest suite
```

`data/` is git-ignored. Run the pipeline to rebuild it.

## How to run

Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # use the root file, see note below
pip install -r requirements-dev.txt      # only for running tests
```

> Install from the **root** `requirements.txt`. It pins `numpy<2` and `pyarrow<19`. The demo's own
> `05_Deployment_Demo/requirements.txt` does not pin pyarrow, so a fresh install from it pulls a pyarrow that needs
> NumPy 2 and the dashboard crashes on `st.dataframe`.

**1. Build the dataset** (needs network access to the vnstock/VCI data source):

```bash
python 02_Data_Pipeline/run_pipeline.py --start 2018-01-01 --end 2025-12-31
```

This writes `data/processed/feature_matrix.parquet` and `data/raw/benchmarks/VN30.csv`.

**2. Train**

```bash
# one run
python 03_Model_Development/train.py --config ppo_baseline.yaml --reward A3 --window window_3 --seed 42

# cost-aware setting
python 03_Model_Development/train.py --config ppo_baseline.yaml --reward A3 --window window_3 --seed 42 \
    --rebalance-speed 0.1 --lambda-turnover 0.03

# full grid
python 03_Model_Development/run_experiments.py --config ppo_baseline.yaml --rewards A0 A1 A2 A3 \
    --windows window_2 window_3 window_4 --seeds 42 --parallel 1

# quick smoke test on synthetic data
python 03_Model_Development/train.py --dry-run
```

One 500k-step PPO run takes roughly 17 minutes on CPU.

**3. Backtest**

```bash
python 04_Backtesting_Evaluation/backtester.py --window window_1   # window_1 ... window_4
```

Results go to `outputs/backtest_results/` and `outputs/figures/`.

**4. Demo dashboard**

```bash
streamlit run 05_Deployment_Demo/app.py
```

The dashboard only reads the CSVs already in `outputs/backtest_results/`, so it works without data or checkpoints.
It currently shows window 1 (test 2022).

**Tests**

```bash
pytest
```

## References

1. Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement. *IEEE Transactions on Neural Networks*, 12(4), 875–889.
2. Kahneman, D., & Tversky, A. (1979). Prospect Theory: An analysis of decision under risk. *Econometrica*, 47(2), 263–291.
3. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
4. Sortino, F. A., & van der Meer, R. (1991). Downside risk. *The Journal of Portfolio Management*, 17(4), 27–31.
