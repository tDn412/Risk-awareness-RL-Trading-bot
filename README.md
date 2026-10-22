# 🛡️ Risk-Aware Reinforcement Learning for VN30 Portfolio Allocation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29-brightgreen.svg)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Stable--Baselines3](https://img.shields.io/badge/Stable--Baselines3-v2.3-red.svg)](https://stable-baselines3.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academic and practical **Risk-Aware Reinforcement Learning (RL)** quantitative portfolio allocation framework for the **VN30 Index** (Ho Chi Minh Stock Exchange - HOSE). Designed to address the fundamental pitfalls of traditional return-maximizing RL agents—such as catastrophic drawdowns, policy overfitting, and non-stationary market crashes—by integrating **multi-objective reward shaping**, **behavioral utility theory (Prospect Theory)**, and **realistic exchange execution constraints (T+2 settlement, ±7% price bands, 100-share lot sizes, taxes & fees)**.

> 📖 **Authoritative Specifications:**
> - [SOURCE_OF_TRUTH.md](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/SOURCE_OF_TRUTH.md): Canonical Single Source of Truth for problem definition ($O_1-O_6$, $RQ_1-RQ_3$, $H_1-H_3$), mathematical models, HOSE market constraints, and verification protocols.
> - [AGENTS.md](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/AGENTS.md): Operational guide, strict coding standards, and Definition of Done (DoD) for AI agents and human contributors.

---

## 📑 Table of Contents
1. [Core Innovations & CS Depth](#-core-innovations--cs-depth)
2. [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
3. [Mathematical Formulations](#-mathematical-formulations)
4. [Quick Start Guide](#-quick-start-guide)
5. [Evaluation & Quantitative Metrics](#-evaluation--quantitative-metrics)
6. [Research Documentation & References](#-research-documentation--references)

---

## 🔬 Core Innovations & CS Depth

* **Multi-Objective Risk-Aware Reward Shaping:** Replaces naive profit-seeking signals with mathematically grounded risk formulations:
  - *Differential Sharpe Ratio (DSR)*: Online recursive moment tracking (Moody & Saffell, 2001).
  - *Downside Sortino Semi-Variance*: Penalizes only negative deviations, preserving upside volatility.
  - *Drawdown Barrier Penalty*: Dynamically curbs position sizing when portfolio drops beyond a tolerance threshold $d^* = 15\%$.
  - *Prospect Theory Utility*: Implements Kahneman & Tversky's asymmetric loss-aversion ($\lambda_{\text{loss}} \approx 2.25$).
* **HOSE Exchange Microstructure Realism:** Enforces strict execution dynamics:
  - T+2 settlement queues ($S_{\text{pending\_1}} \to S_{\text{pending\_2}} \to S_{\text{available}}$).
  - Daily price limit clipping ($\pm 7\%$).
  - Lot size rounding to nearest 100 shares.
  - Transaction friction: 0.15% brokerage fee + 0.1% sell tax.
  - Long-only Simplex constraint: $\sum_{i=0}^{30} w_i = 1.0, w_i \ge 0$ with Cash position ($w_0$).
* **Out-of-Sample Walk-Forward Evaluation:** Evaluated across 4 rolling windows covering distinct market regimes (Bear Market 2022, Recovery 2023, Sideway-Up 2024, Forward 2025).

---

## 📂 Project Architecture & Directory Structure

```text
Risk-awareness RL Trading bot/
├── 01_Research_Proposal/              # Academic proposal, literature review & charter
├── 02_Data_Pipeline/                  # Data engineering & feature creation
│   ├── collectors/                    # Raw data collectors (prices, events, benchmarks via vnstock)
│   ├── processors/                    # Cleaning, alignment (gap <= 10d), 28+ features
│   ├── run_pipeline.py                # End-to-end data pipeline entrypoint
│   └── 01_vn30_data_exploration.ipynb # Data exploration & statistical analysis
├── 03_Model_Development/              # Reinforcement Learning core algorithms
│   ├── env/                           # Gymnasium environments (VN30PortfolioEnv, ledger, execution)
│   ├── agents/                        # Agent architectures (PPO, SAC Actor-Critic)
│   ├── configs/                       # Hyperparameter YAML configurations
│   └── train.py                       # Training pipeline with TensorBoard logging
├── 04_Backtesting_Evaluation/         # Performance benchmarking & metrics
│   ├── baselines/                     # 5 Baselines (Buy & Hold, Equal Weight, Cash, 60/40, Eq. Equity)
│   ├── metrics.py                     # Quantitative engine (Sharpe, Sortino, MDD, CVaR 95%)
│   └── backtest_engine.py             # Out-of-sample walk-forward evaluation engine
├── 05_Deployment_Demo/                # Interactive presentation demo
│   ├── app.py                         # Streamlit dashboard with equity curves & underwater drawdown
│   └── components/                    # Plotly chart components
├── 06_Docs_and_Reports/               # Slides (Beamer/Remotion), reports & diagnostic log
├── data/                              # Data storage (raw & processed Parquet files)
├── outputs/                           # Checkpoints, TensorBoard logs, and figures
├── tests/                             # Pytest automated test suite
├── requirements.txt                   # Complete project dependencies
├── SOURCE_OF_TRUTH.md                 # Authoritative Single Source of Truth
├── AGENTS.md                          # AI Agent operational harness guide
└── README.md                          # Project overview and instructions
```

---

## 📐 Mathematical Formulations

### 1. Volatility & Drawdown Barrier Reward
$$R_{\text{RiskAware}}(t) = r_t - \lambda_{\text{vol}} \cdot \sigma_t^2 - \lambda_{\text{dd}} \cdot \max(0, \text{DD}_t - \theta_{\text{tol}})$$
*where $\text{DD}_t = \frac{\max_{\tau \le t} V_\tau - V_t}{\max_{\tau \le t} V_\tau}$ and $\theta_{\text{tol}}$ is user-defined drawdown tolerance.*

### 2. Differential Sharpe Ratio (Moody & Saffell, 2001)
$$D_t = \frac{B_{t-1} \Delta A_t - \frac{1}{2} A_{t-1} \Delta B_t}{(B_{t-1} - A_{t-1}^2)^{3/2}}$$
*where $A_t = A_{t-1} + \eta (r_t - A_{t-1})$ and $B_t = B_{t-1} + \eta (r_t^2 - B_{t-1})$ track exponential running moments.*

### 3. Prospect Theory Utility (Loss Aversion)
$$U(r_t) = \begin{cases} 
r_t^\alpha & \text{if } r_t \ge 0 \\
-\lambda \cdot |r_t|^\beta & \text{if } r_t < 0 
\end{cases}$$
*with standard parameters $\alpha = \beta \approx 0.88$ and $\lambda \in [1.5, 2.5]$.*

---

## 🚀 Quick Start Guide

### 1. Installation
Clone or navigate to the repository and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Step 1: Run End-to-End VN30 Data Pipeline
Ingest raw data via `vnstock`, align trading calendar, and compute 28+ technical & risk features:
```bash
# Run full VN30 data pipeline
python 02_Data_Pipeline/run_pipeline.py

# Generate 6 statistical exploration figures & Jupyter notebook
python 02_Data_Pipeline/generate_exploration_charts.py
```

### 3. Step 2: Train Risk-Aware Agent
Train a PPO or SAC agent with your desired reward shaping mode:
```bash
# Available reward modes: volatility_penalty | prospect_theory | differential_sharpe | downside_sortino | standard
python3 03_Model_Development/train.py --symbol BTC-USD --algo PPO --reward-mode volatility_penalty --timesteps 20000
```

### 4. Step 3: Run Out-of-Sample Backtesting
Evaluate performance against Buy & Hold and Trend Following benchmarks:
```bash
python3 04_Backtesting_Evaluation/backtest_engine.py --symbol BTC-USD
```

### 5. Step 4: Launch Interactive Demo Dashboard
Launch the Streamlit web app for visual presentation and defense:
```bash
streamlit run 05_Deployment_Demo/app.py
```

---

## 📊 Evaluation & Quantitative Metrics

The evaluation suite ([metrics.py](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/04_Backtesting_Evaluation/metrics.py)) computes:
- **CAGR (%)**: Compound Annual Growth Rate.
- **Annualized Sharpe Ratio**: Excess return per unit of total risk.
- **Sortino Ratio**: Excess return per unit of downside risk.
- **Maximum Drawdown (MDD %)**: Worst peak-to-trough decline.
- **Calmar Ratio**: $\text{CAGR} / \text{MDD}$.
- **Value at Risk (VaR 95%) & CVaR 95%**: Tail risk quantification.
- **Win Rate (%) & Profit Factor**.

---

## 📚 Research Documentation & References

1. **Moody, J., & Saffell, M. (2001).** Learning to trade via direct reinforcement. *IEEE Transactions on Neural Networks*, 12(4), 875-889.
2. **Kahneman, D., & Tversky, A. (1979).** Prospect Theory: An Analysis of Decision under Risk. *Econometrica*, 47(2), 263-291.
3. **Schulman, J., et al. (2017).** Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.
4. **Sortino, F. A., & van der Meer, R. (1991).** Downside risk. *The Journal of Portfolio Management*, 17(4), 27-31.
5. Refer to [project_charter.md](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/01_Research_Proposal/project_charter.md) and [literature_review.md](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/01_Research_Proposal/literature_review.md) for full academic details.
