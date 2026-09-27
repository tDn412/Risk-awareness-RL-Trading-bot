# 🛡️ Antigravity: Risk-Aware Reinforcement Learning Trading Bot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29-brightgreen.svg)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Stable--Baselines3](https://img.shields.io/badge/Stable--Baselines3-v2.3-red.svg)](https://stable-baselines3.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academic and practical **Risk-Aware Reinforcement Learning (RL)** quantitative trading framework. Designed to address the fundamental pitfalls of traditional return-maximizing RL agents—such as catastrophic drawdowns, policy overfitting, and non-stationary market crashes—by integrating **multi-objective reward shaping**, **behavioral utility theory (Prospect Theory)**, and **multimodal state representations**.

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
  - *Drawdown Barrier Penalty*: Dynamically curbs position sizing when portfolio drops beyond a tolerance threshold $\theta_{\text{tol}}$.
  - *Prospect Theory Utility*: Implements Kahneman & Tversky's asymmetric loss-aversion ($\lambda \approx 2.25$).
* **Multimodal State Fusion:** Enriches raw OHLCV prices with technical indicators (RSI, MACD, Bollinger Bands, ATR) and real-time news/social sentiment signals.
* **Realistic Execution Dynamics:** Incorporates proportional transaction fees, slippage, and continuous position sizing $w_t \in [-1.0, 1.0]$.
* **Out-of-Sample Regimes:** Evaluated across distinct market regimes (Bull, Bear, and High-Volatility Sideways) on both Crypto (`BTC-USD`, `ETH-USD`) and Equities/VN-Index (`^VNINDEX`, `HPG.VN`).

---

## 📂 Project Architecture & Directory Structure

```text
Risk-awareness RL Trading bot/
├── 01_Research_Proposal/              # Academic proposal and literature review
│   ├── project_charter.md             # Formal research charter, problem statement & methodology
│   └── literature_review.md           # Review of RL in Finance, DSR, and Prospect Theory
│
├── 02_Data_Pipeline/                  # Data engineering & feature creation
│   └── data_collector.py              # Market data fetcher (yfinance) + Technical/Sentiment features
│
├── 03_Model_Development/              # Reinforcement Learning core algorithms
│   ├── env/
│   │   ├── __init__.py
│   │   └── trading_env.py             # Custom Gymnasium Environment with 5 Reward Shaping paradigms
│   ├── agents/                        # Agent architectures (PPO, SAC)
│   └── train.py                       # Training pipeline with TensorBoard logging & checkpointing
│
├── 04_Backtesting_Evaluation/         # Performance benchmarking & metrics
│   ├── metrics.py                     # Quantitative engine (Sharpe, Sortino, MDD, Calmar, VaR/CVaR)
│   └── backtest_engine.py             # Out-of-sample benchmarking vs. Buy & Hold and Trend-Following
│
├── 05_Deployment_Demo/                # Interactive presentation demo
│   ├── app.py                         # Streamlit dashboard with equity curves & underwater drawdown
│   └── requirements.txt               # Dashboard-specific dependencies
│
├── 06_Docs_and_Reports/               # Slides, reports, and academic defense artifacts
├── data/                              # Data storage directory (raw & processed CSVs)
├── requirements.txt                   # Complete project dependencies
├── .gitignore                         # Standard Python & checkpoint ignore rules
└── README.md                          # Comprehensive project documentation
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

### 2. Step 1: Collect & Engineer Data
Download market data and generate normalized technical/sentiment features:
```bash
python3 02_Data_Pipeline/data_collector.py --symbol BTC-USD --start 2021-01-01
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
