# Project Charter: Antigravity – Risk-Aware Reinforcement Learning Trading Bot

**Author:** Quantitative Reinforcement Learning Research Team  
**Domain:** Computer Science / Computational Finance / Artificial Intelligence  
**Target Platform:** Gymnasium, PyTorch, Stable-Baselines3, Streamlit  
**Date:** September 2026  

---

## 1. Executive Summary

Traditional Reinforcement Learning (RL) agents applied to algorithmic financial trading typically maximize cumulative returns ($\sum r_t$). While theoretically sound in stationary Markov Decision Processes (MDPs), financial markets exhibit severe non-stationarity, regime shifts, and fat-tailed return distributions. As a consequence, unconstrained return-maximizing agents frequently suffer from:
1. **Severe Drawdowns & Ruin Risk:** Taking excessive leverage or aggressive long/short positions during volatile market crashes.
2. **Policy Overfitting:** Exploiting historical anomalies that fail out-of-sample.
3. **Execution Churn:** Excessive trading frequency leading to catastrophic transaction cost slippage.

**Project Antigravity** addresses these limitations by engineering a **Risk-Aware Reinforcement Learning Framework** incorporating:
- **Risk-Aware Reward Shaping:** Formulating objective functions grounded in modern portfolio theory and behavioral economics (Differential Sharpe Ratio, Sortino downside penalty, Drawdown barrier penalty, and Kahneman & Tversky's Prospect Theory).
- **Multimodal State Representation:** Integrating technical market indicators (momentum, volatility, trend) with real-time sentiment signals.
- **Robust Out-of-Sample Verification:** Benchmarking against unconstrained RL, Buy & Hold, and traditional quantitative baseline strategies across bull, bear, and high-volatility sideways regimes.

---

## 2. Research Problem & Objectives

### 2.1 Problem Statement
Given market price series $\mathcal{P} = \{p_t\}_{t=1}^T$ and feature vectors $\mathbf{s}_t \in \mathcal{S}$, the objective of the agent is to determine an action policy $\pi(a_t | \mathbf{s}_t)$ where action $a_t \in [-1, 1]$ represents the target portfolio weight or asset allocation.

Standard RL maximizes:
$$\max_{\pi} \mathbb{E}\left[ \sum_{t=0}^T \gamma^t R_t \right] \quad \text{where } R_t = \frac{V_t - V_{t-1}}{V_{t-1}}$$

However, this ignores the second moment (variance) and higher-order moments (skewness, kurtosis, downside risk) of portfolio value $V_t$.

### 2.2 Core Hypotheses
* **Hypothesis 1 (Drawdown Mitigation):** Agents trained with downside-risk penalties (Sortino / Drawdown barriers) will achieve lower maximum drawdowns (MDD) and higher Calmar ratios on out-of-sample test datasets compared to pure return-maximizing agents.
* **Hypothesis 2 (Prospect Theory Utility):** Incorporating asymmetric loss aversion ($\lambda > 1$) in reward formulation prevents panic-induced churn while accelerating capital preservation during market downturns.
* **Hypothesis 3 (Multimodal State Superiority):** Combining technical indicators with sentiment scores enhances the agent's reaction speed to macro trend reversals before they reflect in lagging moving averages.

---

## 3. Mathematical Foundations of Risk-Aware Reward Shaping

### 3.1 Standard Return with Transaction Costs
Let $V_t$ be portfolio value at time $t$, $w_t \in [-1, 1]$ be portfolio weight, and $c_{fee}$ be proportional transaction cost (commission + slippage):
$$r_t = w_{t-1} \cdot \left(\frac{p_t - p_{t-1}}{p_{t-1}}\right) - c_{fee} \cdot |w_t - w_{t-1}|$$

### 3.2 Formulation 1: Volatility & Drawdown Penalized Reward
$$R_{\text{RiskAware}}(t) = r_t - \lambda_1 \cdot \sigma_t^2 - \lambda_2 \cdot \max(0, \text{DD}_t - \theta_{\text{tol}})$$
Where:
- $\sigma_t^2$ is the rolling variance of returns over lookback window $K$.
- $\text{DD}_t = \frac{\max_{\tau \le t} V_\tau - V_t}{\max_{\tau \le t} V_\tau}$ is the current drawdown from portfolio peak.
- $\lambda_1, \lambda_2 \ge 0$ are risk-aversion hyper-coefficients.
- $\theta_{\text{tol}}$ is the acceptable drawdown tolerance threshold (e.g., $0.05$ or $5\%$).

### 3.3 Formulation 2: Differential Sharpe Ratio (Moody & Saffell, 2001)
Instead of batch calculation, the agent optimizes the running exponential moving average of Sharpe Ratio:
$$D_t = \frac{B_{t-1} \Delta A_t - \frac{1}{2} A_{t-1} \Delta B_t}{(B_{t-1} - A_{t-1}^2)^{3/2}}$$
Where $A_t$ and $B_t$ track the first and second moments of returns via exponential decay factor $\eta \in (0, 1)$:
$$A_t = A_{t-1} + \eta (r_t - A_{t-1})$$
$$B_t = B_{t-1} + \eta (r_t^2 - B_{t-1})$$

### 3.4 Formulation 3: Prospect Theory Utility (Loss Aversion)
According to Kahneman & Tversky (1979):
$$U(r_t) = \begin{cases} 
r_t^\alpha & \text{if } r_t \ge 0 \\
-\lambda \cdot (-r_t)^\beta & \text{if } r_t < 0 
\end{cases}$$
With empirical parameters $\alpha = \beta \approx 0.88$ and loss-aversion factor $\lambda \in [1.5, 2.5]$, penalizing negative returns substantially more than rewarding equivalent positive gains.

---

## 4. System Architecture

```text
+-------------------------------------------------------------+
|                     Data Pipeline                           |
|  - Historical Market Data (yfinance, Binance, VN-Index)     |
|  - Feature Engineering (RSI, MACD, Bollinger Bands, ATR)    |
|  - Sentiment Analysis (FinBERT / VnBERT / News Scraper)     |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|              Gymnasium Custom Environment                   |
|  - State Space: [Technical Features + Sentiment + Position] |
|  - Action Space: Continuous [-1, 1] or Discrete {0, 1, 2}   |
|  - Slippage & Realistic Fee Simulation                      |
|  - Risk-Aware Reward Shaping Core Engines                   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                   RL Agents & Training                      |
|  - Proximal Policy Optimization (PPO)                       |
|  - Soft Actor-Critic (SAC)                                  |
|  - Experience Replay & Checkpoint Management                |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 Evaluation & Backtesting                    |
|  - Out-of-Sample Testing across Bull/Bear/Sideway Regimes   |
|  - Quantitative Metrics (Sharpe, Sortino, Calmar, MDD)      |
|  - Baseline Comparisons (Buy & Hold, Pure PPO, MACD Rule)   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 Interactive Deployment                      |
|  - Streamlit Web Dashboard                                  |
|  - Equity Curve, Position Allocation, Drawdown Charting     |
+-------------------------------------------------------------+
```

---

## 5. Quantitative Evaluation Metrics

| Metric | Formula | Target Benchmark |
| :--- | :--- | :--- |
| **Annualized Return (CAGR)** | $(V_T / V_0)^{252/T} - 1$ | $> \text{Benchmark}$ |
| **Annualized Volatility** | $\sigma_{\text{daily}} \cdot \sqrt{252}$ | $< \text{Benchmark}$ |
| **Sharpe Ratio** | $\frac{\mathbb{E}[r - r_f]}{\sigma_r} \cdot \sqrt{252}$ | $> 1.5$ |
| **Sortino Ratio** | $\frac{\mathbb{E}[r - r_f]}{\sigma_{\text{downside}}} \cdot \sqrt{252}$ | $> 2.0$ |
| **Maximum Drawdown (MDD)** | $\max_{t} \left(\frac{\max_{\tau \le t} V_\tau - V_t}{\max_{\tau \le t} V_\tau}\right)$ | $< 15\%$ |
| **Calmar Ratio** | $\frac{\text{CAGR}}{\text{MDD}}$ | $> 1.8$ |
| **Profit Factor** | $\frac{\sum \text{Gains}}{\sum |\text{Losses}|}$ | $> 1.4$ |

---

## 6. Implementation Milestones

1. **Phase 1: Project Setup & Proposal:** Finalize Project Charter and Literature Review.
2. **Phase 2: Data Pipeline & Indicators:** Implement automated downloading, cleaning, technical indicator calculation, and sentiment normalization.
3. **Phase 3: Custom Gymnasium Environment:** Construct vectorized trading environment with selectable risk reward functions.
4. **Phase 4: Agent Training:** Train PPO and SAC agents with hyperparameter sweeps on train splits.
5. **Phase 5: Backtesting & Ablation Study:** Compare Risk-Aware agents vs. Standard Return agent vs. Buy & Hold across market regimes.
6. **Phase 6: Interactive Dashboard & Reporting:** Build Streamlit demo dashboard and compile final defense slides.
