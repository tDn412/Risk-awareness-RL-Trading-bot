# Literature Review: Risk-Aware Reinforcement Learning in Algorithmic Trading

**Project:** Risk-Aware Reinforcement Learning for VN30 Portfolio Allocation  
**Field:** Computational Intelligence & Quantitative Finance  

---

## 1. Classical Portfolio Theory vs. Reinforcement Learning

The modern foundation of risk-adjusted investment began with **Markowitz's Modern Portfolio Theory (MPT)** (Markowitz, 1952), which framed asset allocation as an optimization problem balancing expected return against variance (risk). **Sharpe (1966)** formalized the Sharpe Ratio, dividing excess return over the risk-free rate by portfolio volatility.

However, traditional methods rely on strong assumptions:
- Normally distributed asset returns.
- Stationary covariance matrices.
- Static or periodically rebalanced asset allocations.

Financial returns in reality exhibit severe non-Gaussian characteristics: fat tails (kurtosis), negative skewness, volatility clustering, and regime switching. Deep Reinforcement Learning (DRL) emerged as a model-free alternative capable of learning non-linear mapping directly from state space to policy actions (Zhang et al., 2020).

---

## 2. Risk-Aware Reinforcement Learning Paradigms

### 2.1 Differential Sharpe Ratio (Moody & Saffell, 2001)
Moody and Saffell pioneered direct reinforcement learning for trading by maximizing the **Differential Sharpe Ratio (DSR)**. Instead of waiting for an entire episode to complete before evaluating portfolio variance, DSR defines an online, recursive update of running mean and second-moment estimators:
$$\frac{d S_t}{d \mathbf{w}} = \frac{\partial S_t}{\partial A_t}\frac{dA_t}{d\mathbf{w}} + \frac{\partial S_t}{\partial B_t}\frac{dB_t}{d\mathbf{w}}$$
This allows gradient-based algorithms to directly optimize risk-adjusted returns step-by-step.

### 2.2 Downside Risk and Sortino Ratio (Sortino & van der Meer, 1991)
A major critique of variance-based risk metrics (like Sharpe Ratio) is that they penalize upside volatility (desirable outsized gains) equally to downside volatility (catastrophic losses). The **Sortino Ratio** resolves this by replacing standard deviation with downside semi-deviation (Target Downside Deviation):
$$\delta_{\text{downside}} = \sqrt{\frac{1}{T}\sum_{t=1}^T \min(0, r_t - r_{\text{target}})^2}$$
Incorporating downside semi-deviation into the reward function prevents the RL agent from being overly risk-averse during explosive upward market trends while strictly punishing drawdown behavior.

### 2.3 Prospect Theory in Reinforcement Learning (Kahneman & Tversky, 1979)
Behavioral economics demonstrates that human agents exhibit asymmetric risk tolerance: **losses hurt roughly 2 to 2.5 times more than equivalent gains please** (Loss Aversion). Furthermore, marginal sensitivity diminishes for extreme values:
$$U(x) = \begin{cases} x^\alpha & x \ge 0 \\ -\lambda (-x)^\beta & x < 0 \end{cases}$$
Embedding Prospect Theory utility as the agent's reward signal aligns policy optimization with human-like capital preservation, discouraging the high-risk "gambling" behaviors often observed in vanilla Deep Q-Networks (DQN).

---

## 3. Deep Reinforcement Learning Algorithms in Quantitative Finance

### 3.1 Proximal Policy Optimization (PPO) (Schulman et al., 2017)
PPO is widely regarded as the gold standard for continuous action control due to its clipped surrogate objective:
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
In financial environments where signal-to-noise ratios are inherently low, PPO avoids destructively large policy updates that often destabilize standard Policy Gradient (REINFORCE) or TRPO methods.

### 3.2 Soft Actor-Critic (SAC) (Haarnoja et al., 2018)
SAC optimizes a maximum entropy objective:
$$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$
The entropy regularization $\mathcal{H}$ encourages stochastic exploration, preventing the agent from collapsing prematurely into degenerate or naive buy-and-hold local optima.

---

## 4. Multimodal State Representation (Technical + Sentiment)

Recent empirical literature demonstrates that price-only representations suffer from delayed recognition of macroeconomic shifts. Incorporating NLP-based sentiment signals—extracted using transformer models like FinBERT (Araci, 2019) or domain-specific language models—enables the RL agent to anticipate volatility spikes triggered by news events and corporate earnings releases.

---

## 5. Summary & Research Gap Addressed

While modern studies often test RL models on idealized frictionless environments, our framework bridges theoretical formulation and practical execution by:
1. Formulating and benchmarking **three distinct risk-aware reward functions** (Differential Sharpe, Sortino downside penalty, and Prospect Theory utility) against standard unconstrained return.
2. Modeling realistic **transaction costs, bid-ask spread, and slippage**.
3. Providing an **ablation study** quantifying the exact contribution of each risk component and multimodal sentiment inputs.
