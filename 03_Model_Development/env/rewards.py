"""Risk-aware reward schemes A0–A3 (SOURCE_OF_TRUTH.md §6.4).

* A0 — risk-neutral: ``r_net``.
* A1 — volatility penalty: ``r_net - λ_vol · σ²`` (rolling variance of portfolio returns).
* A2 — downside semi-variance + drawdown barrier:
  ``r_net - λ_down · max(0, m - r_net)² - λ_dd · max(0, DD - d*)``.
* A3 — prospect-theory utility + CVaR barrier + drawdown barrier:
  ``U(r_net) - λ_cvar · max(0, CVaR95 - CVaR_target) - λ_dd · max(0, DD - d*)``.

All rewards are multiplied by ``reward_scale`` so that per-step magnitudes are O(0.1–1),
which keeps PPO/SAC value targets well conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

REWARD_MODES: tuple[str, ...] = ("A0", "A1", "A2", "A3")
CVAR_CONFIDENCE = 0.95
MIN_OBS_FOR_VARIANCE = 2


@dataclass(frozen=True)
class RewardConfig:
    """Hyper-parameters of the reward schemes (loaded from YAML ``reward`` block)."""

    mode: str = "A0"
    reward_scale: float = 100.0
    lambda_vol: float = 5.0
    lambda_down: float = 10.0
    min_return: float = 0.0
    lambda_dd: float = 0.5
    dd_threshold: float = 0.15
    lambda_cvar: float = 1.0
    cvar_target: float = 0.03
    pt_alpha: float = 0.88
    pt_beta: float = 0.88
    pt_lambda_loss: float = 2.25

    def __post_init__(self) -> None:
        if self.mode not in REWARD_MODES:
            raise ValueError(f"Unknown reward mode {self.mode!r}; expected one of {REWARD_MODES}.")


def historical_cvar(returns: NDArray[np.float64], confidence: float = CVAR_CONFIDENCE) -> float:
    """Historical CVaR as a positive loss: ``-E[r | r <= VaR_q]``.

    Args:
        returns: 1-D array of periodic returns.
        confidence: Confidence level (0.95 → worst 5% tail).

    Returns:
        Expected tail loss (>= 0 when the tail is negative); 0.0 for empty input.
    """
    if returns.size == 0:
        return 0.0
    var_threshold = np.quantile(returns, 1.0 - confidence)
    tail = returns[returns <= var_threshold]
    return float(max(0.0, -tail.mean()))


def prospect_utility(x: float, alpha: float, beta: float, lambda_loss: float) -> float:
    """Tversky–Kahneman value function ``U(x)``."""
    if x >= 0.0:
        return float(x**alpha)
    return float(-lambda_loss * (-x) ** beta)


def compute_reward(
    config: RewardConfig,
    net_return: float,
    drawdown: float,
    return_window: NDArray[np.float64],
) -> float:
    """Compute the scaled reward for one step.

    Args:
        config: Reward configuration (selects A0–A3).
        net_return: Portfolio net return of this step (after costs).
        drawdown: Current drawdown from peak NAV, in ``[0, 1]``.
        return_window: Recent net returns including ``net_return`` (rolling window).

    Returns:
        Scaled scalar reward.
    """
    dd_penalty = config.lambda_dd * max(0.0, drawdown - config.dd_threshold)

    if config.mode == "A0":
        reward = net_return
    elif config.mode == "A1":
        variance = float(np.var(return_window)) if return_window.size >= MIN_OBS_FOR_VARIANCE else 0.0
        reward = net_return - config.lambda_vol * variance
    elif config.mode == "A2":
        shortfall = max(0.0, config.min_return - net_return)
        reward = net_return - config.lambda_down * shortfall**2 - dd_penalty
    else:  # A3
        utility = prospect_utility(net_return, config.pt_alpha, config.pt_beta, config.pt_lambda_loss)
        cvar_excess = max(0.0, historical_cvar(return_window) - config.cvar_target)
        reward = utility - config.lambda_cvar * cvar_excess - dd_penalty

    return float(reward * config.reward_scale)
