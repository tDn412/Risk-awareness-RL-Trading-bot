"""
Risk-Aware Gymnasium Trading Environment
Project: Antigravity - Risk-Aware Reinforcement Learning Trading Bot

Implements a full Gymnasium-compliant market environment featuring:
1. Continuous target position sizing [-1.0, 1.0] or long-only [0.0, 1.0].
2. Realistic transaction costs and slippage simulation.
3. Multi-objective Risk-Aware Reward Shaping:
   - 'standard': Pure return maximization (baseline).
   - 'volatility_penalty': Penalizes return volatility and drawdowns exceeding tolerance.
   - 'differential_sharpe': Online Differential Sharpe Ratio (Moody & Saffell, 2001).
   - 'downside_sortino': Asymmetric penalty on downside semi-variance only.
   - 'prospect_theory': Kahneman-Tversky loss-aversion utility (U(r) with lambda > 1).
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Graceful fallback to legacy gym if gymnasium is not yet installed
    try:
        import gym
        from gym import spaces
    except ImportError:
        gym = None
        spaces = None


class RiskAwareTradingEnv:
    """
    Custom Gymnasium Environment for Risk-Aware Quantitative Trading.
    Compatible with Gymnasium v0.29+ and Stable-Baselines3.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        initial_balance: float = 10_000.0,
        transaction_fee: float = 0.001,  # 0.1% commission + slippage
        reward_mode: str = "volatility_penalty",
        allow_short: bool = False,
        lookback_window: int = 20,
        lambda_vol: float = 2.0,
        lambda_dd: float = 5.0,
        dd_tolerance: float = 0.05,  # 5% acceptable drawdown
        prospect_lambda: float = 2.25,  # Loss aversion coefficient
        prospect_alpha: float = 0.88,
        dsr_decay: float = 0.05,  # EMA decay for Differential Sharpe
    ):
        """
        Args:
            df: DataFrame containing price and feature columns. Must contain 'Close'.
            feature_cols: List of column names used as state observation.
            initial_balance: Starting cash balance.
            transaction_fee: Proportional fee per unit traded.
            reward_mode: One of ['standard', 'volatility_penalty', 'differential_sharpe',
                                 'downside_sortino', 'prospect_theory'].
            allow_short: If True, action in [-1, 1]. If False, action in [0, 1].
            lookback_window: Rolling window length for volatility estimation.
            lambda_vol: Weight for portfolio volatility penalty.
            lambda_dd: Weight for drawdown penalty.
            dd_tolerance: Drawdown threshold beyond which penalties trigger.
            prospect_lambda: Kahneman-Tversky loss aversion factor (> 1 means losses hurt more).
            prospect_alpha: Power coefficient for diminishing sensitivity in Prospect Theory.
            dsr_decay: Exponential moving average decay for Differential Sharpe Ratio.
        """
        self.df = df.reset_index(drop=True).copy()
        assert "Close" in self.df.columns, "DataFrame must contain a 'Close' price column."
        
        self.close_prices = self.df["Close"].values.astype(np.float32)
        self.num_samples = len(self.df)
        
        # Feature columns setup
        if feature_cols is None:
            excluded = ["Open", "High", "Low", "Close", "Volume", "Date", "datetime"]
            self.feature_cols = [c for c in self.df.columns if c not in excluded]
            if not self.feature_cols:
                # If only Close exists, use 1-day return
                self.df["feature_ret"] = self.df["Close"].pct_change().fillna(0.0)
                self.feature_cols = ["feature_ret"]
        else:
            self.feature_cols = feature_cols

        self.features = self.df[self.feature_cols].values.astype(np.float32)
        
        # Observation includes: [Features, Current Position Weight, Current Drawdown]
        self.obs_dim = self.features.shape[1] + 2

        self.initial_balance = float(initial_balance)
        self.transaction_fee = float(transaction_fee)
        self.reward_mode = reward_mode
        self.allow_short = allow_short
        self.lookback_window = lookback_window
        
        # Risk hyperparameters
        self.lambda_vol = lambda_vol
        self.lambda_dd = lambda_dd
        self.dd_tolerance = dd_tolerance
        self.prospect_lambda = prospect_lambda
        self.prospect_alpha = prospect_alpha
        self.dsr_decay = dsr_decay

        # Define Gym spaces if gym is available
        if spaces is not None:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
            act_low = -1.0 if self.allow_short else 0.0
            self.action_space = spaces.Box(
                low=act_low, high=1.0, shape=(1,), dtype=np.float32
            )

        # State trackers
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.position = 0.0  # Current asset exposure [-1.0, 1.0]
        self.returns_history: List[float] = []
        
        # Differential Sharpe trackers (Moody & Saffell)
        self.dsr_a = 0.0  # First moment EMA
        self.dsr_b = 0.0  # Second moment EMA

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.lookback_window
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.position = 0.0
        self.returns_history = [0.0] * self.lookback_window
        
        self.dsr_a = 0.0
        self.dsr_b = 1e-4

        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "drawdown": 0.0,
            "position": self.position
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        """Construct multimodal state vector: [Market Features, Current Position, Drawdown]."""
        feature_slice = self.features[self.current_step]
        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-9)
        obs = np.append(feature_slice, [self.position, drawdown]).astype(np.float32)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one transition step given target position action.
        """
        # 1. Parse target position action
        if isinstance(action, (np.ndarray, list)):
            target_pos = float(action[0])
        else:
            target_pos = float(action)

        min_pos = -1.0 if self.allow_short else 0.0
        target_pos = float(np.clip(target_pos, min_pos, 1.0))

        # 2. Price transition & asset return
        prev_price = self.close_prices[self.current_step - 1]
        curr_price = self.close_prices[self.current_step]
        asset_return = (curr_price - prev_price) / (prev_price + 1e-9)

        # 3. Transaction costs incurred from rebalancing position
        pos_delta = abs(target_pos - self.position)
        fee_cost = self.portfolio_value * pos_delta * self.transaction_fee

        # 4. Portfolio value update
        gross_return = self.position * asset_return
        net_portfolio_return = gross_return - (fee_cost / (self.portfolio_value + 1e-9))
        self.portfolio_value = max(1e-4, self.portfolio_value * (1.0 + net_portfolio_return))

        # Update high-water mark peak and drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        current_drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-9)

        # Update position for next step
        self.position = target_pos
        self.returns_history.append(net_portfolio_return)

        # 5. Compute Risk-Aware Reward
        reward, reward_info = self._calculate_reward(net_portfolio_return, current_drawdown)

        # 6. Step advancement and termination condition
        self.current_step += 1
        terminated = self.current_step >= (self.num_samples - 1)
        truncated = bool(self.portfolio_value <= (self.initial_balance * 0.1))  # Ruin barrier (90% loss)

        next_obs = self._get_observation() if not (terminated or truncated) else np.zeros(self.obs_dim, dtype=np.float32)

        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "drawdown": current_drawdown,
            "position": self.position,
            "net_return": net_portfolio_return,
            "fee_cost": fee_cost,
            **reward_info
        }

        return next_obs, reward, terminated, truncated, info

    def _calculate_reward(self, net_return: float, drawdown: float) -> Tuple[float, Dict[str, float]]:
        """
        Core Risk-Aware Reward Shaping dispatch.
        Supports 5 academic paradigms.
        """
        recent_returns = np.array(self.returns_history[-self.lookback_window:])
        rolling_std = float(np.std(recent_returns))

        if self.reward_mode == "standard":
            # Baseline: Raw net portfolio return
            reward = net_return
            info = {"reward_type": "standard", "raw_return": net_return}

        elif self.reward_mode == "volatility_penalty":
            # Return penalized by rolling volatility and excess drawdown
            excess_dd = max(0.0, drawdown - self.dd_tolerance)
            penalty = (self.lambda_vol * rolling_std) + (self.lambda_dd * excess_dd)
            reward = net_return - penalty
            info = {
                "reward_type": "volatility_penalty",
                "vol_penalty": self.lambda_vol * rolling_std,
                "dd_penalty": self.lambda_dd * excess_dd
            }

        elif self.reward_mode == "differential_sharpe":
            # Differential Sharpe Ratio (Moody & Saffell, 2001)
            eta = self.dsr_decay
            delta_a = net_return - self.dsr_a
            delta_b = (net_return ** 2) - self.dsr_b
            
            # Update moments
            self.dsr_a += eta * delta_a
            self.dsr_b += eta * delta_b

            var = self.dsr_b - (self.dsr_a ** 2)
            var = max(1e-6, var)
            denominator = var ** 1.5
            
            # Differential D_t
            d_t = (self.dsr_b * delta_a - 0.5 * self.dsr_a * delta_b) / denominator
            reward = float(np.clip(d_t, -5.0, 5.0))
            info = {"reward_type": "differential_sharpe", "dsr_val": reward}

        elif self.reward_mode == "downside_sortino":
            # Penalizes only negative return variance (semi-variance)
            downside_sq = (min(0.0, net_return) ** 2)
            excess_dd = max(0.0, drawdown - self.dd_tolerance)
            reward = net_return - (self.lambda_vol * downside_sq * 100.0) - (self.lambda_dd * excess_dd)
            info = {
                "reward_type": "downside_sortino",
                "downside_penalty": downside_sq,
                "dd_penalty": excess_dd
            }

        elif self.reward_mode == "prospect_theory":
            # Kahneman & Tversky (1979) Loss Aversion Utility Function
            # Positive: r^alpha | Negative: -lambda * (-r)^beta
            abs_ret = abs(net_return)
            if net_return >= 0:
                utility = (abs_ret ** self.prospect_alpha)
            else:
                utility = -self.prospect_lambda * (abs_ret ** self.prospect_alpha)
            
            # Additional penalty if current drawdown exceeds critical tolerance
            excess_dd = max(0.0, drawdown - self.dd_tolerance)
            reward = utility - (self.lambda_dd * excess_dd)
            info = {"reward_type": "prospect_theory", "utility": utility, "excess_dd": excess_dd}

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        return float(reward), info


# Compatibility alias
if gym is not None:
    class GymRiskAwareTradingEnv(RiskAwareTradingEnv, gym.Env):
        """Gymnasium Subclass wrapper."""
        pass
