"""
Training Script for Risk-Aware RL Trading Agents
Project: Antigravity - Risk-Aware Reinforcement Learning Trading Bot

Supports:
- Stable-Baselines3 PPO / SAC
- Multi-reward experiments ('standard', 'volatility_penalty', 'differential_sharpe', 'prospect_theory')
- TensorBoard logging & checkpoint saving
"""

import os
import argparse
import numpy as np
import pandas as pd
import sys

# Add subdirectories to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ["02_Data_Pipeline", "03_Model_Development", "04_Backtesting_Evaluation"]:
    sub_path = os.path.join(PROJECT_ROOT, sub)
    if sub_path not in sys.path:
        sys.path.insert(0, sub_path)

from env.trading_env import RiskAwareTradingEnv
from data_collector import fetch_market_data, add_features


def get_gym_env(df, reward_mode: str = "volatility_penalty"):
    """Instantiate the environment with Gymnasium compatibility."""
    try:
        import gymnasium as gym
        
        class SB3TradingEnv(gym.Env):
            def __init__(self, raw_env):
                super().__init__()
                self.raw_env = raw_env
                self.observation_space = raw_env.observation_space
                self.action_space = raw_env.action_space

            def reset(self, seed=None, options=None):
                return self.raw_env.reset(seed=seed, options=options)

            def step(self, action):
                return self.raw_env.step(action)

        raw = RiskAwareTradingEnv(df, reward_mode=reward_mode)
        return SB3TradingEnv(raw)
    except Exception:
        return RiskAwareTradingEnv(df, reward_mode=reward_mode)


def run_random_agent_benchmark(env, num_episodes: int = 1):
    """Simple baseline test when SB3 is not yet installed."""
    print("[*] Running Random/Heuristic Agent Baseline...")
    for ep in range(num_episodes):
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            # Target position random walk around 0.5 (slight long bias)
            action = np.array([np.clip(np.random.normal(0.5, 0.3), 0.0, 1.0)], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        print(f"[+] Episode {ep+1} Completed | Steps: {steps} | "
              f"Final Portfolio: ${info['portfolio_value']:,.2f} | "
              f"Max Drawdown: {info['drawdown']*100:.2f}% | "
              f"Total Reward: {total_reward:.4f}")


def train(args):
    # 1. Load or Generate Dataset
    data_path = args.data
    if data_path and os.path.exists(data_path):
        print(f"[*] Loading preprocessed data from {data_path}")
        df = pd.read_csv(data_path, index_col=0)
    else:
        print(f"[*] Fetching dataset for symbol: {args.symbol}")
        raw = fetch_market_data(args.symbol, start=args.start, end=args.end)
        df = add_features(raw)

    # 2. Train / Test Split (Time-series chronological split)
    split_idx = int(len(df) * args.train_split)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    print(f"[+] Dataset Split -> Train: {len(df_train)} candles | Test: {len(df_test)} candles")

    # 3. Initialize Environment
    env = get_gym_env(df_train, reward_mode=args.reward_mode)

    # 4. Attempt Stable-Baselines3 Training
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import CheckpointCallback

        model_dir = "models/checkpoints"
        os.makedirs(model_dir, exist_ok=True)

        algo = args.algo.upper()
        print(f"[*] Initializing {algo} agent with reward_mode: {args.reward_mode}")

        if algo == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=args.lr,
                n_steps=256,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log="./runs/"
            )
        elif algo == "SAC":
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=args.lr,
                buffer_size=50_000,
                batch_size=64,
                gamma=0.99,
                tau=0.005,
                verbose=1,
                tensorboard_log="./runs/"
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

        print(f"[*] Training for {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps)

        save_path = os.path.join(model_dir, f"{algo.lower()}_{args.reward_mode}_{args.symbol.replace('-', '_')}.zip")
        model.save(save_path)
        print(f"[+] Model saved successfully to: {save_path}")

    except ImportError:
        print("[!] Note: stable-baselines3 is not currently installed.")
        print("[!] Running baseline agent simulation on the custom environment to verify logic...")
        run_random_agent_benchmark(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Risk-Aware RL Trading Agent")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol")
    parser.add_argument("--data", type=str, default=None, help="Path to precomputed features CSV")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Start date")
    parser.add_argument("--end", type=str, default=None, help="End date")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/test chronological split ratio")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"], help="RL algorithm")
    parser.add_argument("--reward-mode", type=str, default="volatility_penalty",
                        choices=["standard", "volatility_penalty", "differential_sharpe", "downside_sortino", "prospect_theory"],
                        help="Risk-Aware reward formulation")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()
    train(args)
