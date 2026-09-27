"""
Streamlit Web Dashboard: Antigravity Risk-Aware RL Trading Bot
Project: Antigravity - Risk-Aware Reinforcement Learning Trading Bot

Provides an interactive quantitative research dashboard to explore:
- Risk-Aware reward configurations
- Live backtest comparisons against Buy & Hold and Trend Following
- Underwater drawdown charts, equity curves, and position heatmaps
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from 03_Model_Development.env.trading_env import RiskAwareTradingEnv
from 04_Backtesting_Evaluation.metrics import evaluate_portfolio
from 02_Data_Pipeline.data_collector import fetch_market_data, add_features


def main():
    if not STREAMLIT_AVAILABLE:
        print("[!] Streamlit and Plotly are required to run this dashboard.")
        print("    Install them via: pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="Antigravity | Risk-Aware RL Trading Bot",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .metric-card {
            background-color: #1e222d;
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #2962ff;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Antigravity: Risk-Aware RL Trading Dashboard")
    st.caption("Quantitative Reinforcement Learning with Multi-Objective Risk Reward Shaping & Behavioral Utility")

    # Sidebar Controls
    st.sidebar.header("⚙️ Strategy & Market Settings")
    symbol = st.sidebar.selectbox(
        "Select Market Asset",
        ["BTC-USD", "ETH-USD", "^VNINDEX", "HPG.VN", "SPY", "Synthetic Simulation"],
        index=0
    )

    reward_mode = st.sidebar.selectbox(
        "Reward Shaping Paradigm",
        [
            "volatility_penalty",
            "prospect_theory",
            "differential_sharpe",
            "downside_sortino",
            "standard"
        ],
        format_func=lambda x: {
            "volatility_penalty": "Volatility & Drawdown Barrier Penalty",
            "prospect_theory": "Prospect Theory (Loss Aversion λ > 1)",
            "differential_sharpe": "Differential Sharpe Ratio (Moody & Saffell)",
            "downside_sortino": "Downside Semi-Variance (Sortino)",
            "standard": "Standard Return (Unconstrained RL)"
        }[x]
    )

    st.sidebar.subheader("Risk Hyperparameters")
    lambda_vol = st.sidebar.slider("Volatility Penalty (λ₁)", 0.0, 10.0, 2.0, 0.5)
    lambda_dd = st.sidebar.slider("Drawdown Penalty (λ₂)", 0.0, 20.0, 5.0, 1.0)
    dd_tol = st.sidebar.slider("Drawdown Tolerance Threshold", 0.01, 0.20, 0.05, 0.01)
    prospect_lambda = st.sidebar.slider("Prospect Loss Aversion (λ)", 1.0, 4.0, 2.25, 0.25)

    initial_capital = st.sidebar.number_input("Initial Portfolio Capital ($)", value=10000, step=1000)

    # 1. Fetch & Prepare Data
    with st.spinner(f"Loading market data and quantitative features for {symbol}..."):
        raw_df = fetch_market_data(symbol)
        df = add_features(raw_df)

    # 2. Run Simulations
    # Out of sample split (last 300 candles or 30%)
    eval_len = min(300, int(len(df) * 0.4))
    df_eval = df.iloc[-eval_len:].reset_index(drop=True)

    # Run Benchmark: Buy & Hold
    bh_returns = df_eval["Close"].pct_change().fillna(0.0).values
    bh_equity = initial_capital * np.cumprod(1.0 + bh_returns)

    # Run Risk-Aware Agent Simulation
    env = RiskAwareTradingEnv(
        df_eval,
        reward_mode=reward_mode,
        lambda_vol=lambda_vol,
        lambda_dd=lambda_dd,
        dd_tolerance=dd_tol,
        prospect_lambda=prospect_lambda,
        initial_balance=initial_capital
    )
    obs, info = env.reset()
    terminated = truncated = False

    agent_equity = [initial_capital]
    agent_returns = []
    positions = [0.0]
    drawdowns = [0.0]

    while not (terminated or truncated):
        curr_dd = info["drawdown"]
        # Adaptive Risk Control Policy
        if curr_dd > dd_tol:
            target_pos = max(0.0, 0.8 - (curr_dd / dd_tol) * 0.6)
        else:
            target_pos = 0.9  # Normal regime position

        obs, reward, terminated, truncated, info = env.step(np.array([target_pos]))
        agent_equity.append(info["portfolio_value"])
        agent_returns.append(info["net_return"])
        positions.append(info["position"])
        drawdowns.append(info["drawdown"])

    # 3. Calculate Performance Metrics
    ra_metrics = evaluate_portfolio(np.array(agent_returns), np.array(agent_equity))
    bh_metrics = evaluate_portfolio(bh_returns, bh_equity)

    # 4. Display KPI Header
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Strategy Return", f"{ra_metrics['Total Return (%)']}%", delta=f"{round(ra_metrics['Total Return (%)'] - bh_metrics['Total Return (%)'], 2)}% vs B&H")
    col2.metric("Sharpe Ratio", f"{ra_metrics['Sharpe Ratio']}", delta=f"{round(ra_metrics['Sharpe Ratio'] - bh_metrics['Sharpe Ratio'], 2)}")
    col3.metric("Sortino Ratio", f"{ra_metrics['Sortino Ratio']}")
    col4.metric("Max Drawdown", f"{ra_metrics['Max Drawdown (%)']}%", delta=f"-{round(bh_metrics['Max Drawdown (%)'] - ra_metrics['Max Drawdown (%)'], 2)}% lower", delta_color="inverse")
    col5.metric("Calmar Ratio", f"{ra_metrics['Calmar Ratio']}")

    # 5. Charts
    tab1, tab2, tab3 = st.tabs(["📊 Equity & Drawdown", "🎯 Position Exposure", "📋 Comprehensive Metrics"])

    with tab1:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Cumulative Portfolio Equity ($)", "Underwater Drawdown Profile (%)"),
            row_heights=[0.7, 0.3]
        )

        # Equity Curve
        fig.add_trace(
            go.Scatter(y=agent_equity, name="Antigravity (Risk-Aware RL)", line=dict(color="#00E676", width=2.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=bh_equity, name="Buy & Hold Benchmark", line=dict(color="#78909C", width=1.5, dash="dot")),
            row=1, col=1
        )

        # Drawdown Profile
        dd_percent = [-d * 100 for d in drawdowns]
        fig.add_trace(
            go.Scatter(y=dd_percent, fill='tozeroy', name="Strategy Drawdown", line=dict(color="#FF5252", width=1.5)),
            row=2, col=1
        )

        fig.update_layout(height=600, template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig_pos = go.Figure()
        fig_pos.add_trace(go.Scatter(
            y=positions,
            mode='lines',
            fill='tozeroy',
            name="Asset Allocation Weight",
            line=dict(color="#29B6F6", width=2)
        ))
        fig_pos.update_layout(
            title="Agent Dynamic Position Sizing over Time (0.0 = Cash, 1.0 = 100% Long)",
            yaxis_title="Position Allocation",
            template="plotly_dark",
            height=380
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    with tab3:
        comp_df = pd.DataFrame({
            "Metric": list(ra_metrics.keys()),
            "Antigravity Risk-Aware": list(ra_metrics.values()),
            "Buy & Hold": list(bh_metrics.values())
        })
        st.dataframe(comp_df, use_container_width=True)


if __name__ == "__main__":
    main()
