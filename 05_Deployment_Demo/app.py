"""Streamlit dashboard: Risk-Aware RL for VN30 portfolio allocation.

Read-only viewer of the saved backtest results in ``outputs/backtest_results/`` (written by
``04_Backtesting_Evaluation/backtester.py``). It never trains or simulates anything.

Run:
    streamlit run 05_Deployment_Demo/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "outputs" / "backtest_results"
DEFAULT_WINDOW = "window_1"
TEST_PERIODS = {"window_1": "2022 (bear market)"}

REWARD_COLORS = {"A0": "#2a78d6", "A1": "#eb6834", "A2": "#1baf7a", "A3": "#e87ba4"}
BASELINE_STYLES = {
    "VN30 Index (Buy&Hold)": ("#0b0b0b", "solid"),
    "Equal Weight 1/N": ("#52514e", "dash"),
    "Static 60/40": ("#8a8984", "dot"),
    "100% Cash": ("#b4b3ad", "dashdot"),
}
METRIC_COLUMNS = [
    "Total Return (%)",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Max Drawdown (%)",
    "Calmar Ratio",
    "CVaR (95%)",
    "Annualized Volatility (%)",
    "Avg Cash Weight",
    "Avg Turnover",
    "Friction (%/yr)",
]
CHART_HEIGHT = 380

REWARD_DESCRIPTIONS = {
    "A0": "Risk-neutral: net return only.",
    "A1": "Net return minus a rolling-variance penalty.",
    "A2": "Net return minus downside semi-variance and a drawdown barrier (d* = 15%).",
    "A3": "Prospect-theory utility minus CVaR-95 and drawdown barriers.",
}


def result_paths(window: str) -> tuple[Path, Path]:
    """Return ``(curves_csv, runs_csv)`` for a window."""
    return RESULTS_DIR / f"{window}_test_curves.csv", RESULTS_DIR / f"{window}_test_runs.csv"


def load_curves(path: Path) -> dict[str, pd.DataFrame]:
    """Load seed-averaged curves (``nav``, ``drawdown``, ``cash_weight``) per strategy."""
    frame = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return {str(name): frame[name].copy() for name in frame.columns.get_level_values(0).unique()}


def load_runs(path: Path) -> pd.DataFrame:
    """Load per-run metrics (agents, EEB and baselines)."""
    return pd.read_csv(path)


def metric_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per strategy with the number of seeds, EEB rows excluded."""
    own = runs[runs["kind"] != "eeb"]
    grouped = own.groupby("group", sort=False)
    table = grouped[METRIC_COLUMNS].mean().round(2)
    table.insert(0, "Seeds", grouped.size())
    return table


def eeb_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Agent vs its Equivalent Equity Benchmark (same equity exposure, equal-weight stocks)."""
    agents = runs[runs["kind"] == "agent"].copy()
    eeb = runs[runs["kind"] == "eeb"].copy()
    eeb["group"] = eeb["group"].str.replace("EEB of ", "", regex=False)
    agent_ret = agents.groupby("group", sort=False)["Total Return (%)"].mean()
    eeb_ret = eeb.groupby("group", sort=False)["Total Return (%)"].mean()
    table = pd.DataFrame(
        {
            "Agent return (%)": agent_ret,
            "EEB return (%)": eeb_ret,
            "Agent − EEB (pp)": agent_ret - eeb_ret,
            "Agent friction (%/yr)": agents.groupby("group", sort=False)["Friction (%/yr)"].mean(),
            "EEB friction (%/yr)": eeb.groupby("group", sort=False)["Friction (%/yr)"].mean(),
            "p-value (paired t, daily)": agents.groupby("group", sort=False)["p_vs_EEB"].mean(),
        }
    )
    return table.round(3)


INDEX_NAME = "VN30 Index (Buy&Hold)"


def result_summary(table: pd.DataFrame) -> str:
    """One-paragraph summary of agents vs the VN30 index, computed from the metric table."""
    agents = table[[name.split()[-1] in REWARD_COLORS for name in table.index]]
    if agents.empty or INDEX_NAME not in table.index:
        return ""
    index_ret = float(table["Total Return (%)"].astype(float)[INDEX_NAME])
    beat = [name for name in agents.index if agents.loc[name, "Total Return (%)"] > index_ret]
    best = str(agents["Total Return (%)"].idxmax())
    turnover = agents["Avg Turnover"] * 100
    friction = agents["Friction (%/yr)"]
    beat_text = ", ".join(beat) if beat else "none"
    return (
        f"**Result on the test period:** VN30 index returned {index_ret:.1f}%. "
        f"Agents beating it: {beat_text}. Best agent: {best} "
        f"({agents.loc[best, 'Total Return (%)']:.1f}%, max drawdown {agents.loc[best, 'Max Drawdown (%)']:.1f}%). "
        f"Agent daily turnover {turnover.min():.1f}–{turnover.max():.1f}%, "
        f"fees and tax {friction.min():.1f}–{friction.max():.1f}% per year."
    )


def series_style(name: str) -> tuple[str, str, float]:
    """Return ``(color, dash, width)`` for a strategy (agents by reward, baselines neutral)."""
    if name in BASELINE_STYLES:
        color, dash = BASELINE_STYLES[name]
        return color, dash, 2.0
    reward = name.split()[-1]
    return REWARD_COLORS.get(reward, "#52514e"), "dot" if name.startswith("SAC") else "solid", 2.0


def line_chart(curves: dict[str, pd.DataFrame], names: list[str], column: str, title: str, unit: str) -> go.Figure:
    """Line chart of one curve column for the selected strategies."""
    fig = go.Figure()
    for name in names:
        if column == "cash_weight" and curves[name][column].isna().all():
            continue
        series = curves[name][column].iloc[1:] if column == "cash_weight" else curves[name][column]
        color, dash, width = series_style(name)
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                name=name,
                mode="lines",
                line={"color": color, "dash": dash, "width": width},
                hovertemplate=f"{name}: %{{y:.1f}}{unit}<extra></extra>",
            )
        )
    fig.update_layout(
        title={"text": title, "x": 0.0, "font": {"size": 15}},
        height=CHART_HEIGHT,
        margin={"l": 8, "r": 8, "t": 44, "b": 8},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.3},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", ticksuffix=unit)
    return fig


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="Risk-Aware RL · VN30", page_icon="📉", layout="wide")
    st.title("Risk-Aware RL for VN30 portfolio allocation")
    st.caption("INT3011E · PPO / SAC agents, four reward schemes (A0–A3), HOSE execution rules · demo of saved results")

    window = DEFAULT_WINDOW
    curves_path, runs_path = result_paths(window)
    if not curves_path.exists() or not runs_path.exists():
        st.error(
            f"No backtest results found in `{RESULTS_DIR}`. Run "
            "`python 04_Backtesting_Evaluation/backtester.py --window window_1` first."
        )
        return
    curves = load_curves(curves_path)
    runs = load_runs(runs_path)
    table = metric_table(runs)

    agent_names = [name for name in curves if name.split()[-1] in REWARD_COLORS]
    baseline_names = [name for name in curves if name in BASELINE_STYLES]

    with st.sidebar:
        st.header("View")
        st.write(f"**Test period:** {TEST_PERIODS[window]}")
        st.caption("Agents trained 2018–2020, checkpoint chosen on 2021 validation, evaluated on unseen 2022.")
        chosen_agents = st.multiselect("Agents", agent_names, default=[n for n in agent_names if n.startswith("PPO")])
        chosen_baselines = st.multiselect("Baselines", baseline_names, default=baseline_names[:2])
        st.divider()
        st.markdown("**Reward schemes**")
        for key, text in REWARD_DESCRIPTIONS.items():
            st.markdown(f"- **{key}** — {text}")

    selected = chosen_agents + chosen_baselines
    if not selected:
        st.info("Select at least one strategy in the sidebar.")
        return

    summary_text = result_summary(table)
    if summary_text:
        st.info(summary_text)

    cols = st.columns(min(len(selected), 4))
    for col, name in zip(cols, selected[:4]):
        row = table.loc[name]
        col.metric(name, f"{row['Total Return (%)']:.1f}%", f"MDD {row['Max Drawdown (%)']:.1f}%", delta_color="off")

    left, right = st.columns(2)
    left.plotly_chart(line_chart(curves, selected, "nav", "NAV (start = 100)", ""), use_container_width=True)
    right.plotly_chart(line_chart(curves, selected, "drawdown", "Drawdown", "%"), use_container_width=True)
    if chosen_agents:
        st.plotly_chart(
            line_chart(curves, chosen_agents, "cash_weight", "Agent cash weight", "%"), use_container_width=True
        )

    st.subheader("Performance metrics (test period, mean over seeds)")
    st.dataframe(table.loc[selected], use_container_width=True)
    with st.expander("All strategies"):
        st.dataframe(table, use_container_width=True)

    st.subheader("Agent vs Equivalent Equity Benchmark (H3)")
    st.caption(
        "EEB holds equal-weight stocks at the agent's own daily equity exposure, so the gap isolates "
        "stock-selection and timing from simple de-risking — and, here, from transaction costs."
    )
    st.dataframe(eeb_table(runs), use_container_width=True)

    with st.expander("HOSE rules enforced in the simulator"):
        st.markdown(
            "- T+2 settlement: shares bought on day T cannot be sold on T or T+1\n"
            "- ±7% daily price band; no buying at the ceiling, no selling at the floor\n"
            "- 100-share lots, long-only, cash as asset 0 (31-dim simplex action)\n"
            "- 0.15% brokerage on both sides and 0.1% tax on sells"
        )


if __name__ == "__main__":
    main()
