"""Streamlit demo: Risk-Aware RL for VN30 portfolio allocation.

Read-only viewer of the saved backtest results in ``outputs/backtest_results/`` (written by
``04_Backtesting_Evaluation/backtester.py``). It never trains or simulates anything, so it runs
on a free host with only streamlit, pandas and plotly.

Run:
    streamlit run 05_Deployment_Demo/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "outputs" / "backtest_results"
REPO_URL = "https://github.com/tDn412/Risk-awareness-RL-Trading-bot"

YEARS = {
    "2022": ("window_1", "Thị trường sập", "VN30 giảm 35,5%, có lúc mất 42% từ đỉnh."),
    "2023": ("window_2", "Hồi phục nhẹ", "VN30 tăng 8%, dao động mạnh giữa năm."),
    "2024": ("window_3", "Phân hoá", "VN30 tăng 18,8%, nhóm ngân hàng dẫn dắt."),
    "2025": ("window_4", "Tăng mạnh", "VN30 tăng 51%, gần như mã nào cũng lên."),
}
DEFAULT_YEAR = "2024"

# Raw group names in the CSVs -> names a reader understands.
LABELS = {
    "PPO A0": "Bot A0 (chỉ thích lời)",
    "PPO A1": "Bot A1 (ghét dao động)",
    "PPO A2": "Bot A2 (ghét ngày lỗ)",
    "PPO A3": "Bot A3 (sợ lỗ)",
    "PPO-T0.03-K0.1 A2": "Bot A2 + giảm phí",
    "PPO-T0.03-K0.1 A3": "Bot A3 + giảm phí",
    "SAC A0": "SAC A0",
    "SAC A3": "SAC A3",
    "VN30 Index (Buy&Hold)": "VN30 (mua và giữ)",
    "Equal Weight 1/N": "Chia đều 30 mã",
    "Static 60/40": "60% cổ phiếu, 40% tiền",
    "100% Cash": "Giữ 100% tiền mặt",
}
BEST_BOT = ("PPO-T0.03-K0.1 A3", "PPO A3")  # first one present in a window is the headline bot
VN30 = "VN30 Index (Buy&Hold)"
EQUAL = "Equal Weight 1/N"

BOT_COLOR = "#1f6feb"
LINE_STYLES = {
    VN30: ("#1f2328", "solid", 2.0),
    EQUAL: ("#8c959f", "dash", 2.0),
    "Static 60/40": ("#b6bcc4", "dot", 1.6),
    "100% Cash": ("#d0d7de", "dot", 1.4),
}
OTHER_BOT_COLORS = ["#e8590c", "#2f9e44", "#ae3ec9", "#f59f00", "#0c8599", "#c2255c", "#5c7cfa"]

REWARDS = [
    ("A0", "Chỉ thích lời", "Thưởng = lợi nhuận sau phí. Không quan tâm rủi ro."),
    ("A1", "Ghét dao động", "Trừ điểm khi giá trị danh mục lên xuống thất thường."),
    ("A2", "Ghét ngày lỗ", "Phạt nặng ngày lỗ và phạt thêm khi sụt quá 15% từ đỉnh."),
    ("A3", "Sợ lỗ như người thật", "Lỗ đau gấp 2,25 lần lãi, canh 5% ngày tệ nhất."),
]


def pct(value: float, digits: int = 1, sign: bool = False) -> str:
    """Format a percentage the Vietnamese way (comma decimal)."""
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text.replace(".", ",") + "%"


@st.cache_data
def load_year(window: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Seed-averaged curves per strategy and per-run metrics for one window."""
    curves_raw = pd.read_csv(
        RESULTS_DIR / f"{window}_test_curves.csv", header=[0, 1], index_col=0, parse_dates=True
    )
    curves = {str(n): curves_raw[n].copy() for n in curves_raw.columns.get_level_values(0).unique()}
    runs = pd.read_csv(RESULTS_DIR / f"{window}_test_runs.csv")
    return curves, runs


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per strategy (EEB rows excluded), indexed by raw group name."""
    own = runs[runs["kind"] != "eeb"]
    table = own.groupby("group", sort=False).mean(numeric_only=True)
    table["seeds"] = own.groupby("group", sort=False).size()
    return table


def headline_bot(table: pd.DataFrame) -> str:
    """The bot shown by default: cost-aware A3 when it was run, plain A3 otherwise."""
    return next(name for name in BEST_BOT if name in table.index)


def verdict(year: str, bot: str, table: pd.DataFrame) -> str:
    """One plain sentence comparing the bot with VN30 and equal weight."""
    b, v, e = (table.loc[n, "Total Return (%)"] for n in (bot, VN30, EQUAL))
    beats = [LABELS[n] for n, r in ((VN30, v), (EQUAL, e)) if b > r]
    loses = [LABELS[n] for n, r in ((VN30, v), (EQUAL, e)) if b <= r]
    head = f"Năm {year}, {LABELS[bot]} {'lãi' if b >= 0 else 'lỗ'} {pct(abs(b))}"
    if not beats:
        return f"{head}, thua cả {' và '.join(loses)}."
    if not loses:
        return f"{head}, hơn cả {' và '.join(beats)}."
    return f"{head}: hơn {beats[0]} nhưng vẫn thua {loses[0]}."


def style_for(name: str, bot: str, other_bots: list[str]) -> tuple[str, str, float]:
    if name == bot:
        return BOT_COLOR, "solid", 3.0
    if name in LINE_STYLES:
        return LINE_STYLES[name]
    return OTHER_BOT_COLORS[other_bots.index(name) % len(OTHER_BOT_COLORS)], "solid", 1.8


def curve_chart(
    curves: dict[str, pd.DataFrame], names: list[str], bot: str, column: str, title: str, height: int
) -> go.Figure:
    """Line chart of NAV or drawdown for the chosen strategies."""
    other_bots = [n for n in names if n != bot and n not in LINE_STYLES]
    fig = go.Figure()
    for name in names:
        series = curves[name][column]
        if column == "drawdown":
            series = -series.abs()
        color, dash, width = style_for(name, bot, other_bots)
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                name=LABELS.get(name, name),
                mode="lines",
                line={"color": color, "dash": dash, "width": width},
                hovertemplate="%{y:.1f}",
            )
        )
    if column == "nav":
        fig.add_hline(y=100, line={"color": "#d0d7de", "width": 1})
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15}},
        height=height,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1, "xanchor": "right"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, tickformat="%m/%Y")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", ticksuffix="%" if column == "drawdown" else "")
    return fig


def replay_chart(curves: dict[str, pd.DataFrame], bot: str, year: str, step: int = 3) -> go.Figure:
    """Animated day-by-day replay: NAV of bot / VN30 / equal weight and the bot's cash weight."""
    names = [bot, VN30, EQUAL]
    nav = pd.DataFrame({n: curves[n]["nav"] for n in names})
    cash = curves[bot]["cash_weight"].bfill().fillna(0).rename("cash")  # already in %
    dates = nav.index
    ends = list(range(1, len(dates), step))
    if ends[-1] != len(dates) - 1:
        ends.append(len(dates) - 1)
    styles = {bot: (BOT_COLOR, "solid", 3.0), VN30: LINE_STYLES[VN30], EQUAL: LINE_STYLES[EQUAL]}

    def traces(k: int) -> list[go.Scatter]:
        out = [
            go.Scatter(x=dates[: k + 1], y=nav[n].iloc[: k + 1], mode="lines", name=LABELS[n],
                       line={"color": styles[n][0], "dash": styles[n][1], "width": styles[n][2]})
            for n in names
        ]
        out.append(go.Scatter(x=dates[: k + 1], y=cash.iloc[: k + 1], mode="lines", name="Tiền mặt bot giữ",
                              line={"color": BOT_COLOR, "width": 1.5}, fill="tozeroy",
                              fillcolor="rgba(31,111,235,0.15)", showlegend=False))
        return out

    def label(k: int) -> str:
        parts = [f"<b>{dates[k]:%d/%m/%Y}</b>"] + [
            f"{LABELS[n]}: {pct(nav[n].iloc[k] - 100, sign=True)}" for n in names
        ] + [f"Tiền mặt bot giữ: {pct(cash.iloc[k], 0)}"]
        return "<br>".join(parts)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.06,
                        subplot_titles=(f"Phát lại năm {year}: giá trị danh mục (bắt đầu = 100)",
                                        "Tỷ lệ tiền mặt bot giữ (%)"))
    for i, tr in enumerate(traces(ends[0])):
        fig.add_trace(tr, row=1 if i < 3 else 2, col=1)
    annotation = {"x": 0.01, "y": 0.98, "xref": "paper", "yref": "paper", "xanchor": "left", "yanchor": "top",
                  "align": "left", "showarrow": False, "bgcolor": "rgba(255,255,255,0.85)",
                  "bordercolor": "#d0d7de", "borderwidth": 1, "font": {"size": 12}}
    base_annotations = list(fig.layout.annotations)
    fig.frames = [
        go.Frame(data=traces(k), name=str(k), traces=[0, 1, 2, 3],
                 layout={"annotations": base_annotations + [dict(annotation, text=label(k))]})
        for k in ends
    ]
    lo = min(nav.min().min(), 100) - 3
    hi = max(nav.max().max(), 100) + 3
    fig.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        annotations=base_annotations + [dict(annotation, text=label(ends[0]))],
        legend={"orientation": "h", "y": 1.13, "x": 1, "xanchor": "right"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.0, "y": 1.13, "xanchor": "left", "showactive": False,
            "buttons": [
                {"label": "▶ Phát", "method": "animate",
                 "args": [None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True,
                                 "transition": {"duration": 0}}]},
                {"label": "⏸ Dừng", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "x": 0.0, "len": 1.0, "y": -0.02, "pad": {"t": 30}, "currentvalue": {"visible": False},
            "steps": [{"label": f"{dates[k]:%m/%Y}" if dates[k].month != dates[max(k - step, 0)].month else "", "method": "animate",
                       "args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                      for k in ends],
        }],
    )
    fig.update_xaxes(range=[dates[0], dates[-1]], showgrid=False, tickformat="%m/%Y")
    fig.update_yaxes(range=[lo, hi], gridcolor="rgba(128,128,128,0.18)", row=1, col=1)
    fig.update_yaxes(range=[0, max(cash.max() * 1.1, 10)], ticksuffix="%", gridcolor="rgba(128,128,128,0.18)",
                     row=2, col=1)
    return fig


def four_year_chart(rows: pd.DataFrame) -> go.Figure:
    """Grouped bars: return of the headline bot, VN30 and equal weight in each test year."""
    fig = go.Figure()
    for column, color in (("Bot", BOT_COLOR), ("VN30", "#1f2328"), ("Chia đều 30 mã", "#8c959f")):
        fig.add_trace(
            go.Bar(
                x=rows.index,
                y=rows[column],
                name="Bot A3" if column == "Bot" else column,
                marker_color=color,
                text=[pct(v) for v in rows[column]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"{column}: %{{y:.1f}}%<extra></extra>",
            )
        )
    fig.add_hline(y=0, line={"color": "#8c959f", "width": 1})
    fig.update_layout(
        barmode="group",
        height=360,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(type="category")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", ticksuffix="%", title="Lợi nhuận cả năm")
    return fig


def all_years() -> pd.DataFrame:
    """Return per year for the headline bot, VN30 and equal weight, plus fee rows for the cost section."""
    rows = {}
    for year, (window, _, _) in YEARS.items():
        table = summary(load_year(window)[1])
        bot = headline_bot(table)
        rows[year] = {
            "Bot": table.loc[bot, "Total Return (%)"],
            "Bot name": LABELS[bot],
            "VN30": table.loc[VN30, "Total Return (%)"],
            "Chia đều 30 mã": table.loc[EQUAL, "Total Return (%)"],
            "A3 return": table.loc["PPO A3", "Total Return (%)"],
            "A3 fee": table.loc["PPO A3", "Friction (%/yr)"],
            "A3c return": table.loc[bot, "Total Return (%)"] if bot != "PPO A3" else None,
            "A3c fee": table.loc[bot, "Friction (%/yr)"] if bot != "PPO A3" else None,
        }
    return pd.DataFrame(rows).T


def stat_card(col, title: str, row: pd.Series, highlight: bool = False) -> None:
    with col.container(border=True):
        st.markdown(f"**{'🔵 ' if highlight else ''}{title}**")
        st.metric("Lợi nhuận cả năm", pct(row["Total Return (%)"], sign=True))
        a, b = st.columns(2)
        a.metric("Sụt tối đa", pct(row["Max Drawdown (%)"]), help="Mức giảm lớn nhất từ đỉnh xuống đáy trong năm.")
        fee = row["Friction (%/yr)"]
        b.metric("Phí + thuế/năm", "—" if pd.isna(fee) else pct(fee), help="Phần vốn mất vào phí môi giới và thuế bán.")


CSS = """
<style>
.block-container {max-width: 1100px; padding-top: 3rem;}
.lead {font-size: 1.1rem; opacity: .85; margin-bottom: .2rem;}
.small {font-size: .85rem; opacity: .65;}
.verdict {font-size: 1.15rem; padding: .9rem 1.1rem; border-left: 4px solid #1f6feb;
          background: rgba(31,111,235,.08); border-radius: 6px; margin: .5rem 0 1rem;}
</style>
"""


def main() -> None:
    st.set_page_config(page_title="Bot đầu tư VN30 nhận thức rủi ro", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    if not (RESULTS_DIR / "window_1_test_runs.csv").exists():
        st.error(f"Không tìm thấy kết quả backtest trong `{RESULTS_DIR}`.")
        return

    # ---- Header
    st.title("Bot phân bổ danh mục VN30 có nhận thức rủi ro", anchor=False)
    st.markdown(
        '<div class="lead">Một tác tử học tăng cường (PPO) mỗi ngày quyết định chia 1 tỷ đồng vào 30 cổ phiếu VN30 '
        "và tiền mặt. Trang này cho xem nó đã làm được gì trên 4 năm dữ liệu thật mà nó chưa từng thấy khi học.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="small">Nhóm 02 · INT3011E Các vấn đề hiện đại trong KHMT · UET, ĐHQGHN · '
        "Kết quả mô phỏng quá khứ, không phải khuyến nghị đầu tư.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    with st.container(border=True):
        st.markdown(
            "**Cách xem trong 30 giây:** "
            "① chọn một năm ở bên dưới → "
            "② so đường xanh (bot) với đường đen (VN30) → "
            "③ đọc câu kết luận và phần tổng kết 4 năm ở cuối trang."
        )

    # ---- Problem
    st.subheader("Bài toán")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**Bot làm gì?**")
        st.markdown(
            "Sau mỗi phiên, bot nhìn 422 con số (giá, chỉ báo của 30 mã, danh mục đang cầm) "
            "rồi đưa ra 31 tỷ trọng: 30 mã + tiền mặt, tổng 100%."
        )
    with c2.container(border=True):
        st.markdown("**Luật sàn HOSE được mô phỏng**")
        st.markdown("Mua hôm nay T+2 mới bán được · biên độ ±7% · lô 100 cổ phiếu · phí 0,15% mỗi chiều · thuế bán 0,1%.")
    with c3.container(border=True):
        st.markdown("**Vì sao \"nhận thức rủi ro\"?**")
        st.markdown(
            "Bot chỉ ham lời sẽ dồn vào mã biến động mạnh và cháy khi thị trường sập. "
            "Nhóm thử 4 cách chấm điểm (A0→A3) từ \"chỉ thích lời\" đến \"sợ lỗ\"."
        )
    with st.expander("Bốn cách chấm điểm (hàm thưởng) A0–A3"):
        for key, name, text in REWARDS:
            st.markdown(f"- **{key} · {name}:** {text}")
        st.markdown(
            "\"**+ giảm phí**\" là cùng bot đó nhưng mỗi ngày chỉ đi 10% quãng đường tới tỷ trọng mới, "
            "kèm phạt nhẹ khi giao dịch nhiều. Cách này đưa phí từ 9–12% xuống còn 1,6–4,3% vốn mỗi năm."
        )

    # ---- Year view
    st.subheader("Kết quả từng năm")
    year = st.segmented_control(
        "Chọn năm kiểm thử",
        list(YEARS),
        default=DEFAULT_YEAR,
        format_func=lambda y: f"{y} · {YEARS[y][1]}",
    ) or DEFAULT_YEAR
    window, regime, note = YEARS[year]
    curves, runs = load_year(window)
    table = summary(runs)
    bot = headline_bot(table)
    st.caption(
        f"{note} Bot học trên dữ liệu 2018–{int(year) - 2}, chọn phiên bản tốt nhất trên năm {int(year) - 1}, "
        f"rồi chạy thử trên {year} mà không được xem trước."
    )

    st.markdown(f'<div class="verdict">{verdict(year, bot, table)}</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    stat_card(cols[0], LABELS[bot], table.loc[bot], highlight=True)
    stat_card(cols[1], LABELS[VN30], table.loc[VN30])
    stat_card(cols[2], LABELS[EQUAL], table.loc[EQUAL])
    if bot == "PPO A3":
        st.caption(
            f"Năm 2022 chưa chạy bản \"giảm phí\" nên đây là Bot A3 thường, trung bình {int(table.loc[bot, 'seeds'])} lần huấn luyện."
        )

    others = [n for n in curves if n not in (bot, VN30, EQUAL) and n in LABELS]
    extra = st.multiselect(
        "So thêm với chiến lược khác (tuỳ chọn)",
        others,
        format_func=lambda n: LABELS[n],
        placeholder="Ví dụ: Bot A0 để thấy bot chỉ ham lời lỗ thế nào",
    )
    names = [bot, VN30, EQUAL] + extra
    replay_tab, full_tab = st.tabs(["▶ Phát lại từng ngày", "Cả năm và mức sụt"])
    with replay_tab:
        st.caption(
            "Bấm ▶ Phát để xem bot đi qua từng phiên của năm, so với VN30 và chia đều; "
            "kéo thanh trượt để nhảy tới một ngày bất kỳ. Ô góc trái cho biết lãi/lỗ tới ngày đó."
        )
        st.plotly_chart(replay_chart(curves, bot, year), use_container_width=True, config={"displayModeBar": False})
    with full_tab:
        st.plotly_chart(
            curve_chart(curves, names, bot, "nav", f"Giá trị danh mục năm {year} (bắt đầu = 100)", 400),
            use_container_width=True,
        )
        st.plotly_chart(
            curve_chart(curves, names, bot, "drawdown", "Mức sụt so với đỉnh gần nhất (%)", 260),
            use_container_width=True,
        )

    # ---- Four years
    st.subheader("Tổng kết 4 năm")
    years = all_years()
    st.plotly_chart(four_year_chart(years), use_container_width=True)
    st.markdown(
        "- **Thưởng nhận thức rủi ro có ích:** năm nào Bot A3 \"sợ lỗ\" cũng tốt hơn Bot A0 \"chỉ thích lời\", "
        "ví dụ năm 2024 là +7,5% so với −14,9%.\n"
        "- **Chưa thắng thị trường một cách thuyết phục:** năm 2022 bot lỗ ít hơn VN30 (−32,0% so với −35,5%) "
        "nhưng vẫn thua cách chia đều; 2023–2025 bot đều thua VN30. Năm 2024 bot lần đầu thắng chia đều "
        "(16,6% so với 15,7%).\n"
        "- Cột xanh năm 2022 là Bot A3 thường; từ 2023 là Bot A3 + giảm phí."
    )

    st.subheader("Nút thắt là phí giao dịch")
    fees = years.loc[years["A3c fee"].notna(), ["A3 return", "A3 fee", "A3c return", "A3c fee"]]
    fees = fees.astype(float).map(pct)
    fees.columns = ["Lợi nhuận (thường)", "Phí/năm (thường)", "Lợi nhuận (giảm phí)", "Phí/năm (giảm phí)"]
    fees.index.name = "Năm"
    st.markdown(
        "Bot thường đổi 10–30% danh mục mỗi ngày nên mất rất nhiều vào phí. "
        "Chỉ cho bot đi 10% quãng đường mỗi ngày làm phí giảm 3–6 lần và lợi nhuận tăng ở cả 3 năm."
    )
    st.table(fees)

    st.subheader("Hạn chế (nói thẳng)")
    st.markdown(
        "- Mỗi năm 2023–2025 mới huấn luyện **1 lần** (seed 42); năm 2022 có 3 lần. Kết quả có thể đổi khi chạy lại.\n"
        "- So với chiến lược cùng mức nắm giữ cổ phiếu (EEB), chênh lệch **chưa có ý nghĩa thống kê** (p = 0,28 năm 2024).\n"
        "- Lệnh được khớp ở giá đóng cửa của chính phiên bot vừa quan sát, nên kết quả có thể hơi lạc quan.\n"
        "- Danh sách VN30 cố định theo hiện tại, chưa tính các lần đổi rổ trong quá khứ."
    )

    with st.expander("Chi tiết kỹ thuật: bảng đủ chỉ số và so với EEB"):
        detail = table.drop(columns=["seed", "p_vs_EEB", "mean_daily_excess_vs_EEB (%)"], errors="ignore")
        detail.index = [LABELS.get(n, n) for n in detail.index]
        st.dataframe(detail.round(2), use_container_width=True)
        agents = runs[runs["kind"] == "agent"]
        eeb = runs[runs["kind"] == "eeb"].assign(group=lambda d: d["group"].str.replace("EEB of ", "", regex=False))
        eeb_table = pd.DataFrame(
            {
                "Bot (%)": agents.groupby("group", sort=False)["Total Return (%)"].mean(),
                "EEB (%)": eeb.groupby("group", sort=False)["Total Return (%)"].mean(),
                "p-value": agents.groupby("group", sort=False)["p_vs_EEB"].mean(),
            }
        )
        eeb_table.index = [LABELS.get(n, n) for n in eeb_table.index]
        st.caption(
            "EEB giữ cổ phiếu chia đều với đúng tỷ lệ cổ phiếu/tiền mặt mà bot chọn mỗi ngày, "
            "để tách phần chọn mã của bot khỏi việc đơn giản là giữ nhiều tiền mặt."
        )
        st.dataframe(eeb_table.round(3), use_container_width=True)

    st.divider()
    st.markdown(
        f'<div class="small">Mã nguồn và dữ liệu kết quả: <a href="{REPO_URL}">{REPO_URL}</a> · '
        "Nhữ Đình Tú, Nguyễn Văn Mạnh, Đoàn Thái Hùng</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
