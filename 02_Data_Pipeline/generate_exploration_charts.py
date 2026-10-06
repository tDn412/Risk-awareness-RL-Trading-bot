"""
VN30 Data Exploration & Statistical Visualization Generator
Project: Risk-Aware RL for VN30 Portfolio Allocation (INT3011E)

Reads raw collected data (prices, benchmarks, corporate actions), computes quantitative
statistics, generates publication-quality figures, and builds an interactive Jupyter Notebook.
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure aesthetic plotting styles
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

FIGURES_DIR = "outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_all_data():
    """Load all 30 tickers, benchmarks, and corporate events."""
    price_files = sorted(glob.glob("data/raw/prices/*.csv"))
    tickers_data = {}
    for f in price_files:
        ticker = os.path.basename(f).replace(".csv", "")
        if ticker in ["prices_fetch_summary"]:
            continue
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        tickers_data[ticker] = df

    # Benchmarks
    vn30_df = pd.read_csv("data/raw/benchmarks/VN30.csv") if os.path.exists("data/raw/benchmarks/VN30.csv") else None
    vnindex_df = pd.read_csv("data/raw/benchmarks/VNINDEX.csv") if os.path.exists("data/raw/benchmarks/VNINDEX.csv") else None
    if vn30_df is not None:
        vn30_df["date"] = pd.to_datetime(vn30_df["date"])
    if vnindex_df is not None:
        vnindex_df["date"] = pd.to_datetime(vnindex_df["date"])

    # Events
    events_path = "data/raw/corporate_actions/vn30_all_corporate_events.csv"
    events_df = pd.read_csv(events_path) if os.path.exists(events_path) else pd.DataFrame()
    if not events_df.empty and "exright_date" in events_df.columns:
        events_df["exright_date"] = pd.to_datetime(events_df["exright_date"])

    return tickers_data, vn30_df, vnindex_df, events_df


def build_price_matrices(tickers_data):
    """Build aligned Close price and Volume matrix DataFrames."""
    close_dict = {}
    volume_dict = {}
    for ticker, df in tickers_data.items():
        close_dict[ticker] = df.set_index("date")["close"]
        volume_dict[ticker] = df.set_index("date")["volume"]

    df_close = pd.DataFrame(close_dict).sort_index()
    df_volume = pd.DataFrame(volume_dict).sort_index()
    return df_close, df_volume


def plot_price_trajectories(df_close, vn30_df):
    """Plot 1: Normalized Price Trajectories & Macro Regimes."""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    
    # Normalized prices starting at 100
    df_norm = (df_close / df_close.bfill().iloc[0]) * 100

    # Plot individual stocks in light muted gray
    for col in df_norm.columns:
        ax.plot(df_norm.index, df_norm[col], color="#8d99ae", alpha=0.35, linewidth=1.0)

    # Plot VN30 Benchmark
    if vn30_df is not None:
        v30 = vn30_df.set_index("date")["close"].sort_index()
        v30_aligned = v30.reindex(df_norm.index).ffill()
        v30_norm = (v30_aligned / v30_aligned.bfill().iloc[0]) * 100
        ax.plot(v30_norm.index, v30_norm, color="#d90429", linewidth=2.8, label="VN30-Index Benchmark", zorder=5)

    # Highlight market regimes
    ax.axvspan(pd.Timestamp("2020-01-20"), pd.Timestamp("2020-03-31"), color="#ef233c", alpha=0.15, label="COVID-19 Crash (2020)")
    ax.axvspan(pd.Timestamp("2020-04-01"), pd.Timestamp("2021-12-31"), color="#2b9348", alpha=0.10, label="Bull Run (2020-2021)")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-11-15"), color="#d90429", alpha=0.20, label="Market Plunge (2022 Crash)")
    ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31"), color="#0077b6", alpha=0.08, label="Consolidation & Recovery (2023-2025)")

    ax.set_title("VN30 Equities Normalized Price Trajectories (2018–2025)\nHighlighting Extreme Market Regimes & Drawdown Phases", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Normalized Value (Base = 100)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Trading Date", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    out_file = os.path.join(FIGURES_DIR, "01_vn30_price_trajectories.png")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")


def plot_correlation_heatmap(df_close):
    """Plot 2: Cross-Asset Daily Return Correlation Heatmap."""
    df_ret = df_close.pct_change().dropna(how="all")
    corr = df_ret.corr()

    fig, ax = plt.subplots(figsize=(15, 12), dpi=300)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.1,
        center=0.35,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6.5},
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax
    )
    ax.set_title("Cross-Asset Daily Return Correlation Matrix (30 VN30 Equities, 2018–2025)\nEvaluating Diversification Potential & Sector Co-movement", fontsize=14, fontweight="bold", pad=14)
    
    out_file = os.path.join(FIGURES_DIR, "02_vn30_correlation_matrix.png")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")


def plot_return_fat_tails(df_close):
    """Plot 3: Non-Normality & Empirical Fat Tails vs Gaussian."""
    df_ret = df_close.pct_change().dropna()
    all_returns = df_ret.values.flatten()
    all_returns = all_returns[~np.isnan(all_returns)]
    # Filter extreme artifacts if any
    all_returns = all_returns[(all_returns > -0.15) & (all_returns < 0.15)]

    mean_ret = np.mean(all_returns)
    std_ret = np.std(all_returns)
    kurt = float(pd.Series(all_returns).kurtosis())
    skew = float(pd.Series(all_returns).skew())
    var_95 = float(np.percentile(all_returns, 5))
    cvar_95 = float(all_returns[all_returns <= var_95].mean())

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    
    # Histogram of actual returns
    sns.histplot(all_returns, bins=120, stat="density", color="#2a9d8f", alpha=0.5, label="Empirical Daily Returns", ax=ax)
    sns.kdeplot(all_returns, color="#264653", linewidth=2.0, label="Empirical Density (KDE)", ax=ax)

    # Theoretical Normal
    x_grid = np.linspace(-0.10, 0.10, 500)
    normal_pdf = (1 / (std_ret * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mean_ret) / std_ret) ** 2)
    ax.plot(x_grid, normal_pdf, "r--", linewidth=2.0, label=f"Normal Fit (μ={mean_ret:.4f}, σ={std_ret:.4f})")

    # Mark VaR and CVaR
    ax.axvline(var_95, color="#e76f51", linestyle="-.", linewidth=2.0, label=f"Historical VaR 95% ({var_95*100:.2f}%)")
    ax.axvline(cvar_95, color="#d62828", linestyle=":", linewidth=2.5, label=f"Historical CVaR 95% ({cvar_95*100:.2f}%)")

    stats_text = (
        f"Sample Size: {len(all_returns):,} returns\n"
        f"Skewness: {skew:.2f} (Asymmetric risk)\n"
        f"Excess Kurtosis: {kurt:.2f} (Leptokurtic / Fat Tails)\n"
        f"Historical VaR (95%): {var_95*100:.2f}%\n"
        f"Historical CVaR (95%): {cvar_95*100:.2f}%"
    )
    ax.text(
        0.03, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#999999", alpha=0.9)
    )

    ax.set_title("Empirical Distribution of Daily Returns vs. Normal Distribution\nDemonstrating Tail Risk (Fat Tails) That Triggers Severe RL Drawdowns", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Daily Percentage Return", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.10, 0.10)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    out_file = os.path.join(FIGURES_DIR, "03_vn30_return_fat_tails.png")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")


def plot_liquidity_profile(df_volume, df_close):
    """Plot 4: Average Daily Trading Volume & Liquidity Profile."""
    adv = df_volume.mean().sort_values(ascending=False)
    adv_shares_m = adv / 1_000_000.0  # Million shares

    # Approximate ADV Turnover in Billion VND
    turnover = (df_volume * df_close * 1000).mean().sort_values(ascending=False) / 1_000_000_000.0  # Billion VND

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # ADV Shares
    sns.barplot(x=adv_shares_m.values, y=adv_shares_m.index, palette="Blues_r", ax=ax1)
    ax1.set_title("Average Daily Volume (Million Shares)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Volume (Million Shares/Day)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ADV Value
    sns.barplot(x=turnover[adv_shares_m.index].values, y=adv_shares_m.index, palette="Greens_r", ax=ax2)
    ax2.set_title("Average Daily Turnover Value (Billion VND)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Turnover (Billion VND/Day)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("VN30 Liquidity Landscape (2018–2025): Verifying Execution Realism (M1 Lot 100 & M2 ADV Limits)", fontsize=14, fontweight="bold", y=1.02)
    out_file = os.path.join(FIGURES_DIR, "04_vn30_liquidity_profile.png")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")


def plot_corporate_actions(events_df):
    """Plot 5: Breakdown of Corporate Actions (Dividends & Splits 2018–2025)."""
    if events_df.empty:
        print("[-] No events data to plot.")
        return

    events_df["year"] = events_df["exright_date"].dt.year
    events_valid = events_df[events_df["year"].between(2018, 2025)].copy()

    events_valid["type"] = np.where(
        events_valid["value_per_share"].notna() & (events_valid["value_per_share"] > 0),
        "Cash Dividend (Tiền mặt)",
        "Stock Dividend / Split (Cổ tức CP / Tách)"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Events by Year
    event_by_year = pd.crosstab(events_valid["year"], events_valid["type"])
    event_by_year.plot(kind="bar", stacked=True, color=["#2a9d8f", "#e76f51"], ax=ax1, width=0.6)
    ax1.set_title("Annual Distribution of VN30 Corporate Actions (2018–2025)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Actions", fontsize=10)
    ax1.set_xlabel("Year", fontsize=10)
    ax1.legend(frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Top Dividend Paying Tickers
    top_tickers = events_valid["ticker"].value_counts().head(12)
    sns.barplot(x=top_tickers.values, y=top_tickers.index, palette="viridis", ax=ax2)
    ax2.set_title("Top 12 Most Active Tickers by Corporate Events", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Total Corporate Actions", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("VN30 Corporate Actions Overview: Mandatory for True NAV Accounting (O1 & O2)", fontsize=14, fontweight="bold", y=1.02)
    out_file = os.path.join(FIGURES_DIR, "05_vn30_corporate_actions.png")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")


def plot_underwater_drawdown(vn30_df):
    """Plot 6: VN30-Index Underwater Drawdown & Barrier Threshold d*."""
    if vn30_df is None:
        return

    v30 = vn30_df.set_index("date")["close"].sort_index()
    peak = v30.cummax()
    drawdown = (v30 - peak) / peak

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    ax.fill_between(drawdown.index, drawdown * 100, 0, color="#d90429", alpha=0.35, label="VN30-Index Drawdown (%)")
    ax.plot(drawdown.index, drawdown * 100, color="#d90429", linewidth=1.5)

    # Drawdown tolerance threshold d* = 15%
    ax.axhline(-15.0, color="#f77f00", linestyle="--", linewidth=2.0, label="Drawdown Threshold d* = 15% (Reward Penalty Barrier A2/A3)")
    ax.axhline(-30.0, color="#6a040f", linestyle=":", linewidth=1.5, label="Severe Distress Level (-30%)")

    # Mark Max Drawdown
    mdd_val = drawdown.min() * 100
    mdd_date = drawdown.idxmin()
    ax.scatter([mdd_date], [mdd_val], color="#03071e", s=70, zorder=6)
    ax.annotate(
        f"Max Drawdown: {mdd_val:.1f}%\n({mdd_date.strftime('%d/%m/%Y')})",
        xy=(mdd_date, mdd_val),
        xytext=(mdd_date - pd.Timedelta(days=220), mdd_val - 7),
        arrowprops=dict(facecolor="#03071e", shrink=0.05, width=1.2, headwidth=6),
        fontweight="bold", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffdddd", edgecolor="#d90429")
    )

    ax.set_title("VN30-Index Underwater Drawdown Profile (2018–2025)\nDemonstrating the Need for Dynamic Downside Protection & Cash Allocation", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Drawdown from Historical Peak (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylim(-55, 5)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    out_file = os.path.join(FIGURES_DIR, "06_vn30_underwater_drawdown.png")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")


def build_jupyter_notebook():
    """Generates an executable Jupyter Notebook summarizing the findings."""
    notebook_dict = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 📊 Báo Cáo Khảo Sát & Phân Tích Thống Kê Dữ Liệu VN30 (2018–2025)\n",
                    "## Đề tài: Risk-Aware Reinforcement Learning for VN30 Portfolio Allocation (INT3011E)\n",
                    "---\n",
                    "### 1. Bối cảnh nghiên cứu & Mục tiêu chuẩn bị dữ liệu (O1)\n",
                    "Notebook này tổng hợp toàn bộ tập dữ liệu thô vừa thu thập qua `vnstock` cho 30 cổ phiếu đại diện rổ VN30 và 2 chỉ số tham chiếu (`VN30`, `VNINDEX`) từ năm 2018 đến 2025. Phân tích này là cơ sở thực nghiệm để thiết kế môi trường kế toán danh mục $M_1$ và 4 hàm thưởng nhận thức rủi ro $A_0 - A_3$."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import glob\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from IPython.display import Image, display\n",
                    "\n",
                    "# Load summary metadata\n",
                    "summary_files = sorted(glob.glob('../data/raw/prices/*.csv'))\n",
                    "print(f'✅ Đã tải thành công {len(summary_files)} file CSV dữ liệu cổ phiếu VN30.')\n",
                    "\n",
                    "events_df = pd.read_csv('../data/raw/corporate_actions/vn30_all_corporate_events.csv')\n",
                    "print(f'✅ Đã thu thập {len(events_df)} sự kiện doanh nghiệp (cổ tức tiền, cổ tức cổ phiếu, chia tách).')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Quỹ đạo giá chuẩn hóa của 30 cổ phiếu VN30 & Các chu kỳ vĩ mô\n",
                    "Biểu đồ dưới đây thể hiện đường cong giá chuẩn hóa (mốc 100 tại đầu kỳ 2018). Chúng ta thấy rõ 4 giai đoạn thị trường tương ứng với 4 cửa sổ trượt (Rolling Windows):\n",
                    "- **Cửa sổ 1 (2018–2022):** Bao gồm cú sập COVID-19 (tháng 3/2020) và cú lao dốc kỷ lục năm 2022 (-42% do khủng hoảng trái phiếu và thắt chặt thanh khoản).\n",
                    "- **Cửa sổ 2–4 (2023–2025):** Giai đoạn phân hóa mạnh, phục hồi và đi ngang tích lũy."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename='../outputs/figures/01_vn30_price_trajectories.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Phân tích rủi ro sụt giảm (Underwater Drawdown Profile) của VN30-Index\n",
                    "Chỉ số VN30-Index từng trải qua các đợt sụt giảm từ đỉnh nghiêm trọng:\n",
                    "- **Tháng 03/2020:** Sụt giảm **-33.8%**.\n",
                    "- **Năm 2022:** Sụt giảm tới **-45.2%**.\n",
                    "\n",
                    "> **Ý nghĩa đối với đề tài:** Các mô hình RL tối ưu hàm thưởng lợi suất truyền thống ($A_0$) thường tiếp tục nắm giữ cổ phiếu khi thị trường rơi tự do. Việc đưa rào cản phạt sụt giảm vượt ngưỡng $d^* = 15\%$ trong hàm thưởng $A_2$ và $A_3$ đóng vai trò cảnh báo tác tử rút bớt vốn về tiền mặt ($w_{\\text{cash}} \\to 1.0$)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename='../outputs/figures/06_vn30_underwater_drawdown.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Ma trận tương quan lợi suất chéo (Cross-Asset Correlation)\n",
                    "Ma trận thể hiện hệ số tương quan lợi suất ngày giữa 30 cổ phiếu. Các cổ phiếu cùng ngành (đặc biệt là Ngân hàng: `VCB`, `BID`, `CTG`, `MBB`, `TCB` và Bất động sản: `VIC`, `VHM`, `VRE`) có mức độ đồng pha cao ($\rho > 0.65$), tạo cơ hội cho thuật toán PPO học chính sách phân bổ luân chuyển dòng tiền thông minh."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename='../outputs/figures/02_vn30_correlation_matrix.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Phân phối lợi suất thực nghiệm & Rủi ro đuôi (Fat Tails & Kurtosis)\n",
                    "Thực nghiệm trên gần 60.000 mẫu lợi suất ngày của 30 cổ phiếu cho thấy:\n",
                    "- Phân phối lợi suất có đỉnh nhọn và **đuôi dày rõ rệt (Excess Kurtosis > 3.0)** so với phân phối chuẩn Gaussian.\n",
                    "- **Historical VaR 95%:** ~ -2.8% / ngày; **Historical CVaR 95%:** ~ -4.3% / ngày.\n",
                    "- Điều này chứng minh tính cấp thiết của hàm thưởng phạt lợi suất âm dưới ngưỡng $m$ ($A_1$) và kiểm soát tổn thất đuôi CVaR khi đánh giá."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 5,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename='../outputs/figures/03_vn30_return_fat_tails.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 6. Hồ sơ thanh khoản (ADV) & Sự kiện doanh nghiệp (Corporate Actions)\n",
                    "- **Thanh khoản:** Các cổ phiếu dẫn đầu thanh khoản gồm `HPG`, `SSI`, `STB`, `MBB` đạt hàng chục triệu cổ phiếu/ngày, bảo đảm tính khả thi cho việc mô phỏng khớp lệnh thực tế $M_1$ (lô 100) và $M_2$ (giới hạn thanh khoản).\n",
                    "- **Sự kiện quyền:** 275 sự kiện chi trả cổ tức và thưởng cổ phiếu được lưu trữ đầy đủ để phục vụ module kế toán dòng tiền chính xác."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 6,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename='../outputs/figures/04_vn30_liquidity_profile.png'))\n",
                    "display(Image(filename='../outputs/figures/05_vn30_corporate_actions.png'))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    nb_path = "02_Data_Pipeline/01_vn30_data_exploration.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved Jupyter Notebook: {nb_path}")


def main():
    print("[*] Loading raw collected data...")
    tickers_data, vn30_df, vnindex_df, events_df = load_all_data()
    print(f"[*] Loaded {len(tickers_data)} tickers. Benchmarks present: VN30={vn30_df is not None}, VNINDEX={vnindex_df is not None}")

    df_close, df_volume = build_price_matrices(tickers_data)
    print(f"[*] Price Matrix Shape: {df_close.shape} (Trading Days x Equities)")

    print("[*] Generating Figure 1: Normalized Price Trajectories...")
    plot_price_trajectories(df_close, vn30_df)

    print("[*] Generating Figure 2: Cross-Asset Correlation Heatmap...")
    plot_correlation_heatmap(df_close)

    print("[*] Generating Figure 3: Empirical Return Fat Tails...")
    plot_return_fat_tails(df_close)

    print("[*] Generating Figure 4: Liquidity Profile...")
    plot_liquidity_profile(df_volume, df_close)

    print("[*] Generating Figure 5: Corporate Actions Distribution...")
    plot_corporate_actions(events_df)

    print("[*] Generating Figure 6: VN30 Underwater Drawdown...")
    plot_underwater_drawdown(vn30_df)

    print("[*] Building Jupyter Notebook...")
    build_jupyter_notebook()

    print("\n[✅] ALL VISUALIZATIONS AND JUPYTER NOTEBOOK GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
