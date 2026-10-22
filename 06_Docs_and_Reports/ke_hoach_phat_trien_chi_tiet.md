# BẢN KẾ HOẠCH PHÁT TRIỂN CHI TIẾT (DETAILED DEVELOPMENT ROADMAP)

## Đề tài: Risk-Aware Reinforcement Learning for VN30 Portfolio Allocation

- **Học phần:** INT3011E
- **Phạm vi:** Từ sau mốc nghiệm thu giữa kỳ đến nghiệm thu cuối kỳ toàn bộ đề tài
- **Tài liệu tham chiếu:** `06_Docs_and_Reports/bao_cao_giua_ky_de_tai_Tu.md`, `06_Docs_and_Reports/kich_ban_trao_doi_nhom.md`, `AGENTS.md`
- **Nguyên tắc nghiệm thu:** Mọi task chỉ được coi là hoàn thành khi thỏa mãn đầy đủ Definition of Done (DoD) tương ứng tại `AGENTS.md §6`. "Chạy được trên máy local" **không** phải là DoD.

---

## 0. CẢNH BÁO CHẶN (CRITICAL BLOCKERS) — XỬ LÝ TRƯỚC SPRINT 1

Khảo sát mã nguồn thực tế phát hiện 8 vấn đề chặn tiến độ. Các mục này phải được xử lý trước khi khởi động Giai đoạn 1, nếu không toàn bộ chuỗi verification tại `AGENTS.md §5` sẽ không thể thực thi.

| # | Vấn đề phát hiện | Bằng chứng kiểm chứng | Mức độ | Hành động bắt buộc |
|---|---|---|---|---|
| B1 | `.venv` thiếu gần như toàn bộ dependency cốt lõi: `torch`, `gymnasium`, `stable-baselines3`, `streamlit`, `pytest`, `mypy`, `pyarrow`, `ta` | `./.venv/bin/pip list` chỉ trả về `numpy`, `pandas`, `plotly`, `vnstock`, `vnstock_ezchart` | **Chặn** | `pip install -r requirements.txt` + bổ sung `requirements-dev.txt` |
| B2 | `numpy 2.2.6` đã cài, **vi phạm** ràng buộc `numpy>=1.24.0,<2.0` trong `requirements.txt`. SB3/torch có thể lỗi ABI | `pip list \| grep numpy` → `2.2.6` | **Chặn** | Pin lại `numpy<2.0`, tái tạo venv sạch |
| B3 | `requirements.txt` thiếu `vnstock`, `pyarrow`, `PyYAML` dù mã nguồn và `AGENTS.md §4.1` đều đang dùng | Collectors `import vnstock`; DoD yêu cầu Parquet; config là YAML | **Cao** | Bổ sung và pin version |
| B4 | Collectors ghi **CSV**, trong khi `AGENTS.md §6.1` và `§9` bắt buộc **Parquet** | `data/raw/prices/VIC.csv`; DoD: `<ticker>_<start>_<end>.parquet` | **Cao** | Chuyển sang Parquet hoặc cập nhật AGENTS.md (chốt 1 chuẩn) |
| B5 | `run_pipeline.py` **không có** cờ `--dry-run` mà `AGENTS.md §5` bước 5 yêu cầu | `argparse` chỉ có `--start/--end/--delay/--skip-*` | **Cao** | Thêm `--dry-run` |
| B6 | Lệnh verification tại `AGENTS.md §5` bước 4 là **SyntaxError**, không thể chạy | `from 03_Model_Development.env... ` → `SyntaxError: invalid decimal literal` (tên module bắt đầu bằng chữ số) | **Chặn** | Sửa AGENTS.md sang `importlib.import_module` hoặc dùng `sys.path` + alias |
| B7 | `tests/` **rỗng** (chỉ có `__init__.py`) → coverage hiện tại **0%**, trong khi DoD yêu cầu ≥ 80% | `find tests/ -type f` | **Chặn** | Dựng test suite theo `AGENTS.md §6` |
| B8 | Dữ liệu thực tế kết thúc **2025-12-31**, không có dữ liệu 2026 như mục tiêu ghi "OOS 2022-2026" | `max(date)` trên 30 file = `2025-12-31` | **Cao** | Chốt lại phạm vi OOS = **2022-01-01 → 2025-12-31** |

### 0.1 Giải trình về B6 (lỗi trong chính tài liệu harness)

Các thư mục nguồn bắt đầu bằng chữ số (`02_`, `03_`, `04_`) **không phải** Python identifier hợp lệ, nên không thể dùng trong cú pháp `import`/`from`. Hệ quả kéo theo:

- `AGENTS.md §5` bước 4 (`check_env`) không thể thực thi nguyên văn.
- `mypy --strict 02_Data_Pipeline/ ...` và `pytest --cov=02_Data_Pipeline` chạy theo đường dẫn thì được, nhưng mọi `import` chéo giữa các tầng đều phải đi qua hack `sys.path` (hiện `train.py`, `backtest_engine.py`, `app.py` đều đang làm vậy).

**Phương án chốt (khuyến nghị):** giữ nguyên tên thư mục (vì gắn với cấu trúc báo cáo học phần) và bổ sung một package phẳng `src/` chứa các module `import`-able, các thư mục số chỉ giữ entrypoint script. Ghi quyết định này vào `AGENTS.md` để agent khác không lặp lại lỗi.

---

## 1. TỔNG QUAN DỰ ÁN & MỤC TIÊU ĐỊNH LƯỢNG (KPIs)

### 1.1 Mục tiêu tổng quát

Xây dựng, huấn luyện và kiểm định thực nghiệm hệ thống phân bổ danh mục đầu tư tự động ứng dụng **Risk-Aware Deep Reinforcement Learning** cho rổ cổ phiếu VN30, tối ưu hóa tỷ suất sinh lời điều chỉnh theo rủi ro (risk-adjusted return) và giới hạn Maximum Drawdown dưới 15% trong các pha thị trường suy thoái.

### 1.2 Bốn mục tiêu SMART (kế thừa từ Báo cáo giữa kỳ §4.2)

| Mã | Mục tiêu SMART | Tiêu chí đo lường | Giai đoạn triển khai |
|---|---|---|---|
| **MT1** | Xây dựng Pipeline Dữ liệu VN30 và môi trường `VN30PortfolioEnv` chuẩn Gymnasium v0.29+ | 30 ticker OHLCV 2018–2025 đã đồng bộ; feature matrix đầy đủ; env pass `check_env` không warning; tích hợp 4 reward modes | Giai đoạn 1 + 2 |
| **MT2** | Triển khai và huấn luyện PPO & SAC với policy phân bổ tỷ trọng trên simplex | Train In-Sample 2018–2021; TensorBoard log Policy Loss / Value Loss / entropy / reward curve | Giai đoạn 3 |
| **MT3** | Kiểm định Out-of-Sample và đối chuẩn định lượng với 3 baselines | Toàn bộ KPI tại §1.3 đạt ngưỡng trên tập OOS; so sánh VN30-Index, Equal Weight, Markowitz MPT | Giai đoạn 4 |
| **MT4** | Dashboard Streamlit trực quan hóa + hoàn thiện báo cáo và đóng gói mã nguồn | 4 thành phần biểu đồ hoạt động; `mypy --strict` 0 lỗi; `pytest` coverage ≥ 80% | Giai đoạn 5 + 6 |

> **Lưu ý thống nhất số liệu (phát hiện mâu thuẫn tài liệu):**
> - Báo cáo giữa kỳ §4.2 (MT2) ghi *"tối thiểu 100,000 timesteps"*; `AGENTS.md §6.4` và `§8` ghi *"1M steps"*; `configs/default_config.yaml` ghi `total_timesteps: 150_000`.
> - **Chốt:** `150,000` steps cho vòng *smoke/ablation* (khớp config hiện có), `1,000,000` steps cho *final production run* của PPO và SAC. Ngưỡng DoD nghiệm thu dùng mốc **1M**.
> - Báo cáo giữa kỳ đặt mục tiêu `Sharpe ≥ 1.2 / Sortino ≥ 1.8 / Calmar ≥ 1.5`; `kich_ban_trao_doi_nhom.md` lại ghi `Sharpe ≥ 1.5 / Sortino ≥ 2.0 / Calmar ≥ 1.8`. **Chốt theo Báo cáo giữa kỳ** (bản đã nộp, ngưỡng thấp hơn và khả thi hơn); bộ số của kịch bản nhóm coi là *stretch goal*.

### 1.3 Bảng chỉ số hiệu năng mục tiêu bắt buộc (Mandatory KPI Table)

Đo trên tập **Out-of-Sample 2022-01-01 → 2025-12-31**, annualized với `annual_trading_days = 252`.

| KPI | Ngưỡng bắt buộc | Stretch goal | Tham chiếu VN30-Index | Hàm tính |
|---|---|---|---|---|
| **Sharpe Ratio** (annualized) | ≥ 1.2 | ≥ 1.5 | < 0.5 | `metrics.sharpe_ratio()` |
| **Sortino Ratio** | ≥ 1.8 | ≥ 2.0 | — | `metrics.sortino_ratio()` |
| **Maximum Drawdown (MDD)** | ≤ 15% | ≤ 12% | ~ -45% (2022) | `metrics.max_drawdown()` |
| **Calmar Ratio** | ≥ 1.5 | ≥ 1.8 | < 0.3 | `metrics.calmar_ratio()` |
| **Turnover Cost** | < 2% NAV/năm | < 1.5% | 0% (passive) | `metrics.annualized_turnover_cost()` |
| **CVaR 95%** (daily) | ≥ -3.0% | ≥ -2.5% | — | `metrics.cvar()` |
| **Profit Factor** | ≥ 1.4 | ≥ 1.6 | — | `metrics.profit_factor()` |

**Tiêu chí nghiệm thu bổ sung (robustness):** mỗi KPI phải đạt ngưỡng trên **trung vị (median) của 5 random seeds** `[42, 101, 2024, 777, 999]`, không phải trên seed tốt nhất. Đây là điều kiện chống cherry-picking.

### 1.4 Ràng buộc cứng của thị trường HOSE

Mọi env/backtester bỏ qua các ràng buộc này đều **không hợp lệ** (`AGENTS.md §3`).

| Ràng buộc | Giá trị | Công thức / Ghi chú triển khai |
|---|---|---|
| **Settlement** | T+2 | Cổ phiếu mua ngày `T` chỉ khả dụng để bán từ ngày `T+2` |
| **Daily price limit** | ±7% | `price_band = [ref × 0.93, ref × 1.07]`; `ref` = giá tham chiếu (close phiên trước) |
| **Lot size** | 100 shares | `shares = floor(target_value / (price × 100)) × 100` |
| **Transaction tax (sell)** | 0.1% | Chỉ tính trên giá trị bán |
| **Brokerage fee** | 0.15% | Tính cả hai chiều mua và bán |
| **Short selling** | Không cho phép | `w_i ≥ 0 ∀i` |
| **Cash as asset** | `weight[0]` = cash | `Σ w = 1.0` (simplex N+1 = 31 chiều) |

**Công thức chi phí giao dịch tổng hợp:**

```
cost_buy(v)  = v × 0.0015
cost_sell(v) = v × (0.0015 + 0.0010) = v × 0.0025
total_cost_t = Σ_i [ cost_buy(buy_value_i) + cost_sell(sell_value_i) ]
```

Vòng mua-bán khứ hồi (round-trip) chịu tổng ma sát **0.40%**. Đây là lý do kỹ thuật khiến KPI Turnover Cost < 2%/năm giới hạn số lần đảo danh mục toàn phần xuống dưới ~5 lượt/năm.

---

## 2. PHÂN TÍCH HIỆN TRẠNG (AS-IS vs TO-BE GAP ANALYSIS)

### 2.1 Những gì ĐÃ hoàn thành (AS-IS — kiểm chứng trực tiếp trên mã nguồn)

| Hạng mục | File | Trạng thái kiểm chứng |
|---|---|---|
| Collector giá VN30 | `02_Data_Pipeline/collectors/fetch_vn30_prices.py` (214 dòng) | ✅ Hoạt động. Đã tải **30/30 ticker**, 2018-01-02 → 2025-12-31, tối đa 1.999 phiên/mã. Có `skip_existing`, retry, `VNSTOCK_TELEMETRY=off`, dùng đúng `vnstock.api.quote.Quote` |
| Collector sự kiện doanh nghiệp | `collectors/fetch_corporate_events.py` (180 dòng) | ✅ Đã tải cổ tức/chia tách cho các mã |
| Collector benchmark | `collectors/fetch_benchmarks.py` (100 dòng) | ✅ `VN30.csv` + `VNINDEX.csv`, 1.999 phiên khớp với ticker |
| Pipeline orchestrator | `02_Data_Pipeline/run_pipeline.py` (110 dòng) | ⚠️ Chạy được 3 collector nhưng **thiếu `--dry-run`** và chưa có registry cho processors |
| Notebook khảo sát dữ liệu | `02_Data_Pipeline/01_vn30_data_exploration.ipynb` | ✅ Sinh 6 biểu đồ tại `outputs/figures/` (price trajectories, correlation, fat tails, liquidity, corporate actions, underwater drawdown) |
| Config thực nghiệm | `03_Model_Development/configs/default_config.yaml` | ✅ **Chất lượng cao**: đã có 30 ticker, `market_rules` đúng chuẩn HOSE, 4 reward configs A0–A3, 4 rolling windows, 5 seeds, PPO hyperparams |
| Metrics engine | `04_Backtesting_Evaluation/metrics.py` (138 dòng) | ⚠️ Có đủ Sharpe/Sortino/MDD/Calmar/VaR/CVaR/Profit Factor nhưng là **một hàm khối `evaluate_portfolio()`**, chưa tách hàm đơn lẻ theo DoD `§6.5`, thiếu Turnover |
| PPO / SAC wrapper | `agents/ppo_agent.py`, `agents/sac_agent.py` | ⚠️ Có `train/predict/save/load` nhưng **hardcode hyperparameters**, chưa đọc YAML (vi phạm `§4.4`) |

### 2.2 Những gì CÒN THIẾU hoặc PHẢI TÁI CẤU TRÚC (TO-BE)

#### A. Data Processors — **chưa tồn tại** (thư mục rỗng)

`02_Data_Pipeline/processors/` chỉ có `__init__.py` trống. Cần xây mới hoàn toàn:
- `clean_alignment.py` → `data/processed/aligned_prices.parquet`
- `build_features.py` → `data/processed/feature_matrix.parquet`

**Rủi ro dữ liệu đã phát hiện:** `SSB` chỉ có **1.195 phiên** (niêm yết 2021-03-24) và `TCB` có **1.898 phiên** (2018-06-04), trong khi các mã khác có 1.999 phiên. Đây là vấn đề **survivorship / late-listing bias** chưa được xử lý. Nếu forward-fill thô theo quy tắc "gap ≤ 5 ngày" thì `SSB` sẽ tạo gap **hơn 3 năm** và làm `ValueError` đánh sập pipeline.

#### B. Môi trường Gymnasium — **phải viết lại từ đầu**

File hiện tại `03_Model_Development/env/trading_env.py` (301 dòng) **không dùng được** cho đề tài VN30 vì sai bản chất bài toán:

| Khía cạnh | `trading_env.py` hiện tại (SAI) | `vn30_portfolio_env.py` cần có (ĐÚNG) |
|---|---|---|
| Lớp tài sản | **Đơn tài sản**, yêu cầu cột `Close` | **31 tài sản** (30 cổ phiếu + cash) |
| Action space | `Box(shape=(1,))` — vị thế scalar | `Box(shape=(31,))` → softmax → simplex |
| Kế thừa | `class RiskAwareTradingEnv:` — **không kế thừa `gym.Env`** | `class VN30PortfolioEnv(gym.Env)` |
| T+2 settlement | **Không có** | Hàng đợi settlement bắt buộc |
| Biên độ ±7% | **Không có** | Clip giá khớp theo price band |
| Lô 100 shares | **Không có** | Làm tròn xuống bội số 100 |
| Thuế/phí HOSE | `transaction_fee=0.001` phẳng | 0.15% hai chiều + 0.1% thuế bán |
| `OBS_SCHEMA` | **Không có** (vi phạm `§4.3`) | Class attribute mô tả mọi feature slice |
| `info` dict | Thiếu `cvar_95`, `turnover`, `transaction_cost` | Đủ 5 khóa theo `§6.3` |
| Reward modes | 5 modes nhưng tên lệch config | 4 modes khớp `A0–A3` |

**Mâu thuẫn danh pháp reward cần chốt:** env dùng tên mô tả (`standard`, `volatility_penalty`, `differential_sharpe`, `downside_sortino`, `prospect_theory`), còn `default_config.yaml` dùng mã `A0–A3`. Báo cáo giữa kỳ yêu cầu đúng **4 modes**. Bảng mapping chốt tại §3.2.3.

#### C. Backtester — **phải viết lại**

`04_Backtesting_Evaluation/backtest_engine.py` (117 dòng) vẫn thuộc thời kỳ "Antigravity/BTC":
- Mặc định `--symbol BTC-USD`, gọi `fetch_market_data` từ `data_collector.py` (Yahoo Finance).
- Baselines là **Buy & Hold** và **SMA crossover** — **không phải** 3 baselines mà đề tài cam kết.
- Phần "RL agent" thực chất là **heuristic if-else** (`if curr_dd > 0.05: action = 0.2`), **không nạp checkpoint** PPO/SAC.
- Split OOS là tỷ lệ `70/30` cố định, **không phải walk-forward** theo 4 windows trong config.
- **Chưa có** `04_Backtesting_Evaluation/baselines/` (thư mục rỗng).

#### D. Streamlit Dashboard — **phải viết lại**

`05_Deployment_Demo/app.py` (214 dòng) còn chọn tài sản `["BTC-USD", "ETH-USD", "^VNINDEX", "HPG.VN", "SPY", ...]`, import `trading_env` đơn tài sản, tiêu đề vẫn là "Antigravity". Thiếu stacked area chart tỷ trọng danh mục. Ngoài ra `AGENTS.md §2` quy định tên file là `streamlit_app.py` còn thực tế là `app.py` (báo cáo giữa kỳ cũng ghi `app.py`) — cần chốt một tên.

#### E. Test suite — **0% coverage**

`tests/` rỗng. Cần tối thiểu 8 file test để phủ DoD `§6.1`–`§6.5`.

#### F. Di sản "Antigravity" cần làm sạch

Toàn bộ docstring của `trading_env.py`, `metrics.py`, `backtest_engine.py`, `app.py`, `ppo_agent.py`, `sac_agent.py` còn ghi *"Project: Antigravity"*, và class name là `AntigravityPPOAgent` / `AntigravitySACAgent`. Commit `f143821` đã bắt đầu loại bỏ naming này nhưng chưa xử lý tầng mã nguồn.

### 2.3 Ma trận Gap tổng hợp theo tầng kiến trúc

| Tầng | AS-IS | TO-BE | Khối lượng | Ưu tiên |
|---|---|---|---|---|
| Collectors | 85% | Chuyển Parquet, thêm `--dry-run`, test | Nhỏ | P1 |
| Processors | **0%** | 2 module mới + xử lý late-listing | Lớn | **P0** |
| Environment | **10%** (sai bài toán) | Viết mới `vn30_portfolio_env.py` | **Rất lớn** | **P0** |
| Agents | 40% | Nạp YAML, bỏ hardcode, đổi tên class | Trung bình | P1 |
| Backtester | **15%** | Walk-forward + 3 baselines + nạp checkpoint | Lớn | P1 |
| Metrics | 70% | Tách hàm đơn lẻ, thêm Turnover | Nhỏ | P1 |
| Dashboard | **20%** | Viết lại cho đa tài sản VN30 | Trung bình | P2 |
| Tests | **0%** | ~8 file test, coverage ≥ 80% | Lớn | **P0** |

---

## 3. KẾ HOẠCH CHI TIẾT THEO GIAI ĐOẠN (PHASES & SPRINTS)

Lộ trình 6 tuần bám sát `AGENTS.md §8`, bổ sung **Sprint 0** để xử lý các blocker tại §0.

### SPRINT 0 — Khôi phục môi trường & Thống nhất harness (2 ngày)

**Mục tiêu:** đưa chuỗi verification `AGENTS.md §5` về trạng thái chạy được.

| Task | File tác động | Tiêu chí hoàn thành |
|---|---|---|
| S0.1 | `requirements.txt` | Pin `numpy>=1.24,<2.0`; bổ sung `vnstock`, `pyarrow>=14.0`, `PyYAML>=6.0` |
| S0.2 | `requirements-dev.txt` (mới) | `pytest`, `pytest-cov`, `pytest-mock`, `mypy`, `types-PyYAML` |
| S0.3 | `.venv` | Tái tạo sạch; `python -c "import torch, gymnasium, stable_baselines3"` thành công |
| S0.4 | `AGENTS.md §5` | Sửa lệnh `check_env` thành dạng chạy được (xem khối mã dưới) |
| S0.5 | `AGENTS.md §6.1`, `§9` | Chốt định dạng lưu trữ: **Parquet** cho `data/processed/`, giữ CSV cho `data/raw/` đã tải (tránh tải lại 30 mã) |
| S0.6 | `pytest.ini` / `pyproject.toml` (mới) | Cấu hình `pythonpath` để pytest import được các tầng |
| S0.7 | `02_Data_Pipeline/run_pipeline.py` | Thêm `--dry-run`: kiểm tra kết nối + liệt kê file sẽ ghi, **không** gọi API |

**Lệnh verification thay thế cho `AGENTS.md §5` bước 4:**

```bash
python -c "
import importlib, sys
sys.path.insert(0, '03_Model_Development')
from gymnasium.utils.env_checker import check_env
mod = importlib.import_module('env.vn30_portfolio_env')
env = mod.VN30PortfolioEnv(data_path='data/processed/feature_matrix.parquet')
check_env(env)
print('OK: Env check passed')
"
```

**Gate ra khỏi Sprint 0:** cả 5 bước trong `AGENTS.md §5` thực thi không lỗi cú pháp (test có thể còn trống).

---

### GIAI ĐOẠN 1 — Chuẩn hóa Data Pipeline & Feature Engineering (Tuần 1)

**Deliverable chính:** `data/processed/aligned_prices.parquet` + `data/processed/feature_matrix.parquet`

#### 1.1 Task P1.1 — `processors/clean_alignment.py`

- **Input:** `data/raw/prices/*.csv` (30 ticker), `data/raw/benchmarks/VN30.csv`
- **Output:** `data/processed/aligned_prices.parquet`
- **Trục thời gian chuẩn:** lấy tập ngày giao dịch của `VN30.csv` (1.999 phiên) làm `trading_calendar` chuẩn, **không** dùng union/intersection của 30 mã (tránh ngày lỗi của một mã làm lệch cả rổ).

**Output schema (bắt buộc khớp chính xác theo `AGENTS.md §6.2`):**

| Cột | Dtype | Ghi chú |
|---|---|---|
| `date` | `datetime64[ns]` | Không bao giờ là string |
| `ticker` | `str` | 30 giá trị |
| `close` | `float64` | Đã điều chỉnh split/cổ tức |
| `volume` | `int64` | Khối lượng khớp |
| `adj_factor` | `float64` | Hệ số điều chỉnh từ `corporate_actions` |

**Quy tắc xử lý khuyết dữ liệu (3 tầng, mở rộng quy tắc `§4.2`):**

```python
# Tầng 1: gap <= 5 phiên liên tiếp -> forward-fill
# Tầng 2: gap > 5 phiên NẰM TRONG vòng đời cổ phiếu -> raise ValueError(ticker, start, end)
# Tầng 3: gap TRƯỚC ngày niêm yết đầu tiên -> KHÔNG ffill, gán listing_mask = False
```

> **Xử lý bắt buộc cho `SSB` và `TCB` (đã phát hiện ở §2.2.A):** quy tắc "gap > 5 → `ValueError`" nguyên văn sẽ **đánh sập pipeline** vì `SSB` thiếu hơn 3 năm đầu. Giải pháp: bổ sung cột boolean `is_listed` (hoặc `listing_mask`) phân biệt *"chưa niêm yết"* với *"lỗi thiếu dữ liệu"*. Env phải cưỡng chế `w_i = 0` cho mọi mã có `is_listed == False` tại thời điểm `t`. Đây là điều kiện bắt buộc để loại bỏ **look-ahead bias**.

**Unit tests — `tests/test_clean_alignment.py`:**

| Test | Nội dung |
|---|---|
| `test_gap_ffill_short()` | Gap ≤ 5 phiên được ffill đúng giá trị cuối |
| `test_gap_raises_long()` | Gap > 5 phiên **trong** vòng đời → `ValueError` chứa tên mã + khoảng ngày |
| `test_output_schema()` | Cột và dtype khớp tuyệt đối bảng trên |
| `test_pre_listing_not_ffilled()` | `SSB` trước 2021-03-24 có `is_listed == False` và `close` là `NaN`, **không** bị ffill |
| `test_calendar_alignment()` | Mọi ticker có đúng số phiên = độ dài `trading_calendar` |

#### 1.2 Task P1.2 — `processors/build_features.py`

- **Input:** `data/processed/aligned_prices.parquet`
- **Output:** `data/processed/feature_matrix.parquet`

**Bộ đặc trưng (khớp Báo cáo giữa kỳ MT1), tính riêng từng ticker:**

| Nhóm | Feature | Tham số | Thư viện |
|---|---|---|---|
| Return | `ret_1d`, `ret_5d`, `ret_20d` | log return | pandas |
| Momentum | `rsi_14` | window 14 | `ta.momentum.RSIIndicator` |
| Trend | `macd_hist` | 12/26/9 | `ta.trend.MACD` |
| Volatility band | `bb_pctb`, `bb_width` | window 20, 2σ | `ta.volatility.BollingerBands` |
| Volatility | `atr_norm` | ATR14 / close | `ta.volatility.AverageTrueRange` |
| Volatility | `roll_vol_20` | std(ret_1d, 20) × √252 | pandas |
| Risk | `downside_vol_20` | std của return âm, window 20 | pandas |
| Liquidity | `turnover_ratio` | `volume × close` normalized | pandas |

**Quy tắc chống look-ahead (bắt buộc):** mọi indicator chỉ dùng dữ liệu đến `t`, mọi chuẩn hóa (z-score) dùng **rolling statistics** chứ không dùng mean/std toàn tập. Vi phạm điểm này làm toàn bộ kết quả OOS mất giá trị khoa học.

**Unit tests — `tests/test_build_features.py`:**

| Test | Nội dung |
|---|---|
| `test_no_lookahead()` | Cắt data tại `t`, tính lại feature → giá trị tại `t` không đổi |
| `test_rsi_known_value()` | RSI trên series đã biết, khớp 4 chữ số thập phân |
| `test_no_nan_after_warmup()` | Sau 20 phiên warmup, không còn `NaN` (trừ vùng pre-listing) |
| `test_output_schema()` | Đủ số cột = `n_features × 30`, dtype `float64` |

#### 1.3 DoD Giai đoạn 1

- [ ] 2 file processor có module docstring + type annotation đầy đủ
- [ ] `aligned_prices.parquet` và `feature_matrix.parquet` được sinh ra, schema khớp spec
- [ ] Cả 2 module đã đăng ký vào registry của `run_pipeline.py`
- [ ] 9 unit test pass
- [ ] `mypy --strict` → 0 lỗi trên `02_Data_Pipeline/`
- [ ] Coverage ≥ 80% trên 2 module mới
- [ ] `python 02_Data_Pipeline/run_pipeline.py --dry-run` exit code 0

---

### GIAI ĐOẠN 2 — Xây dựng & Kiểm chuẩn Môi trường Gymnasium (Tuần 2–3)

**Deliverable:** `03_Model_Development/env/vn30_portfolio_env.py` pass `check_env` không warning.

> Đây là giai đoạn **rủi ro cao nhất** của dự án: file hiện tại phải viết lại từ đầu (§2.2.B), và đây là nút cổ chai mà Giai đoạn 3, 4, 5 đều phụ thuộc.

#### 2.1 Task P2.1 — Thiết kế Observation Space & `OBS_SCHEMA`

Với `N = 30` cổ phiếu, `F = 11` feature/mã, `W = 20` phiên lookback:

```python
class VN30PortfolioEnv(gym.Env):
    """Môi trường phân bổ danh mục VN30 mô phỏng đầy đủ ràng buộc sàn HOSE."""

    N_ASSETS: int = 30
    N_FEATURES: int = 11
    LOOKBACK: int = 20

    OBS_SCHEMA = {
        "market_features": {
            "slice": (0, 330),          # 30 ticker × 11 feature
            "desc":  "Feature kỹ thuật đã z-score rolling tại phiên t",
        },
        "current_weights": {
            "slice": (330, 361),        # 31 = 30 cổ phiếu + cash
            "desc":  "Tỷ trọng danh mục hiện tại trước khi rebalance",
        },
        "pending_settlement": {
            "slice": (361, 391),        # 30
            "desc":  "Tỷ lệ cổ phiếu đang trong hàng đợi T+2 (chưa bán được)",
        },
        "portfolio_state": {
            "slice": (391, 395),
            "desc":  "[current_drawdown, roll_vol_20, cvar_95, cash_ratio]",
        },
        "regime_features": {
            "slice": (395, 399),
            "desc":  "[vn30_ret_1d, vn30_ret_20d, vn30_vol_20, breadth_ratio]",
        },
        "listing_mask": {
            "slice": (399, 429),        # 30
            "desc":  "1.0 nếu mã đã niêm yết tại t, 0.0 nếu chưa (chống look-ahead)",
        },
    }
    # Tổng obs_dim = 429
```

- `observation_space = Box(low=-np.inf, high=np.inf, shape=(429,), dtype=np.float32)`
- **Lưu ý `check_env`:** `check_env` cảnh báo khi bound là vô hạn. Để pass **không warning**, dùng bound hữu hạn sau khi đã clip feature: `Box(low=-10.0, high=10.0, ...)` và clip giá trị trong `_get_observation()`.

#### 2.2 Task P2.2 — Action Space & phép chiếu lên Simplex

```python
action_space = Box(low=-1.0, high=1.0, shape=(31,), dtype=np.float32)
```

Hành động thô `a_t ∈ R^31` được chuyển thành tỷ trọng hợp lệ:

```
w_raw = softmax(a_t / τ)                  # τ = temperature, mặc định 1.0
w_masked = w_raw × [1, listing_mask]      # cưỡng chế 0 cho mã chưa niêm yết
w_t = w_masked / Σ(w_masked)              # chuẩn hóa lại về simplex
```

Softmax bảo đảm `w_i ≥ 0` và `Σ w = 1` **theo cấu trúc** (long-only được thỏa mãn tự động, không cần clip hậu kỳ). Lý do chọn `Box(-1, 1)` thay vì `Box(0, 1)`: SB3 khuyến nghị action space đối xứng chuẩn hóa cho PPO/SAC để ổn định gradient.

#### 2.3 Task P2.3 — Cơ chế mô phỏng sàn HOSE

Thứ tự thực thi trong `step()` **phải** theo đúng trình tự sau (sai thứ tự làm sai toàn bộ chi phí):

```
Bước 1. Nhận a_t → chiếu lên simplex → w_target
Bước 2. Tính giá khớp P_exec cho phiên t+1:
          P_ref  = close[t]
          band   = [P_ref × 0.93, P_ref × 1.07]
          P_exec = clip(open[t+1] hoặc close[t+1], band)
Bước 3. Quy đổi w_target → số cổ phiếu mục tiêu, làm tròn lô 100:
          shares_target_i = floor(w_target_i × NAV / (P_exec_i × 100)) × 100
Bước 4. Kiểm tra hàng đợi T+2:
          sellable_i = holdings_i - pending_i   # pending = mua trong 2 phiên gần nhất
          lệnh bán bị giới hạn bởi sellable_i
Bước 5. Tính chi phí:
          cost = Σ buy_value × 0.0015 + Σ sell_value × 0.0025
Bước 6. Cập nhật NAV, đẩy hàng đợi settlement tiến 1 phiên
Bước 7. Tính reward theo mode đang chọn
Bước 8. Trả về (obs, reward, terminated, truncated, info)
```

**Cấu trúc dữ liệu hàng đợi T+2 (tối ưu hiệu năng — xem Rủi ro R4):**

```python
# Ma trận tròn (circular buffer) thay vì list-of-dict:
# settlement_queue.shape == (2, 30);  hàng 0 = mua tại t-1, hàng 1 = mua tại t-2
self.settlement_queue = np.zeros((2, self.N_ASSETS), dtype=np.float64)
pending = self.settlement_queue.sum(axis=0)       # O(1) vectorized
# Cuối mỗi step: np.roll(..., shift=1, axis=0) rồi ghi hàng 0 = 0
```

#### 2.4 Task P2.4 — Bốn chế độ hàm thưởng nhận thức rủi ro

Bảng mapping chốt giữa danh pháp env và mã config `A0–A3` (giải quyết mâu thuẫn tại §2.2.B):

| Mã config | `reward_mode` | Công thức | Tham số |
|---|---|---|---|
| **A0** | `standard` | `r_t = log(NAV_t / NAV_{t-1})` | — |
| **A1** | `volatility_penalty` | `r_t = log-ret − λ·σ_20 − η·max(0, DD_t − d*)` | `λ=2.0`, `η=5.0`, `d*=0.15` |
| **A2** | `differential_sharpe` | DSR đệ quy (Moody & Saffell, 2001) | `η_decay=0.05` |
| **A3** | `prospect_theory_cvar` | `U(r) − η·max(0, DD−d*) − κ·max(0, −CVaR_95 − c*)` | `λ_loss=2.25`, `α=0.88`, `κ=3.0` |

Trong đó hàm hữu dụng Prospect Theory (Kahneman–Tversky, 1979):

```
U(r) =  r^α                      nếu r ≥ 0
U(r) = −λ_loss · (−r)^α          nếu r < 0        (λ_loss = 2.25 > 1 ⇒ loss aversion)
```

**Ghi chú kỹ thuật về A2 (DSR):** công thức DSR trong `trading_env.py` hiện tại chia cho `var**1.5` với `var = max(1e-6, var)`. Khi variance tiến về 0 (giai đoạn thị trường cực tĩnh), mẫu số `1e-9` làm reward nổ lên hàng triệu rồi bị clip cứng ở `±5.0`, tạo gradient rác. Khi port sang env mới phải **chuẩn hóa lại thang đo** và ghi test riêng cho trường hợp variance suy biến.

**Reward scaling:** log-return hàng ngày có độ lớn ~1e-3, quá nhỏ so với khởi tạo mạng của SB3. Nhân toàn bộ reward với `reward_scale = 100.0` (ghi vào YAML, không hardcode).

#### 2.5 Task P2.5 — Kiểm chuẩn (5 test DoD theo `AGENTS.md §6.3`)

`tests/test_vn30_env.py`:

| Test | Nội dung nghiệm thu |
|---|---|
| `test_check_env()` | `check_env(env)` pass, **0 warning** (bắt warning bằng `pytest.warns(None)` / `warnings.catch_warnings`) |
| `test_t2_settlement()` | Mua tại `T` → `sellable == 0` tại `T` và `T+1` → `sellable > 0` tại `T+2` |
| `test_price_limit_clip()` | Inject giá `t+1` tăng 15% → `P_exec` bị clip đúng tại `+7%` |
| `test_fee_deduction()` | Giao dịch đã biết: mua 100 cp giá 20.000 → phí `0.0015 × 2.000.000 = 3.000` VND; bán lại → `0.0025 × value`. Khớp đến đơn vị đồng |
| `test_episode_deterministic()` | Cùng `seed` → trajectory NAV giống hệt (so sánh `np.allclose`) |

**Test bổ sung ngoài DoD (khuyến nghị mạnh):**

| Test | Lý do |
|---|---|
| `test_simplex_invariant()` | Với 1.000 action ngẫu nhiên: luôn `Σw == 1 ± 1e-6` và `w_i ≥ 0` |
| `test_lot_size_rounding()` | Mọi số cổ phiếu khớp đều là bội số của 100 |
| `test_no_lookahead_in_obs()` | Obs tại `t` không chứa thông tin giá `t+1` |
| `test_listing_mask_enforced()` | `SSB` trước 2021-03-24 luôn có `w == 0` |
| `test_info_keys()` | `info` chứa đủ `portfolio_value`, `daily_return`, `cvar_95`, `turnover`, `transaction_cost` |

#### 2.6 DoD Giai đoạn 2

- [ ] `VN30PortfolioEnv(gym.Env)` tồn tại, có `OBS_SCHEMA` đầy đủ
- [ ] `check_env` pass 0 warning
- [ ] 5 test DoD + 5 test bổ sung đều pass
- [ ] `mypy --strict` 0 lỗi
- [ ] Coverage ≥ 80% trên file env
- [ ] Hiệu năng: ≥ 2.000 `step()`/giây (điều kiện để train 1M steps trong thời gian hợp lý)
- [ ] File cũ `trading_env.py` được đánh dấu deprecated hoặc xóa, không còn module nào import nó

---

### GIAI ĐOẠN 3 — Huấn luyện Tác tử Deep RL (PPO & SAC) (Tuần 4)

#### 3.1 Task P3.1 — Cấu hình YAML

Tạo 2 file mới, kế thừa `default_config.yaml` (đã có sẵn chất lượng tốt):

`03_Model_Development/configs/ppo_baseline.yaml`:

```yaml
agent:
  algo: "PPO"
  policy: "MlpPolicy"
  policy_kwargs:
    net_arch:
      pi: [256, 128]      # Actor
      vf: [256, 128]      # Critic
    activation_fn: "Tanh" # Tanh ổn định hơn ReLU cho dữ liệu tài chính nhiễu
  hyperparameters:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.01        # Giữ entropy để tránh co cụm 100% cash (Rủi ro R2)
    vf_coef: 0.5
    max_grad_norm: 0.5
  training:
    total_timesteps: 1000000
    reward_mode: "volatility_penalty"   # A1
    reward_scale: 100.0
    seed: 42
```

`03_Model_Development/configs/sac_baseline.yaml`:

```yaml
agent:
  algo: "SAC"
  policy: "MlpPolicy"
  policy_kwargs:
    net_arch: [256, 256]
  hyperparameters:
    learning_rate: 0.0003
    buffer_size: 300000
    batch_size: 256
    gamma: 0.99
    tau: 0.005
    train_freq: 1
    gradient_steps: 1
    learning_starts: 10000
    ent_coef: "auto"
  training:
    total_timesteps: 1000000
    reward_mode: "volatility_penalty"
    reward_scale: 100.0
    seed: 42
```

#### 3.2 Task P3.2 — Tái cấu trúc Agent wrappers

- Đổi tên `AntigravityPPOAgent` → `PPOPortfolioAgent`, `AntigravitySACAgent` → `SACPortfolioAgent`.
- Bỏ **toàn bộ** hyperparameter hardcode trong `__init__`; nhận duy nhất `config: dict` nạp từ YAML (`AGENTS.md §4.4`).
- Loại bỏ pattern `try: import ... except ImportError: SB3_AVAILABLE = False` và fallback `return np.array([0.5])`. Pattern này **che giấu lỗi cài đặt** và trả về hành động giả, cực kỳ nguy hiểm khi lẫn vào kết quả backtest. Thay bằng để `ImportError` nổ ra tự nhiên.
- Giữ nguyên 4 phương thức theo `§4.4`: `train(total_timesteps)`, `predict(obs)`, `save(path)`, `load(path)`.

#### 3.3 Task P3.3 — Chiến lược huấn luyện

| Thông số | Giá trị |
|---|---|
| Tập In-Sample | **2018-01-01 → 2021-12-31** (~993 phiên) |
| Tập Validation | 2021 (dùng chọn checkpoint tốt nhất, không dùng để test) |
| Tổng timesteps | 1.000.000 (production) / 150.000 (ablation) |
| Số seed | 5 — `[42, 101, 2024, 777, 999]` |
| Vectorized env | `DummyVecEnv` + `VecNormalize(norm_obs=True, norm_reward=False)` |

> **Cảnh báo quan trọng về `VecNormalize`:** phải đặt `norm_reward=False`. Nếu chuẩn hóa reward, tín hiệu phạt drawdown của A1/A3 bị rescale động theo thời gian, làm mất ý nghĩa ngưỡng `d* = 0.15` và vô hiệu hóa toàn bộ thiết kế risk-aware. Thống kê normalize của obs phải được `save`/`load` kèm checkpoint, nếu không kết quả OOS sẽ sai lệch hoàn toàn.

**Ma trận thực nghiệm:** 2 thuật toán × 4 reward modes × 4 walk-forward windows × 5 seeds = **160 runs**. Nếu không đủ tài nguyên tính toán, giảm theo thứ tự ưu tiên sau: (1) giữ đủ 5 seeds cho cấu hình tốt nhất; (2) giảm ablation reward mode xuống 150k steps; (3) giảm số window từ 4 xuống 2.

#### 3.4 Task P3.4 — Giám sát TensorBoard

Log bắt buộc tại `outputs/logs/<agent>_<run_id>/`:

| Nhóm | Metric |
|---|---|
| Loss | `train/policy_gradient_loss`, `train/value_loss`, `train/entropy_loss` |
| Convergence | `rollout/ep_rew_mean`, `rollout/ep_len_mean` |
| Policy health | `train/approx_kl`, `train/clip_fraction`, `train/std` |
| **Custom callback** | `portfolio/final_nav`, `portfolio/max_drawdown`, `portfolio/turnover`, `portfolio/cash_weight_mean`, `portfolio/n_active_positions` |

`portfolio/cash_weight_mean` là **chỉ báo cảnh báo sớm** cho Rủi ro R2: nếu vượt 0.9 và duy trì, tác tử đang co cụm về 100% tiền mặt và cần can thiệp ngay.

#### 3.5 DoD Giai đoạn 3

- [ ] 2 file YAML có mọi hyperparameter được chú thích
- [ ] `train.py --config ppo_baseline.yaml --dry-run` chạy 1.000 steps không lỗi
- [ ] Training 1M steps hoàn tất, exit code 0
- [ ] Checkpoint tại `outputs/models/ppo_baseline_<timestamp>.zip` (kèm file `VecNormalize` stats)
- [ ] TensorBoard log tại `outputs/logs/ppo_baseline_<timestamp>/`
- [ ] `ep_rew_mean` cuối cùng được log và **không âm**
- [ ] `train.py` không còn tham chiếu `BTC-USD` hay `data_collector.py`

---

### GIAI ĐOẠN 4 — Backtesting Walk-Forward Đa Chế độ Thị trường (Tuần 5)

#### 4.1 Task P4.1 — Hoàn thiện `metrics.py`

Tách `evaluate_portfolio()` thành các hàm đơn lẻ có signature rõ ràng theo DoD `§6.5`:

```python
def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                 annual_trading_days: int = 252) -> float: ...
def sortino_ratio(returns: pd.Series, target: float = 0.0) -> float: ...
def max_drawdown(equity: pd.Series) -> float: ...
def calmar_ratio(returns: pd.Series, equity: pd.Series) -> float: ...
def cvar(returns: pd.Series, alpha: float = 0.95) -> float: ...
def annualized_turnover_cost(weights: pd.DataFrame) -> float: ...   # CÒN THIẾU
def profit_factor(returns: pd.Series) -> float: ...
```

**Hai lỗi cần sửa trong `metrics.py` hiện tại:**

1. `sharpe_ratio` dùng `excess_std = returns_series.std()` (std của return thô) làm mẫu số thay vì std của excess return. Sai lệch nhỏ nhưng không đúng định nghĩa chuẩn.
2. `sortino_ratio` trả `0.0` khi không có downside, trong khi DoD `§6.5` yêu cầu trả `float('inf')`.

Edge case bắt buộc xử lý: `profit_factor` hiện trả hằng số `999.0` — nên trả `float('inf')` cho nhất quán.

#### 4.2 Task P4.2 — Ba baselines (`04_Backtesting_Evaluation/baselines/`)

| File | Chiến lược | Đặc tả |
|---|---|---|
| `vn30_index.py` | VN30-Index passive | Đọc `data/raw/benchmarks/VN30.csv`, buy & hold, phí = 0 |
| `equal_weight.py` | Equal Weight 1/N | `w_i = 1/30`, rebalance hàng tháng, **chịu đầy đủ phí/thuế HOSE** |
| `markowitz_mpt.py` | Markowitz Mean-Variance | Max Sharpe trên rolling covariance 252 phiên, long-only, rebalance hàng tháng, `scipy.optimize.minimize` SLSQP |

> **Nguyên tắc công bằng trong đối chuẩn (fair comparison):** Equal Weight và Markowitz **phải** chạy qua cùng engine chi phí của `VN30PortfolioEnv` (T+2, lô 100, phí 0.15%, thuế 0.1%). Nếu baseline được miễn phí giao dịch trong khi tác tử RL phải chịu, kết quả so sánh mất giá trị khoa học. VN30-Index là ngoại lệ hợp lý vì là chỉ số tham chiếu thụ động.

#### 4.3 Task P4.3 — Walk-Forward Engine

Viết lại `backtest_engine.py` thành `04_Backtesting_Evaluation/backtester.py` (đúng tên theo `AGENTS.md §2`):

Sử dụng đúng 4 window đã khai báo trong `default_config.yaml`:

| Window | Train | Validation | Test (OOS) |
|---|---|---|---|
| 1 | 2018-01-01 → 2020-12-31 | 2021 | **2022** (Bear) |
| 2 | 2018-01-01 → 2021-12-31 | 2022 | **2023** (Phục hồi) |
| 3 | 2018-01-01 → 2022-12-31 | 2023 | **2024** (Sideway) |
| 4 | 2018-01-01 → 2023-12-31 | 2024 | **2025** (OOS cuối) |

Yêu cầu bắt buộc: engine phải **nạp checkpoint PPO/SAC thật** (`model.predict(obs, deterministic=True)`), loại bỏ hoàn toàn heuristic `if curr_dd > 0.05` hiện có.

#### 4.4 Task P4.4 — Phân tích theo 3 pha thị trường

| Pha | Khoảng thời gian | Đặc điểm | Tiêu chí nghiệm thu riêng |
|---|---|---|---|
| **Bull** | 2020-04 → 2021-12 | VN-Index tăng mạnh | Agent không được thua VN30-Index quá 20% tổng return (chứng minh không quá rụt rè) |
| **Bear** | 2022-01 → 2022-12 | VN-Index −42.8% | **MDD agent ≤ 50% MDD của VN30-Index** — đây là KPI cốt lõi của đề tài |
| **Sideway/Phục hồi** | 2023-01 → 2025-12 | Giằng co, whipsaw | Turnover cost < 2% NAV/năm (chứng minh không churning) |

> **Lưu ý về pha Bull:** giai đoạn 2020–2021 nằm trong tập In-Sample của Window 1–2. Kết quả pha Bull vì vậy **không phải** Out-of-Sample thuần túy và **phải được ghi rõ** trong báo cáo cuối kỳ. Chỉ Window 1 (test 2022) trở đi mới là OOS thực sự. Trình bày thiếu ghi chú này là lỗi phương pháp luận nghiêm trọng khi phản biện.

#### 4.5 DoD Giai đoạn 4

- [ ] 7 hàm metric đơn lẻ + test với giá trị đã biết đến 4 chữ số thập phân
- [ ] 3 baseline chạy được, dùng chung engine chi phí
- [ ] Walk-forward 4 window × 2 thuật toán × 5 seeds hoàn tất
- [ ] Bảng kết quả tổng hợp `outputs/backtest_results/summary.csv` (median qua 5 seeds)
- [ ] Biểu đồ: equity curve, underwater drawdown, phân bổ tỷ trọng → `outputs/figures/`
- [ ] Toàn bộ KPI tại §1.3 đạt ngưỡng trên **median 5 seeds**
- [ ] `mypy --strict` 0 lỗi; coverage ≥ 80%

---

### GIAI ĐOẠN 5 — Dashboard Streamlit & Triển khai (Tuần 6, nửa đầu)

#### 5.1 Task P5.1 — Kiến trúc UI

Viết lại `05_Deployment_Demo/app.py` (giữ tên `app.py` theo Báo cáo giữa kỳ MT4; cập nhật `AGENTS.md §2` cho khớp thay vì đổi tên file đã cam kết trong báo cáo).

**Sidebar — Interactive selectors:**

| Control | Lựa chọn |
|---|---|
| Thuật toán | PPO / SAC |
| Reward mode | A0 `standard` / A1 `volatility_penalty` / A2 `differential_sharpe` / A3 `prospect_theory_cvar` |
| Walk-forward window | Window 1–4 |
| Baselines hiển thị | Multi-select: VN30-Index, Equal Weight, Markowitz MPT |
| Khoảng thời gian | Date range slider |
| Vốn ban đầu | Number input, mặc định 100.000.000 VND |

**Main panel — 4 thành phần bắt buộc (theo Báo cáo giữa kỳ MT4):**

| # | Thành phần | Thư viện | Ghi chú |
|---|---|---|---|
| 1 | **Equity Curve Comparison** | `plotly.graph_objects.Scatter` | Agent vs 3 baselines trên cùng trục; tô vùng pha Bear 2022 |
| 2 | **Stacked Area — Portfolio Weights** | `plotly.express.area` | 30 cổ phiếu + cash theo thời gian; cash tách màu riêng nổi bật |
| 3 | **Underwater Drawdown Chart** | `Scatter` + `fill='tozeroy'` | Vẽ đường ngưỡng `-15%` để đối chiếu KPI |
| 4 | **Real-time Metrics Table** | `st.dataframe` | Toàn bộ KPI §1.3; tô xanh/đỏ theo đạt/không đạt ngưỡng |

#### 5.2 Task P5.2 — Tối ưu hiệu năng

> **Nguyên tắc thiết kế cốt lõi:** dashboard **chỉ đọc kết quả đã tính sẵn** từ `outputs/backtest_results/`, **không** huấn luyện và **không** chạy backtest trong request của người dùng. Chạy inference trực tiếp trên web sẽ làm UI treo nhiều giây mỗi lần thao tác.

```python
@st.cache_data(ttl=3600)
def load_backtest_results(window: int, algo: str, reward_mode: str) -> pd.DataFrame:
    """Nạp kết quả backtest đã tính sẵn (Parquet) — cache theo tham số."""
    ...

@st.cache_resource
def load_agent_checkpoint(path: str):
    """Chỉ dùng cho tab 'Live Inference' tùy chọn; cache model nặng."""
    ...
```

- Tiền tính toán toàn bộ 160 run ra Parquet ở Giai đoạn 4.
- Downsample equity curve xuống ≤ 1.000 điểm khi vẽ.
- Mục tiêu: **thời gian tải trang < 3 giây**, thao tác đổi selector phản hồi **< 1 giây**.

#### 5.3 DoD Giai đoạn 5

- [ ] `streamlit run 05_Deployment_Demo/app.py` khởi động không lỗi
- [ ] Đủ 4 thành phần biểu đồ, dữ liệu khớp với `summary.csv`
- [ ] Mọi selector hoạt động, không có lựa chọn nào gây exception
- [ ] Không còn tham chiếu `BTC-USD`, `ETH-USD`, `SPY`, "Antigravity"
- [ ] Tải trang < 3s; đã bật `@st.cache_data`
- [ ] `05_Deployment_Demo/requirements.txt` đồng bộ với root

---

### GIAI ĐOẠN 6 — Báo cáo Cuối kỳ, Nghiệm thu & Đóng gói (Tuần 6, nửa sau)

#### 6.1 Task P6.1 — Báo cáo toàn văn

`06_Docs_and_Reports/final_report/bao_cao_cuoi_ky.md`, cấu trúc:

| Chương | Nội dung | Nguồn |
|---|---|---|
| 1 | Đặt vấn đề & Stakeholders | Báo cáo giữa kỳ §1 |
| 2 | Khảo sát 5 papers / 5 patents / 5 products & Research Gap | Báo cáo giữa kỳ §2 |
| 3 | Câu hỏi & Mục tiêu nghiên cứu | Báo cáo giữa kỳ §3, §4 |
| 4 | Phương pháp: MDP formulation, `OBS_SCHEMA`, simplex, 4 reward modes | Giai đoạn 2 |
| 5 | Mô phỏng ràng buộc HOSE (T+2, ±7%, lô 100, thuế/phí) | Giai đoạn 2 |
| 6 | Thiết lập thực nghiệm: walk-forward, 5 seeds, 160 runs | Giai đoạn 3 |
| 7 | Kết quả & Ablation 4 reward modes; phân tích 3 pha thị trường | Giai đoạn 4 |
| 8 | Trả lời trực tiếp 3 RQ bằng số liệu thực nghiệm | Tổng hợp |
| 9 | Hạn chế & hướng phát triển | Mới |
| 10 | Tài liệu tham khảo | Mới |

**Chương 9 phải nêu trung thực các hạn chế sau:**
- Pha Bull 2020–2021 không phải OOS thuần túy (xem §4.4).
- Dữ liệu chỉ đến 2025-12-31, không có 2026 như mục tiêu ban đầu nêu.
- Rổ VN30 dùng danh sách **cố định hiện tại**, chưa tái tạo lịch sử thay đổi thành phần rổ → còn tồn tại survivorship bias.
- Không mô phỏng market impact và thanh khoản động (chỉ dừng ở mức M1 trong `simulation_levels`).
- FinBERT sentiment nêu trong `kich_ban_trao_doi_nhom.md` RQ2 **đã được loại khỏi phạm vi**; Báo cáo giữa kỳ đã thu hẹp còn 3 RQ không gồm sentiment. Cần thống nhất trong báo cáo cuối.

#### 6.2 Task P6.2 — Slide & Video demo

- Slide thuyết trình (~15 trang) tại `06_Docs_and_Reports/presentation/`.
- Video demo Remotion: tái sử dụng hạ tầng từ commit `28176d0` và `8a1e1e1`; cập nhật số liệu thực tế từ Giai đoạn 4.

#### 6.3 Task P6.3 — Đóng gói & nghiệm thu mã nguồn

Chạy đầy đủ chuỗi verification `AGENTS.md §5`:

```bash
source .venv/bin/activate

# 1. Type check — bắt buộc 0 lỗi
mypy --strict --ignore-missing-imports \
  02_Data_Pipeline/ 03_Model_Development/ 04_Backtesting_Evaluation/

# 2. Test + coverage — bắt buộc >= 80%
pytest tests/ -v \
  --cov=02_Data_Pipeline --cov=03_Model_Development --cov=04_Backtesting_Evaluation \
  --cov-report=term-missing

# 3. Env sanity check (dùng bản đã sửa tại Sprint 0 S0.4)
python -c "
import importlib, sys
sys.path.insert(0, '03_Model_Development')
from gymnasium.utils.env_checker import check_env
env = importlib.import_module('env.vn30_portfolio_env').VN30PortfolioEnv()
check_env(env); print('OK: Env check passed')
"

# 4. Pipeline smoke test
python 02_Data_Pipeline/run_pipeline.py --dry-run
```

**Checklist đóng gói:**
- [ ] `README.md` cập nhật: hướng dẫn cài đặt, chạy pipeline, train, backtest, demo
- [ ] Xóa toàn bộ `__pycache__`, `.pyc` khỏi git; kiểm tra `.gitignore`
- [ ] Loại bỏ hết naming "Antigravity" trong mã nguồn (§2.2.F)
- [ ] `data/archive/BTC_USD_features.csv` và `data_collector.py` (BTC/Yahoo legacy) được xóa hoặc chuyển vào `archive/`
- [ ] Mọi incident trong quá trình phát triển đã ghi vào `06_Docs_and_Reports/diagnostic_log.md` theo template `§7`
- [ ] Tất cả PR đã merge vào `main` (không push trực tiếp — `AGENTS.md §10`)

---

## 4. MA TRẬN PHÂN CÔNG NHIỆM VỤ (RACI MATRIX)

**Quy ước:** **R** = Responsible (người thực thi) · **A** = Accountable (người chịu trách nhiệm cuối, duy nhất 1 người/task) · **C** = Consulted (được hỏi ý kiến) · **I** = Informed (được thông báo)

### 4.1 Vai trò thành viên

| Thành viên | Vai trò | Phạm vi chuyên môn |
|---|---|---|
| **Như Đình Tú** (`@nhudinhlu1`) | Lead — Quant Theory & Reward Shaping | Thiết kế hàm thưởng, MDP formulation, lý thuyết tài chính định lượng, nghiệm thu KPI |
| **Nguyễn Văn Mạnh** (`@23021619`) | ML Engineer — PPO/SAC & Backtesting | Thuật toán RL, cấu hình hyperparameter, walk-forward engine, metrics |
| **Trần Minh Đức** (`@duc_tm22`) | Data Engineer — Pipeline & Features | Collectors, processors, feature engineering, chất lượng dữ liệu |
| **Lê Hoàng Long** (`@long_lh01`) | System Engineer — Gym Env & Streamlit | `VN30PortfolioEnv`, mô phỏng HOSE, dashboard, tổng hợp báo cáo |

### 4.2 Ma trận RACI theo task

| Mã | Task | Tú (Lead) | Mạnh (ML) | Đức (Data) | Long (System) |
|---|---|:---:|:---:|:---:|:---:|
| **S0.1–S0.3** | Khôi phục venv & pin dependency | I | C | **A/R** | C |
| **S0.4–S0.6** | Sửa harness, pytest config | **A** | C | C | **R** |
| **S0.7** | `--dry-run` cho pipeline | I | I | **A/R** | I |
| **P1.1** | `clean_alignment.py` | C | I | **A/R** | C |
| **P1.2** | `build_features.py` | **C** | C | **A/R** | I |
| **P1.x** | Xử lý late-listing SSB/TCB | **A** | C | **R** | C |
| **P2.1** | `OBS_SCHEMA` & Observation Space | **A** | C | C | **R** |
| **P2.2** | Action Space & simplex projection | **A/R** | C | I | **R** |
| **P2.3** | Mô phỏng HOSE (T+2, lô, band, phí) | **A** | C | C | **R** |
| **P2.4** | 4 reward modes nhận thức rủi ro | **A/R** | C | I | C |
| **P2.5** | 10 test cho env | C | C | I | **A/R** |
| **P3.1** | YAML configs PPO/SAC | C | **A/R** | I | I |
| **P3.2** | Tái cấu trúc agent wrappers | I | **A/R** | I | C |
| **P3.3** | Chạy training 1M steps × 5 seeds | C | **A/R** | I | C |
| **P3.4** | TensorBoard & custom callback | C | **A/R** | I | C |
| **P4.1** | Hoàn thiện `metrics.py` | **A** | **R** | I | I |
| **P4.2** | 3 baselines | **A** | **R** | C | C |
| **P4.3** | Walk-forward backtester | C | **A/R** | I | C |
| **P4.4** | Phân tích 3 pha thị trường | **A/R** | **R** | I | I |
| **P5.1** | Streamlit UI 4 thành phần | C | I | I | **A/R** |
| **P5.2** | Tối ưu hiệu năng dashboard | I | C | I | **A/R** |
| **P6.1** | Báo cáo cuối kỳ | **A** | C | C | **R** |
| **P6.2** | Slide & video demo | C | I | I | **A/R** |
| **P6.3** | Đóng gói, mypy, coverage | **A** | **R** | **R** | **R** |

### 4.3 Nguyên tắc phối hợp & chống nghẽn

| Rủi ro phối hợp | Biện pháp |
|---|---|
| **Giai đoạn 2 là nút cổ chai** — Mạnh (P3) và Long (P5) đều chờ env xong | Long và Tú làm P2 song song ngay từ Tuần 2. Trong lúc chờ, Mạnh triển khai P4.1 (`metrics.py`) và P4.2 (baselines) vì **không phụ thuộc env** |
| Đức xong việc sớm (hết Tuần 1) | Chuyển sang hỗ trợ P2.5 (viết test env) và tiền tính toán kết quả cho dashboard |
| Interface giữa Data và Env chưa rõ | **Tuần 1 phải chốt trước** schema `feature_matrix.parquet`. Long viết env dựa trên schema đã chốt, dùng fixture dữ liệu giả để không phải chờ Đức |
| Xung đột khi merge | Mỗi task 1 feature branch `feat/<desc>`, PR phải liệt kê checkbox DoD đã thỏa (`AGENTS.md §10`) |

**Nghi thức làm việc:** standup ngắn 2 lần/tuần; mọi incident ghi vào `diagnostic_log.md` theo template `§7`; không ai được tự ý sửa `AGENTS.md` mà không thông báo (A = Tú).

---

## 5. QUẢN TRỊ RỦI RO KỸ THUẬT & PHƯƠNG ÁN DỰ PHÒNG

### 5.1 Sổ đăng ký rủi ro (Risk Register)

Thang điểm: Xác suất (P) và Tác động (I) từ 1–5; Mức độ = P × I.

| ID | Rủi ro | P | I | Mức | Chủ trì |
|---|---|:---:|:---:|:---:|---|
| **R1** | Rate limit / gián đoạn API vnstock | 4 | 2 | 8 | Đức |
| **R2** | Overfitting hoặc policy co cụm 100% cash | **5** | **5** | **25** | Tú |
| **R3** | Churning — chi phí giao dịch bào mòn lợi nhuận | 4 | 4 | 16 | Tú |
| **R4** | T+2 queue làm chậm simulation step | 3 | 3 | 9 | Long |
| **R5** | `check_env` không pass 0 warning | 3 | 4 | 12 | Long |
| **R6** | Không đạt KPI (Sharpe ≥ 1.2, MDD ≤ 15%) | **4** | **5** | **20** | Tú + Mạnh |
| **R7** | Thiếu thời gian tính toán cho 160 runs | 4 | 3 | 12 | Mạnh |
| **R8** | Survivorship bias do rổ VN30 cố định | 3 | 3 | 9 | Đức |

### 5.2 Phân tích chi tiết & phương án dự phòng

#### R1 — Rate limit và gián đoạn API vnstock (Mức 8)

**Hiện trạng:** đã có 2 incident được ghi trong `diagnostic_log.md` (API deprecated, HTTP 429). Dữ liệu 30 mã **đã tải xong và cache**, nên rủi ro hiện tại thấp.

| Biện pháp | Chi tiết |
|---|---|
| Phòng ngừa | Giữ `skip_existing=True`; `time.sleep(3.5)` (`AGENTS.md §1`: 20 req/phút) |
| Dự phòng 1 | **Commit dữ liệu raw vào git** (30 file CSV, dung lượng nhỏ) để không bao giờ phải tải lại |
| Dự phòng 2 | Nếu vnstock chết hoàn toàn: fallback sang `yfinance` với hậu tố `.VN` |
| Phát hiện sớm | `run_pipeline.py --dry-run` kiểm tra kết nối trước mỗi phiên làm việc |

#### R2 — Overfitting / policy co cụm 100% cash (Mức 25 — **NGHIÊM TRỌNG NHẤT**)

**Cơ chế phát sinh:** đây là hệ quả trực tiếp của thiết kế risk-aware. Khi `η` (hệ số phạt drawdown) hoặc `λ` (phạt volatility) quá lớn, nghiệm tối ưu toàn cục của bài toán trở thành `w_cash = 1.0` — tác tử không giao dịch gì cả thì không có drawdown, reward xấp xỉ 0 nhưng vẫn cao hơn mọi chiến lược có rủi ro. Tác tử "đạt" KPI MDD = 0% nhưng hoàn toàn vô dụng.

| Biện pháp | Chi tiết |
|---|---|
| Phát hiện sớm | Log `portfolio/cash_weight_mean` mỗi rollout. **Ngưỡng báo động: > 0.9 liên tục 50k steps** |
| Chặn bằng KPI kép | Nghiệm thu **bắt buộc** đồng thời `MDD ≤ 15%` **VÀ** `CAGR > CAGR(VN30-Index)`. Chỉ xét MDD đơn lẻ là lỗ hổng nghiệm thu |
| Ràng buộc cấu trúc | Giới hạn `w_cash ≤ 0.6` trong pha huấn luyện (ràng buộc mềm bằng penalty), nới ra khi đánh giá |
| Giữ exploration | `ent_coef = 0.01` cho PPO, `ent_coef = "auto"` cho SAC |
| Chống overfitting | 5 seeds + walk-forward; chỉ tin **median**, không tin seed tốt nhất; early stopping theo validation |
| Dự phòng | Hạ `η` từ 5.0 → 2.0; chuyển reward mode từ A3 sang A1; thu hẹp ngưỡng `d*` |

#### R3 — Churning (Mức 16)

**Cơ chế:** ma sát round-trip 0.40% (§1.4). Nếu tác tử đảo danh mục toàn phần hàng ngày trong 252 phiên, chi phí lý thuyết lên tới hàng chục phần trăm NAV/năm — triệt tiêu toàn bộ lợi nhuận. Đây chính là hiện tượng Đức đã nêu trong `kich_ban_trao_doi_nhom.md` Chặng 1.

| Biện pháp | Chi tiết |
|---|---|
| Nội sinh hóa chi phí | Phí/thuế trừ **trực tiếp vào NAV** trong `step()`, nên tự động phản ánh vào reward — tác tử học tránh churning một cách tự nhiên |
| Phạt turnover tường minh | Thêm số hạng `− ζ · turnover_t` vào reward nếu chi phí vẫn vượt ngưỡng |
| Vùng chết (no-trade band) | Bỏ qua lệnh nếu `|w_target − w_current| < 2%` — giảm mạnh giao dịch vụn |
| Giãn tần suất rebalance | Nếu vẫn churning: chuyển từ rebalance hàng ngày sang hàng tuần/tháng |
| Nghiệm thu | KPI Turnover Cost < 2% NAV/năm, đo riêng trong pha Sideway 2023–2025 (whipsaw nặng nhất) |

#### R4 — T+2 queue làm chậm simulation (Mức 9)

**Phân tích định lượng:** train 1M steps × 2 thuật toán × 5 seeds = 10M step. Ở tốc độ 500 step/s cần ~5,5 giờ; ở 100 step/s cần **27+ giờ** — không khả thi trong 1 tuần.

| Biện pháp | Chi tiết |
|---|---|
| Vectorize | Dùng `np.ndarray (2, 30)` + `np.roll` thay vì list-of-dict (§2.3) — toàn bộ logic O(1) |
| Tiền tính toán | Nạp sẵn giá và price band thành ma trận NumPy trong `__init__`, không truy vấn DataFrame trong `step()` (lookup `.loc` trong vòng lặp là nguyên nhân chậm phổ biến nhất) |
| Benchmark gate | Test hiệu năng bắt buộc: **≥ 2.000 step/s**, chạy trong CI như một test |
| Dự phòng | `SubprocVecEnv` chạy song song nhiều env; hoặc giảm `total_timesteps` xuống 300k và ghi rõ trong báo cáo |

#### R5 — `check_env` không pass 0 warning (Mức 12)

Nguyên nhân thường gặp: observation bound vô hạn, dtype không khớp `float32`, `reset(seed=...)` không gọi `super().reset(seed=seed)`, `info` không phải dict.

**Biện pháp:** dùng bound hữu hạn `Box(-10, 10)` + clip obs (§2.1); gọi `super().reset(seed=seed)` và dùng `self.np_random`; chạy `check_env` trong CI ngay từ commit đầu tiên của env, không để đến cuối.

#### R6 — Không đạt KPI (Mức 20)

**Đây là rủi ro học thuật lớn nhất.** Các ngưỡng `Sharpe ≥ 1.2`, `Sortino ≥ 1.8`, `MDD ≤ 15%` là cam kết tham vọng trên một thị trường đã giảm 42.8% trong giai đoạn test.

| Thứ tự xử lý | Hành động |
|---|---|
| 1 | Ablation 4 reward modes, chọn mode tốt nhất theo validation |
| 2 | Tinh chỉnh `λ`, `η`, `d*` theo grid nhỏ trên tập validation (**không** trên test) |
| 3 | Mở rộng feature: thêm regime indicator, breadth, VN30 momentum |
| 4 | Thử PPO thay SAC hoặc ngược lại; tăng timesteps |
| **Dự phòng cuối** | Nếu vẫn không đạt: **báo cáo trung thực kết quả thực tế** kèm phân tích nguyên nhân, và trình bày KPI đạt được tương đối so với 3 baselines. Một nghiên cứu trung thực "đạt Sharpe 0.9 nhưng vượt cả 3 baseline và giảm MDD 40% so với VN30-Index" có giá trị khoa học cao hơn số liệu được tô vẽ. **Tuyệt đối không** tinh chỉnh tham số trên tập test để đạt KPI — đó là gian lận phương pháp luận |

#### R7 — Thiếu tài nguyên cho 160 runs (Mức 12)

Thứ tự giảm quy mô (đã nêu §3.3): giữ 5 seeds cho cấu hình chính → giảm ablation xuống 150k steps → giảm 4 window xuống 2. Chạy ablation song song nhiều process nếu CPU cho phép. Mọi quyết định giảm quy mô phải ghi vào báo cáo cuối kỳ.

#### R8 — Survivorship bias (Mức 9)

Rổ VN30 dùng danh sách cố định hiện tại, trong khi thành phần rổ được HOSE review 2 lần/năm. Các mã bị loại khỏi rổ trong quá khứ (thường là mã kém hiệu năng) không có trong dữ liệu → kết quả bị thiên lệch lạc quan.

**Biện pháp:** `listing_mask` (§1.1) đã xử lý được phần late-listing (SSB, TCB). Phần tái tạo lịch sử thay đổi rổ đầy đủ **nằm ngoài phạm vi** đồ án học phần — phải **ghi rõ hạn chế này trong Chương 9** báo cáo cuối kỳ.

---

## 6. BẢNG THEO DÕI TIẾN ĐỘ TỔNG HỢP

| Tuần | Giai đoạn | Deliverable nghiệm thu | Gate ra |
|---|---|---|---|
| **0** (2 ngày) | Sprint 0 | venv sạch, harness chạy được | 5 bước `AGENTS.md §5` không lỗi cú pháp |
| **1** | GĐ 1 — Data | `aligned_prices.parquet`, `feature_matrix.parquet` | 9 test pass, mypy 0 lỗi, schema đã chốt |
| **2–3** | GĐ 2 — Env | `vn30_portfolio_env.py` | `check_env` 0 warning, 10 test pass, ≥ 2.000 step/s |
| **4** | GĐ 3 — Training | Checkpoint PPO + SAC 1M steps | exit 0, `ep_rew_mean` ≥ 0, TensorBoard đầy đủ |
| **5** | GĐ 4 — Backtest | `summary.csv`, 3 baselines, figures | KPI §1.3 đạt trên median 5 seeds |
| **6a** | GĐ 5 — Demo | Streamlit 4 thành phần | Tải < 3s, mọi selector hoạt động |
| **6b** | GĐ 6 — Nghiệm thu | Báo cáo, slide, video, mã nguồn sạch | mypy 0 lỗi, coverage ≥ 80%, mọi DoD đã tick |

### 6.1 Thứ tự ưu tiên tuyệt đối

Nếu thời gian bị thu hẹp, giữ theo thứ tự sau (bỏ từ dưới lên):

1. **`VN30PortfolioEnv` + test** — đây là đóng góp khoa học cốt lõi của đề tài (Research Gap mục 2)
2. **Data pipeline** — không có dữ liệu thì không có gì cả
3. **PPO 1 reward mode + 3 baselines** — tối thiểu để trả lời RQ
4. **Backtest walk-forward** — bằng chứng thực nghiệm
5. **Báo cáo cuối kỳ** — sản phẩm nghiệm thu bắt buộc
6. SAC + ablation 4 reward modes — làm mạnh thêm luận cứ
7. Streamlit dashboard — trình diễn
8. Video Remotion — tùy chọn

---

*Tài liệu lập trên cơ sở khảo sát trực tiếp mã nguồn, dữ liệu và tài liệu dự án. Mọi ngưỡng nghiệm thu tuân thủ `AGENTS.md §6` (Definition of Done).*







