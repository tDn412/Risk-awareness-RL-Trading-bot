# Kế hoạch cải thiện v2: kiểm soát turnover

> Dành cho agent (Claude) chạy trên máy khác. Đọc kèm `AGENTS.md` và `SOURCE_OF_TRUTH.md`.
> Nhánh làm việc: `feat/turnover-control`. **Không thêm dòng `Co-Authored-By: Claude` vào commit.**

## 1. Bối cảnh (kết quả v1, tuần 5)

Backtest trên tập test 2022 (cửa sổ 1), trung bình 3 seed — chi tiết: `06_Docs_and_Reports/week6_window1_test_summary.md`, lưu trữ ở `outputs/backtest_results/v1_week5/`.

| Chiến lược | Lợi nhuận | MDD | Turnover/phiên | Phí ma sát/năm |
|---|---|---|---|---|
| PPO A0 / A1 / A2 / A3 | −47.8 / −40.2 / −34.8 / −32.0% | 53 / 48 / 45 / 40% | 18–32% | 17–30% |
| VN30 Index | −35.5% | 42% | — | — |
| Equal Weight 1/N | −26.3% | 37% | 0.6% | 0.3% |
| EEB (cùng mức giải ngân với agent) | ≈ −26% | ≈ 37% | 2–5% | 1–2% |

**Chẩn đoán:** agent đảo danh mục quá nhiều, phí và thuế ăn mất phần lớn lợi nhuận. EEB có cùng mức giải ngân nhưng ít giao dịch, và đã thắng VN30.

## 2. Đã làm sẵn trên nhánh (commit `5a6b3e4` + commit chứa file này)

- `VN30PortfolioEnv(no_trade_band=..., action_smoothing=...)`
  - `no_trade_band`: chỉ giao dịch một mã khi tỷ trọng mục tiêu lệch ít nhất *band* so với hiện tại.
  - `action_smoothing`: `w = α·w_target + (1−α)·w_current`.
- `RewardConfig.lambda_turn`: trừ `λ_turn · turnover` ở mọi biến thể A0–A3.
- `03_Model_Development/callbacks.py`: `SharpeCheckpointCallback` chọn checkpoint `_best` theo **Sharpe trên validation 2021** (thay cho `EvalCallback` chọn theo reward). Tiêu chí này so sánh được giữa các biến thể.
- Config `ppo_baseline.yaml` / `sac_baseline.yaml`: `no_trade_band: 0.02`, `action_smoothing: 1.0`, `lambda_turn: 0.005`.
- Backtester và train đọc các tham số mới; run cũ (không có các khoá này) mặc định về 0 nên tái lập được v1.
- Demo: khung tóm tắt kết quả giờ tự tính từ bảng chỉ số.
- Test: 47 test đều qua, mypy strict sạch trên các module đã đổi.

## 3. Việc cần làm trên máy kia

### Bước 0 — Môi trường (chỉ làm một lần)
```bash
git fetch origin && git checkout feat/turnover-control
python -m venv .venv          # Python 3.12
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
grep -v vnstock requirements.txt > /tmp/req.txt && .venv/Scripts/python -m pip install -r /tmp/req.txt -r requirements-dev.txt pandas-stubs tensorboard tabulate
```
`vnstock` và `vnai` không còn trên PyPI; collector tự dùng `collectors/vci_client.py`. Trên Linux/macOS thay `.venv/Scripts/` bằng `.venv/bin/`.

### Bước 1 — Dữ liệu (không có trong git)
```bash
.venv/Scripts/python 02_Data_Pipeline/run_pipeline.py --skip-events --delay 3.5
```
Kiểm tra: `data/processed/feature_matrix.parquet` có 30 mã × 1.999 phiên (2018-01-02 → 2025-12-31).

### Bước 2 — Kiểm tra nhanh
```bash
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python 03_Model_Development/train.py --config ppo_baseline.yaml --dry-run
```

### Bước 3 — Sanity check turnover trước khi train hết (khoảng 2 phút)
```bash
.venv/Scripts/python 03_Model_Development/train.py --reward A2 --seed 1 --timesteps 30000
```
Trong `outputs/models/ppo_A2_window_1_s1_*.json`, mục `validation.best["Avg Turnover"]` phải **< 0.10** (v1 là 0.18–0.32). Nếu vẫn cao, tăng `no_trade_band` lên 0.03–0.05 hoặc `lambda_turn` lên 0.01–0.02 rồi thử lại. Xoá run thử trước khi sang bước 4.

### Bước 4 — Train lại (khoảng 1 giờ với 8 lõi CPU)
```bash
.venv/Scripts/python 03_Model_Development/run_experiments.py --config ppo_baseline.yaml --parallel 4
```
12 run: A0–A3 × seed {42, 101, 2024}, mỗi run 500k bước. Log nằm ở `outputs/logs/train_ppo_*.log`. Giữ máy thức trong lúc chạy.
Tuỳ chọn, nếu còn thời gian: chạy SAC với `--config sac_baseline.yaml --rewards A0 A3` (SAC chỉ khoảng 35 bước/giây trên CPU).

### Bước 5 — Đánh giá
```bash
.venv/Scripts/python 03_Model_Development/summarize_runs.py --out 06_Docs_and_Reports/v2_validation_summary.md
.venv/Scripts/python 04_Backtesting_Evaluation/backtester.py --window window_1
cp outputs/backtest_results/window_1_test_summary.md 06_Docs_and_Reports/v2_window1_test_summary.md
```
Backtester chỉ đọc `outputs/models/*.json` ở cấp gốc (run v1 nằm trong `outputs/models/v1_week5/`, không bị lẫn).

### Bước 6 — Demo, kiểm tra và commit
```bash
.venv/Scripts/python -m pytest tests -q        # test_app chạy khi đã có kết quả backtest
.venv/Scripts/streamlit run 05_Deployment_Demo/app.py   # xem lại giao diện
```
Commit kết quả (`outputs/backtest_results/`, `outputs/figures/window_1_test_nav.png`, các file summary trong `06_Docs_and_Reports/`), push nhánh rồi mở PR vào `main`. **Không thêm dòng `Co-Authored-By`.**

## 4. Tiêu chí thành công

| Chỉ số (test 2022, trung bình seed) | v1 | Mục tiêu v2 |
|---|---|---|
| Avg Turnover/phiên | 18–32% | **< 5%** (SSOT §7.2) |
| Phí ma sát/năm | 17–30% | **≤ 2%** |
| Lợi nhuận PPO A2/A3 | −34.8 / −32.0% | > VN30 (−35.5%), lý tưởng ≥ EEB |
| MDD PPO A2/A3 | 45 / 40% | giảm rõ, hướng tới ≤ 15% (H1) |

Nếu turnover đã thấp mà agent vẫn thua EEB thì kết quả v2 vẫn hợp lệ. Khi đó ghi vào báo cáo rằng lợi thế không đến từ chọn mã/timing (H3 bị bác bỏ).

## 5. Bước tiếp theo nếu v2 vẫn chưa đạt
1. Thêm đặc trưng cấp thị trường (lợi suất/độ biến động/drawdown của VN30, độ rộng thị trường) vào observation.
2. Tune λ trên validation 2021: `λ_dd ∈ {0.5, 2, 5}`, `λ_cvar ∈ {1, 5}`, `λ_turn ∈ {0.005, 0.01, 0.02}`.
3. Giảm overfit: `net_arch` [64, 64] và 200–300k bước, hoặc train các cửa sổ 2–4 (nhiều năm train hơn).
4. Ghi mọi sự cố vào `06_Docs_and_Reports/diagnostic_log.md` theo mẫu ở AGENTS.md §7.
