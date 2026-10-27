# 📁 Kiến Trúc Thư Mục Dự Án (Project Directory Structure)
## Đề tài: Risk-Aware Reinforcement Learning for VN30 Portfolio Allocation
**Học phần:** INT3011E | **Nhóm:** INT3011E 2 _ Risk - aware RL

Cấu trúc thư mục được tái thiết kế đồng bộ theo đúng các mục tiêu $O_1 - O_6$, câu hỏi nghiên cứu $RQ_1 - RQ_3$ và kế hoạch triển khai 6 tuần trong Báo cáo giữa kỳ.

---

```text
Risk-awareness RL Trading bot/
├── 01_Research_Proposal/              # [O6] Hồ sơ bài toán & Đề xuất nghiên cứu
│   ├── INT3011E-2_Risk-aware-RL.pdf   # Báo cáo giữa kỳ chính thức (PDF)
│   ├── project_charter.md             # Điều lệ dự án, câu hỏi RQ1-RQ3, giả thuyết H1-H3
│   └── literature_review.md           # Khảo sát 5 papers, 5 patents, 5 products
│
├── 02_Data_Pipeline/                  # [O1] Thu thập & Tiền xử lý dữ liệu VN30 (Tuần 1 & 2)
│   ├── collectors/                    # Các module thu thập dữ liệu qua vnstock / API
│   │   ├── fetch_vn30_prices.py       # Tải nến ngày OHLCV 30 mã (2018–2025)
│   │   ├── fetch_corporate_events.py  # Tải sự kiện cổ tức, chia tách cổ phiếu
│   │   └── fetch_benchmarks.py        # Tải chỉ số VN30-Index / VN-Index
│   ├── processors/                    # Các module làm sạch & chuẩn hóa dữ liệu
│   │   ├── clean_alignment.py         # Căn chỉnh lịch giao dịch HOSE & xử lý missing data
│   │   ├── feature_engineering.py     # Tính toán chỉ báo kỹ thuật, rolling volatility
│   │   └── split_windows.py           # Chia 4 cửa sổ trượt (Train/Val/Test) theo Bảng 4
│   └── run_pipeline.py                # Script chạy toàn bộ pipeline dữ liệu 1-click
│
├── 03_Model_Development/              # [O2, O3] Môi trường Gymnasium & Huấn luyện PPO (Tuần 3 & 4)
│   ├── env/                           # Môi trường mô phỏng giao dịch thực tế
│   │   ├── vn30_portfolio_env.py      # Môi trường Gymnasium đa tài sản Simplex [N+1]
│   │   ├── portfolio_ledger.py        # Kế toán tiền mặt, cổ phiếu khả dụng, chu kỳ T+2
│   │   ├── execution_engine.py        # Lớp thực thi 3 cấp độ: M0 (cơ sở), M1 (thực tế), M2 (bất lợi)
│   │   └── reward_schemes.py          # 4 Biến thể hàm thưởng bóc tách: A0, A1, A2, A3
│   ├── agents/                        # Kiến trúc tác tử học tăng cường
│   │   ├── ppo_agent.py               # Thuật toán chính: PPO Actor-Critic cho simplex
│   │   └── sac_agent.py               # Thuật toán phụ: SAC (nếu còn tài nguyên)
│   ├── configs/                       # Quản lý siêu tham số tập trung
│   │   └── default_config.yaml        # Cấu hình phí HOSE, thuế, ngưỡng phạt m, d*, cửa sổ
│   ├── train.py                       # Huấn luyện mô hình đơn lẻ (1 window, 1 config, 1 seed)
│   └── run_experiments.py             # Script tự động chạy batch 80 runs (4 configs x 4 windows x 5 seeds)
│
├── 04_Backtesting_Evaluation/         # [O4] Đánh giá Ngoài mẫu & Kiểm định Giả thuyết (Tuần 5)
│   ├── baselines/                     # Bộ chiến lược đối chuẩn bắt buộc
│   │   ├── buy_and_hold_vn30.py       # Đối chuẩn thị trường VN30-Index
│   │   ├── equal_weight.py            # Phân bổ đều Equal-Weight (1/N)
│   │   ├── cash_benchmark.py          # Đối chuẩn 100% Tiền mặt
│   │   ├── static_stock_cash.py       # Tỷ lệ cổ phiếu / tiền mặt cố định (e.g., 60/40)
│   │   └── equiv_equity_benchmark.py  # Đối chuẩn tỷ trọng cổ phiếu bình quân tương đương (H3)
│   ├── metrics.py                     # Tính toán CAGR, MDD, Sortino, Sharpe, 95% CVaR, Turnover
│   ├── backtest_engine.py             # Động cơ backtest ngoài mẫu trên tập Test (2022, 2023, 2024, 2025)
│   └── hypothesis_testing.py          # Script phân tích thống kê và kiểm định tự động H1, H2, H3
│
├── 05_Deployment_Demo/                # [O5] Web Dashboard Tra cứu & Trực quan hóa (Tuần 6)
│   ├── app.py                         # Ứng dụng tương tác Streamlit
│   ├── components/                    # Biểu đồ Plotly: NAV, Underwater Drawdown, Stacked Weights
│   └── requirements.txt               # Gói phụ thuộc riêng cho dashboard
│
├── data/                              # Quản lý dữ liệu tập trung (được gitignore trừ sample)
│   ├── raw/                           # Dữ liệu thô tải từ vnstock
│   │   ├── prices/                    # 30 file CSV OHLCV theo ngày
│   │   ├── corporate_actions/         # Sự kiện cổ tức, chia tách cổ phiếu
│   │   └── benchmarks/                # Dữ liệu chỉ số VN30-Index, VN-Index
│   ├── processed/                     # Dữ liệu sạch đã căn chỉnh theo 4 cửa sổ trượt
│   │   ├── window_1/                  # Train (2018-2020), Val (2021), Test (2022)
│   │   ├── window_2/                  # Train (2018-2021), Val (2022), Test (2023)
│   │   ├── window_3/                  # Train (2018-2022), Val (2023), Test (2024)
│   │   └── window_4/                  # Train (2018-2023), Val (2024), Test (2025)
│   └── archive/                       # Dữ liệu mẫu cũ lưu trữ
│
├── outputs/                           # Lưu trữ toàn bộ kết quả thực nghiệm tự động
│   ├── checkpoints/                   # Checkpoints mô hình PPO của 80 runs
│   ├── logs/                          # Nhật ký TensorBoard phục vụ phân tích hội tụ
│   ├── backtest_results/              # Bảng số liệu NAV, danh mục giao dịch xuất ra CSV
│   └── figures/                       # Biểu đồ so sánh A0-A3 và baselines
│
├── tests/                             # Bộ kiểm thử chất lượng (Unit Tests)
│   ├── test_portfolio_ledger.py       # Kiểm thử bảo toàn tiền mặt, chu kỳ T+2, lô chẵn 100
│   ├── test_environment.py            # Kiểm thử tính tuân thủ Gymnasium API
│   └── test_rewards.py                # Kiểm thử độ nhạy và thang đo của 4 hàm thưởng
│
├── SOURCE_OF_TRUTH.md                 # Single Source of Truth chuẩn tối cao của toàn dự án
├── AGENTS.md                          # Quy chuẩn vận hành và tiêu chí DoD cho AI Agents
├── DIRECTORY_STRUCTURE.md             # Bản đồ kiến trúc thư mục
├── README.md                          # Tài liệu hướng dẫn chung của toàn bộ dự án
└── requirements.txt                   # Danh mục thư viện Python bắt buộc
```
