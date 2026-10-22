# 🌟 SOURCE OF TRUTH (SSOT)
## Đề tài: Risk-Aware Reinforcement Learning for VN30 Portfolio Allocation
### Tối ưu hóa Phân bổ Danh mục Cổ phiếu VN30 Thích ứng Rủi ro bằng Học tăng cường Sâu

- **Học phần:** INT3011E — Trường Đại học Công nghệ, Đại học Quốc gia Hà Nội (VNU-UET)
- **Nhóm nghiên cứu:** `INT3011E 2 _ Risk - aware RL`
- **Tác giả / Duy trì:** Lê Đỗ Minh Tú & Nhóm nghiên cứu
- **Phiên bản:** v2.0 (Tháng 10/2026)
- **Kho lưu trữ:** [Risk-awareness RL Trading bot](https://github.com/tDn412/Risk-awareness-RL-Trading-bot)

> [!IMPORTANT]
> **QUY CHẾ SINGLE SOURCE OF TRUTH (SSOT):**  
> Tài liệu này là **nguồn chân lý tối cao duy nhất** của toàn bộ dự án. Mọi tài liệu thiết kế ([README.md](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/README.md), [AGENTS.md](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/AGENTS.md), [DIRECTORY_STRUCTURE.md](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/DIRECTORY_STRUCTURE.md)), mã nguồn trong các thư mục `01` đến `06`, quy chuẩn môi trường Gymnasium, siêu tham số, dữ liệu thực nghiệm và chỉ số kiểm định đều **bắt buộc tuân thủ 100%** nội dung tại đây.

---

## 📑 Mục Lục

1. [Tuyên ngôn Sứ mệnh & Tóm tắt Đề tài](#1-tuyên-ngôn-sứ-mệnh--tóm-tắt-đề-tài)
2. [Câu hỏi Nghiên cứu (RQ) & Giả thuyết Khoa học (H)](#2-câu-hỏi-nghiên-cứu-rq--giả-thuyết-khoa-học-h)
3. [Mục tiêu Nghiên cứu Cụ thể (O1 – O6)](#3-mục-tiêu-nghiên-cứu-cụ-thể-o1--o6)
4. [Ràng buộc Thị trường HOSE (Market Constraints)](#4-ràng-buộc-thị-trường-hose-market-constraints)
5. [Quy chuẩn Dữ liệu & Pipeline (Data Pipeline)](#5-quy-chuẩn-dữ-liệu--pipeline-data-pipeline)
6. [Môi trường Gymnasium & Công thức Toán học](#6-môi-trường-gymnasium--công-thức-toán-học)
7. [Bộ Chiến lược Đối chuẩn & Chỉ số Định lượng](#7-bộ-chiến-lược-đối-chuẩn--chỉ-số-định-lượng)
8. [Cấu trúc Thư mục Chuẩn (Canonical Architecture)](#8-cấu-trúc-thư-mục-chuẩn-canonical-architecture)
9. [Công nghệ & Quy chuẩn Mã nguồn](#9-công-nghệ--quy-chuẩn-mã-nguồn)
10. [Quy trình Kiểm định & Tiêu chí Hoàn thành (DoD)](#10-quy-trình-kiểm-định--tiêu-chí-hoàn-thành-dod)
11. [Hướng dẫn Khởi chạy Nhanh (Quick Start)](#11-hướng-dẫn-khởi-chạy-nhanh-quick-start)

---

## 1. Tuyên ngôn Sứ mệnh & Tóm tắt Đề tài

### 1.1 Sứ mệnh (Mission Statement)
Xây dựng một hệ thống phân bổ danh mục đầu tư tự động ứng dụng **Học tăng cường sâu nhận thức rủi ro (Risk-Aware Deep Reinforcement Learning)** cho rổ 30 cổ phiếu vốn hóa lớn thị trường chứng khoán Việt Nam (**VN30**). 

Hệ thống tối ưu hóa tỷ suất sinh lời điều chỉnh theo rủi ro (Sharpe, Sortino, Calmar) và chủ động kiểm soát tổn thất đuôi (**$95\%$ CVaR**) cùng mức sụt giảm vốn tối đa (**Max Drawdown $\le 15\%$**) trong các chu kỳ khủng hoảng, dưới đầy đủ các ràng buộc vi cấu trúc thực tế của Sở Giao dịch Chứng khoán TP.HCM (**HOSE**).

### 1.2 Bối cảnh & Vấn đề Cốt lõi
- **Hạn chế của RL truyền thống:** Các mô hình RL tối ưu lợi nhuận thuần túy thường tích cực nắm giữ cổ phiếu beta cao trong pha tăng trưởng (Bull market). Khi thị trường đảo chiều sang suy thoái (Bear market như năm 2022: VN-Index giảm $-42.8\%$, nhiều mã VN30 giảm $>60\%$), tác tử tiếp tục bắt đáy gây "cháy tài khoản" (catastrophic drawdowns).
- **Ràng buộc sàn HOSE:** Thị trường cơ sở Việt Nam chỉ cho phép mua nắm giữ (**long-only**, không được short-selling), biên độ dao động giá **$\pm 7\%$**, chu kỳ thanh toán bù trừ **T+2**, đơn vị lô chẵn **100 cổ phiếu**, và chịu thuế bán **0.1%** cùng phí môi giới **0.15%**. Bỏ qua các ràng buộc này sẽ tạo ra độ lệch thực thi (execution gap) khổng lồ khi triển khai thực tế.
- **Giải pháp bóc tách:** Đề tài thiết kế **4 biến thể hàm thưởng nhận thức rủi ro ($A_0 - A_3$)** kết hợp lý thuyết hữu dụng triển vọng (**Prospect Theory**) và cơ chế rào cản tổn thất đuôi, tích hợp cùng thuật toán **PPO** (chính) và **SAC** (phụ).

---

## 2. Câu hỏi Nghiên cứu (RQ) & Giả thuyết Khoa học (H)

| Mã | Câu hỏi Nghiên cứu (Research Question) | Giả thuyết Khoa học (Hypothesis) | Phương pháp Kiểm chứng Thực nghiệm |
|:---|:---|:---|:---|
| **$RQ_1$** | Việc tích hợp hàm thưởng nhận thức rủi ro đa mục tiêu (Downside semi-variance, Drawdown barrier, Prospect Theory) có giúp tác tử PPO kiềm chế mức sụt giảm tài sản tối đa ($\text{MDD} \le 15\%$) trong giai đoạn thị trường suy thoái (2022) mà vẫn bảo toàn tỷ suất sinh lời vượt trội so với VN30-Index không? | **$H_1$:** Tác tử PPO với hàm thưởng nhận thức rủi ro ($A_2, A_3$) đạt chỉ số Sharpe $\ge 1.2$, Sortino $\ge 1.8$ và cắt giảm ít nhất $50\%$ mức sụt giảm tối đa so với đối chuẩn VN30-Index trong đợt sập năm 2022 ($\text{MDD}_{\text{VN30}} = 45.2\%$). | So sánh đường cong vốn NAV, Drawdown Profile và ma trận chỉ số định lượng giữa $A_0, A_1, A_2, A_3$ và VN30-Index trên tập Test Cửa sổ 1 (2022). |
| **$RQ_2$** | Việc áp đặt các ràng buộc vi cấu trúc HOSE (T+2, biên độ $\pm 7\%$, lô 100, phí/thuế) có làm suy giảm hiệu năng trên lý thuyết nhưng giúp hạn chế tối đa độ lệch thực thi (execution gap) giữa môi trường mô phỏng và thực tế không? | **$H_2$:** Mặc dù các ràng buộc HOSE làm giảm CAGR danh nghĩa từ $3\% - 6\%$, mô hình vẫn duy trì hiệu năng dương ổn định và triệt tiêu hoàn toàn hiện tượng tái cơ cấu quá mức (portfolio churning, chi phí giao dịch $< 2\%$/năm). | So sánh hiệu năng giữa 3 cấp độ mô phỏng khớp lệnh: $M_0$ (cơ sở lý thuyết), $M_1$ (thực tế HOSE), $M_2$ (bất lợi thanh khoản). |
| **$RQ_3$** | Hiệu năng vượt trội và mức sụt giảm thấp của tác tử Risk-Aware RL đến từ khả năng lựa chọn cổ phiếu và timing thị trường thông minh, hay đơn thuần từ việc phòng thủ nắm giữ tỷ trọng tiền mặt lớn ($w_{\text{cash}}$)? | **$H_3$:** Tác tử $A_2, A_3$ đạt tỷ suất sinh lời và chỉ số Sortino cao hơn có ý nghĩa thống kê ($p < 0.05$ qua paired t-test) so với đối chuẩn **Equivalent Equity Benchmark** (danh mục nắm giữ tỷ lệ cổ phiếu VN30 tương đương tỷ lệ giải ngân bình quân của tác tử). | Xây dựng thuật toán đối chuẩn Equivalent Equity Benchmark tự động tái lập theo đúng tỷ trọng giải ngân thực tế của tác tử qua từng phiên. |

---

## 3. Mục tiêu Nghiên cứu Cụ thể (O1 – O6)

- **$O_1$ — Data Pipeline (Tuần 1 & 2):** Xây dựng pipeline tự động thu thập lịch sử OHLCV, sự kiện quyền (cổ tức tiền, cổ phiếu thưởng, chia tách) cho 30 mã VN30 và chỉ số tham chiếu (2018–2025) từ `vnstock`. Căn chỉnh lịch giao dịch HOSE (1,996 phiên), xử lý gián đoạn giao dịch (gap $\le 10$ ngày), tính toán ma trận 28+ chỉ báo kỹ thuật và chia 4 cửa sổ trượt Walk-Forward.
- **$O_2$ — Môi trường Gymnasium (Tuần 3):** Xây dựng môi trường đa tài sản `VN30PortfolioEnv` chuẩn Gymnasium v0.29+ với module kế toán sổ cái `portfolio_ledger.py` mô phỏng đầy đủ: chu kỳ T+2, biên độ $\pm 7\%$, lô 100, phí $0.15\%$, thuế $0.1\%$, và không gian hành động Simplex $[N+1]$.
- **$O_3$ — Tác tử RL & Huấn luyện (Tuần 4):** Triển khai thuật toán PPO và SAC với mạng Actor-Critic xuất vector tỷ trọng qua Softmax Policy. Huấn luyện thực nghiệm bóc tách 4 hàm thưởng ($A_0 - A_3$) trên các cửa sổ trượt với nhiều random seeds.
- **$O_4$ — Đánh giá Ngoài mẫu & Kiểm định Giả thuyết (Tuần 5):** Thực hiện backtest Out-of-Sample trên các tập Test (2022, 2023, 2024, 2025). So sánh với 5 đối chuẩn bắt buộc và kiểm định thống kê $H_1, H_2, H_3$.
- **$O_5$ — Web Dashboard Trực quan hóa (Tuần 6):** Xây dựng ứng dụng tương tác Streamlit (`05_Deployment_Demo/app.py`) trực quan hóa đường cong NAV, Underwater Drawdown, phân bổ tỷ trọng theo thời gian thực và bảng chỉ số định lượng.
- **$O_6$ — Tài liệu & Công bố Khoa học (Xuyên suốt):** Hoàn thiện báo cáo khoa học, slide Beamer LaTeX, bài thuyết trình tương tác Remotion, và video tóm tắt phục vụ bảo vệ đề tài.

---

## 4. Ràng buộc Thị trường HOSE (Market Constraints)

Tất cả các ràng buộc sau đây là **quy tắc cứng** được lập trình trong môi trường mô phỏng. Bất kỳ môi trường hay kịch bản backtest nào vi phạm đều bị coi là **không hợp lệ (invalid)**:

| Ràng buộc | Giá trị quy định | Giải thích cơ chế kỹ thuật | Vị trí hiện thực |
|:---|:---|:---|:---|
| **Chu kỳ thanh toán** | **T+2** | Cổ phiếu mua tại phiên $t$ phải đến đầu phiên $t+2$ mới khả dụng để bán. Module sổ cái duy trì hàng đợi trạng thái: $S_{\text{pending\_1}} \to S_{\text{pending\_2}} \to S_{\text{available}}$. | `env/portfolio_ledger.py` |
| **Biên độ dao động giá** | **$\pm 7\%$** | Giá đặt lệnh không được vượt ngoài dải trần/sàn: $[P_{\text{ref}} \times 0.93, P_{\text{ref}} \times 1.07]$. Nếu cổ phiếu chạm sàn không có thanh khoản mua, lệnh bán không được khớp toàn bộ. | `env/execution_engine.py` |
| **Đơn vị giao dịch** | **Lô 100 cổ phiếu** | Khối lượng mua/bán cổ phiếu bắt buộc làm tròn xuống bội số nguyên của 100 shares. Phần tiền dư thừa được hoàn trả tự động vào tài khoản tiền mặt. | `env/portfolio_ledger.py` |
| **Thuế thu nhập (Bán)** | **0.1%** | Khấu trừ trực tiếp $0.1\%$ trên tổng giá trị khớp lệnh bán của cổ phiếu. | `env/vn30_portfolio_env.py` |
| **Phí môi giới** | **0.15%** | Khấu trừ $0.15\%$ trên giá trị giao dịch cho cả hai chiều mua và bán (tổng phí 1 vòng mua-bán là $0.3\%$). | `env/vn30_portfolio_env.py` |
| **Không bán khống** | **$w_i \ge 0$** | Thị trường cơ sở Việt Nam là Long-only. Không cho phép tỷ trọng âm ($w_i \ge 0, \forall i$). | Simplex Action Space |
| **Tài sản tiền mặt** | **$w_0 \equiv w_{\text{cash}}$** | Tiền mặt là tài sản an toàn trong danh mục ($i=0$). Tỷ suất sinh lời tiền mặt bằng $0$ (hoặc lãi suất tiền gửi không kỳ hạn). | Simplex: $\sum_{i=0}^{30} w_i = 1.0$ |

---

## 5. Quy chuẩn Dữ liệu & Pipeline (Data Pipeline)

### 5.1 Danh sách 30 Cổ phiếu VN30 Chính thức
Danh mục 30 mã cổ phiếu đại diện được chuẩn hóa:
```
ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG, 
MBB, MSN, MWG, PLX, POW, SAB, SHB, SSB, SSI, STB, 
TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE
```
- **Chỉ số tham chiếu:** `VN30-Index` (chỉ số cơ sở chính) và `VN-Index` (chỉ số thị trường chung).
- **Giai đoạn thu thập:** Từ `2018-01-01` đến `2025-12-31`.
- **Tổng số phiên giao dịch chuẩn:** 1,996 phiên theo lịch giao dịch của `VN30.csv`.

### 5.2 Xử lý Gián đoạn Giao dịch & Chuyển sàn (Trading Halts)
- **Cơ chế chuyển sàn UPCoM $\to$ HOSE:** Một số cổ phiếu lớn có thời gian tạm ngừng giao dịch 6–9 phiên làm thủ tục niêm yết (POW: 9 ngày, BCM: 7 ngày, VIB: 7 ngày, GVR: 6 ngày). Do đó, ngưỡng cho phép forward-fill hợp lệ được thiết lập là:
  $$\text{MAX\_ALLOWED\_GAP\_DAYS} = 10 \text{ phiên}$$
- **Cổ phiếu niêm yết muộn:** Cổ phiếu niêm yết sau thời điểm bắt đầu (ví dụ: `SSB` niêm yết tháng 03/2021, `TCB` tháng 06/2018) được gắn mặt nạ boolean `is_listed = False` trước ngày chào sàn chính thức. Không được forward-fill dữ liệu trước ngày niêm yết.

### 5.3 Bộ 28+ Chỉ báo Kỹ thuật & Đặc trưng Rủi ro (Feature Matrix)
Ma trận đặc trưng được chuẩn hóa Z-score qua rolling window tránh rò rỉ dữ liệu (data leakage):
1. **Lợi suất & Đà giá (Momentum):** `ret_1d`, `ret_5d`, `ret_20d`, `log_ret_1d`.
2. **Xu hướng & Dao động (Trend & Oscillators):** `rsi_14`, `macd`, `macd_signal`, `macd_diff`, `sma_20_ratio`, `sma_50_ratio`.
3. **Độ biến động & Rủi ro (Volatility & Risk):** `rolling_vol_5d`, `rolling_vol_20d`, `rolling_vol_60d`, `bb_high`, `bb_low`, `bb_width`, `atr_14_norm`.
4. **Rủi ro đuôi (Tail Risk):** `rolling_var_95` (20d), `rolling_cvar_95` (20d), `rolling_max_dd_60d`.
5. **Thanh khoản & Khối lượng (Liquidity):** `volume_sma_ratio_20d`, `turnover_ratio_norm`.

### 5.4 Phân chia 4 Cửa sổ Trượt Walk-Forward (Rolling Windows)

| Cửa sổ | Tập Huấn luyện (Train) | Tập Xác thực (Validation) | Tập Kiểm thử (Test - Out-of-Sample) | Đặc trưng Chế độ Thị trường Tập Test |
|:---|:---|:---|:---|:---|
| **Window 1** | 2018–2020 (3 năm) | 2021 (1 năm) | **2022** (1 năm) | **Bear Market cực đoan:** VN-Index sập $-42.8\%$, khủng hoảng trái phiếu doanh nghiệp và thanh khoản. |
| **Window 2** | 2018–2021 (4 năm) | 2022 (1 năm) | **2023** (1 năm) | **Recovery & Tích lũy:** Thị trường tạo đáy, dòng tiền phục hồi phân hóa, biến động trung bình. |
| **Window 3** | 2018–2022 (5 năm) | 2023 (1 năm) | **2024** (1 năm) | **Sideway-Up / Phân hóa:** VN-Index tiệm cận vùng kháng cự 1.300 điểm, nhóm ngân hàng dẫn dắt. |
| **Window 4** | 2018–2023 (6 năm) | 2024 (1 năm) | **2025** (1 năm) | **Out-of-sample forward test:** Đánh giá độ bền vững của chính sách trong điều kiện hiện tại. |

---

## 6. Môi trường Gymnasium & Công thức Toán học

### 6.1 Không gian Trạng thái (Observation Space)
Tại mỗi bước thời gian $t$, quan sát $s_t$ bao gồm:
$$s_t = \left[ X_t, w_{t-1}, \text{cash\_ratio}_{t-1}, \text{DD}_t, \tau_{\text{pending}} \right]$$
- $X_t \in \mathbb{R}^{30 \times F}$: Ma trận đặc trưng chuẩn hóa của 30 cổ phiếu ($F \ge 28$).
- $w_{t-1} \in \Delta^{30}$: Tỷ trọng nắm giữ cổ phiếu thực tế phiên trước.
- $\text{cash\_ratio}_{t-1} = w_{t-1, \text{cash}} \in [0, 1]$: Tỷ lệ tiền mặt hiện có.
- $\text{DD}_t = \frac{\max_{\tau \le t} V_\tau - V_t}{\max_{\tau \le t} V_\tau} \in [0, 1]$: Mức sụt giảm tài sản từ đỉnh lịch sử đến thời điểm $t$.
- $\tau_{\text{pending}}$: Trạng thái các lô cổ phiếu đang chờ khớp T+1 và T+2.

### 6.2 Không gian Hành động (Action Space)
Hành động thô của mô hình là vector logit $a_t \in \mathbb{R}^{31}$ (gồm 30 cổ phiếu + 1 tiền mặt).  
Tỷ trọng mục tiêu $w_t$ được ánh xạ lên Simplex $\Delta^{30}$ thông qua hàm **Softmax có nhiệt độ (Temperature-scaled Softmax)**:
$$w_{t,i} = \frac{\exp(a_{t,i} / \tau)}{\sum_{j=0}^{30} \exp(a_{t,j} / \tau)}, \quad \sum_{i=0}^{30} w_{t,i} = 1.0, \quad w_{t,i} \ge 0$$

### 6.3 Kế toán Danh mục & Chi phí Tái cơ cấu (Portfolio Ledger)
Tỷ suất sinh lời gộp của danh mục tại bước $t$:
$$r_t^{\text{gross}} = \sum_{i=1}^{30} w_{t,i} \cdot r_{t,i} + w_{t,0} \cdot r_{\text{cash}}$$
Chi phí giao dịch $c_t$ (gồm phí mua $0.15\%$, phí bán $0.15\%$, thuế bán $0.1\%$):
$$c_t = \sum_{i=1}^{30} \left( 0.0015 \cdot \max(0, w_{t,i} - w_{t-1,i}') + (0.0015 + 0.0010) \cdot \max(0, w_{t-1,i}' - w_{t,i}) \right)$$
Lợi nhuận ròng danh mục:
$$r_t^{\text{net}} = r_t^{\text{gross}} - c_t$$
Giá trị tài sản ròng (NAV):
$$V_t = V_{t-1} \cdot (1 + r_t^{\text{net}})$$

### 6.4 Bốn Biến thể Hàm Thưởng Bóc tách (Reward Schemes)

#### Biến thể $A_0$ — Baseline Tối đa Lợi nhuận Thuần (Risk-Neutral)
$$R_t^{(A_0)} = r_t^{\text{net}}$$

#### Biến thể $A_1$ — Phạt Phương sai Biến động Tổng thể (Volatility Penalty)
$$R_t^{(A_1)} = r_t^{\text{net}} - \lambda_{\text{vol}} \cdot \sigma_t^2$$
*với $\sigma_t^2$ là phương sai lợi suất tính trên cửa sổ trượt 20 ngày gần nhất.*

#### Biến thể $A_2$ — Phạt Độ lệch Âm Sortino & Rào cản Sụt vốn (Downside Semi-Variance & Drawdown Barrier)
$$R_t^{(A_2)} = r_t^{\text{net}} - \lambda_{\text{down}} \cdot \max(0, m - r_t^{\text{net}})^2 - \lambda_{\text{dd}} \cdot \max(0, \text{DD}_t - d^*)$$
*trong đó:*
- $m$: Mức lợi suất mục tiêu tối thiểu (mặc định $m = 0.0$).
- $d^*$: Ngưỡng chịu đựng sụt giảm vốn chấp nhận được (mặc định $d^* = 0.15 \equiv 15\%$).
- Khi $\text{DD}_t > d^*$, hàm phạt kích hoạt lũy tiến, thúc đẩy tác tử chuyển tài sản về tiền mặt ($w_{\text{cash}} \to 1.0$).

#### Biến thể $A_3$ — Hữu dụng Thuyết Triển vọng & Phạt Rủi ro Đuôi CVaR (Prospect Theory & CVaR Barrier)
$$R_t^{(A_3)} = U(r_t^{\text{net}}) - \lambda_{\text{cvar}} \cdot \max(0, \text{CVaR}_{95}(t) - \text{CVaR}_{\text{target}}) - \lambda_{\text{dd}} \cdot \max(0, \text{DD}_t - d^*)$$
*Hàm hữu dụng giá trị Tversky-Kahneman ($U$):*
$$U(x) = \begin{cases} 
x^\alpha & \text{khi } x \ge 0 \\
-\lambda_{\text{loss}} \cdot |x|^\beta & \text{khi } x < 0 
\end{cases}$$
*với $\alpha = \beta = 0.88$, hệ số sợ lỗ $\lambda_{\text{loss}} = 2.25$. Tác tử phản ứng mạnh gấp 2.25 lần trước mỗi đơn vị tổn thất so với cùng một đơn vị lợi nhuận.*

---

## 7. Bộ Chiến lược Đối chuẩn & Chỉ số Định lượng

### 7.1 Năm Chiến lược Đối chuẩn Bắt buộc (5 Baselines)
1. **Buy & Hold VN30-Index:** Mua và nắm giữ chỉ số thị trường VN30 toàn thời gian.
2. **Equal Weight (1/N):** Tái cân bằng định kỳ phân bổ tỷ trọng bằng nhau ($w_i = 1/30$) cho 30 cổ phiếu, $w_{\text{cash}} = 0$.
3. **100% Cash Benchmark:** Giữ toàn bộ $100\%$ tiền mặt (bảo toàn vốn danh nghĩa, sinh lời $0\%$).
4. **Static 60/40 Stock-Cash:** Duy trì tỷ lệ $60\%$ cổ phiếu (phân bổ đều) và $40\%$ tiền mặt.
5. **Equivalent Equity Benchmark (Kiểm định $H_3$):** Danh mục thụ động mô phỏng tỷ lệ cổ phiếu bình quân chính xác bằng tỷ lệ giải ngân cổ phiếu của tác tử trong cùng phiên, kiểm chứng xem lợi thế đến từ kỹ năng chọn mã/timing hay do nắm giữ tiền mặt.

### 7.2 Bộ 11 Chỉ số Định lượng Chuẩn (Performance Metrics)

| Chỉ số | Ký hiệu / Công thức | Ý nghĩa định lượng | Mục tiêu tối thiểu |
|:---|:---|:---|:---|
| **CAGR** | $(V_T / V_0)^{252/T} - 1$ | Tỷ suất sinh lời kép hàng năm | $> \text{CAGR}_{\text{VN30}}$ |
| **Annualized Volatility** | $\sigma_{\text{daily}} \times \sqrt{252}$ | Độ biến động hàng năm của danh mục | $< 20\%$ |
| **Sharpe Ratio** | $\frac{\mathbb{E}[r - r_f]}{\sigma} \times \sqrt{252}$ | Lợi suất vượt trội trên mỗi đơn vị rủi ro tổng thể | $\ge 1.2$ |
| **Sortino Ratio** | $\frac{\mathbb{E}[r - r_f]}{\sigma_{\text{down}}} \times \sqrt{252}$ | Lợi suất vượt trội trên mỗi đơn vị rủi ro sụt giảm | $\ge 1.8$ |
| **Max Drawdown (MDD)** | $\max_{t} \left( \frac{\max_{\tau \le t} V_\tau - V_t}{\max_{\tau \le t} V_\tau} \right)$ | Mức sụt giảm vốn lớn nhất từ đỉnh lịch sử | $\le 15\%$ (năm 2022) |
| **Calmar Ratio** | $\text{CAGR} / \text{MDD}$ | Tỷ số giữa tăng trưởng kép và mức sụt giảm tối đa | $\ge 1.5$ |
| **Historical VaR (95%)** | $\text{Percentile}(r, 5\%)$ | Mức tổn thất xấu nhất tại ngưỡng tin cậy $95\%$ | Kiểm soát rủi ro |
| **Historical CVaR (95%)** | $\mathbb{E}[r \mid r \le \text{VaR}_{95}]$ | Kỳ vọng tổn thất khi rơi vào đuôi phân phối $5\%$ | Cắt giảm tổn thất đuôi |
| **Turnover Rate** | $\frac{1}{2} \sum \|w_t - w_{t-1}'\|$ | Tỷ lệ đảo danh mục trung bình mỗi phiên | $< 5\%$/phiên |
| **Chi phí ma sát/năm** | $\sum c_t \times (252 / T)$ | Tỷ lệ thuế và phí bào mòn tài sản trên năm | $\le 2.0\%$/năm |
| **Win Rate** | $N(r_t > 0) / N_{\text{total}}$ | Tỷ lệ số ngày giao dịch có lợi nhuận dương | $> 52\%$ |

---

## 8. Cấu trúc Thư mục Chuẩn (Canonical Architecture)

Cấu trúc thư mục chuẩn mực được tổ chức đồng bộ theo 6 tuần triển khai và 6 mục tiêu nghiên cứu:

```text
Risk-awareness RL Trading bot/
├── 01_Research_Proposal/              # [O6] Hồ sơ học thuật & Đề xuất nghiên cứu
│   ├── INT3011E-2_Risk-aware-RL.pdf   # Báo cáo giữa kỳ chính thức (PDF)
│   ├── full_report.txt                # Bản thảo nội dung toàn văn báo cáo
│   ├── project_charter.md             # Điều lệ dự án, câu hỏi RQ1-RQ3, giả thuyết H1-H3
│   └── literature_review.md           # Khảo sát 5 bài báo, 5 sáng chế, 5 sản phẩm
│
├── 02_Data_Pipeline/                  # [O1] Thu thập & Tiền xử lý dữ liệu VN30 (Tuần 1 & 2)
│   ├── collectors/                    # Các module thu thập dữ liệu qua vnstock / API
│   │   ├── __init__.py                # Khởi tạo package collectors
│   │   ├── fetch_vn30_prices.py       # Tải nến ngày OHLCV 30 mã VN30 (2018–2025)
│   │   ├── fetch_corporate_events.py  # Tải sự kiện cổ tức, chia tách cổ phiếu
│   │   └── fetch_benchmarks.py        # Tải chỉ số VN30-Index và VN-Index
│   ├── processors/                    # Các module làm sạch, căn chỉnh và trích xuất đặc trưng
│   │   ├── __init__.py                # Khởi tạo package processors
│   │   ├── clean_alignment.py         # Căn chỉnh lịch HOSE, xử lý khoảng trống niêm yết (gap <= 10)
│   │   └── build_features.py          # Tính toán ma trận 28+ chỉ báo kỹ thuật & rủi ro đuôi
│   ├── 01_vn30_data_exploration.ipynb # Notebook trực quan hóa phân phối & thống kê dữ liệu
│   ├── generate_exploration_charts.py # Script xuất 6 biểu đồ khảo sát chất lượng cao
│   ├── data_collector.py              # Collector phụ (Yahoo Finance phục vụ đối chuẩn crypto/US)
│   ├── run_pipeline.py                # Entrypoint: Chạy toàn bộ pipeline dữ liệu 1-click
│   └── __init__.py                    # Khởi tạo package Data Pipeline
│
├── 03_Model_Development/              # [O2, O3] Môi trường Gymnasium & Huấn luyện PPO (Tuần 3 & 4)
│   ├── env/                           # Môi trường mô phỏng giao dịch thực tế sàn HOSE
│   │   ├── __init__.py                # Khởi tạo package env
│   │   ├── vn30_portfolio_env.py      # Môi trường Gymnasium đa tài sản Simplex [N+1]
│   │   ├── portfolio_ledger.py        # Kế toán tiền mặt, cổ phiếu khả dụng, chu kỳ T+2
│   │   ├── execution_engine.py        # Mô phỏng khớp lệnh 3 cấp độ: M0, M1, M2
│   │   ├── reward_schemes.py          # Hiện thực 4 hàm thưởng nhận thức rủi ro A0, A1, A2, A3
│   │   └── trading_env.py             # Môi trường đơn tài sản (dùng tham chiếu)
│   ├── agents/                        # Kiến trúc mạng nơ-ron Actor-Critic
│   │   ├── __init__.py                # Khởi tạo package agents
│   │   ├── ppo_agent.py               # Thuật toán chính: PPO Actor-Critic cho simplex
│   │   └── sac_agent.py               # Thuật toán phụ: SAC cho không gian liên tục
│   ├── configs/                       # Quản lý siêu tham số tập trung (YAML)
│   │   ├── __init__.py                # Khởi tạo package configs
│   │   └── default_config.yaml        # Cấu hình phí HOSE, thuế, ngưỡng phạt m, d*, window
│   ├── train.py                       # Huấn luyện mô hình đơn lẻ (1 config, 1 window, 1 seed)
│   └── run_experiments.py             # Script tự động batch 80 runs (4 configs x 4 windows x 5 seeds)
│
├── 04_Backtesting_Evaluation/         # [O4] Đánh giá Ngoài mẫu & Kiểm định Giả thuyết (Tuần 5)
│   ├── baselines/                     # Bộ 5 chiến lược đối chuẩn bắt buộc
│   │   ├── __init__.py                # Khởi tạo package baselines
│   │   ├── buy_and_hold_vn30.py       # Đối chuẩn thị trường VN30-Index
│   │   ├── equal_weight.py            # Đối chuẩn phân bổ đều (1/N)
│   │   ├── cash_benchmark.py          # Đối chuẩn 100% Tiền mặt
│   │   ├── static_stock_cash.py       # Tỷ lệ cố định 60/40 cổ phiếu/tiền mặt
│   │   └── equiv_equity_benchmark.py  # Đối chuẩn tỷ trọng cổ phiếu bình quân tương đương (H3)
│   ├── metrics.py                     # Động cơ tính 11 chỉ số định lượng (Sharpe, Sortino, MDD, CVaR)
│   ├── backtest_engine.py             # Động cơ backtest ngoài mẫu trên 4 tập Test
│   └── hypothesis_testing.py          # Phân tích thống kê & kiểm định giả thuyết H1, H2, H3
│
├── 05_Deployment_Demo/                # [O5] Web Dashboard Tra cứu & Trực quan hóa (Tuần 6)
│   ├── app.py                         # Ứng dụng Streamlit tương tác
│   ├── components/                    # Biểu đồ Plotly: NAV, Underwater Drawdown, Tỷ trọng
│   └── requirements.txt               # Gói phụ thuộc riêng cho dashboard
│
├── 06_Docs_and_Reports/               # [O6] Tài liệu Báo cáo, Slide & Thuyết trình
│   ├── bao_cao_giua_ky_de_tai_Tu.md   # Báo cáo giữa kỳ markdown chi tiết
│   ├── ke_hoach_phat_trien_chi_tiet.md# Kế hoạch phát triển 6 tuần chi tiết
│   ├── kich_ban_trao_doi_nhom.md      # Kịch bản họp nhóm & bảo vệ đề tài
│   ├── diagnostic_log.md              # Nhật ký sự cố, nguyên nhân gốc & giải pháp phòng ngừa
│   ├── beamer_presentation/           # Slide thuyết trình LaTeX Beamer
│   │   ├── main.tex                   # Mã nguồn chính Beamer
│   │   ├── preamble.tex               # Cấu hình gói và màu sắc chuẩn học thuật
│   │   └── figures/                   # Biểu đồ nhúng trong Beamer
│   ├── presentation/                  # Slide thuyết trình động & Video Remotion (TypeScript/React)
│   │   ├── src/                       # Các scenes Remotion và video dọc TikTok 9:16
│   │   └── package.json               # Cấu hình dự án Remotion
│   ├── learning/                      # Tài liệu học tập & giải thích trực quan đề tài
│   │   └── hoc_du_an_rl_vn30.html     # Cẩm nang tương tác giải thích toàn diện dự án
│   ├── figures/                       # Hình ảnh và biểu đồ phục vụ báo cáo
│   ├── final_report/                  # Thư mục chứa báo cáo cuối khóa luận
│   └── README.md                      # Hướng dẫn tài liệu báo cáo
│
├── data/                              # Dữ liệu tập trung (Được gitignore, chỉ commit .gitkeep)
│   ├── raw/                           # Dữ liệu thô tải từ vnstock
│   │   ├── prices/                    # 30 file CSV OHLCV theo ngày
│   │   ├── corporate_actions/         # Sự kiện cổ tức, chia tách cổ phiếu
│   │   └── benchmarks/                # Dữ liệu chỉ số VN30.csv và VNINDEX.csv
│   ├── processed/                     # Dữ liệu sạch đã căn chỉnh
│   │   ├── aligned_prices.parquet     # Bảng giá đã làm sạch và đồng bộ lịch giao dịch
│   │   ├── feature_matrix.parquet     # Ma trận 28+ đặc trưng kỹ thuật & rủi ro đuôi
│   │   ├── window_1/                  # Dữ liệu chia sẵn cho Cửa sổ 1 (Test 2022)
│   │   ├── window_2/                  # Dữ liệu chia sẵn cho Cửa sổ 2 (Test 2023)
│   │   ├── window_3/                  # Dữ liệu chia sẵn cho Cửa sổ 3 (Test 2024)
│   │   └── window_4/                  # Dữ liệu chia sẵn cho Cửa sổ 4 (Test 2025)
│   └── archive/                       # Dữ liệu cũ lưu trữ
│
├── outputs/                           # Kết quả thực nghiệm (Được gitignore checkpoints và logs)
│   ├── checkpoints/                   # Trọng số mô hình PPO/SAC của các lượt huấn luyện
│   ├── logs/                          # Nhật ký TensorBoard
│   ├── backtest_results/              # Bảng số liệu NAV, danh mục giao dịch xuất ra CSV
│   └── figures/                       # 6 biểu đồ phân tích thống kê xuất ra từ pipeline
│
├── tests/                             # Bộ kiểm thử tự động (Unit Tests chuẩn Pytest)
│   ├── test_clean_alignment.py        # Kiểm thử căn chỉnh lịch giao dịch & xử lý gián đoạn
│   ├── test_build_features.py         # Kiểm thử tính toán 28+ chỉ báo kỹ thuật & rủi ro
│   └── __init__.py                    # Khởi tạo package tests
│
├── .gitignore                         # Loại bỏ cache, parquet dữ liệu lớn, model checkpoints
├── pytest.ini                         # Cấu hình bộ chạy kiểm thử Pytest
├── requirements.txt                   # Danh mục thư viện Python bắt buộc
├── requirements-dev.txt               # Thư viện kiểm thử & công cụ phát triển (pytest, mypy)
├── DIRECTORY_STRUCTURE.md             # Bản đồ kiến trúc thư mục
├── AGENTS.md                          # Hướng dẫn vận hành dành cho AI Agents
├── SOURCE_OF_TRUTH.md                 # TÀI LIỆU NÀY (Single Source of Truth)
└── README.md                          # Tài liệu giới thiệu tổng quan dự án trên GitHub
```

---

## 9. Công nghệ & Quy chuẩn Mã nguồn

### 9.1 Bảng Phiên bản Thư viện Cốt lõi

| Lớp công nghệ | Thư viện | Phiên bản | Mục đích sử dụng |
|:---|:---|:---|:---|
| **Ngôn ngữ** | Python | `3.12.x` | Ngôn ngữ phát triển tiêu chuẩn |
| **Deep Learning** | PyTorch | `≥ 2.0.0` | Xây dựng mạng nơ-ron Actor-Critic |
| **RL Framework** | Stable-Baselines3 | `≥ 2.3.0` | Thuật toán PPO, SAC chuẩn hóa |
| **Môi trường** | Gymnasium | `≥ 0.29.1` | API chuẩn hóa môi trường mô phỏng |
| **Dữ liệu tài chính** | vnstock | `≥ 4.0.0` | Tải dữ liệu chứng khoán HOSE chính thức |
| **Đối chuẩn thị trường** | yfinance | `≥ 0.2.38` | Tải dữ liệu quốc tế / chỉ số so sánh |
| **Xử lý số liệu** | pandas, numpy | `pandas >= 2.0`, `numpy >= 1.24, < 2.0` | Thao tác ma trận giá và vector đặc trưng |
| **Chỉ báo kỹ thuật** | ta | `≥ 0.11.0` | Thư viện Technical Analysis tính RSI, MACD, BB |
| **Lưu trữ hiệu năng cao**| pyarrow | `≥ 14.0.0, < 19.0.0` | Định dạng Parquet tối ưu nén dữ liệu |
| **Trực quan hóa** | matplotlib, seaborn, plotly | Xem `requirements.txt` | Xuất biểu đồ tĩnh và tương tác |
| **Web Dashboard** | Streamlit | `≥ 1.30.0` | Giao diện tra cứu và trình diễn trực quan |
| **Kiểm thử & Type** | pytest, pytest-cov, mypy | Mới nhất | Đảm bảo chất lượng mã nguồn nghiêm ngặt |

### 9.2 Quy chuẩn Mã nguồn (Code Conventions)
1. **Kiểu dữ liệu tĩnh (Strict Typing):** Tất cả các hàm và phương thức công khai bắt buộc có đầy đủ type annotations. Phải vượt qua lệnh kiểm tra `mypy --strict`.
2. **Không dùng số ma thuật (No Magic Numbers):** Tất cả các ngưỡng (thuế $0.001$, phí $0.0015$, biên độ $0.07$, ngưỡng sụt giảm $d^* = 0.15$) phải được định nghĩa bằng hằng số viết hoa hoặc tải từ file cấu hình `configs/default_config.yaml`.
3. **Ghi nhật ký chuẩn (Standard Logging):** Sử dụng thư viện `logging` của Python thay cho lệnh `print` thông thường. Logger name luôn là `__name__`.
4. **Bất biến đối với dữ liệu thô (Immutability):** Toàn bộ tệp tin trong `data/raw/` là chỉ đọc (read-only). Các bước tiền xử lý chỉ được ghi ra `data/processed/`.

---

## 10. Quy trình Kiểm định & Tiêu chí Hoàn thành (DoD)

### 10.1 Lệnh Xác minh Bắt buộc (Verification Pipeline)
Trước khi hợp nhất bất kỳ tính năng hoặc commit mã nguồn nào, kỹ sư (hoặc AI agent) **bắt buộc chạy thành công toàn bộ chuỗi lệnh sau theo thứ tự**:

```bash
# 1. Kích hoạt môi trường ảo
source .venv/bin/activate

# 2. Kiểm tra kiểu tĩnh (bắt buộc 0 lỗi)
mypy --strict --ignore-missing-imports 02_Data_Pipeline/ 04_Backtesting_Evaluation/metrics.py

# 3. Chạy toàn bộ Unit Tests với báo cáo độ bao phủ
pytest tests/ -v --cov=02_Data_Pipeline --cov=04_Backtesting_Evaluation --cov-report=term-missing

# 4. Kiểm tra tính hợp lệ của môi trường Gymnasium (khi chỉnh sửa env)
python -c "
import sys, importlib
sys.path.insert(0, '03_Model_Development')
from gymnasium.utils.env_checker import check_env
mod = importlib.import_module('env.trading_env')
env = mod.StockTradingEnv()
check_env(env)
print('✅ Env check passed!')
"

# 5. Smoke test toàn bộ pipeline dữ liệu (không gọi mạng)
python 02_Data_Pipeline/run_pipeline.py --dry-run
```

### 10.2 Tiêu chí Hoàn thành Từng Hạng mục (Definition of Done - DoD)
- **Data Collector / Processor:** Phải có kiểm thử schema cột, xử lý thiếu dữ liệu hợp lệ (gap $\le 10$ ngày), có cờ `skip_existing=True`, cơ chế hạn chế tần suất gọi (rate limit guard) và vượt qua 100% unit tests.
- **Gymnasium Environment:** Phải vượt qua `gymnasium.utils.env_checker.check_env` không cảnh báo; trả về đủ các trường trong `info` (`portfolio_value`, `daily_return`, `cvar_95`, `turnover`); bảo toàn tính đơn ánh và tất định theo random seed.
- **Model Training:** File cấu hình YAML đầy đủ tham số; log TensorBoard lưu trữ tự động; hoàn thành số bước chỉ định không lỗi; checkpoint được lưu vào `outputs/checkpoints/`.
- **Backtesting & Evaluation:** Kiểm thử Out-of-sample đối chiếu đủ 5 Baselines; xuất bảng chỉ số định lượng 11 metrics; kiểm định thống kê $p$-value của các giả thuyết $H_1, H_2, H_3$.

### 10.3 Chu trình Xử lý Sự cố (Diagnostic Loop)
Mỗi khi phát sinh lỗi trong quá trình thực thi, không được thử lại một cách mù quáng (blind retry) mà phải tuân thủ mẫu ghi nhận tại [diagnostic_log.md](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/06_Docs_and_Reports/diagnostic_log.md):
- Xác định chính xác lớp phát sinh lỗi (Collector, Processor, Environment, Agent, Backtest).
- Phân tích nguyên nhân gốc rễ (Root Cause).
- Áp dụng biện pháp khắc phục triệt để.
- Bổ sung Unit Test mới để đảm bảo không tái diễn (Prevention).

---

## 11. Hướng dẫn Khởi chạy Nhanh (Quick Start)

### Bước 1: Thiết lập Môi trường
```bash
# Clone kho lưu trữ
git clone https://github.com/tDn412/Risk-awareness-RL-Trading-bot.git
cd "Risk-awareness RL Trading bot"

# Tạo và kích hoạt môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt toàn bộ thư viện phụ thuộc
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Bước 2: Chạy Toàn bộ Pipeline Dữ liệu VN30
```bash
# Chạy pipeline dữ liệu tự động (tải dữ liệu, căn chỉnh lịch, tạo 28+ features)
python 02_Data_Pipeline/run_pipeline.py

# Xuất 6 biểu đồ khảo sát thống kê và tạo Jupyter Notebook
python 02_Data_Pipeline/generate_exploration_charts.py
```

### Bước 3: Chạy Kiểm thử Chất lượng Mã nguồn
```bash
# Chạy toàn bộ unit tests
pytest tests/ -v

# Kiểm tra tĩnh mypy
mypy --strict --ignore-missing-imports 02_Data_Pipeline/ 04_Backtesting_Evaluation/metrics.py
```

### Bước 4: Huấn luyện Tác tử RL Nhận thức Rủi ro
```bash
# Huấn luyện mô hình PPO với hàm thưởng nhận thức rủi ro
python 03_Model_Development/train.py --algo PPO --reward-mode prospect_theory --timesteps 50000
```

### Bước 5: Chạy Backtest Ngoài Mẫu & Trực quan hóa Dashboard
```bash
# Đánh giá hiệu năng và tính toán 11 chỉ số định lượng
python 04_Backtesting_Evaluation/backtest_engine.py

# Khởi chạy Dashboard tương tác Streamlit
streamlit run 05_Deployment_Demo/app.py
```

---

*Tài liệu được phê duyệt và duy trì bởi Nhóm nghiên cứu INT3011E-2. Mọi cập nhật cấu trúc hoặc quy tắc kỹ thuật đều phải thông qua việc chỉnh sửa tệp tin này.*
