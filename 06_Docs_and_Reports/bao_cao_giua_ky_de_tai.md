# BÁO CÁO GIỮA KỲ: XÂY DỰNG BÀI TOÁN NGHIÊN CỨU ĐỀ TÀI
## Đề tài: Risk-Aware Reinforcement Learning for VN30 Portfolio Allocation

- **Học phần:** INT3011E
- **Thực hiện:** Chúng tôi
- **Kho lưu trữ mã nguồn:** `Risk-awareness RL Trading bot`

---

## 1. ĐẶT VẤN ĐỀ (PROBLEM STATEMENT)

### 1.1 Mô tả vấn đề
Phân bổ danh mục đầu tư đa tài sản cho rổ cổ phiếu VN30 là bài toán ra quyết định tuần tự. Không gian hành động của bài toán là một simplex tỷ trọng liên tục:
$$\sum_{i=1}^{N} w_{t,i} + w_{t,\text{cash}} = 1, \quad w_{t,i} \ge 0$$

Các mô hình Reinforcement Learning (RL) tiêu chuẩn thường tối ưu hóa tổng lợi nhuận tích lũy:
$$\max_{\pi} \mathbb{E}\left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

Cách tiếp cận này bộc lộ các vấn đề kỹ thuật nghiêm trọng:
- **Tác tử mù rủi ro (Risk-Blindness):** Tác tử tập trung dồn tỷ trọng vào các cổ phiếu beta cao nhằm tìm kiếm phần thưởng lớn. Danh mục phải gánh chịu độ biến động và mức sụt giảm vốn cực đoan.
- **Ràng buộc thị trường chứng khoán Việt Nam:** Sàn HOSE áp dụng biên độ dao động giá $\pm 7\%$/ngày. Thị trường cơ sở chỉ cho phép vị thế mua nắm giữ (long-only), không cho phép bán khống (short-selling).
- **Trôi dạt chế độ thị trường (Regime Shift):** Tác tử học tốt trong pha tăng trưởng (Bull market). Khi thị trường đảo chiều sang pha suy thoái (Bear market), tác tử tiếp tục giải ngân bắt đáy, gây sụt giảm tài khoản sâu.
- **Bào mòn vốn do chi phí ma sát:** Mỗi giao dịch bán tại Việt Nam chịu thuế 0.1% giá trị giao dịch. Phí môi giới dao động 0.1% - 0.15%. Tác tử đảo danh mục liên tục sẽ bị chi phí giao dịch triệt tiêu toàn bộ lợi nhuận.

### 1.2 Đối tượng gặp vấn đề (Target Stakeholders)
1. **Nhà đầu tư cá nhân giao dịch rổ VN30:**
   - Thiếu công cụ tái cơ cấu tỷ trọng tự động.
   - Thường xuyên giữ vị thế thụ động khi thị trường sụt giảm sâu.
   - Không có kỷ luật quản trị drawdown danh mục.
2. **Quỹ đầu tư nội địa và nhóm quản lý tài sản vừa và nhỏ:**
   - Cần thuật toán phân bổ tỷ trọng thích ứng với từng chu kỳ vĩ mô.
   - Bắt buộc kiểm soát các chỉ số rủi ro danh mục (Drawdown, Sortino, CVaR) theo cam kết với khách hàng.
   - Chi phí thuê đội ngũ kỹ sư định lượng (Quant) vận hành hệ thống riêng rất cao.
3. **Kỹ sư FinTech và nhóm nghiên cứu học máy tài chính:**
   - Thiếu môi trường mô phỏng chuẩn Gymnasium tích hợp đúng cơ chế giao dịch Việt Nam.
   - Thiếu framework mã nguồn mở tích hợp cơ chế hàm thưởng nhận thức rủi ro cho bài toán phân bổ danh mục.

### 1.3 Các ví dụ cụ thể trong thực tế
- **Đợt sụt giảm của thị trường chứng khoán Việt Nam năm 2022:**
  - Chỉ số VN-Index rơi từ 1.528 điểm (tháng 04/2022) về 873 điểm (tháng 11/2022), giảm 42.8%.
  - Nhiều cổ phiếu trong rổ VN30 giảm hơn 60% (HPG giảm từ 44.000 về 12.000 VNĐ; SSI giảm từ 45.000 về 13.000 VNĐ).
  - Các thuật toán phân bổ theo đà tăng trưởng (momentum) hoặc RL tối đa hóa lợi nhuận đã duy trì tỷ trọng cổ phiếu cao, khiến danh mục bốc hơi hơn phân nửa giá trị.
- **Cú sập do đại dịch COVID-19 (Tháng 03/2020):**
  - VN-Index giảm 33% trong 4 tuần.
  - Hàng loạt mã VN30 giảm sàn liên tục chạm biên độ -7%.
  - Các mô hình phân bổ cổ điển (Markowitz Mean-Variance) sử dụng ma trận hiệp phương sai lịch sử bị trễ pha, không kịp thời chuyển dịch tài sản về tiền mặt ($w_{\text{cash}} \to 1.0$).
- **Đợt điều chỉnh quý 3/2023:**
  - VN-Index giảm nhanh từ 1.250 điểm về 1.020 điểm trong vòng 8 tuần.
  - Các bot giao dịch bán lẻ dựa trên chỉ báo kỹ thuật rời rạc (SMA, RSI) bị nhiễu tín hiệu (whipsaw).
  - Bot thực hiện tái cơ cấu danh mục quá nhiều lần, làm thuế phí bào mòn 5% - 8% tổng tài sản.

---

## 2. KHOẢNG TRỐNG NGHIÊN CỨU & CÔNG NGHỆ (GAP ANALYSIS)

### 2.1 Khảo sát 05 Bài báo khoa học (Scientific Papers)

1. **Moody, J., & Saffell, M. (2001).** *"Learning to trade via direct reinforcement."* IEEE Transactions on Neural Networks, 12(4), 875-889.
   - **Giải pháp:** Đề xuất Differential Sharpe Ratio (DSR), cho phép cập nhật trực tuyến đệ quy đạo hàm Sharpe theo từng step.
   - **Điểm yếu:** Sử dụng phương sai tổng thể làm mẫu số, vô tình phạt cả biến động chiều tăng (upside gains); không có cơ chế rào cản kiểm soát drawdown tức thời; chỉ thử nghiệm trên đơn tài sản.
2. **Jiang, Z., Xu, D., & Liang, J. (2017).** *"A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem."* arXiv:1706.08824.
   - **Giải pháp:** Xây dựng mạng nơ-ron tích hợp CNN-RNN (EIIE) kết hợp thuật toán DDPG để xuất vector tỷ trọng phân bổ danh mục.
   - **Điểm yếu:** Hàm mục tiêu chỉ tối đa hóa log return thuần túy (risk-neutral); Max Drawdown vượt quá 60% khi thị trường suy thoái; bỏ qua ma sát trượt giá động.
3. **Wang, Z., Huang, B., Tu, S., Zhang, K., & Xu, L. (2021).** *"DeepTrader: A Deep Reinforcement Learning Approach to Risk-Return Balanced Portfolio Management with Market Conditions Embedding."* AAAI-21.
   - **Giải pháp:** Nhúng điều kiện thị trường vĩ mô bằng đồ thị không-thời gian; dùng negative maximum drawdown làm tín hiệu thưởng để cân bằng rủi ro và lợi nhuận.
   - **Điểm yếu:** Chi phí tính toán cấu trúc đồ thị nhân quả lớn, latency cao; giả định cho phép mở vị thế long/short tự do, không phù hợp thị trường cơ sở chỉ long-only của Việt Nam.
4. **Enkhsaikhan, B., & Jo, O. (2025).** *"Augmented Lagrangian Risk-constrained Reinforcement Learning for Portfolio Optimization."* AAAI-25.
   - **Giải pháp:** Dùng Augmented Lagrangian Multiplier (ALM) đưa ràng buộc cứng Value-at-Risk (VaR) vào quá trình học của thuật toán DDPG.
   - **Điểm yếu:** Đòi hỏi mạng phụ (cost network) ước lượng rủi ro; độ nhạy siêu tham số cao; dễ mất ổn định khi phân phối lợi nhuận có đuôi dày; chưa kiểm chứng trên thị trường mới nổi.
5. **Chow, Y., Tamar, A., Mannor, S., & Pavone, M. (2015).** *"Risk-Sensitive and Robust Decision-Making: A CVaR Constrained MDP Approach."* NeurIPS 2015.
   - **Giải pháp:** Mô hình hóa bài toán quyết định tuần tự dưới dạng Constrained MDP với ràng buộc Conditional Value-at-Risk (CVaR).
   - **Điểm yếu:** Tốc độ hội tụ của thuật toán Primal-Dual Policy Gradient chậm; khó mở rộng quy mô khi không gian trạng thái lên tới 30 mã tài sản; không xét đến chi phí tái cơ cấu danh mục.

---

### 2.2 Khảo sát 05 Sáng chế kỹ thuật (Technical Patents)

1. **US 12,094,005 B2 (BlackRock, Inc.; Cấp ngày 17/09/2024; Inventors: Stephen Boyd, Mykel Kochenderfer, et al.):**
   - **Tên sáng chế:** *"Systems and methods for a user interface and application for simulating construction and optimization of an investment portfolio"*.
   - **Giải pháp:** Hệ thống mô phỏng dòng tiền và tối ưu hóa phân bổ tài sản dựa trên phương pháp Markowitz với giao diện điều chỉnh trục rủi ro động.
   - **Điểm yếu:** Dựa trên mô hình tối ưu hóa lồi tĩnh; thiếu khả năng tự thích ứng liên tục của RL trước các cú sốc thị trường; không hỗ trợ mô hình hóa biên độ giá $\pm 7\%$ của sàn HOSE.
2. **US 10,776,692 B2 (DeepMind Technologies Ltd; Cấp ngày 15/09/2020; Inventors: Timothy Lillicrap et al.):**
   - **Tên sáng chế:** *"Continuous control with deep reinforcement learning"*.
   - **Giải pháp:** Kiến trúc học tăng cường sâu Actor-Critic cho không gian hành động liên tục với Experience Replay và Target Network.
   - **Điểm yếu:** Thiết kế cho bài toán điều khiển vật lý dừng; cơ chế lấy mẫu ngẫu nhiên đồng đều làm mất các mẫu dữ liệu khủng hoảng thanh khoản hiếm gặp; hàm thưởng không có cơ chế quản trị rủi ro đuôi tài chính.
3. **US 2022/0129925 A1 (JPMorgan Chase Bank, N.A.; Công bố ngày 28/04/2022):**
   - **Tên sáng chế:** *"Method and system for simulation and calibration of markets using reinforcement learning"*.
   - **Giải pháp:** Nền tảng mô phỏng thị trường đa tác tử phục vụ phân tách và thực thi khối lệnh lớn (hệ thống LOXM).
   - **Điểm yếu:** Chuyên biệt hóa cho việc khớp lệnh vi mô (execution level); không giải quyết bài toán phân bổ tỷ trọng danh mục tài sản trung hạn (portfolio allocation level); hệ thống đóng và yêu cầu hạ tầng tính toán lớn.
4. **US 9,940,673 B2 (Hudson Bay Capital Management LP; Cấp ngày 10/04/2018; Inventor: Sander Gerber):**
   - **Tên sáng chế:** *"Systems and methods for measuring relationships between investments and other variables"*.
   - **Giải pháp:** Hệ thống tính toán và giám sát cấu trúc tương quan động giữa các tài sản nhằm cảnh báo rủi ro danh mục.
   - **Điểm yếu:** Chỉ phục vụ phân tích dữ liệu thụ động; không có tác tử tự động ra quyết định tái cơ cấu danh mục; không tích hợp phản hồi từ hành vi thị trường.
5. **US 2019/0370649 A1 (Công bố ngày 05/12/2019):**
   - **Tên sáng chế:** *"Reinforcement learning neural network for automated trading"*.
   - **Giải pháp:** Ứng dụng mạng nơ-ron học tăng cường sâu để phát sinh tín hiệu giao dịch mua bán tự động.
   - **Điểm yếu:** Hàm thưởng chỉ dựa trên lợi nhuận thuần túy; không gian hành động rời rạc (chỉ chọn mua/bán đơn lẻ); không hỗ trợ phân bổ tỷ trọng trên simplex đa tài sản; thiếu cơ chế giới hạn drawdown.

---

### 2.3 Khảo sát 05 Sản phẩm thực tế trên thị trường (Commercial & Open-Source Products)

1. **FinRL (AI4Finance Foundation):**
   - **Giải pháp:** Thư viện mã nguồn mở chuẩn hóa môi trường Gymnasium và tích hợp các thuật toán DRL (PPO, A2C, DDPG, SAC) cho thị trường chứng khoán.
   - **Điểm yếu:** Hàm thưởng mặc định chỉ là biến động giá trị tài sản ròng thuần túy; thiếu cơ chế phạt sụt giảm vốn tức thời (drawdown barrier); chưa có môi trường chuẩn hóa cho dữ liệu và quy định giao dịch Việt Nam (HOSE).
2. **Qlib (Microsoft Research):**
   - **Giải pháp:** Nền tảng định lượng AI hỗ trợ trích xuất đặc trưng chuỗi thời gian và backtest danh mục quy mô lớn.
   - **Điểm yếu:** Tập trung chủ yếu vào các mô hình học có giám sát (LightGBM, Alpha158); module học tăng cường còn sơ khai; không hỗ trợ hàm thưởng điều chỉnh rủi ro bất đối xứng theo thuyết triển vọng (Prospect Theory).
3. **QuantConnect (Lean Engine):**
   - **Giải pháp:** Nền tảng thuật toán hỗ trợ backtest và giao dịch live đa tài sản dựa trên dữ liệu tick và thanh nến.
   - **Điểm yếu:** Công cụ quản trị rủi ro chủ yếu là các quy tắc dừng lỗ tĩnh (stop-loss rules) viết bằng mã lệnh cứng, tách rời khỏi quá trình huấn luyện mô hình; phụ thuộc vào hạ tầng đám mây độc quyền.
4. **Composer.trade:**
   - **Giải pháp:** Nền tảng tự động hóa phân bổ danh mục theo các quy tắc logic điều kiện thị trường (ví dụ: chuyển dịch tỷ trọng khi RSI hoặc đường SMA cắt nhau).
   - **Điểm yếu:** Vận hành hoàn toàn bằng quy tắc tĩnh (heuristic rules); có độ trễ lớn khi thị trường biến động nhanh; không có khả năng tự học tối ưu tỷ trọng phi tuyến tính.
5. **Wealthfront / Betterment:**
   - **Giải pháp:** Nền tảng Robo-advisor quản lý danh mục tự động dựa trên Modern Portfolio Theory và tái cân bằng theo ngưỡng lệch (drift-based rebalancing).
   - **Điểm yếu:** Mô hình đầu tư thụ động, chu kỳ điều chỉnh chậm; không có cơ chế phòng vệ chủ động trước các cú sập thị trường ngắn hạn; chỉ hỗ trợ chứng chỉ quỹ ETF quốc tế, không hoạt động trên cổ phiếu đơn lẻ rổ VN30.

---

### 2.4 Bảng tổng hợp đối sánh và Khoảng trống cốt lõi (Core Research Gap)

| Tiêu chí so sánh | Bài báo lý thuyết (Moody, Jiang, Wang) | Sáng chế kỹ thuật (BlackRock, DeepMind) | Sản phẩm thực tế (FinRL, Qlib, Composer) | **Giải pháp chúng tôi đề xuất: Risk-Aware RL** |
| :--- | :--- | :--- | :--- | :--- |
| **Mục tiêu tối ưu** | Tối đa lợi nhuận hoặc Sharpe đối xứng | Tối ưu hóa lồi tĩnh / Khớp lệnh vi mô | Heuristic rules / Lợi nhuận thuần túy | **Hàm thưởng nhận thức rủi ro đa mục tiêu (Sortino, Drawdown Barrier, Prospect Theory)** |
| **Không gian hành động** | Đơn tài sản hoặc tỷ trọng không ràng buộc | Rời rạc hoặc không kiểm soát ma sát | Quy tắc cố định theo bước | **Vector tỷ trọng liên tục trên simplex ($\sum w_i = 1$) qua Softmax Policy** |
| **Kiểm soát sụt giảm vốn** | Không có hoặc ràng buộc lý thuyết | Cắt lỗ cứng bên ngoài mô hình | Phụ thuộc người dùng thiết lập | **Rào cản sụt vốn động (Drawdown Barrier Penalty $\theta_{\text{tol}}$)** |
| **Phí ma sát & Quy chế sàn** | Thường bỏ qua hoặc giả định phí cố định | Không mô phỏng thuế bán nội địa | Phí cố định theo lệnh | **Mô phỏng đầy đủ thuế bán 0.1%, phí giao dịch và biên độ $\pm 7\%$ sàn HOSE** |
| **Lớp tài sản mục tiêu** | US Equities / Crypto | Tài sản thay thế / Sổ lệnh ngoại hối | US Equities / ETFs | **Cổ phiếu rổ VN30 (Thị trường chứng khoán Việt Nam)** |

> **KHOẢNG TRỐNG CỐT LÕI (CORE RESEARCH GAP):**
> Hiện chưa có một framework mã nguồn mở hoàn chỉnh nào giải quyết đồng thời:
> 1. Tích hợp trực tiếp **cơ chế hàm thưởng nhận thức rủi ro đa mục tiêu** (Sortino Downside semi-variance, Drawdown Barrier Penalty và Prospect Theory Utility) vào thuật toán Deep RL (PPO/SAC) để xuất vector phân bổ tỷ trọng liên tục trên simplex cho danh mục cổ phiếu.
> 2. Mô phỏng đúng **các ràng buộc thực tế của thị trường chứng khoán Việt Nam**: danh mục cổ phiếu VN30, ràng buộc long-only ($w_i \ge 0$), biên độ giá $\pm 7\%$, thuế bán 0.1% và phí giao dịch thực tế.
> 3. Kiểm định thực nghiệm đối sánh đa chế độ thị trường (Bull 2020-2021, Bear 2022, Sideway 2023-2024) so với các baseline chuẩn (VN30-Index, Equal Weight 1/N, Markowitz MPT).

---

## 3. XÂY DỰNG CÂU HỎI NGHIÊN CỨU (RESEARCH QUESTIONS)

Chúng tôi xác định 03 câu hỏi nghiên cứu kỹ thuật trọng tâm:

- **Câu hỏi 1 (Về Thiết kế Hàm thưởng Nhận thức Rủi ro trên Simplex):**
  Làm thế nào để thiết kế và tối ưu hóa hàm thưởng nhận thức rủi ro (kết hợp Sortino downside semi-variance, Drawdown Barrier Penalty và Prospect Theory) giúp tác tử RL phân bổ tỷ trọng trên simplex ($\sum w_i = 1, w_i \ge 0$) kiềm chế mức sụt giảm tài sản tối đa (Max Drawdown $\le 15\%$) trên rổ VN30 mà vẫn bảo toàn tỷ suất sinh lời vượt trội so với VN30-Index?

- **Câu hỏi 2 (Về Tính Thích ứng trước Ma sát Thị trường Việt Nam):**
  Thuật toán Policy Gradient liên tục (PPO với Softmax Policy) thích ứng như thế nào trước tác động của thuế bán 0.1% và phí giao dịch sàn HOSE, nhằm triệt tiêu hiện tượng tái cơ cấu danh mục quá mức (portfolio churning) so với chiến lược Buy & Hold?

- **Câu hỏi 3 (Về Tính Tổng quát hóa qua các Chế độ Thị trường):**
  Chính sách phân bổ tỷ trọng học được có duy trì các chỉ số hiệu năng điều chỉnh theo rủi ro (Sharpe Ratio $\ge 1.2$, Sortino Ratio $\ge 1.8$, Calmar Ratio $\ge 1.5$) khi kiểm thử Out-of-Sample qua 3 giai đoạn thị trường đối nghịch của Việt Nam (Tăng trưởng 2020-2021, Suy thoái 2022, Đi ngang 2023-2024) so với danh mục Equal Weight (1/N) và Markowitz MPT hay không?

---

## 4. XÂY DỰNG MỤC TIÊU NGHIÊN CỨU (RESEARCH OBJECTIVES)

### 4.1 Mục tiêu tổng quát (General Objective)
Xây dựng, huấn luyện và kiểm định thực nghiệm một hệ thống phân bổ danh mục đầu tư tự động ứng dụng học tăng cường sâu nhận thức rủi ro (Risk-Aware Deep RL) cho rổ cổ phiếu VN30, tối ưu hóa tỷ suất sinh lời điều chỉnh theo rủi ro và giới hạn mức sụt giảm vốn tối đa dưới 15% trong các pha thị trường suy thoái.

### 4.2 Mục tiêu cụ thể (Specific Objectives theo chuẩn SMART)

1. **Mục tiêu 1: Xây dựng Pipeline Dữ liệu Rổ VN30 và Môi trường Gymnasium (Specific & Measurable)**
   - Xây dựng pipeline tự động tải, làm sạch và đồng bộ chuỗi dữ liệu OHLCV của rổ cổ phiếu VN30 giai đoạn 2018 - 2026.
   - Tính toán ma trận đặc trưng kỹ thuật (Return 1d/5d/20d, RSI 14, MACD Histogram, Bollinger Bands, Normalized ATR, Rolling Volatility).
   - Phát triển môi trường tùy biến `VN30PortfolioEnv` chuẩn **Gymnasium v0.29+**:
     - Không gian hành động liên tục $a_t \in \mathbb{R}^{N+1}$ chuyển đổi qua Softmax thành vector tỷ trọng hợp lệ ($w_i \ge 0, \sum w_i = 1$).
     - Mô phỏng chính xác thuế bán 0.1%, phí giao dịch 0.15% và biên độ giá $\pm 7\%$.
     - Tích hợp 4 chế độ hàm thưởng: `standard`, `volatility_penalty`, `differential_sharpe`, `prospect_theory`.

2. **Mục tiêu 2: Triển khai và Huấn luyện Thuật toán Deep RL (Achievable & Relevant)**
   - Triển khai thuật toán **Proximal Policy Optimization (PPO)** và **Soft Actor-Critic (SAC)** với mạng policy phân bổ tỷ trọng trên PyTorch.
   - Huấn luyện tác tử trên tập In-Sample (2018 - 2021) với tối thiểu 100,000 timesteps.
   - Sử dụng TensorBoard giám sát các hàm mất mát (Policy Loss, Value Loss), entropy và đường cong tích lũy phần thưởng.

3. **Mục tiêu 3: Kiểm định Out-of-Sample và So sánh Đối chuẩn Định lượng (Measurable & Time-bound)**
   - Xây dựng module kiểm định định lượng độc lập ([metrics.py](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/04_Backtesting_Evaluation/metrics.py) và [backtest_engine.py](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/04_Backtesting_Evaluation/backtest_engine.py)).
   - Kiểm thử mô hình trên tập Out-of-Sample (2022 - 2026) qua các chế độ thị trường đặc thù của Việt Nam.
   - **Các chỉ tiêu định lượng bắt buộc đạt được:**
     - **Sharpe Ratio (Annualized):** $\ge 1.2$ (so với VN30-Index $< 0.5$).
     - **Sortino Ratio:** $\ge 1.8$ (kiểm chứng khả năng triệt tiêu tổn thất chiều giảm).
     - **Maximum Drawdown (MDD):** $\le 15\%$ (cắt giảm tối thiểu $50\%$ mức sụt giảm của VN30-Index trong đợt sập năm 2022).
     - **Calmar Ratio:** $\ge 1.5$.
     - **Turnover Rate:** Kiểm soát tỷ lệ đảo danh mục để chi phí giao dịch không vượt quá 2% tổng tài sản/năm.

4. **Mục tiêu 4: Xây dựng Dashboard Trực quan hóa và Hoàn thiện Báo cáo (Relevant & Time-bound)**
   - Xây dựng Dashboard tương tác bằng **Streamlit** ([app.py](file:///home/tu/Documents/Risk-awareness%20RL%20Trading%20bot/05_Deployment_Demo/app.py)):
     - Trực quan hóa đường cong vốn (Equity Curve) của tác tử Risk-Aware RL so với VN30-Index, Equal Weight (1/N) và Markowitz MPT.
     - Biểu đồ diện tích xếp chồng (Stacked Area Chart) thể hiện biến động phân bổ tỷ trọng cổ phiếu và tiền mặt theo thời gian.
     - Biểu đồ sụt giảm vốn tức thời (Underwater Drawdown Chart).
     - Bảng so sánh các chỉ số thống kê định lượng theo thời gian thực.
   - Đóng gói mã nguồn có unit test và hoàn thiện toàn bộ tài liệu báo cáo kỹ thuật đề tài.
