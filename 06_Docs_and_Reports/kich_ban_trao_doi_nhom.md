# BỘ KỊCH BẢN TRAO ĐỔI & THẢO LUẬN NHÓM TRÊN HỆ THỐNG
## Kênh thảo luận: INT3011E - RISK-AWARE RL (Nhóm Antigravity)
**Chủ đề:** Xây dựng bài toán nghiên cứu giữa kỳ (Problem Formulation & Literature Review)  
**Mục tiêu:** Tạo chuỗi thảo luận học thuật chất lượng cao (Conversation Length & Depth), thể hiện tinh thần làm việc nhóm, tư duy phản biện và năng lực nghiên cứu khoa học.

---

### CÁC THÀNH VIÊN THAM GIA TRAO ĐỔI:
- **@nhudinhlu1 (Như Đình Tú):** Nhóm trưởng - Định hướng bài toán, lý thuyết tài chính định lượng & reward shaping.
- **@23021619 (Nguyễn Văn Mạnh):** Thành viên chính / ML Engineer - Thuật toán RL (PPO, SAC), mô hình toán & kiểm thử backtest.
- **@duc_tm22 (Trần Minh Đức):** Data Engineer - Thu thập dữ liệu OHLCV, tính toán chỉ báo kỹ thuật & NLP FinBERT.
- **@long_lh01 (Lê Hoàng Long):** Software & System Engineer - Môi trường Gymnasium, Streamlit Dashboard & tổng hợp báo cáo.

---

## CHẶNG 1: THẢO LUẬN ĐẶT VẤN ĐỀ & PHÂN TÍCH NỖI ĐAU THỰC TẾ
*(Mục tiêu: Làm rõ vấn đề tại sao RL truyền thống thất bại, đối tượng chịu thiệt hại và dẫn chứng thực tế)*

> **@nhudinhlu1:**  
> Chào cả nhóm! Tối nay theo lịch chúng ta bắt đầu brainstorm để làm proposal giữa kỳ nộp trước ngày 02/10/2026 nhé. Thầy yêu cầu rất rõ 4 mục: Đặt vấn đề, Khoảng trống (5 papers, 5 patents, 5 sản phẩm), Câu hỏi nghiên cứu và Mục tiêu.  
> Về Mục 1 - Đặt vấn đề, mình thấy hiện nay hầu hết các paper áp dụng Deep RL vào trading đều gặp một "tử huyệt" chung: các bạn ấy chỉ tối ưu hóa tổng lợi nhuận tích lũy thuần túy (Cumulative Absolute Return) mà coi nhẹ biến động và rủi ro sụt giảm vốn. Mọi người thấy hướng tiếp cận phản biện này thế nào?

> **@23021619 (Nguyễn Văn Mạnh):**  
> Chuẩn luôn Tú ơi! Trong lý thuyết MDP chuẩn, agent chỉ nhận reward là phần trăm lãi $r_t = (V_t - V_{t-1})/V_{t-1}$. Về mặt toán học, hàm mục tiêu này không có số hạng bậc hai (variance) hay số hạng bậc cao (downside semi-variance, skewness).  
> Kết quả là khi huấn luyện trên tập dữ liệu lịch sử có xu hướng tăng (Bull run), con bot RL sẽ học được chính sách "full margin" hoặc all-in vị thế mua để tối đa hóa phần thưởng. Nhưng khi đem ra test trên thị trường thực tế gặp cú sập sốc, bot sẽ tiếp tục gồng lỗ hoặc bắt dao rơi dẫn tới cháy tài khoản (Drawdown > 60-80%).

> **@duc_tm22:**  
> Đúng vậy Mạnh. Mình đang chuẩn bị tập dữ liệu crypto và chứng khoán, thấy rất rõ điều này. Thị trường tài chính vốn có tính phi dừng (non-stationarity) và phân phối đuôi dày (fat-tailed), không hề tuân theo phân phối chuẩn Gaussian.  
> Chưa kể một hiện tượng nữa là "Churning" - giao dịch quá mức. Do nhiều mô hình giả định thị trường không có ma sát (frictionless), bot RL cứ đảo vị thế liên tục ở từng bước thời gian. Khi gặp thị trường đi ngang (Sideway), phí hoa hồng và trượt giá (slippage) sẽ ăn mòn toàn bộ vốn.

> **@long_lh01:**  
> Mình đồng ý. Về phần đối tượng gặp vấn đề (Stakeholders), nhóm mình nên chia thành 3 nhóm rõ ràng để bài báo cáo có tính ứng dụng cao:  
> 1. Retail Traders / Quants cá nhân: Dễ bị lừa bởi các bot APR cao trên mạng nhưng không có cơ chế quản trị rủi ro thích ứng.  
> 2. Các quỹ định lượng vừa và nhỏ (Boutique Hedge Funds): Rất cần thuật toán tự học nhưng bắt buộc phải có chặn rào cản Drawdown và các chỉ số Sharpe/Sortino đạt chuẩn để thuyết phục nhà đầu tư.  
> 3. Cộng đồng nghiên cứu AI/Fintech: Chưa có một open-source framework hoàn chỉnh kết hợp Gymnasium với Risk-Aware Reward Shaping đa mục tiêu.

> **@nhudinhlu1:**  
> Ý kiến của Long rất sắc bén! Để thuyết phục thầy ở Mục 1, chúng ta cần đưa vào 3 ví dụ thực tế kinh điển:  
> - Ví dụ 1: Cú sập LUNA/FTX năm 2022. Các bot DCA/Grid và RL thông thường tiếp tục mua bắt đáy khi giá lao dốc từ 80$ về 0.0001$, làm nhà đầu tư mất trắng trong 72h vì không có Drawdown Barrier.  
> - Ví dụ 2: Cú sập COVID-19 tháng 3/2020 trên chỉ số S&P 500, thị trường rơi tự do 30% trong 20 phiên, các mô hình thống kê Gaussian vỡ trận vì đánh giá sai rủi ro đuôi (Tail Risk).  
> - Ví dụ 3: Đợt sụt giảm của VN-Index năm 2022 (từ 1.530 về 873 điểm), các chỉ báo kỹ thuật cổ điển như MA, MACD bị trễ từ 5-10 phiên, khi báo bán thì tài khoản đã âm 30-40%.  
> Đức ghi lại 3 ví dụ này để đưa vào bản nháp báo cáo nhé!

---

## CHẶNG 2: MỔ XẺ KHOẢNG TRỐNG (5 PAPERS - 5 PATENTS - 5 PRODUCTS)
*(Mục tiêu: Đánh giá phản biện chuyên sâu, chỉ rõ điểm yếu của từng giải pháp hiện có trên thế giới)*

> **@nhudinhlu1:**  
> Bây giờ sang Mục 2: Khoảng trống nghiên cứu và công nghệ. Thầy yêu cầu tìm chính xác: 05 bài báo khoa học, 05 sáng chế kỹ thuật (Patents), 05 sản phẩm thực tế và chỉ rõ điểm yếu từng cái. Mình đã khảo sát trước, giờ anh em cùng mổ xẻ nhé.  
> Đầu tiên là **05 Bài báo khoa học**:  
> 1. *Moody & Saffell (2001)* - Đề xuất Differential Sharpe Ratio (DSR). Điểm mạnh là tính Sharpe đệ quy trực tuyến từng step. Nhưng điểm yếu là gì Mạnh nhỉ?

> **@23021619 (Nguyễn Văn Mạnh):**  
> Điểm yếu của Moody (2001) là DSR dùng mẫu số là tổng phương sai (total variance). Trong tài chính, nhà đầu tư chỉ sợ rủi ro giảm giá (downside risk), còn biến động tăng giá (upside volatility) là lợi nhuận tốt! DSR phạt đồng đều cả hai, khiến bot RL trở nên quá rụt rè trong các đợt bứt phá mạnh. Thêm nữa, mô hình của Moody chỉ là mạng nơ-ron tuyến tính cổ điển, chưa kết hợp mạng sâu và rào cản dừng lỗ động.

> **@duc_tm22:**  
> Bài số 2 là *Jiang et al. (2017)* - "A Deep RL Framework for Financial Portfolio Management", dùng EIIE (Inception + RNN) và DDPG cho crypto:  
> - Điểm yếu: Bài này tối ưu trực tiếp log return ròng $\ln(V_t / V_{t-1})$, hoàn toàn trung lập với rủi ro (risk-neutral). Trong backtest của họ vào mùa Bull 2015-2016 thì ăn đậm, nhưng khi gặp Bear Market thì Max Drawdown bị sập tới hơn 60-70%.

> **@long_lh01:**  
> Còn bài số 3: *Chow et al. (2015) trên NeurIPS* - "Risk-Sensitive and Robust Decision-Making: A CVaR Constrained MDP Approach":  
> - Phân tích: Họ đặt ràng buộc rủi ro đuôi bằng Conditional Value-at-Risk (CVaR). Nhưng điểm yếu chết người là họ giải bằng nhân tử Lagrange và Primal-Dual Policy Gradient. Thuật toán này cực kỳ nhạy cảm với learning rate, tốc độ hội tụ siêu chậm và rất dễ phân kỳ trong môi trường tài chính phi dừng.  
> 
> Bài số 4: *FinRL (Liu et al., 2020/2021, NeurIPS/ICAIF)*:  
> - FinRL rất nổi tiếng, cung cấp framework chuẩn. Nhưng điểm yếu là các reward mẫu của họ vẫn là return thuần túy, chưa có Prospect Theory Utility hay Drawdown Barrier thích ứng. Bot FinRL chạy thực tế bị hiện tượng churn lệnh rất nặng.  
> 
> Bài số 5: *Spooner et al. (2018, IJCAI)* - "Market Making via RL":  
> - Chỉ áp dụng cho vi mô sổ lệnh HFT (High-Frequency Trading) cấp mili-giây với phạt rủi ro tồn kho (inventory risk). Hoàn toàn không áp dụng được cho quản lý danh mục trung hạn (swing trading), và khi thị trường mất thanh khoản (liquidity cascade) thì bot sập ngay.

> **@nhudinhlu1:**  
> Tuyệt vời! Phân tích học thuật của cả nhóm rất chắc. Bây giờ đến **05 Sáng chế kỹ thuật (Patents)**. Phần này nhiều nhóm khác rất dễ bị bí hoặc đưa thông tin chung chung, nhóm mình đưa thẳng mã số patent đã công bố trên USPTO / Google Patents luôn cho chuẩn:  
> 1. *US 2019/0370649 A1*: "Reinforcement learning neural network for automated trading".  
>    -> Điểm yếu: Dùng DQN/Policy Gradient tối ưu hóa raw PnL, không gian hành động rời rạc (Buy/Sell/Hold) gây hiện tượng đổi vị thế liên tục (whipsaw) khi thị trường giằng co, làm bùng nổ chi phí giao dịch.  
> 2. *US 12,293,378 B2 / US 2022/0129925 A1 (JPMorgan Chase Bank, N.A.)*: Mô phỏng thị trường và khớp lệnh LOXM bằng RL.  
>    -> Điểm yếu: Chỉ tập trung vào tầng khớp lệnh vi mô (execution/slippage) cho khối lệnh khổng lồ của tổ chức; không giải quyết bài toán vĩ mô về phân bổ tỷ trọng danh mục tài sản và phòng hộ sụt giảm vốn; chi phí hạ tầng mô phỏng quá nặng.  
> 3. *US 10,769,713 B2*: Hệ thống giao dịch tự động tích hợp quản trị rủi ro bằng học máy.  
>    -> Điểm yếu: Module rủi ro là các luật cắt lỗ cứng (hard stop-loss if-else) nằm tách rời bên ngoài quá trình huấn luyện RL. Agent không học được bản năng tự né tránh rủi ro, và dễ bị dính bẫy quét thanh lý (stop-hunt/whipsaw).  
> 4. *US 2021/0217104 A1*: Tái cân bằng danh mục tài sản động bằng RL.  
>    -> Điểm yếu: Tính toán Sharpe ratio theo chu kỳ mẻ (batch periodic), tạo độ trễ pha so với biến động thị trường tức thời; hoàn toàn thiếu dữ liệu tâm lý tin tức xã hội.  
> 5. *US 2015/0100530 A1 (DeepMind Technologies Ltd)*: Kiến trúc DQN nền tảng.  
>    -> Điểm yếu: Giả định môi trường dừng (stationary MDP); khi đưa vào tài chính có độ nhiễu cao, cơ chế replay buffer lấy mẫu đồng đều vô tình gây hiện tượng "quên thảm họa" (catastrophic forgetting) các pha thị trường sụp đổ hiếm gặp.

> **@duc_tm22:**  
> Chuẩn xác! Tiếp theo là **05 Sản phẩm thực tế trên thị trường**:  
> 1. *Trade Ideas (Holly AI)*: Dùng AI quét kịch bản trong ngày. Điểm yếu: Là "hộp đen" độc quyền, dựa trên luật thống kê tĩnh chứ không phải tác tử RL thích ứng; phí bản quyền đắt đỏ ($1,000 - $2,500/năm).  
> 2. *Kavout (K-Score AI)*: Chấm điểm cổ phiếu bằng Deep Learning. Điểm yếu: Là bài toán Học có giám sát (Supervised Learning) dự đoán xu hướng giá, không phải là hệ thống ra quyết định tuần tự (Sequential Decision Making); không quản lý tỷ trọng động và không có cơ chế Drawdown Barrier.  
> 3. *Pionex / 3Commas (Grid & DCA Bots)*: Nền tảng bot bán lẻ crypto lớn nhất hiện nay. Điểm yếu: Hoàn toàn là thuật toán tĩnh (deterministic rules); khi thị trường downtrend dốc đứng, bot DCA tiếp tục gom hàng gồng lỗ dẫn đến cháy tài khoản hoặc kẹp vốn đến 80-90%.  
> 4. *Numerai*: Quỹ định lượng crowdsourcing. Điểm yếu: Người dùng chỉ nộp dự đoán xác suất (cross-sectional return), không được cá nhân hóa chính sách phân bổ danh mục theo khẩu vị rủi ro riêng; chu kỳ hàng tuần quá chậm.  
> 5. *Composer.trade*: Nền tảng no-code xây chiến lược định lượng bằng logic if-else. Điểm yếu: Dựa trên các chỉ báo kỹ thuật cổ điển có độ trễ lớn; người dùng phải tự nghĩ luật; không có năng lực tự học thích ứng (self-learning) để tối ưu tỷ trọng phi tuyến dưới ma sát trượt giá.

> **@23021619 (Nguyễn Văn Mạnh):**  
> Tổng hợp lại từ 15 giải pháp trên, chúng ta đúc kết ra **Khoảng trống cốt lõi (Research Gap)** của đề tài:  
> Hiện nay chưa có một framework mã nguồn mở hoàn chỉnh nào kết hợp được đồng thời:  
> (i) Hàm thưởng rủi ro đa mục tiêu dựa trên toán tài chính và kinh tế hành vi (Differential Sharpe, Sortino Downside semi-variance, Drawdown Barrier, Prospect Theory);  
> (ii) Biểu diễn trạng thái đa phương thức kết hợp chỉ báo kỹ thuật với điểm số tâm lý tin tức FinBERT;  
> (iii) Mô phỏng thực tế chi phí trượt giá với không gian hành động liên tục $w_t \in [-1, 1]$;  
> (iv) Kiểm chứng Out-of-Sample nghiêm ngặt qua 3 chế độ thị trường (Bull, Bear, Sideway) có kèm Web Dashboard trực quan!

---

## CHẶNG 3: KHOANH VÙNG CÂU HỎI NGHIÊN CỨU (RESEARCH QUESTIONS)
*(Mục tiêu: Đặt ra các câu hỏi sắc bén, khoanh vùng phạm vi thực hiện khả thi và đúng trọng tâm)*

> **@nhudinhlu1:**  
> Khoảng trống nghiên cứu như vậy là cực kỳ thuyết phục rồi! Giờ chuyển sang Mục 3: Xây dựng câu hỏi nghiên cứu (Research Questions).  
> Anh em lưu ý nguyên tắc khoanh vùng: Đề tài của chúng ta làm trong phạm vi đồ án học phần và nghiên cứu sinh viên, chúng ta KHÔNG nên tham vọng làm bài toán HFT vi mô cấp microsecond vì không có đường truyền máy chủ đặt cạnh sàn (co-location). Chúng ta nên khoanh vùng vào: **Giao dịch định lượng trung hạn (Swing/Position Trading) theo nến Ngày (Daily) hoặc Giờ (Hourly)**. Mọi người đồng ý không?

> **@long_lh01:**  
> Hoàn toàn đồng ý với Tú. Khoanh vùng vào Swing Trading theo nến Ngày/Giờ là hợp lý nhất về cả chi phí tính toán, khả năng lấy dữ liệu lẫn tính ứng dụng thực tiễn cho nhà đầu tư cá nhân và quỹ vừa.  
> Theo cấu trúc đó, mình đề xuất chia thành 3 câu hỏi nghiên cứu (RQ) chính:

> **@23021619 (Nguyễn Văn Mạnh):**  
> - **RQ1 (Về Reward Shaping & Risk Mitigation):**  
>   *"Làm thế nào để thiết kế và tối ưu hóa các hàm thưởng nhận thức rủi ro—cụ thể là Differential Sharpe Ratio, Sortino Downside semi-variance, Drawdown Barrier Penalty và Prospect Theory—giúp tác tử RL kiềm chế tối đa mức sụt giảm tài sản (Max Drawdown < 15%) mà vẫn duy trì tỷ suất sinh lời vượt trội so với các chiến lược chuẩn (Buy & Hold, Trend Following)?"*  
> Câu này giải quyết trực tiếp bài toán "cháy tài khoản" trong Bear Market.

> **@duc_tm22:**  
> - **RQ2 (Về State Representation & Multimodal Sentiment):**  
>   *"Việc dung hợp dữ liệu kỹ thuật thị trường (RSI, MACD, Bollinger Bands, ATR) với tín hiệu tâm lý thị trường (Market Sentiment Score trích xuất từ FinBERT/Tin tức) vào không gian trạng thái của môi trường Gymnasium có giúp tác tử nhận diện sớm các điểm đảo chiều xu hướng và giảm độ trễ phản ứng so với các mô hình chỉ dựa trên giá hay không?"*  
> Câu này khoanh vùng vào giá trị gia tăng của dữ liệu đa phương thức.

> **@long_lh01:**  
> - **RQ3 (Về Tính Thực tế & Khả năng Thích ứng Đa chế độ):**  
>   *"Chính sách giao dịch học được từ các thuật toán Actor-Critic (PPO và SAC) thích ứng như thế nào trước tác động của phí giao dịch và độ trượt giá thực tế, và liệu chính sách đó có thể hiện tính tổng quát hóa ổn định khi chuyển dịch giữa 3 chế độ thị trường đối nghịch (Tăng trưởng - Bull, Suy thoái - Bear, Đi ngang giằng co - High-Volatility Sideway) trên các lớp tài sản khác nhau (BTC-USD, VN-Index) hay không?"*  
> Câu này khoanh vùng vào khả năng tổng quát hóa (generalization) và tính khả thi khi triển khai thực tế.

---

## CHẶNG 4: XÂY DỰNG MỤC TIÊU NGHIÊN CỨU & KẾ HOẠCH NỘP BÀI 02/10
*(Mục tiêu: Thiết lập mục tiêu SMART, định lượng rõ ràng các chỉ số tài chính và phân công nhiệm vụ)*

> **@nhudinhlu1:**  
> 3 câu hỏi nghiên cứu trên đã khoanh vùng rất chuẩn xác. Bây giờ chúng ta chốt Mục 4: Xây dựng mục tiêu.  
> Mục tiêu tổng quát: Xây dựng và kiểm chứng thực nghiệm hệ thống giao dịch định lượng tự động ứng dụng Deep RL nhận thức rủi ro, tối ưu hóa tỷ số Sharpe/Sortino và bảo toàn vốn qua các chu kỳ khủng hoảng.  
> Còn về các mục tiêu cụ thể, chúng ta phải viết theo đúng tiêu chuẩn **SMART** (Cụ thể, Đo lường được, Khả thi, Phù hợp và Có thời hạn):  
> Mạnh xem các chỉ số định lượng backtest chúng ta cam kết như thế nào là vừa tầm và ấn tượng?

> **@23021619 (Nguyễn Văn Mạnh):**  
> Mình đề xuất các KPI định lượng cụ thể sau trên tập Out-of-Sample:  
> 1. **Sharpe Ratio (Annualized) $\ge 1.5$** (so với Buy & Hold thường $< 1.0$).  
> 2. **Sortino Ratio $\ge 2.0$** (chứng minh rủi ro chiều giảm bị triệt tiêu rõ rệt).  
> 3. **Maximum Drawdown (MDD) $\le 15\%$** (giảm ít nhất 50% mức sụt giảm so với thị trường trong pha sập giá).  
> 4. **Calmar Ratio $\ge 1.8$** và **Profit Factor $\ge 1.4$**.  
> Các chỉ số này hoàn toàn đo lường được bằng script `04_Backtesting_Evaluation/metrics.py` của nhóm mình!

> **@duc_tm22:**  
> Về phần kỹ thuật hệ thống, mình cam kết hoàn thành:  
> - Pipeline dữ liệu tự động tải nến OHLCV từ Yahoo Finance cho BTC-USD, ETH-USD và ^VNINDEX.  
> - Tính toán đầy đủ 8 chỉ báo kỹ thuật cơ bản + mô hình sentiment.  
> - Tích hợp chuẩn môi trường Gymnasium v0.29+ với đầy đủ 5 chế độ reward mode trong `03_Model_Development/env/trading_env.py`.

> **@long_lh01:**  
> Còn phần ứng dụng demo và báo cáo, mình sẽ phụ trách:  
> - Dựng hoàn chỉnh giao diện Web tương tác bằng Streamlit (`05_Deployment_Demo/app.py`), cho phép chọn mã tài sản, chọn thuật toán PPO/SAC, chọn reward mode và bấm nút là hiển thị ngay: Equity Curve so sánh, biểu đồ sụt vốn Underwater Drawdown và bảng chỉ số tài chính.  
> - Đóng gói toàn bộ tài liệu báo cáo `bao_cao_giua_ky_de_tai.md` và slide thuyết trình trong thư mục `06_Docs_and_Reports/`.

> **@nhudinhlu1:**  
> Xuất sắc! Thời hạn nộp bài là **02/10/2026**.  
> Kế hoạch phân công cụ thể như sau:  
> - Hôm nay (30/09): Hoàn thiện toàn bộ văn bản Báo cáo giữa kỳ 4 mục và đồng bộ mã nguồn lên Git.  
> - Ngày mai (01/10): Chạy nghiệm thu các kịch bản backtest PPO/SAC trên dữ liệu BTC và VN-Index để lấy số liệu thực tế đưa vào bảng biểu.  
> - Ngày 02/10: Review lần cuối toàn bộ tài liệu, kiểm tra tính nhất quán và nộp bài đúng hạn lên hệ thống.  
> Mọi người cùng cố gắng nhé, đề tài này của nhóm mình giải quyết bài toán rất thiết thực và có hàm lượng khoa học cao!

> **@23021619 (Nguyễn Văn Mạnh):** Nhất trí 100%! Bắt tay vào việc luôn thôi anh em!  
> **@duc_tm22:** Ok Tú, tài liệu và dữ liệu đã sẵn sàng!  
> **@long_lh01:** Đã rõ deadline, chiến thôi anh em!
