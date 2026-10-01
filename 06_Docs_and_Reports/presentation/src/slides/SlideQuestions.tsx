import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Header } from "../components/Header";

export const SlideQuestions: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const opacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: "clamp" });
  const translateY = spring({ frame, fps, config: { damping: 15 }, from: 30, to: 0 });

  const questions = [
    {
      qNum: "01",
      tag: "Hàm Thưởng Rủi Ro Trên Simplex",
      color: "#2563EB",
      text: "Làm thế nào để thiết kế và tối ưu hóa hàm thưởng nhận thức rủi ro (kết hợp Sortino downside semi-variance, Drawdown Barrier Penalty và Prospect Theory) giúp tác tử RL phân bổ tỷ trọng trên simplex (∑ w_i = 1, w_i ≥ 0) kiềm chế mức sụt giảm tài sản tối đa (Max Drawdown ≤ 15%) trên rổ VN30 mà vẫn bảo toàn tỷ suất sinh lời vượt trội so với VN30-Index?",
      target: "Kiểm soát Max Drawdown ≤ 15% | Sharpe Ratio ≥ 1.2",
    },
    {
      qNum: "02",
      tag: "Thích Ứng Ma Sát Sàn HOSE",
      color: "#F59E0B",
      text: "Thuật toán Policy Gradient liên tục (PPO với Softmax Policy) thích ứng như thế nào trước tác động của thuế bán 0.1% và phí giao dịch sàn HOSE, nhằm triệt tiêu hiện tượng tái cơ cấu danh mục quá mức (portfolio churning) so với chiến lược Buy & Hold?",
      target: "Turnover Cost ≤ 2%/năm | Triệt tiêu Whipsaw Churning",
    },
    {
      qNum: "03",
      tag: "Tổng Quát Hóa Đa Chế Độ Thị Trường",
      color: "#10B981",
      text: "Chính sách phân bổ tỷ trọng học được có duy trì các chỉ số hiệu năng điều chỉnh theo rủi ro (Sharpe Ratio ≥ 1.2, Sortino Ratio ≥ 1.8, Calmar Ratio ≥ 1.5) khi kiểm thử Out-of-Sample qua 3 giai đoạn thị trường đối nghịch của Việt Nam (Tăng trưởng 2020-2021, Suy thoái 2022, Đi ngang 2023-2024) so với danh mục Equal Weight (1/N) và Markowitz MPT hay không?",
      target: "Sortino ≥ 1.8 | Calmar ≥ 1.5 qua Bull, Bear & Sideway",
    },
  ];

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        padding: "40px 60px",
        background: "#0B0F19",
        color: "#FFFFFF",
        opacity,
        transform: `translateY(${translateY}px)`,
      }}
    >
      <Header
        sectionNumber="03"
        category="Khoanh Vùng Nghiên Cứu"
        sectionTitle="Xây Dựng 03 Câu Hỏi Nghiên Cứu (Research Questions)"
      />

      <div style={{ display: "flex", flexDirection: "column", gap: 16, flex: 1, justifyContent: "center" }}>
        {questions.map((q, idx) => (
          <div
            key={idx}
            style={{
              background: "rgba(30, 41, 59, 0.5)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: 14,
              padding: "18px 24px",
              display: "flex",
              alignItems: "center",
              gap: 24,
            }}
          >
            <div
              style={{
                fontSize: 28,
                fontWeight: 900,
                color: q.color,
                background: "rgba(15, 23, 42, 0.8)",
                padding: "12px 18px",
                borderRadius: 12,
                border: `1px solid ${q.color}40`,
              }}
            >
              Q{q.qNum}
            </div>

            <div style={{ flex: 1 }}>
              <div
                style={{
                  display: "inline-block",
                  color: q.color,
                  fontSize: 12,
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: 1,
                  marginBottom: 6,
                }}
              >
                {q.tag}
              </div>
              <div style={{ fontSize: 16, color: "#F8FAFC", lineHeight: 1.5, fontWeight: 500 }}>
                {q.text}
              </div>
              <div
                style={{
                  marginTop: 8,
                  fontSize: 13,
                  color: "#94A3B8",
                  fontWeight: 600,
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                <span style={{ color: "#38BDF8" }}>🎯 Chỉ tiêu kiểm chứng:</span> {q.target}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
