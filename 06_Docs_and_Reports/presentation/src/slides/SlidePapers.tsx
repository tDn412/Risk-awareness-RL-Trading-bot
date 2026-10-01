import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Header } from "../components/Header";

export const SlidePapers: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const opacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: "clamp" });
  const translateY = spring({ frame, fps, config: { damping: 15 }, from: 30, to: 0 });

  const papers = [
    {
      title: "Moody & Saffell (2001) • IEEE TNN",
      name: "Learning to Trade via Direct Reinforcement",
      solution: "Đề xuất Differential Sharpe Ratio (DSR) cập nhật trực tuyến theo từng step.",
      weakness: "Phạt cả biến động chiều tăng (upside gains); không có rào cản drawdown tức thời; chỉ thử nghiệm đơn tài sản.",
    },
    {
      title: "Jiang et al. (2017) • arXiv",
      name: "Deep RL Framework for Financial Portfolio Management",
      solution: "Mạng CNN-RNN (EIIE) kết hợp DDPG xuất vector tỷ trọng phân bổ danh mục.",
      weakness: "Hàm mục tiêu là log return thuần túy (risk-neutral); Max Drawdown > 60% trong pha giảm; bỏ qua trượt giá động.",
    },
    {
      title: "Wang et al. (2021) • AAAI-21",
      name: "DeepTrader: Risk-Return Balanced Portfolio Management",
      solution: "Nhúng điều kiện vĩ mô bằng đồ thị không-thời gian; dùng negative MDD làm reward.",
      weakness: "Đồ thị nhân quả phức tạp, latency cao; giả định cho phép long/short tự do, không phù hợp quy chế HOSE.",
    },
    {
      title: "Enkhsaikhan & Jo (2025) • AAAI-25",
      name: "Augmented Lagrangian Risk-constrained RL for Portfolio Optimization",
      solution: "Dùng Augmented Lagrangian Multiplier (ALM) đưa ràng buộc cứng Value-at-Risk (VaR) vào DDPG.",
      weakness: "Mạng phụ (cost network) nhạy cảm siêu tham số; dễ mất ổn định với đuôi dày; chưa kiểm chứng thị trường mới nổi.",
    },
    {
      title: "Chow et al. (2015) • NeurIPS",
      name: "Risk-Sensitive and Robust Decision-Making: A CVaR Constrained MDP",
      solution: "Mô hình hóa bài toán Constrained MDP với ràng buộc Conditional Value-at-Risk (CVaR).",
      weakness: "Thuật toán Primal-Dual hội tụ chậm; khó mở rộng cho 30 cổ phiếu; không tính chi phí ma sát rebalance.",
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
        sectionNumber="02"
        category="Khoảng Trống Nghiên Cứu"
        sectionTitle="Khảo Sát 05 Bài Báo Khoa Học (Scientific Papers)"
      />

      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 16, flex: 1 }}>
        {papers.map((p, idx) => (
          <div
            key={idx}
            style={{
              background: "rgba(30, 41, 59, 0.5)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: 14,
              padding: 18,
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
            }}
          >
            <div>
              <div
                style={{
                  background: "rgba(59, 130, 246, 0.2)",
                  color: "#60A5FA",
                  fontSize: 12,
                  fontWeight: 700,
                  padding: "4px 8px",
                  borderRadius: 6,
                  display: "inline-block",
                  marginBottom: 10,
                }}
              >
                {p.title}
              </div>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#F8FAFC", marginBottom: 12, lineHeight: 1.3 }}>
                {p.name}
              </div>

              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 11, color: "#10B981", fontWeight: 700, textTransform: "uppercase" }}>
                  Giải pháp cốt lõi
                </div>
                <div style={{ fontSize: 12, color: "#CBD5E1", marginTop: 4, lineHeight: 1.4 }}>
                  {p.solution}
                </div>
              </div>
            </div>

            <div style={{ background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.2)", borderRadius: 8, padding: 10 }}>
              <div style={{ fontSize: 11, color: "#EF4444", fontWeight: 700, textTransform: "uppercase" }}>
                Điểm yếu & Giới hạn
              </div>
              <div style={{ fontSize: 12, color: "#FDA4AF", marginTop: 4, lineHeight: 1.35 }}>
                {p.weakness}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
