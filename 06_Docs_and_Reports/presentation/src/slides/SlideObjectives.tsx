import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Header } from "../components/Header";

export const SlideObjectives: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const opacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: "clamp" });
  const translateY = spring({ frame, fps, config: { damping: 15 }, from: 30, to: 0 });

  const objectives = [
    {
      num: "01",
      title: "Pipeline Dữ Liệu & Môi Trường Gymnasium",
      badge: "Specific & Measurable",
      color: "#2563EB",
      items: [
        "Thu thập dữ liệu rổ VN30 (2018 - 2026) với 8+ chỉ báo kỹ thuật.",
        "Môi trường VN30PortfolioEnv chuẩn Gymnasium v0.29+.",
        "Mô phỏng Long-Only, biên độ ±7%, thuế 0.1% và phí 0.15%.",
      ],
    },
    {
      num: "02",
      title: "Triển Khai & Huấn Luyện Deep RL",
      badge: "Achievable & Relevant",
      color: "#F59E0B",
      items: [
        "Thuật toán PPO và SAC với policy phân bổ Softmax trên PyTorch.",
        "Huấn luyện tập In-Sample (2018 - 2021) với ≥ 100,000 timesteps.",
        "Giám sát Policy/Value Loss và Entropy qua TensorBoard.",
      ],
    },
    {
      num: "03",
      title: "Kiểm Định Out-of-Sample & Benchmark",
      badge: "Measurable & Time-bound",
      color: "#10B981",
      items: [
        "Sharpe Ratio ≥ 1.2 (thị trường < 0.5) | Sortino Ratio ≥ 1.8.",
        "Max Drawdown ≤ 15% (giảm > 50% so với VN30-Index năm 2022).",
        "Calmar Ratio ≥ 1.5 | Chi phí giao dịch ≤ 2% tài sản/năm.",
      ],
    },
    {
      num: "04",
      title: "Streamlit Dashboard & Báo Cáo",
      badge: "Relevant & Time-bound",
      color: "#8B5CF6",
      items: [
        "Dashboard Streamlit: Equity curve, Stacked Area tỷ trọng, MDD.",
        "So sánh trực quan với VN30-Index, Equal Weight và Markowitz MPT.",
        "Đóng gói mã nguồn và hoàn thiện báo cáo trước 02/10/2026.",
      ],
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
        sectionNumber="04"
        category="Kế Hoạch Thực Hiện"
        sectionTitle="Xây Dựng Mục Tiêu Nghiên Cứu (SMART Objectives)"
      />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, flex: 1 }}>
        {objectives.map((obj, idx) => (
          <div
            key={idx}
            style={{
              background: "rgba(30, 41, 59, 0.5)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: 14,
              padding: 20,
              display: "flex",
              flexDirection: "column",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span
                  style={{
                    background: obj.color,
                    color: "#FFFFFF",
                    fontSize: 14,
                    fontWeight: 800,
                    padding: "3px 8px",
                    borderRadius: 6,
                  }}
                >
                  Mục tiêu {obj.num}
                </span>
                <span style={{ fontSize: 16, fontWeight: 700, color: "#F8FAFC" }}>
                  {obj.title}
                </span>
              </div>
              <span
                style={{
                  fontSize: 11,
                  color: "#94A3B8",
                  background: "rgba(15, 23, 42, 0.7)",
                  padding: "4px 8px",
                  borderRadius: 6,
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                }}
              >
                {obj.badge}
              </span>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 4 }}>
              {obj.items.map((item, i) => (
                <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, fontSize: 13.5, color: "#CBD5E1", lineHeight: 1.4 }}>
                  <span style={{ color: obj.color, fontSize: 14, marginTop: 1 }}>•</span>
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
