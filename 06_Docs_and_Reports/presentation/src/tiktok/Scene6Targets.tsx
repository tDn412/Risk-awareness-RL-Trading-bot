import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

export const Scene6Targets: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleSpring = spring({ frame, fps, config: { damping: 10, mass: 0.6 }, from: 0, to: 1 });

  const kpi1 = spring({ frame: frame - 15, fps, config: { damping: 10 }, from: 0.6, to: 1 });
  const kpi2 = spring({ frame: frame - 25, fps, config: { damping: 10 }, from: 0.6, to: 1 });
  const kpi3 = spring({ frame: frame - 35, fps, config: { damping: 10 }, from: 0.6, to: 1 });
  const kpi4 = spring({ frame: frame - 45, fps, config: { damping: 10 }, from: 0.6, to: 1 });

  return (
    <div
      style={{
        width: 1080,
        height: 1920,
        backgroundColor: "#080B11",
        color: "#FFFFFF",
        padding: "100px 60px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        alignItems: "center",
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Background glow */}
      <div
        style={{
          position: "absolute",
          top: "40%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 800,
          height: 800,
          background: "radial-gradient(circle, rgba(56, 189, 248, 0.25) 0%, rgba(8, 11, 17, 0) 70%)",
          filter: "blur(80px)",
        }}
      />

      {/* Top Badge */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          background: "rgba(56, 189, 248, 0.15)",
          border: "2px solid rgba(56, 189, 248, 0.5)",
          padding: "12px 28px",
          borderRadius: 40,
        }}
      >
        <span style={{ fontSize: 24 }}>🎯</span>
        <span
          style={{
            color: "#38BDF8",
            fontSize: 20,
            fontWeight: 800,
            letterSpacing: 2,
            textTransform: "uppercase",
          }}
        >
          CAM KẾT KẾT QUẢ ĐỒ ÁN
        </span>
      </div>

      {/* Title */}
      <div style={{ textAlign: "center", transform: `scale(${titleSpring})`, opacity: titleSpring }}>
        <div style={{ fontSize: 40, fontWeight: 700, color: "#94A3B8", marginBottom: 12 }}>
          Đo lường bằng số liệu
        </div>
        <div
          style={{
            fontSize: 68,
            fontWeight: 950,
            lineHeight: 1.15,
            color: "#FFFFFF",
            letterSpacing: -1,
            textTransform: "uppercase",
          }}
        >
          MỤC TIÊU ĐỊNH LƯỢNG{" "}
          <span
            style={{
              background: "linear-gradient(135deg, #38BDF8, #A855F7)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            SỐNG CÒN
          </span>
        </div>
      </div>

      {/* 4 Massive Glowing KPI Cards */}
      <div
        style={{
          width: "100%",
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 24,
        }}
      >
        {/* KPI 1 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.85)",
            border: "2px solid #EF4444",
            borderRadius: 24,
            padding: 30,
            textAlign: "center",
            boxShadow: "0 10px 30px rgba(239, 68, 68, 0.2)",
            transform: `scale(${kpi1})`,
          }}
        >
          <div style={{ color: "#FDA4AF", fontSize: 18, fontWeight: 700, textTransform: "uppercase" }}>
            Max Drawdown (MDD)
          </div>
          <div style={{ color: "#EF4444", fontSize: 56, fontWeight: 950, margin: "12px 0" }}>
            ≤ 15%
          </div>
          <div style={{ color: "#94A3B8", fontSize: 16 }}>
            Cắt giảm &gt;50% sụt giảm so với VN30-Index năm 2022 (-42.8%)
          </div>
        </div>

        {/* KPI 2 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.85)",
            border: "2px solid #10B981",
            borderRadius: 24,
            padding: 30,
            textAlign: "center",
            boxShadow: "0 10px 30px rgba(16, 185, 129, 0.2)",
            transform: `scale(${kpi2})`,
          }}
        >
          <div style={{ color: "#A7F3D0", fontSize: 18, fontWeight: 700, textTransform: "uppercase" }}>
            Sharpe Ratio
          </div>
          <div style={{ color: "#10B981", fontSize: 56, fontWeight: 950, margin: "12px 0" }}>
            ≥ 1.2
          </div>
          <div style={{ color: "#94A3B8", fontSize: 16 }}>
            Thị trường chỉ đạt &lt;0.5 trong giai đoạn 2022 - 2024
          </div>
        </div>

        {/* KPI 3 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.85)",
            border: "2px solid #38BDF8",
            borderRadius: 24,
            padding: 30,
            textAlign: "center",
            boxShadow: "0 10px 30px rgba(56, 189, 248, 0.2)",
            transform: `scale(${kpi3})`,
          }}
        >
          <div style={{ color: "#BAE6FD", fontSize: 18, fontWeight: 700, textTransform: "uppercase" }}>
            Sortino Ratio
          </div>
          <div style={{ color: "#38BDF8", fontSize: 56, fontWeight: 950, margin: "12px 0" }}>
            ≥ 1.8
          </div>
          <div style={{ color: "#94A3B8", fontSize: 16 }}>
            Chứng minh năng lực triệt tiêu phương sai chiều giảm
          </div>
        </div>

        {/* KPI 4 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.85)",
            border: "2px solid #F59E0B",
            borderRadius: 24,
            padding: 30,
            textAlign: "center",
            boxShadow: "0 10px 30px rgba(245, 158, 11, 0.2)",
            transform: `scale(${kpi4})`,
          }}
        >
          <div style={{ color: "#FDE68A", fontSize: 18, fontWeight: 700, textTransform: "uppercase" }}>
            Chi Phí Rebalance
          </div>
          <div style={{ color: "#F59E0B", fontSize: 56, fontWeight: 950, margin: "12px 0" }}>
            ≤ 2% / năm
          </div>
          <div style={{ color: "#94A3B8", fontSize: 16 }}>
            Kiểm soát turnover rate, không bị thuế phí bào mòn vốn
          </div>
        </div>
      </div>

      {/* Outro Project Card */}
      <div
        style={{
          width: "100%",
          background: "linear-gradient(135deg, rgba(37, 99, 235, 0.3), rgba(168, 85, 247, 0.3))",
          border: "2px solid rgba(255, 255, 255, 0.2)",
          borderRadius: 24,
          padding: "24px 36px",
          textAlign: "center",
        }}
      >
        <div style={{ color: "#38BDF8", fontSize: 16, fontWeight: 800, textTransform: "uppercase", letterSpacing: 1.5 }}>
          DỰ ÁN ANTIGRAVITY • INT3011E
        </div>
        <div style={{ color: "#FFFFFF", fontSize: 26, fontWeight: 900, marginTop: 4 }}>
          Risk-Aware RL for VN30 Portfolio Allocation
        </div>
        <div style={{ color: "#CBD5E1", fontSize: 16, marginTop: 8 }}>
          Nhóm thực hiện: Như Đình Tú • Nguyễn Văn Mạnh • Hạn bảo vệ: 02/10/2026
        </div>
      </div>
    </div>
  );
};
