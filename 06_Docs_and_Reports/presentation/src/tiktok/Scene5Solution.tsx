import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

export const Scene5Solution: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleSpring = spring({ frame, fps, config: { damping: 10, mass: 0.6 }, from: 0, to: 1 });

  // Animation for asset weight shift: from 85% stocks to 15% stocks (flight to cash!)
  const stockWeight = interpolate(frame, [30, 80], [85, 15], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const cashWeight = 100 - stockWeight;

  const card1 = spring({ frame: frame - 15, fps, config: { damping: 12 }, from: 60, to: 0 });
  const card2 = spring({ frame: frame - 25, fps, config: { damping: 12 }, from: 60, to: 0 });
  const card3 = spring({ frame: frame - 35, fps, config: { damping: 12 }, from: 60, to: 0 });

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
          top: "35%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 700,
          height: 700,
          background: "radial-gradient(circle, rgba(16, 185, 129, 0.25) 0%, rgba(8, 11, 17, 0) 70%)",
          filter: "blur(70px)",
        }}
      />

      {/* Top Badge */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          background: "rgba(16, 185, 129, 0.15)",
          border: "2px solid rgba(16, 185, 129, 0.5)",
          padding: "12px 28px",
          borderRadius: 40,
        }}
      >
        <span style={{ fontSize: 24 }}>🛡️</span>
        <span
          style={{
            color: "#10B981",
            fontSize: 20,
            fontWeight: 800,
            letterSpacing: 2,
            textTransform: "uppercase",
          }}
        >
          GIẢI PHÁP ĐỀ TÀI ANTIGRAVITY
        </span>
      </div>

      {/* Title */}
      <div style={{ textAlign: "center", transform: `scale(${titleSpring})`, opacity: titleSpring }}>
        <div style={{ fontSize: 40, fontWeight: 700, color: "#94A3B8", marginBottom: 12 }}>
          Lớp phòng thủ 3 tầng
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
          RISK-AWARE{" "}
          <span
            style={{
              background: "linear-gradient(135deg, #10B981, #38BDF8)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            REWARD SHAPING
          </span>
        </div>
      </div>

      {/* Dynamic Animated Asset Weight Allocation Bar */}
      <div
        style={{
          width: "100%",
          background: "rgba(18, 24, 38, 0.9)",
          border: "2px solid rgba(255, 255, 255, 0.15)",
          borderRadius: 24,
          padding: "24px 30px",
          boxShadow: "0 10px 40px rgba(0, 0, 0, 0.5)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
          <div style={{ fontSize: 18, fontWeight: 800, color: "#38BDF8" }}>
            📊 Cổ Phiếu VN30: {Math.round(stockWeight)}%
          </div>
          <div style={{ fontSize: 18, fontWeight: 800, color: "#10B981" }}>
            💵 Tiền Mặt (Cash): {Math.round(cashWeight)}%
          </div>
        </div>

        {/* Progress Bar */}
        <div
          style={{
            width: "100%",
            height: 36,
            background: "#1E293B",
            borderRadius: 18,
            overflow: "hidden",
            display: "flex",
            border: "1px solid rgba(255, 255, 255, 0.1)",
          }}
        >
          <div
            style={{
              width: `${stockWeight}%`,
              background: "linear-gradient(90deg, #3B82F6, #06B6D4)",
              transition: "width 0.1s linear",
            }}
          />
          <div
            style={{
              width: `${cashWeight}%`,
              background: "linear-gradient(90deg, #10B981, #34D399)",
              transition: "width 0.1s linear",
            }}
          />
        </div>

        <div style={{ textAlign: "center", color: "#F59E0B", fontSize: 15, fontWeight: 700, marginTop: 12 }}>
          {stockWeight < 30 ? "⚠️ CẢNH BÁO RỦI RO: Tác tử tự động rút về Tiền Mặt!" : "🟢 THỊ TRƯỜNG ỔN ĐỊNH: Tối ưu phân bổ rổ VN30"}
        </div>
      </div>

      {/* 3 Core Reward Mechanism Cards */}
      <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 16 }}>
        {/* Mechanism 1 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.8)",
            border: "1px solid rgba(16, 185, 129, 0.4)",
            borderRadius: 20,
            padding: "20px 24px",
            display: "flex",
            alignItems: "center",
            gap: 20,
            transform: `translateY(${Math.max(0, card1)}px)`,
            opacity: interpolate(frame, [15, 30], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div style={{ fontSize: 36, background: "rgba(16, 185, 129, 0.2)", padding: 12, borderRadius: 14 }}>
            🎯
          </div>
          <div>
            <div style={{ fontSize: 22, fontWeight: 800, color: "#F8FAFC" }}>1. Sortino Downside Penalty</div>
            <div style={{ fontSize: 16, color: "#94A3B8", marginTop: 4 }}>
              Chỉ phạt phương sai chiều giảm! Giữ trọn vẹn phần thưởng lãi đột biến.
            </div>
          </div>
        </div>

        {/* Mechanism 2 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.8)",
            border: "1px solid rgba(239, 68, 68, 0.4)",
            borderRadius: 20,
            padding: "20px 24px",
            display: "flex",
            alignItems: "center",
            gap: 20,
            transform: `translateY(${Math.max(0, card2)}px)`,
            opacity: interpolate(frame, [25, 40], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div style={{ fontSize: 36, background: "rgba(239, 68, 68, 0.2)", padding: 12, borderRadius: 14 }}>
            🛑
          </div>
          <div>
            <div style={{ fontSize: 22, fontWeight: 800, color: "#F8FAFC" }}>2. Drawdown Barrier Penalty</div>
            <div style={{ fontSize: 16, color: "#94A3B8", marginTop: 4 }}>
              Vượt ngưỡng sụt giảm vốn (ví dụ &gt;5%) sẽ phạt nặng, ép tác tử rút về Cash.
            </div>
          </div>
        </div>

        {/* Mechanism 3 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.8)",
            border: "1px solid rgba(245, 158, 11, 0.4)",
            borderRadius: 20,
            padding: "20px 24px",
            display: "flex",
            alignItems: "center",
            gap: 20,
            transform: `translateY(${Math.max(0, card3)}px)`,
            opacity: interpolate(frame, [35, 50], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div style={{ fontSize: 36, background: "rgba(245, 158, 11, 0.2)", padding: 12, borderRadius: 14 }}>
            🧠
          </div>
          <div>
            <div style={{ fontSize: 22, fontWeight: 800, color: "#F8FAFC" }}>3. Prospect Theory Utility (λ = 2.25)</div>
            <div style={{ fontSize: 16, color: "#94A3B8", marginTop: 4 }}>
              Tâm lý sợ lỗ gấp 2.25 lần mong lãi, triệt tiêu thói quen "đánh bạc" bắt đáy.
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Hook Footer */}
      <div
        style={{
          background: "rgba(56, 189, 248, 0.1)",
          border: "1px solid rgba(56, 189, 248, 0.3)",
          color: "#38BDF8",
          fontSize: 22,
          fontWeight: 700,
          padding: "16px 32px",
          borderRadius: 16,
          textAlign: "center",
          width: "100%",
        }}
      >
        Mục tiêu định lượng cụ thể thế nào? 👇
      </div>
    </div>
  );
};
