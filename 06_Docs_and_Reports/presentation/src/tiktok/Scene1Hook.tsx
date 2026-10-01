import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

export const Scene1Hook: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Entrance spring for main title
  const titleSpring = spring({
    frame,
    fps,
    config: { damping: 10, mass: 0.6, stiffness: 120 },
    from: 0,
    to: 1,
  });

  // Pulse animation for crash badge
  const pulse = Math.sin(frame * 0.2) * 0.05 + 1;

  // Chart drop progress
  const chartDrop = interpolate(frame, [25, 75], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Card slide up
  const cardSlide = spring({
    frame: frame - 45,
    fps,
    config: { damping: 12 },
    from: 80,
    to: 0,
  });

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
      {/* Background glowing gradient orb */}
      <div
        style={{
          position: "absolute",
          top: "20%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 700,
          height: 700,
          background: "radial-gradient(circle, rgba(239, 68, 68, 0.25) 0%, rgba(8, 11, 17, 0) 70%)",
          filter: "blur(60px)",
          pointerEvents: "none",
        }}
      />

      {/* Top Header Tag */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          background: "rgba(239, 68, 68, 0.15)",
          border: "2px solid rgba(239, 68, 68, 0.5)",
          padding: "14px 28px",
          borderRadius: 40,
          transform: `scale(${pulse})`,
        }}
      >
        <span style={{ fontSize: 24 }}>🚨</span>
        <span
          style={{
            color: "#EF4444",
            fontSize: 22,
            fontWeight: 900,
            letterSpacing: 2,
            textTransform: "uppercase",
          }}
        >
          NGHỊCH LÝ GIAO DỊCH AI
        </span>
      </div>

      {/* Massive Punchy Hook Title */}
      <div
        style={{
          textAlign: "center",
          transform: `scale(${titleSpring})`,
          opacity: titleSpring,
        }}
      >
        <div
          style={{
            fontSize: 48,
            fontWeight: 800,
            color: "#94A3B8",
            marginBottom: 16,
            textTransform: "uppercase",
            letterSpacing: 1,
          }}
        >
          Tại sao 90% Bot RL
        </div>
        <div
          style={{
            fontSize: 76,
            fontWeight: 950,
            lineHeight: 1.15,
            color: "#FFFFFF",
            textTransform: "uppercase",
            letterSpacing: -1,
          }}
        >
          CHÁY TÀI KHOẢN KHI{" "}
          <span
            style={{
              background: "linear-gradient(135deg, #EF4444 0%, #F59E0B 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              textShadow: "0 0 40px rgba(239, 68, 68, 0.5)",
            }}
          >
            VN30 SẬP?
          </span>
        </div>
      </div>

      {/* Dynamic Animated Crash Chart Box */}
      <div
        style={{
          width: "100%",
          background: "rgba(18, 24, 38, 0.85)",
          border: "2px solid rgba(239, 68, 68, 0.4)",
          borderRadius: 32,
          padding: 40,
          boxShadow: "0 20px 60px rgba(239, 68, 68, 0.15)",
          transform: `translateY(${Math.max(0, cardSlide)}px)`,
          opacity: interpolate(frame, [30, 45], [0, 1], { extrapolateRight: "clamp" }),
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
          <div>
            <div style={{ color: "#94A3B8", fontSize: 18, fontWeight: 700, textTransform: "uppercase" }}>
              Cú Sập Lịch Sử
            </div>
            <div style={{ color: "#F8FAFC", fontSize: 36, fontWeight: 900, marginTop: 4 }}>
              VN-Index 2022
            </div>
          </div>
          <div
            style={{
              background: "#EF4444",
              color: "#FFFFFF",
              fontSize: 32,
              fontWeight: 900,
              padding: "10px 24px",
              borderRadius: 16,
              boxShadow: "0 8px 24px rgba(239, 68, 68, 0.5)",
            }}
          >
            -42.8%
          </div>
        </div>

        {/* SVG Crash Line */}
        <svg width="100%" height="240" viewBox="0 0 900 240" fill="none">
          <path
            d="M 50 40 Q 250 30 450 120 T 850 220"
            stroke="#EF4444"
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray="900"
            strokeDashoffset={900 * (1 - chartDrop)}
          />
          {/* Highlight Dot */}
          {chartDrop > 0.8 && (
            <circle cx="850" cy="220" r="14" fill="#EF4444" stroke="#FFFFFF" strokeWidth="4" />
          )}
        </svg>

        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16 }}>
          <div style={{ color: "#10B981", fontSize: 24, fontWeight: 800 }}>Đỉnh: 1.528 Điểm</div>
          <div style={{ color: "#EF4444", fontSize: 24, fontWeight: 800 }}>Đáy: 873 Điểm</div>
        </div>
      </div>

      {/* Impact Stat Cards */}
      <div
        style={{
          width: "100%",
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 20,
        }}
      >
        <div
          style={{
            background: "rgba(30, 41, 59, 0.7)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            padding: 24,
            borderRadius: 20,
            textAlign: "center",
          }}
        >
          <div style={{ color: "#94A3B8", fontSize: 16, fontWeight: 600 }}>Cổ phiếu HPG</div>
          <div style={{ color: "#EF4444", fontSize: 38, fontWeight: 900, margin: "6px 0" }}>-72%</div>
          <div style={{ color: "#64748B", fontSize: 14 }}>44.000 ➔ 12.000 VNĐ</div>
        </div>

        <div
          style={{
            background: "rgba(30, 41, 59, 0.7)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            padding: 24,
            borderRadius: 20,
            textAlign: "center",
          }}
        >
          <div style={{ color: "#94A3B8", fontSize: 16, fontWeight: 600 }}>Cổ phiếu SSI</div>
          <div style={{ color: "#EF4444", fontSize: 38, fontWeight: 900, margin: "6px 0" }}>-71%</div>
          <div style={{ color: "#64748B", fontSize: 14 }}>45.000 ➔ 13.000 VNĐ</div>
        </div>
      </div>

      {/* Bottom Hook Footer */}
      <div
        style={{
          background: "rgba(245, 158, 11, 0.1)",
          border: "1px solid rgba(245, 158, 11, 0.3)",
          color: "#FBBF24",
          fontSize: 22,
          fontWeight: 700,
          padding: "16px 32px",
          borderRadius: 16,
          textAlign: "center",
          width: "100%",
        }}
      >
        Tác tử RL tối ưu lợi nhuận thuần túy đều cháy tài khoản! 👇
      </div>
    </div>
  );
};
