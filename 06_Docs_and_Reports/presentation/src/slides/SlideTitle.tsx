import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

export const SlideTitle: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const opacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });
  const translateY = spring({
    frame,
    fps,
    config: { damping: 14 },
    from: 40,
    to: 0,
  });

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        height: "100%",
        padding: "0 80px",
        background: "radial-gradient(circle at 50% 40%, #172554 0%, #0B0F19 80%)",
        textAlign: "center",
        opacity,
        transform: `translateY(${translateY}px)`,
        color: "#FFFFFF",
      }}
    >
      <div
        style={{
          display: "inline-block",
          background: "rgba(37, 99, 235, 0.2)",
          border: "1px solid rgba(59, 130, 246, 0.5)",
          color: "#60A5FA",
          fontSize: 18,
          fontWeight: 700,
          padding: "8px 24px",
          borderRadius: 30,
          letterSpacing: 2,
          textTransform: "uppercase",
          marginBottom: 32,
        }}
      >
        INT3011E • Báo Cáo Giữa Kỳ Đề Tài
      </div>

      <h1
        style={{
          fontSize: 60,
          fontWeight: 900,
          lineHeight: 1.15,
          letterSpacing: -1.5,
          margin: "0 0 24px 0",
          background: "linear-gradient(135deg, #FFFFFF 30%, #93C5FD 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          maxWidth: 1200,
        }}
      >
        Risk-Aware Reinforcement Learning for VN30 Portfolio Allocation
      </h1>

      <p
        style={{
          fontSize: 24,
          color: "#94A3B8",
          maxWidth: 950,
          lineHeight: 1.6,
          margin: "0 0 48px 0",
        }}
      >
        Hệ thống phân bổ danh mục đầu tư rổ VN30 ứng dụng Học tăng cường sâu nhận thức rủi ro,
        kiểm soát sụt giảm vốn và thích ứng với ma sát thị trường Việt Nam
      </p>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          background: "rgba(15, 23, 42, 0.7)",
          border: "1px solid rgba(255, 255, 255, 0.12)",
          padding: "16px 48px",
          borderRadius: 16,
        }}
      >
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 13, color: "#94A3B8", textTransform: "uppercase", fontWeight: 700, letterSpacing: 1.5 }}>
            Thực hiện
          </div>
          <div style={{ fontSize: 22, color: "#38BDF8", fontWeight: 700, marginTop: 4 }}>
            Chúng tôi
          </div>
        </div>
      </div>
    </div>
  );
};
