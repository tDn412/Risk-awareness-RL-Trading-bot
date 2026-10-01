import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

export const Scene3HoseRules: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleSpring = spring({ frame, fps, config: { damping: 10, mass: 0.6 }, from: 0, to: 1 });

  const card1 = spring({ frame: frame - 15, fps, config: { damping: 12 }, from: 80, to: 0 });
  const card2 = spring({ frame: frame - 25, fps, config: { damping: 12 }, from: 80, to: 0 });
  const card3 = spring({ frame: frame - 35, fps, config: { damping: 12 }, from: 80, to: 0 });
  const card4 = spring({ frame: frame - 45, fps, config: { damping: 12 }, from: 80, to: 0 });

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
          width: 700,
          height: 700,
          background: "radial-gradient(circle, rgba(16, 185, 129, 0.18) 0%, rgba(8, 11, 17, 0) 70%)",
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
        <span style={{ fontSize: 24 }}>🇻🇳</span>
        <span
          style={{
            color: "#10B981",
            fontSize: 20,
            fontWeight: 800,
            letterSpacing: 2,
            textTransform: "uppercase",
          }}
        >
          ĐẶC THÙ THỊ TRƯỜNG VIỆT NAM
        </span>
      </div>

      {/* Title */}
      <div style={{ textAlign: "center", transform: `scale(${titleSpring})`, opacity: titleSpring }}>
        <div style={{ fontSize: 40, fontWeight: 700, color: "#94A3B8", marginBottom: 12 }}>
          Không thể áp dụng máy móc
        </div>
        <div
          style={{
            fontSize: 70,
            fontWeight: 950,
            lineHeight: 1.15,
            color: "#FFFFFF",
            letterSpacing: -1,
            textTransform: "uppercase",
          }}
        >
          LUẬT CHƠI KHẮC NGHIỆT{" "}
          <span
            style={{
              background: "linear-gradient(135deg, #10B981, #06B6D4)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            SÀN HOSE
          </span>
        </div>
      </div>

      {/* 4 Vietnam Market Constraint Cards */}
      <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 20 }}>
        {/* Card 1 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.8)",
            border: "1px solid rgba(16, 185, 129, 0.3)",
            borderRadius: 20,
            padding: "24px 30px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            transform: `translateY(${Math.max(0, card1)}px)`,
            opacity: interpolate(frame, [15, 30], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ fontSize: 36 }}>🔒</div>
            <div>
              <div style={{ fontSize: 24, fontWeight: 800, color: "#F8FAFC" }}>Chỉ Cho Phép Long-Only</div>
              <div style={{ fontSize: 17, color: "#94A3B8", marginTop: 4 }}>Cấm short-selling ở cơ sở. Chỉ Mua hoặc Cash (w_i ≥ 0).</div>
            </div>
          </div>
          <div style={{ background: "#10B981", color: "#000", fontWeight: 900, padding: "6px 14px", borderRadius: 8, fontSize: 16 }}>
            w_i ≥ 0
          </div>
        </div>

        {/* Card 2 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.8)",
            border: "1px solid rgba(245, 158, 11, 0.3)",
            borderRadius: 20,
            padding: "24px 30px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            transform: `translateY(${Math.max(0, card2)}px)`,
            opacity: interpolate(frame, [25, 40], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ fontSize: 36 }}>⚡</div>
            <div>
              <div style={{ fontSize: 24, fontWeight: 800, color: "#F8FAFC" }}>Biên Độ Trần/Sàn ±7%</div>
              <div style={{ fontSize: 17, color: "#94A3B8", marginTop: 4 }}>Nguy cơ mất thanh khoản khi cổ phiếu rơi sàn liên tục.</div>
            </div>
          </div>
          <div style={{ background: "#F59E0B", color: "#000", fontWeight: 900, padding: "6px 14px", borderRadius: 8, fontSize: 16 }}>
            ±7% / ngày
          </div>
        </div>

        {/* Card 3 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.8)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            borderRadius: 20,
            padding: "24px 30px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            transform: `translateY(${Math.max(0, card3)}px)`,
            opacity: interpolate(frame, [35, 50], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ fontSize: 36 }}>💸</div>
            <div>
              <div style={{ fontSize: 24, fontWeight: 800, color: "#F8FAFC" }}>Thuế Giao Dịch Bán 0.1%</div>
              <div style={{ fontSize: 17, color: "#94A3B8", marginTop: 4 }}>Mỗi lần bán đều mất thuế cố định trên tổng giá trị.</div>
            </div>
          </div>
          <div style={{ background: "#EF4444", color: "#FFF", fontWeight: 900, padding: "6px 14px", borderRadius: 8, fontSize: 16 }}>
            Thuế 0.1%
          </div>
        </div>

        {/* Card 4 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.8)",
            border: "1px solid rgba(139, 92, 246, 0.3)",
            borderRadius: 20,
            padding: "24px 30px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            transform: `translateY(${Math.max(0, card4)}px)`,
            opacity: interpolate(frame, [45, 60], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ fontSize: 36 }}>🔄</div>
            <div>
              <div style={{ fontSize: 24, fontWeight: 800, color: "#F8FAFC" }}>Phí Môi Giới ~0.15%</div>
              <div style={{ fontSize: 17, color: "#94A3B8", marginTop: 4 }}>Tác tử rebalance quá nhiều lần = "Cháy tài khoản vì phí"!</div>
            </div>
          </div>
          <div style={{ background: "#8B5CF6", color: "#FFF", fontWeight: 900, padding: "6px 14px", borderRadius: 8, fontSize: 16 }}>
            Fee 0.15%
          </div>
        </div>
      </div>

      {/* Bottom Hook Footer */}
      <div
        style={{
          background: "rgba(16, 185, 129, 0.1)",
          border: "1px solid rgba(16, 185, 129, 0.3)",
          color: "#34D399",
          fontSize: 22,
          fontWeight: 700,
          padding: "16px 32px",
          borderRadius: 16,
          textAlign: "center",
          width: "100%",
        }}
      >
        Vậy các công trình quốc tế có giải quyết được không? 👇
      </div>
    </div>
  );
};
