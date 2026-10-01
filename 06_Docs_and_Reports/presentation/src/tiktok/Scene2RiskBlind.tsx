import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

export const Scene2RiskBlind: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleSpring = spring({
    frame,
    fps,
    config: { damping: 10, mass: 0.6 },
    from: 0,
    to: 1,
  });

  const card1 = spring({ frame: frame - 15, fps, config: { damping: 12 }, from: 100, to: 0 });
  const card2 = spring({ frame: frame - 30, fps, config: { damping: 12 }, from: 100, to: 0 });
  const card3 = spring({ frame: frame - 45, fps, config: { damping: 12 }, from: 100, to: 0 });

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
          top: "30%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 600,
          height: 600,
          background: "radial-gradient(circle, rgba(234, 88, 12, 0.2) 0%, rgba(8, 11, 17, 0) 70%)",
          filter: "blur(70px)",
        }}
      />

      {/* Top Badge */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          background: "rgba(234, 88, 12, 0.15)",
          border: "2px solid rgba(234, 88, 12, 0.5)",
          padding: "12px 28px",
          borderRadius: 40,
        }}
      >
        <span style={{ fontSize: 24 }}>🧠</span>
        <span
          style={{
            color: "#FB923C",
            fontSize: 20,
            fontWeight: 800,
            letterSpacing: 2,
            textTransform: "uppercase",
          }}
        >
          NGUYÊN NHÂN GỐC RỄ
        </span>
      </div>

      {/* Main Title */}
      <div
        style={{
          textAlign: "center",
          transform: `scale(${titleSpring})`,
          opacity: titleSpring,
        }}
      >
        <div style={{ fontSize: 44, fontWeight: 700, color: "#94A3B8", marginBottom: 12 }}>
          Căn bệnh kinh niên
        </div>
        <div
          style={{
            fontSize: 72,
            fontWeight: 950,
            color: "#FFFFFF",
            letterSpacing: -1,
            textTransform: "uppercase",
          }}
        >
          <span
            style={{
              background: "linear-gradient(135deg, #F97316, #EF4444)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            MÙ RỦI RO
          </span>{" "}
          (RISK-BLIND)
        </div>
      </div>

      {/* Formula Box with Strike-through */}
      <div
        style={{
          width: "100%",
          background: "rgba(18, 24, 38, 0.9)",
          border: "2px dashed rgba(239, 68, 68, 0.6)",
          borderRadius: 24,
          padding: "28px 40px",
          textAlign: "center",
          position: "relative",
        }}
      >
        <div style={{ color: "#94A3B8", fontSize: 18, marginBottom: 8, fontWeight: 600 }}>
          Hàm thưởng RL truyền thống:
        </div>
        <div
          style={{
            fontSize: 42,
            fontWeight: 900,
            fontFamily: "monospace",
            color: "#EF4444",
            textDecoration: "line-through",
            textDecorationThickness: "4px",
          }}
        >
          Reward = Max Return (r_t)
        </div>
        <div style={{ color: "#FDA4AF", fontSize: 18, marginTop: 8, fontWeight: 600 }}>
          ❌ Bỏ qua hoàn toàn Drawdown & Volatility!
        </div>
      </div>

      {/* 3 Killer Consequence Cards */}
      <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 20 }}>
        {/* Card 1 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.7)",
            border: "1px solid rgba(255, 255, 255, 0.12)",
            borderRadius: 20,
            padding: 24,
            display: "flex",
            alignItems: "center",
            gap: 24,
            transform: `translateY(${Math.max(0, card1)}px)`,
            opacity: interpolate(frame, [15, 30], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div
            style={{
              fontSize: 36,
              background: "rgba(239, 68, 68, 0.2)",
              padding: 16,
              borderRadius: 16,
            }}
          >
            🎲
          </div>
          <div>
            <div style={{ fontSize: 24, fontWeight: 800, color: "#F8FAFC" }}>All-in Cổ Phiếu Beta Cao</div>
            <div style={{ fontSize: 18, color: "#94A3B8", marginTop: 4 }}>
              Dồn toàn bộ tỷ trọng vào mã rủi ro để tối đa điểm thưởng ngắn hạn.
            </div>
          </div>
        </div>

        {/* Card 2 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.7)",
            border: "1px solid rgba(255, 255, 255, 0.12)",
            borderRadius: 20,
            padding: 24,
            display: "flex",
            alignItems: "center",
            gap: 24,
            transform: `translateY(${Math.max(0, card2)}px)`,
            opacity: interpolate(frame, [30, 45], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div
            style={{
              fontSize: 36,
              background: "rgba(245, 158, 11, 0.2)",
              padding: 16,
              borderRadius: 16,
            }}
          >
            🕳️
          </div>
          <div>
            <div style={{ fontSize: 24, fontWeight: 800, color: "#F8FAFC" }}>Bắt Đáy Mù Quáng & Gồng Lỗ</div>
            <div style={{ fontSize: 18, color: "#94A3B8", marginTop: 4 }}>
              Càng giảm càng mua vì kỳ vọng "hồi vốn", dẫn đến cháy tài khoản.
            </div>
          </div>
        </div>

        {/* Card 3 */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.7)",
            border: "1px solid rgba(255, 255, 255, 0.12)",
            borderRadius: 20,
            padding: 24,
            display: "flex",
            alignItems: "center",
            gap: 24,
            transform: `translateY(${Math.max(0, card3)}px)`,
            opacity: interpolate(frame, [45, 60], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div
            style={{
              fontSize: 36,
              background: "rgba(139, 92, 246, 0.2)",
              padding: 16,
              borderRadius: 16,
            }}
          >
            📉
          </div>
          <div>
            <div style={{ fontSize: 24, fontWeight: 800, color: "#F8FAFC" }}>Sập Đổ Khi Chuyển Chế Độ</div>
            <div style={{ fontSize: 18, color: "#94A3B8", marginTop: 4 }}>
              Học tốt ở Bull Market nhưng tê liệt hoàn toàn khi bước vào Bear Market!
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Footer */}
      <div
        style={{
          background: "rgba(59, 130, 246, 0.1)",
          border: "1px solid rgba(59, 130, 246, 0.3)",
          color: "#60A5FA",
          fontSize: 22,
          fontWeight: 700,
          padding: "16px 32px",
          borderRadius: 16,
          textAlign: "center",
          width: "100%",
        }}
      >
        Chưa hết: Thị trường Việt Nam còn có luật chơi riêng! 🇻🇳
      </div>
    </div>
  );
};
