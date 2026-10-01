import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

export const Scene4Gaps: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleSpring = spring({ frame, fps, config: { damping: 10, mass: 0.6 }, from: 0, to: 1 });

  const card1 = spring({ frame: frame - 15, fps, config: { damping: 12 }, from: 80, to: 0 });
  const card2 = spring({ frame: frame - 30, fps, config: { damping: 12 }, from: 80, to: 0 });
  const card3 = spring({ frame: frame - 45, fps, config: { damping: 12 }, from: 80, to: 0 });

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
          background: "radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, rgba(8, 11, 17, 0) 70%)",
          filter: "blur(70px)",
        }}
      />

      {/* Top Badge */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          background: "rgba(139, 92, 246, 0.15)",
          border: "2px solid rgba(139, 92, 246, 0.5)",
          padding: "12px 28px",
          borderRadius: 40,
        }}
      >
        <span style={{ fontSize: 24 }}>🔍</span>
        <span
          style={{
            color: "#A78BFA",
            fontSize: 20,
            fontWeight: 800,
            letterSpacing: 2,
            textTransform: "uppercase",
          }}
        >
          KHOẢNG TRỐNG CÔNG NGHỆ
        </span>
      </div>

      {/* Title */}
      <div style={{ textAlign: "center", transform: `scale(${titleSpring})`, opacity: titleSpring }}>
        <div style={{ fontSize: 40, fontWeight: 700, color: "#94A3B8", marginBottom: 12 }}>
          Khảo sát thực nghiệm
        </div>
        <div
          style={{
            fontSize: 66,
            fontWeight: 950,
            lineHeight: 1.15,
            color: "#FFFFFF",
            letterSpacing: -1,
            textTransform: "uppercase",
          }}
        >
          15 GIẢI PHÁP ĐỀU CÓ{" "}
          <span
            style={{
              background: "linear-gradient(135deg, #EC4899, #8B5CF6)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            LỖ HỔNG LỚN!
          </span>
        </div>
      </div>

      {/* 3 Categories Summary Cards */}
      <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 24 }}>
        {/* Papers */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.85)",
            border: "1px solid rgba(59, 130, 246, 0.4)",
            borderRadius: 24,
            padding: 28,
            transform: `translateY(${Math.max(0, card1)}px)`,
            opacity: interpolate(frame, [15, 30], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div style={{ fontSize: 24, fontWeight: 800, color: "#60A5FA" }}>📄 05 Bài Báo Khoa Học Đỉnh Cao</div>
            <div style={{ background: "#2563EB", color: "#FFF", fontSize: 13, fontWeight: 700, padding: "4px 10px", borderRadius: 6 }}>
              Moody, Jiang, Wang...
            </div>
          </div>
          <div style={{ color: "#CBD5E1", fontSize: 18, lineHeight: 1.5 }}>
            • <strong>DSR (Moody):</strong> Phạt cả biến động lãi đột biến.<br />
            • <strong>EIIE (Jiang):</strong> Return thuần túy, Max Drawdown &gt; 60%.<br />
            • <strong>DeepTrader:</strong> Giả định cho phép bán khống (sai luật HOSE).
          </div>
        </div>

        {/* Patents */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.85)",
            border: "1px solid rgba(245, 158, 11, 0.4)",
            borderRadius: 24,
            padding: 28,
            transform: `translateY(${Math.max(0, card2)}px)`,
            opacity: interpolate(frame, [30, 45], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div style={{ fontSize: 24, fontWeight: 800, color: "#FBBF24" }}>💡 05 Bằng Sáng Chế Quốc Tế</div>
            <div style={{ background: "#D97706", color: "#FFF", fontSize: 13, fontWeight: 700, padding: "4px 10px", borderRadius: 6 }}>
              BlackRock, DeepMind, JPMorgan
            </div>
          </div>
          <div style={{ color: "#CBD5E1", fontSize: 18, lineHeight: 1.5 }}>
            • <strong>BlackRock:</strong> Toàn tối ưu hóa lồi tĩnh Markowitz.<br />
            • <strong>JPMorgan LOXM:</strong> Khớp lệnh vi mô, không phân bổ danh mục.<br />
            • Độc quyền, hệ thống đóng, không áp dụng được cho VN30.
          </div>
        </div>

        {/* Products */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.85)",
            border: "1px solid rgba(16, 185, 129, 0.4)",
            borderRadius: 24,
            padding: 28,
            transform: `translateY(${Math.max(0, card3)}px)`,
            opacity: interpolate(frame, [45, 60], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div style={{ fontSize: 24, fontWeight: 800, color: "#34D399" }}>📱 05 Sản Phẩm Thương Mại</div>
            <div style={{ background: "#059669", color: "#FFF", fontSize: 13, fontWeight: 700, padding: "4px 10px", borderRadius: 6 }}>
              FinRL, Qlib, Composer...
            </div>
          </div>
          <div style={{ color: "#CBD5E1", fontSize: 18, lineHeight: 1.5 }}>
            • Cắt lỗ bằng code cứng if-else tách rời quá trình học RL.<br />
            • Chỉ hỗ trợ chứng chỉ quỹ ETF ngoại, không có cổ phiếu VN30!
          </div>
        </div>
      </div>

      {/* Punchline Footer */}
      <div
        style={{
          background: "linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(234, 88, 12, 0.2))",
          border: "2px solid #EF4444",
          color: "#FFFFFF",
          fontSize: 24,
          fontWeight: 800,
          padding: "18px 32px",
          borderRadius: 20,
          textAlign: "center",
          width: "100%",
          boxShadow: "0 10px 30px rgba(239, 68, 68, 0.2)",
        }}
      >
        🎯 Cơ hội nghiên cứu: Tự xây giải pháp riêng cho VN30! 👇
      </div>
    </div>
  );
};
