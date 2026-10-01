import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Header } from "../components/Header";

export const SlideProducts: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const opacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: "clamp" });
  const translateY = spring({ frame, fps, config: { damping: 15 }, from: 30, to: 0 });

  const products = [
    {
      name: "FinRL (AI4Finance)",
      type: "Open-source RL Library",
      weakness: "Hàm thưởng mặc định là lợi nhuận thuần túy; thiếu cơ chế phạt drawdown tức thời; chưa có môi trường HOSE.",
    },
    {
      name: "Qlib (Microsoft)",
      type: "AI Quant Platform",
      weakness: "Trọng tâm là Supervised Learning (LightGBM); module RL sơ khai; không hỗ trợ hàm thưởng Prospect Theory.",
    },
    {
      name: "QuantConnect (Lean)",
      type: "Algorithmic Cloud",
      weakness: "Quản trị rủi ro là các quy tắc dừng lỗ tĩnh tách rời mô hình học; hạ tầng cloud phụ thuộc độc quyền.",
    },
    {
      name: "Composer.trade",
      type: "No-code Rule Engine",
      weakness: "Quy tắc tĩnh if-else có độ trễ lớn; không có khả năng tự học tối ưu hóa tỷ trọng phi tuyến tính.",
    },
    {
      name: "Wealthfront / Betterment",
      type: "Robo-advisor",
      weakness: "Đầu tư thụ động theo ngưỡng lệch drift; chu kỳ điều chỉnh chậm; chỉ hỗ trợ ETF ngoại, không có cổ phiếu VN30.",
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
        category="Khoảng Trống Thị Trường"
        sectionTitle="05 Sản Phẩm Thực Tế & Khoảng Trống Cốt Lõi (Core Gap)"
      />

      <div style={{ display: "flex", flexDirection: "column", gap: 20, flex: 1 }}>
        {/* 5 Products Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 16 }}>
          {products.map((p, idx) => (
            <div
              key={idx}
              style={{
                background: "rgba(30, 41, 59, 0.5)",
                border: "1px solid rgba(255, 255, 255, 0.1)",
                borderRadius: 12,
                padding: 16,
              }}
            >
              <div style={{ fontSize: 14, fontWeight: 700, color: "#38BDF8", marginBottom: 2 }}>
                {p.name}
              </div>
              <div style={{ fontSize: 11, color: "#94A3B8", marginBottom: 8, textTransform: "uppercase" }}>
                {p.type}
              </div>
              <div style={{ background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.2)", borderRadius: 8, padding: 8 }}>
                <div style={{ fontSize: 10, color: "#EF4444", fontWeight: 700, textTransform: "uppercase" }}>
                  Hạn chế chính
                </div>
                <div style={{ fontSize: 11, color: "#FDA4AF", marginTop: 4, lineHeight: 1.35 }}>
                  {p.weakness}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Core Research Gap Box */}
        <div
          style={{
            background: "linear-gradient(135deg, rgba(30, 58, 138, 0.4), rgba(15, 23, 42, 0.8))",
            border: "1px solid #3B82F6",
            borderRadius: 16,
            padding: 24,
            flex: 1,
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              color: "#60A5FA",
              fontSize: 16,
              fontWeight: 800,
              letterSpacing: 1,
              textTransform: "uppercase",
              marginBottom: 12,
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <span>🎯</span> KHOẢNG TRỐNG CỐT LÕI (CORE RESEARCH GAP) CỦA ĐỀ TÀI
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
            <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: 16, borderRadius: 10, border: "1px solid rgba(255, 255, 255, 0.08)" }}>
              <div style={{ color: "#F59E0B", fontWeight: 700, fontSize: 14, marginBottom: 4 }}>1. Hàm thưởng nhận thức rủi ro</div>
              <div style={{ color: "#CBD5E1", fontSize: 13, lineHeight: 1.4 }}>
                Tích hợp trực tiếp Sortino downside, Drawdown Barrier và Prospect Theory vào thuật toán PPO/SAC trên simplex tỷ trọng.
              </div>
            </div>
            <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: 16, borderRadius: 10, border: "1px solid rgba(255, 255, 255, 0.08)" }}>
              <div style={{ color: "#10B981", fontWeight: 700, fontSize: 14, marginBottom: 4 }}>2. Ràng buộc thực tế sàn HOSE</div>
              <div style={{ color: "#CBD5E1", fontSize: 13, lineHeight: 1.4 }}>
                Mô phỏng đúng danh mục VN30, quy định Long-Only (w_i ≥ 0), biên độ trần/sàn ±7%, thuế bán 0.1% và phí môi giới.
              </div>
            </div>
            <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: 16, borderRadius: 10, border: "1px solid rgba(255, 255, 255, 0.08)" }}>
              <div style={{ color: "#8B5CF6", fontWeight: 700, fontSize: 14, marginBottom: 4 }}>3. Kiểm định đa chế độ thị trường</div>
              <div style={{ color: "#CBD5E1", fontSize: 13, lineHeight: 1.4 }}>
                Đối sánh chặt chẽ qua 3 chu kỳ (Bull 2020-2021, Bear 2022, Sideway 2023-2024) so với VN30-Index, Equal Weight (1/N) và Markowitz MPT.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
