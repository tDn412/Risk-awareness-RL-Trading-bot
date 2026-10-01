import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Header } from "../components/Header";

export const SlideProblem: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const opacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: "clamp" });
  const scale = spring({ frame, fps, config: { damping: 15 }, from: 0.96, to: 1 });

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
        transform: `scale(${scale})`,
      }}
    >
      <Header
        sectionNumber="01"
        category="Bài Toán Nghiên Cứu"
        sectionTitle="Đặt Vấn Đề (Problem Statement)"
      />

      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: 32, flex: 1 }}>
        {/* Left Column: Problem Definition & Constraints */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div
            style={{
              background: "rgba(30, 41, 59, 0.5)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: 16,
              padding: 24,
            }}
          >
            <div style={{ color: "#F43F5E", fontSize: 16, fontWeight: 700, textTransform: "uppercase", marginBottom: 8 }}>
              ⚠️ Nghịch lý cốt lõi: Tác tử mù rủi ro (Risk-Blindness)
            </div>
            <div style={{ fontSize: 16, color: "#E2E8F0", lineHeight: 1.6 }}>
              RL tiêu chuẩn chỉ tối ưu hóa tổng lợi nhuận tích lũy (Cumulative Return). Tác tử dồn tỷ trọng vào
              cổ phiếu biến động cao nhằm ăn chênh lệch ngắn hạn, khiến danh mục hứng chịu mức sụt giảm vốn sâu.
            </div>
            <div
              style={{
                marginTop: 12,
                background: "rgba(15, 23, 42, 0.8)",
                padding: "8px 16px",
                borderRadius: 8,
                fontFamily: "monospace",
                color: "#38BDF8",
                fontSize: 15,
              }}
            >
              Simplex Action: ∑ w_i + w_cash = 1.0 (w_i ≥ 0, Long-Only)
            </div>
          </div>

          <div
            style={{
              background: "rgba(30, 41, 59, 0.5)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: 16,
              padding: 24,
            }}
          >
            <div style={{ color: "#38BDF8", fontSize: 16, fontWeight: 700, textTransform: "uppercase", marginBottom: 12 }}>
              🏛️ Ràng buộc đặc thù thị trường chứng khoán Việt Nam (HOSE)
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div style={{ background: "rgba(15, 23, 42, 0.7)", padding: 12, borderRadius: 8 }}>
                <span style={{ color: "#F59E0B", fontWeight: 700 }}>Biên độ ±7%/ngày:</span>
                <span style={{ color: "#94A3B8", fontSize: 14 }}> Giới hạn biến động phiên sàn HOSE.</span>
              </div>
              <div style={{ background: "rgba(15, 23, 42, 0.7)", padding: 12, borderRadius: 8 }}>
                <span style={{ color: "#10B981", fontWeight: 700 }}>Long-Only:</span>
                <span style={{ color: "#94A3B8", fontSize: 14 }}> Không bán khống ở thị trường cơ sở.</span>
              </div>
              <div style={{ background: "rgba(15, 23, 42, 0.7)", padding: 12, borderRadius: 8 }}>
                <span style={{ color: "#EF4444", fontWeight: 700 }}>Thuế bán 0.1%:</span>
                <span style={{ color: "#94A3B8", fontSize: 14 }}> Thuế giao dịch bắt buộc tại Việt Nam.</span>
              </div>
              <div style={{ background: "rgba(15, 23, 42, 0.7)", padding: 12, borderRadius: 8 }}>
                <span style={{ color: "#8B5CF6", fontWeight: 700 }}>Phí ~0.15%:</span>
                <span style={{ color: "#94A3B8", fontSize: 14 }}> Rebalance quá mức gây bào mòn vốn.</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Real-World Evidence & Target Stakeholders */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div
            style={{
              background: "rgba(30, 41, 59, 0.5)",
              border: "1px solid rgba(239, 68, 68, 0.3)",
              borderRadius: 16,
              padding: 24,
            }}
          >
            <div style={{ color: "#EF4444", fontSize: 16, fontWeight: 700, textTransform: "uppercase", marginBottom: 12 }}>
              📉 Bằng chứng thực tế: Các đợt sụp đổ lịch sử
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ borderLeft: "3px solid #EF4444", paddingLeft: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 15, color: "#F8FAFC" }}>VN-Index 2022: 1.528 ➔ 873 (-42.8%)</div>
                <div style={{ fontSize: 13, color: "#94A3B8" }}>Nhiều mã VN30 giảm &gt;60% (HPG, SSI). RL tối đa lợi nhuận mất hơn nửa vốn.</div>
              </div>
              <div style={{ borderLeft: "3px solid #F59E0B", paddingLeft: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 15, color: "#F8FAFC" }}>COVID-19 (03/2020): -33% trong 4 tuần</div>
                <div style={{ fontSize: 13, color: "#94A3B8" }}>Cổ phiếu giảm sàn liên tục; Markowitz trễ pha không kịp rút về tiền mặt.</div>
              </div>
              <div style={{ borderLeft: "3px solid #38BDF8", paddingLeft: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 15, color: "#F8FAFC" }}>Q3/2023 Whipsaw: Giảm 230 điểm</div>
                <div style={{ fontSize: 13, color: "#94A3B8" }}>Bot quy tắc kỹ thuật bị nhiễu, rebalance liên tục làm thuế phí triệt tiêu vốn.</div>
              </div>
            </div>
          </div>

          <div
            style={{
              background: "rgba(30, 41, 59, 0.5)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: 16,
              padding: 20,
            }}
          >
            <div style={{ color: "#10B981", fontSize: 15, fontWeight: 700, textTransform: "uppercase", marginBottom: 8 }}>
              🎯 Đối tượng chịu tác động
            </div>
            <div style={{ fontSize: 14, color: "#CBD5E1", lineHeight: 1.5 }}>
              • <strong>Nhà đầu tư cá nhân:</strong> Thiếu công cụ rebalance tự động, gồng lỗ sâu.<br />
              • <strong>Quỹ vừa & nhỏ:</strong> Bắt buộc kiểm soát trần Drawdown & Sortino theo hợp đồng.<br />
              • <strong>Kỹ sư FinTech / Quant:</strong> Thiếu framework mã nguồn mở chuẩn Gymnasium cho HOSE.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
