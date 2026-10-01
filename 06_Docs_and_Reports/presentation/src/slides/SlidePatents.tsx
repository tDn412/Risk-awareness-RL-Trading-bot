import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Header } from "../components/Header";

export const SlidePatents: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const opacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: "clamp" });
  const translateY = spring({ frame, fps, config: { damping: 15 }, from: 30, to: 0 });

  const patents = [
    {
      id: "US 12,094,005 B2",
      assignee: "BlackRock, Inc. • 17/09/2024",
      inventor: "Stephen Boyd, Mykel Kochenderfer, et al.",
      title: "Simulating Construction and Optimization of Investment Portfolio",
      solution: "Mô phỏng dòng tiền và tối ưu hóa phân bổ tài sản Markowitz với giao diện điều chỉnh trục rủi ro.",
      weakness: "Mô hình tối ưu lồi tĩnh; thiếu khả năng tự thích ứng liên tục của RL; không xử lý biên độ ±7% sàn HOSE.",
    },
    {
      id: "US 10,776,692 B2",
      assignee: "DeepMind Technologies Ltd • 15/09/2020",
      inventor: "Timothy Lillicrap et al.",
      title: "Continuous Control with Deep Reinforcement Learning",
      solution: "Kiến trúc Actor-Critic điều khiển liên tục (DDPG) với Experience Replay và Target Network.",
      weakness: "Thiết kế cho bài toán vật lý dừng; lấy mẫu ngẫu nhiên làm mất mẫu dữ liệu khủng hoảng thanh khoản; thiếu hàm phạt rủi ro đuôi.",
    },
    {
      id: "US 2022/0129925 A1",
      assignee: "JPMorgan Chase Bank, N.A. • 28/04/2022",
      inventor: "Tucker Balch et al. (Framework LOXM)",
      title: "Market Simulation and Calibration Using Reinforcement Learning",
      solution: "Mô phỏng thị trường đa tác tử phục vụ phân tách và thực thi khối lệnh lớn.",
      weakness: "Chuyên biệt hóa cho khớp lệnh vi mô (execution level); không phục vụ phân bổ danh mục trung hạn (portfolio allocation); hệ thống đóng.",
    },
    {
      id: "US 9,940,673 B2",
      assignee: "Hudson Bay Capital Management LP • 10/04/2018",
      inventor: "Sander Gerber",
      title: "Measuring Relationships Between Investments and Other Variables",
      solution: "Đo lường cấu trúc tương quan động giữa các tài sản nhằm giám sát rủi ro danh mục.",
      weakness: "Chỉ phục vụ phân tích dữ liệu thụ động; không có tác tử tự động ra quyết định tỷ trọng; không tích hợp phản hồi thị trường.",
    },
    {
      id: "US 2019/0370649 A1",
      assignee: "USPTO Published Patent Application • 05/12/2019",
      inventor: "K. Zhang et al.",
      title: "Reinforcement Learning Neural Network for Automated Trading",
      solution: "Mạng nơ-ron học tăng cường sâu phát sinh tín hiệu giao dịch mua bán tự động.",
      weakness: "Hàm thưởng chỉ dựa trên lợi nhuận thuần túy; hành động rời rạc (mua/bán đơn lẻ); không hỗ trợ phân bổ tỷ trọng trên simplex đa tài sản.",
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
        category="Khoảng Trống Công Nghệ"
        sectionTitle="Khảo Sát 05 Sáng Chế Kỹ Thuật (Technical Patents)"
      />

      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 16, flex: 1 }}>
        {patents.map((p, idx) => (
          <div
            key={idx}
            style={{
              background: "rgba(30, 41, 59, 0.5)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: 14,
              padding: 18,
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
            }}
          >
            <div>
              <div
                style={{
                  background: "rgba(245, 158, 11, 0.2)",
                  color: "#FBBF24",
                  fontSize: 12,
                  fontWeight: 800,
                  padding: "4px 8px",
                  borderRadius: 6,
                  display: "inline-block",
                  marginBottom: 6,
                }}
              >
                {p.id}
              </div>
              <div style={{ fontSize: 11, color: "#94A3B8", marginBottom: 8 }}>
                {p.assignee}
              </div>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#F8FAFC", marginBottom: 12, lineHeight: 1.3 }}>
                {p.title}
              </div>

              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 11, color: "#10B981", fontWeight: 700, textTransform: "uppercase" }}>
                  Giải pháp kỹ thuật
                </div>
                <div style={{ fontSize: 12, color: "#CBD5E1", marginTop: 4, lineHeight: 1.4 }}>
                  {p.solution}
                </div>
              </div>
            </div>

            <div style={{ background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.2)", borderRadius: 8, padding: 10 }}>
              <div style={{ fontSize: 11, color: "#EF4444", fontWeight: 700, textTransform: "uppercase" }}>
                Điểm yếu & Giới hạn
              </div>
              <div style={{ fontSize: 12, color: "#FDA4AF", marginTop: 4, lineHeight: 1.35 }}>
                {p.weakness}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
