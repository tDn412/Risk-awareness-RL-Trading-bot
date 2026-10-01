import React from "react";

interface HeaderProps {
  sectionNumber?: string;
  sectionTitle: string;
  category: string;
}

export const Header: React.FC<HeaderProps> = ({
  sectionNumber,
  sectionTitle,
  category,
}) => {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        width: "100%",
        paddingBottom: 24,
        marginBottom: 24,
        borderBottom: "1px solid rgba(255, 255, 255, 0.12)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        {sectionNumber && (
          <div
            style={{
              background: "linear-gradient(135deg, #2563EB, #1D4ED8)",
              color: "#FFFFFF",
              fontWeight: 800,
              fontSize: 20,
              padding: "6px 14px",
              borderRadius: 8,
              boxShadow: "0 4px 12px rgba(37, 99, 235, 0.4)",
            }}
          >
            {sectionNumber}
          </div>
        )}
        <div>
          <div
            style={{
              color: "#38BDF8",
              fontSize: 14,
              fontWeight: 600,
              letterSpacing: 1.5,
              textTransform: "uppercase",
              marginBottom: 4,
            }}
          >
            {category}
          </div>
          <div
            style={{
              color: "#F8FAFC",
              fontSize: 32,
              fontWeight: 800,
              letterSpacing: -0.5,
            }}
          >
            {sectionTitle}
          </div>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          background: "rgba(30, 41, 59, 0.7)",
          border: "1px solid rgba(255, 255, 255, 0.1)",
          padding: "8px 16px",
          borderRadius: 20,
        }}
      >
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: "#10B981",
            boxShadow: "0 0 8px #10B981",
          }}
        />
        <span
          style={{
            color: "#94A3B8",
            fontSize: 14,
            fontWeight: 500,
          }}
        >
          VN30 Portfolio Allocation • Risk-Aware RL
        </span>
      </div>
    </div>
  );
};
