import React from "react";
import { Audio, Sequence, staticFile } from "remotion";
import { Scene1Hook } from "./Scene1Hook";
import { Scene2RiskBlind } from "./Scene2RiskBlind";
import { Scene3HoseRules } from "./Scene3HoseRules";
import { Scene4Gaps } from "./Scene4Gaps";
import { Scene5Solution } from "./Scene5Solution";
import { Scene6Targets } from "./Scene6Targets";

export const TikTokPresentation: React.FC = () => {
  // TikTok Scene durations synchronized with Vietnamese AI Voiceover (at 30 fps)
  const DURATION_S1 = 310; // 10.33s Hook: 90% Bot AI sụp đổ
  const DURATION_S2 = 240; // 8.00s Risk-Blindness (Mù rủi ro)
  const DURATION_S3 = 210; // 7.00s Đặc thù thị trường HOSE
  const DURATION_S4 = 230; // 7.67s 15 Giải pháp & Khoảng trống
  const DURATION_S5 = 260; // 8.67s Giải pháp đề xuất (Simplex, Sortino, Prospect)
  const DURATION_S6 = 220; // 7.33s Mục tiêu SMART & Outro

  const START_S1 = 0;
  const START_S2 = START_S1 + DURATION_S1;
  const START_S3 = START_S2 + DURATION_S2;
  const START_S4 = START_S3 + DURATION_S3;
  const START_S5 = START_S4 + DURATION_S4;
  const START_S6 = START_S5 + DURATION_S5;

  return (
    <div
      style={{
        width: 1080,
        height: 1920,
        backgroundColor: "#080B11",
      }}
    >
      <Sequence from={START_S1} durationInFrames={DURATION_S1} name="01. Hook Crash">
        <Scene1Hook />
        <Audio src={staticFile("audio/scene1.mp3")} />
      </Sequence>

      <Sequence from={START_S2} durationInFrames={DURATION_S2} name="02. Risk Blindness">
        <Scene2RiskBlind />
        <Audio src={staticFile("audio/scene2.mp3")} />
      </Sequence>

      <Sequence from={START_S3} durationInFrames={DURATION_S3} name="03. HOSE Rules">
        <Scene3HoseRules />
        <Audio src={staticFile("audio/scene3.mp3")} />
      </Sequence>

      <Sequence from={START_S4} durationInFrames={DURATION_S4} name="04. 15 Gaps Survey">
        <Scene4Gaps />
        <Audio src={staticFile("audio/scene4.mp3")} />
      </Sequence>

      <Sequence from={START_S5} durationInFrames={DURATION_S5} name="05. Proposed Solution">
        <Scene5Solution />
        <Audio src={staticFile("audio/scene5.mp3")} />
      </Sequence>

      <Sequence from={START_S6} durationInFrames={DURATION_S6} name="06. KPI Targets & Outro">
        <Scene6Targets />
        <Audio src={staticFile("audio/scene6.mp3")} />
      </Sequence>
    </div>
  );
};
