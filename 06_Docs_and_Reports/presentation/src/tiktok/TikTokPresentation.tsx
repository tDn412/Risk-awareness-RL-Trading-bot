import React from "react";
import { Sequence } from "remotion";
import { Scene1Hook } from "./Scene1Hook";
import { Scene2RiskBlind } from "./Scene2RiskBlind";
import { Scene3HoseRules } from "./Scene3HoseRules";
import { Scene4Gaps } from "./Scene4Gaps";
import { Scene5Solution } from "./Scene5Solution";
import { Scene6Targets } from "./Scene6Targets";

export const TikTokPresentation: React.FC = () => {
  // TikTok Scene durations at 30 fps
  const DURATION_S1 = 150; // 5.0s Hook
  const DURATION_S2 = 180; // 6.0s Risk-Blindness
  const DURATION_S3 = 180; // 6.0s HOSE Market Rules
  const DURATION_S4 = 210; // 7.0s 15 Solutions & Gaps
  const DURATION_S5 = 210; // 7.0s Antigravity Solution & Shift
  const DURATION_S6 = 210; // 7.0s KPIs & Outro

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
      </Sequence>

      <Sequence from={START_S2} durationInFrames={DURATION_S2} name="02. Risk Blindness">
        <Scene2RiskBlind />
      </Sequence>

      <Sequence from={START_S3} durationInFrames={DURATION_S3} name="03. HOSE Rules">
        <Scene3HoseRules />
      </Sequence>

      <Sequence from={START_S4} durationInFrames={DURATION_S4} name="04. 15 Gaps Survey">
        <Scene4Gaps />
      </Sequence>

      <Sequence from={START_S5} durationInFrames={DURATION_S5} name="05. Antigravity Solution">
        <Scene5Solution />
      </Sequence>

      <Sequence from={START_S6} durationInFrames={DURATION_S6} name="06. KPI Targets & Outro">
        <Scene6Targets />
      </Sequence>
    </div>
  );
};
