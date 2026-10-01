import React from "react";
import { Composition } from "remotion";
import { TikTokPresentation } from "./tiktok/TikTokPresentation";
import { Scene1Hook } from "./tiktok/Scene1Hook";
import { Scene2RiskBlind } from "./tiktok/Scene2RiskBlind";
import { Scene3HoseRules } from "./tiktok/Scene3HoseRules";
import { Scene4Gaps } from "./tiktok/Scene4Gaps";
import { Scene5Solution } from "./tiktok/Scene5Solution";
import { Scene6Targets } from "./tiktok/Scene6Targets";

// Landscape slide deck
import { MainPresentation } from "./MainPresentation";
import { SlideTitle } from "./slides/SlideTitle";
import { SlideProblem } from "./slides/SlideProblem";
import { SlidePapers } from "./slides/SlidePapers";
import { SlidePatents } from "./slides/SlidePatents";
import { SlideProducts } from "./slides/SlideProducts";
import { SlideQuestions } from "./slides/SlideQuestions";
import { SlideObjectives } from "./slides/SlideObjectives";

export const Root: React.FC = () => {
  return (
    <>
      {/* ========================================================= */}
      {/* 1. TIKTOK / REELS / SHORTS VERTICAL VIDEO (9:16 - 1080x1920) */}
      {/* ========================================================= */}
      <Composition
        id="TikTokPresentation"
        component={TikTokPresentation}
        durationInFrames={1140} // 38.00 sec total at 30 fps
        fps={30}
        width={1080}
        height={1920}
      />

      {/* Individual TikTok Scenes */}
      <Composition
        id="TikTok-01-Hook"
        component={Scene1Hook}
        durationInFrames={150}
        fps={30}
        width={1080}
        height={1920}
      />
      <Composition
        id="TikTok-02-RiskBlind"
        component={Scene2RiskBlind}
        durationInFrames={180}
        fps={30}
        width={1080}
        height={1920}
      />
      <Composition
        id="TikTok-03-HoseRules"
        component={Scene3HoseRules}
        durationInFrames={180}
        fps={30}
        width={1080}
        height={1920}
      />
      <Composition
        id="TikTok-04-Gaps"
        component={Scene4Gaps}
        durationInFrames={210}
        fps={30}
        width={1080}
        height={1920}
      />
      <Composition
        id="TikTok-05-Solution"
        component={Scene5Solution}
        durationInFrames={210}
        fps={30}
        width={1080}
        height={1920}
      />
      <Composition
        id="TikTok-06-Targets"
        component={Scene6Targets}
        durationInFrames={210}
        fps={30}
        width={1080}
        height={1920}
      />

      {/* ========================================================= */}
      {/* 2. LANDSCAPE SLIDE DECK (16:9 - 1920x1080)               */}
      {/* ========================================================= */}
      <Composition
        id="MainPresentation"
        component={MainPresentation}
        durationInFrames={1230}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="SlideTitle"
        component={SlideTitle}
        durationInFrames={150}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="SlideProblem"
        component={SlideProblem}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="SlidePapers"
        component={SlidePapers}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="SlidePatents"
        component={SlidePatents}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="SlideProducts"
        component={SlideProducts}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="SlideQuestions"
        component={SlideQuestions}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="SlideObjectives"
        component={SlideObjectives}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
