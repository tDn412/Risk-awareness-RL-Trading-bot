import React from "react";
import { Composition } from "remotion";
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
      {/* Full 7-Slide Video Presentation */}
      <Composition
        id="MainPresentation"
        component={MainPresentation}
        durationInFrames={1230} // 41 seconds total at 30 fps
        fps={30}
        width={1920}
        height={1080}
      />

      {/* Individual Slide Compositions for Quick Inspection & Slide Export */}
      <Composition
        id="01-SlideTitle"
        component={SlideTitle}
        durationInFrames={150}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="02-SlideProblem"
        component={SlideProblem}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="03-SlidePapers"
        component={SlidePapers}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="04-SlidePatents"
        component={SlidePatents}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="05-SlideProducts"
        component={SlideProducts}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="06-SlideQuestions"
        component={SlideQuestions}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="07-SlideObjectives"
        component={SlideObjectives}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
