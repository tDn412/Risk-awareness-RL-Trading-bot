import React from "react";
import { Sequence } from "remotion";
import { SlideTitle } from "./slides/SlideTitle";
import { SlideProblem } from "./slides/SlideProblem";
import { SlidePapers } from "./slides/SlidePapers";
import { SlidePatents } from "./slides/SlidePatents";
import { SlideProducts } from "./slides/SlideProducts";
import { SlideQuestions } from "./slides/SlideQuestions";
import { SlideObjectives } from "./slides/SlideObjectives";

export const MainPresentation: React.FC = () => {
  // Slide durations in frames (at 30 fps)
  const DURATION_TITLE = 150;      // 5.0s
  const DURATION_PROBLEM = 180;    // 6.0s
  const DURATION_PAPERS = 180;     // 6.0s
  const DURATION_PATENTS = 180;    // 6.0s
  const DURATION_PRODUCTS = 180;   // 6.0s
  const DURATION_QUESTIONS = 180;  // 6.0s
  const DURATION_OBJECTIVES = 180; // 6.0s

  const START_TITLE = 0;
  const START_PROBLEM = START_TITLE + DURATION_TITLE;
  const START_PAPERS = START_PROBLEM + DURATION_PROBLEM;
  const START_PATENTS = START_PAPERS + DURATION_PAPERS;
  const START_PRODUCTS = START_PATENTS + DURATION_PATENTS;
  const START_QUESTIONS = START_PRODUCTS + DURATION_PRODUCTS;
  const START_OBJECTIVES = START_QUESTIONS + DURATION_QUESTIONS;

  return (
    <div
      style={{
        flex: 1,
        backgroundColor: "#0B0F19",
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
      }}
    >
      <Sequence from={START_TITLE} durationInFrames={DURATION_TITLE} name="01. Title & Team">
        <SlideTitle />
      </Sequence>

      <Sequence from={START_PROBLEM} durationInFrames={DURATION_PROBLEM} name="02. Đặt Vấn Đề">
        <SlideProblem />
      </Sequence>

      <Sequence from={START_PAPERS} durationInFrames={DURATION_PAPERS} name="03. 05 Bài Báo Khoa Học">
        <SlidePapers />
      </Sequence>

      <Sequence from={START_PATENTS} durationInFrames={DURATION_PATENTS} name="04. 05 Sáng Chế Kỹ Thuật">
        <SlidePatents />
      </Sequence>

      <Sequence from={START_PRODUCTS} durationInFrames={DURATION_PRODUCTS} name="05. 05 Sản Phẩm & Core Gap">
        <SlideProducts />
      </Sequence>

      <Sequence from={START_QUESTIONS} durationInFrames={DURATION_QUESTIONS} name="06. Câu Hỏi Nghiên Cứu">
        <SlideQuestions />
      </Sequence>

      <Sequence from={START_OBJECTIVES} durationInFrames={DURATION_OBJECTIVES} name="07. Mục Tiêu Nghiên Cứu">
        <SlideObjectives />
      </Sequence>
    </div>
  );
};
