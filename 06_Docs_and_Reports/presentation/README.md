# 🎬 Remotion Video & Slide Presentation: VN30 Risk-Aware RL Portfolio Allocation

Dự án Remotion tạo video thuyết trình và bộ slide động cho đề tài **Risk-Aware Reinforcement Learning for VN30 Portfolio Allocation**.

---

## 📂 Cấu Trúc Dự Án

```text
presentation/
├── src/
│   ├── components/
│   │   └── Header.tsx               # Header định danh đề tài và phân mục
│   ├── slides/
│   │   ├── SlideTitle.tsx           # Slide 1: Giới thiệu đề tài & Nhóm nghiên cứu
│   │   ├── SlideProblem.tsx         # Slide 2: 1. Đặt Vấn Đề (Simplex, Ràng buộc HOSE, Sụp đổ 2022)
│   │   ├── SlidePapers.tsx          # Slide 3: 2. Khoảng trống - 05 Bài báo khoa học
│   │   ├── SlidePatents.tsx         # Slide 4: 2. Khoảng trống - 05 Sáng chế kỹ thuật
│   │   ├── SlideProducts.tsx        # Slide 5: 2. Khoảng trống - 05 Sản phẩm & Core Gap
│   │   ├── SlideQuestions.tsx       # Slide 6: 3. Xây dựng 03 Câu hỏi nghiên cứu
│   │   └── SlideObjectives.tsx      # Slide 7: 4. Xây dựng Mục tiêu nghiên cứu (SMART)
│   ├── MainPresentation.tsx         # Xâu chuỗi toàn bộ 7 slide (41 giây video @ 30fps)
│   ├── Root.tsx                     # Đăng ký các Composition chính và từng slide lẻ
│   └── index.ts                     # Entrypoint Remotion
├── out/
│   ├── vn30_risk_aware_rl_presentation.mp4  # Video MP4 Full HD (1920x1080)
│   ├── slide1_title.png             # Ảnh chụp slide 1
│   ├── slide2_problem.png           # Ảnh chụp slide 2
│   ├── slide3_papers.png            # Ảnh chụp slide 3
│   ├── slide4_patents.png           # Ảnh chụp slide 4
│   ├── slide5_products.png          # Ảnh chụp slide 5
│   ├── slide6_questions.png         # Ảnh chụp slide 6
│   └── slide7_objectives.png        # Ảnh chụp slide 7
├── package.json
└── tsconfig.json
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Xem và Điều Khiển Trực Tiếp (Interactive Preview Studio)
Chạy lệnh sau tại thư mục `06_Docs_and_Reports/presentation`:
```bash
npm run dev
```
Trình duyệt sẽ mở giao diện Remotion Preview Studio tại `http://localhost:3000`. Bạn có thể:
- Tua từng khung hình, play/pause video.
- Chuyển đổi giữa video tổng hợp `MainPresentation` hoặc từng slide riêng rẽ (`01-SlideTitle`, `02-SlideProblem`,...).
- Xem và chỉnh sửa code trực tiếp (Hot Reload).

### 2. Xuất Video MP4 (Render Video)
```bash
npm run render
```
Video độ phân giải Full HD (1920x1080) sẽ được xuất ra file `out/vn30_risk_aware_rl_presentation.mp4`.

### 3. Xuất Ảnh Chụp Từng Slide (Export Still Slides)
```bash
npx remotion still src/index.ts 01-SlideTitle out/slide1_title.png
npx remotion still src/index.ts 02-SlideProblem out/slide2_problem.png
npx remotion still src/index.ts 03-SlidePapers out/slide3_papers.png
npx remotion still src/index.ts 04-SlidePatents out/slide4_patents.png
npx remotion still src/index.ts 05-SlideProducts out/slide5_products.png
npx remotion still src/index.ts 06-SlideQuestions out/slide6_questions.png
npx remotion still src/index.ts 07-SlideObjectives out/slide7_objectives.png
```
