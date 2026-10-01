"""
Vietnamese AI Voiceover Generator for VN30 TikTok Video
Uses edge-tts (Microsoft Neural Voice: vi-VN-NamMinhNeural)
and ffmpeg to generate synchronized voiceover audio for all scenes.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "out"
AUDIO_DIR = BASE_DIR / "public" / "audio"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

VOICE = "vi-VN-NamMinhNeural"  # Professional energetic male voice
RATE = "+15%"  # High-energy TikTok pacing

# Scenes matching TikTokPresentation.tsx timeline (at 30 fps)
SCENES = [
    {
        "id": "scene1",
        "frame_start": 0,
        "start": 0.0,
        "duration_budget": 10.33,
        "text": "Tại sao 90% bot AI đều cháy tài khoản khi VN30 sụp đổ? Năm 2022, VN-Index rơi 43%, cuốn bay toàn bộ danh mục!",
    },
    {
        "id": "scene2",
        "frame_start": 310,
        "start": 310 / 30.0,  # 10.33s
        "duration_budget": 8.0,
        "text": "Điểm yếu cốt lõi là sự thờ ơ với rủi ro. Học tăng cường chỉ chăm chăm tối đa lợi nhuận, khiến bot bắt đáy mù quáng và ôm lỗ nặng!",
    },
    {
        "id": "scene3",
        "frame_start": 550,
        "start": 550 / 30.0,  # 18.33s
        "duration_budget": 7.0,
        "text": "Thị trường Việt Nam có luật chơi rất riêng: chỉ được mua nắm giữ, biên độ trần sàn 7%, thuế phí bào mòn vốn cực nhanh!",
    },
    {
        "id": "scene4",
        "frame_start": 760,
        "start": 760 / 30.0,  # 25.33s
        "duration_budget": 7.67,
        "text": "Khảo sát 15 bài báo, bằng sáng chế và sản phẩm lớn trên thế giới cho thấy: chưa có giải pháp nào giải quyết sụt giảm vốn cho VN30!",
    },
    {
        "id": "scene5",
        "frame_start": 990,
        "start": 990 / 30.0,  # 33.00s
        "duration_budget": 8.67,
        "text": "Vũ khí của chúng tôi là hàm thưởng nhận thức rủi ro: phạt lỗ Sortino, rào cản Drawdown ép rút tiền mặt, và tâm lý sợ lỗ Thuyết Triển Vọng!",
    },
    {
        "id": "scene6",
        "frame_start": 1250,
        "start": 1250 / 30.0,  # 41.67s
        "duration_budget": 7.33,
        "text": "Mục tiêu sống còn: khống chế sụt vốn dưới 15%, Sharpe trên 1.2, cắt giảm hơn một nửa rủi ro so với thị trường!",
    },
]


def generate_scene_audio():
    print(f"[*] Generating voice clips with {VOICE} (rate={RATE})...")
    edge_tts_bin = sys.prefix + "/bin/edge-tts" if os.path.exists(sys.prefix + "/bin/edge-tts") else "edge-tts"

    for s in SCENES:
        clip_path = AUDIO_DIR / f"{s['id']}.mp3"
        print(f"  -> Generating {s['id']}: '{s['text'][:40]}...'")
        
        # Call edge-tts with list arguments (no shell escaping issues)
        cmd = [edge_tts_bin, "--voice", VOICE, f"--rate={RATE}", "--text", s["text"], "--write-media", str(clip_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[-] Error generating {s['id']}: {res.stderr}")
            sys.exit(1)
            
        time.sleep(0.5)  # brief pause to avoid network congestion
        
    print("[+] All scene voice clips generated successfully.")


def get_duration(file_path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def assemble_and_merge():
    print("[*] Assembling synchronized master voiceover track...")
    master_audio = OUT_DIR / "voiceover_master.mp3"
    
    # Build ffmpeg filter_complex with adelay for each clip
    inputs = []
    delays = []
    
    for idx, s in enumerate(SCENES):
        clip = AUDIO_DIR / f"{s['id']}.mp3"
        dur = get_duration(clip)
        print(f"  - Clip {s['id']}: starts at {s['start']:.2f}s, actual audio length = {dur:.2f}s (budget: {s['duration_budget']:.2f}s)")
        inputs.extend(["-i", str(clip)])
        delay_ms = int(s["start"] * 1000)
        delays.append(f"[{idx}]adelay={delay_ms}|{delay_ms}[a{idx}]")

    mix_inputs = "".join([f"[a{i}]" for i in range(len(SCENES))])
    filter_complex = f"{';'.join(delays)};{mix_inputs}amix=inputs={len(SCENES)}:dropout_transition=0,volume=1.4[outa]"
    
    cmd_mix = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex, "-map", "[outa]", "-b:a", "192k", str(master_audio)]
    subprocess.run(cmd_mix, check=True, capture_output=True)
    print(f"[+] Master voiceover track created: {master_audio}")

    # Merge with TikTok video if exists
    video_input = OUT_DIR / "vn30_tiktok_video.mp4"
    final_output = OUT_DIR / "vn30_tiktok_with_voice.mp4"

    if video_input.exists():
        print("[*] Merging voiceover with vn30_tiktok_video.mp4...")
        cmd_merge = [
            "ffmpeg", "-y",
            "-i", str(video_input),
            "-i", str(master_audio),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(final_output)
        ]
        subprocess.run(cmd_merge, check=True, capture_output=True)
        print(f"[+] SUCCESS! Final video with voiceover saved to: {final_output}")
    else:
        print(f"[-] Video file {video_input} not found, skip merging.")


if __name__ == "__main__":
    generate_scene_audio()
    assemble_and_merge()
