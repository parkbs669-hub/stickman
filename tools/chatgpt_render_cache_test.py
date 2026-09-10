from __future__ import annotations

import inspect
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2

import tennis_stickman_v12_4_upgraded as main
from stickman_equipment import racket_dimensions

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "examples" / "alcaraz_upgrade_preview_pose_cache.json"
OUT_DIR = ROOT / "renders" / "chatgpt_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW = OUT_DIR / "alcaraz_upgrade3_raw.mp4"
FINAL = OUT_DIR / "alcaraz_upgrade3_browser_safe.mp4"
DIAG = OUT_DIR / "diagnostics.json"

# Linux CI font fallback. This changes text font only, never pose coordinates.
for candidate in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
):
    if Path(candidate).exists():
        main.FONT_PATH = candidate
        break

# The uploaded upgrade.3 currently passes racket_length= into draw_racket(), while
# draw_racket() itself still has the old signature. For the render test only, patch
# that single missing compatibility point using the uploaded stickman_equipment
# dimensions. No joint/pose coordinates are modified.
original_signature = str(inspect.signature(main.draw_racket))
patched_draw_racket = "racket_length" not in inspect.signature(main.draw_racket).parameters

if patched_draw_racket:
    def draw_racket_compat(canvas, wrist, elbow, head_r, nx_racket=None, ny_racket=None,
                           racket_face_ratio=1.0, racket_length=None):
        wx, wy = wrist
        ex, ey = elbow
        if nx_racket is None or ny_racket is None:
            dx, dy = wx - ex, wy - ey
            arm_len = math.hypot(dx, dy) + 1e-6
            nx_racket, ny_racket = dx / arm_len, dy / arm_len

        if racket_length is None:
            # Legacy-only fallback; render_frames in upgrade.3 supplies racket_length.
            racket_length = max(float(head_r) * 3.0, 8.0)
        dims = racket_dimensions(racket_length, racket_face_ratio)
        grip_end = (
            int(round(wx + nx_racket * dims["grip_end"])),
            int(round(wy + ny_racket * dims["grip_end"])),
        )
        hoop_base = (
            int(round(wx + nx_racket * dims["hoop_base"])),
            int(round(wy + ny_racket * dims["hoop_base"])),
        )
        head_center = (
            int(round(wx + nx_racket * dims["center"])),
            int(round(wy + ny_racket * dims["center"])),
        )
        long_r = max(2, int(round(dims["head_long_radius"])))
        short_r = max(2, int(round(dims["head_short_radius"])))
        angle = math.degrees(math.atan2(ny_racket, nx_racket))

        grip_thick = max(int(round(6.0 * main.LW)), 5)
        shaft_thick = max(int(round(2.5 * main.LW)), 2)
        frame_thick = max(int(round(4.0 * main.LW)), 3)
        cv2.line(canvas, (wx, wy), grip_end, main.RACKET_GRIP_COLOR,
                 grip_thick, cv2.LINE_AA)
        cv2.line(canvas, grip_end, hoop_base, (75, 75, 75),
                 shaft_thick, cv2.LINE_AA)
        cv2.ellipse(canvas, head_center, (long_r, short_r), angle, 0, 360,
                    main.RACKET_FRAME_COLOR, frame_thick, cv2.LINE_AA)

        perp_x, perp_y = -ny_racket, nx_racket
        string_thick = max(1, int(round(main.LW * 0.65)))
        for frac in (-0.38, 0.0, 0.38):
            off = short_r * frac
            sx = head_center[0] + perp_x * off
            sy = head_center[1] + perp_y * off
            cv2.line(canvas,
                     (int(round(sx - nx_racket * long_r * .68)),
                      int(round(sy - ny_racket * long_r * .68))),
                     (int(round(sx + nx_racket * long_r * .68)),
                      int(round(sy + ny_racket * long_r * .68))),
                     main.RACKET_STRING_COLOR, string_thick, cv2.LINE_AA)
        for frac in (-0.45, 0.0, 0.45):
            off = long_r * frac
            sx = head_center[0] + nx_racket * off
            sy = head_center[1] + ny_racket * off
            cv2.line(canvas,
                     (int(round(sx - perp_x * short_r * .72)),
                      int(round(sy - perp_y * short_r * .72))),
                     (int(round(sx + perp_x * short_r * .72)),
                      int(round(sy + perp_y * short_r * .72))),
                     main.RACKET_STRING_COLOR, string_thick, cv2.LINE_AA)

    main.draw_racket = draw_racket_compat

with CACHE.open("r", encoding="utf-8") as f:
    raw = json.load(f)

meta = raw["metadata"]
frames = []
for frame in raw["frames"]:
    if frame is None:
        frames.append(None)
        continue
    frames.append([
        SimpleNamespace(x=float(v[0]), y=float(v[1]), z=float(v[2]), v=float(v[3]))
        for v in frame
    ])

fps = float(meta["fps"])
orig_w = int(meta["width"])
orig_h = int(meta["height"])

# Directly exercise upgrade.3 render_frames with the repository's 288-frame
# Alcaraz pose cache. No handcrafted/synthetic motion is used.
width, height, encoded, sources, equipment = main.render_frames(
    actual_input=str(ROOT / "unused_compare_source.mp4"),
    output_path=RAW,
    frames=frames,
    fps=fps,
    orig_w=orig_w,
    orig_h=orig_h,
    height=720,
    ssaa=2,
    speed=1.0,
    is_right_handed=True,
    is_serve=False,
    label=None,
    desc=None,
    strobe=False,
    strobe_frames=32,
    strobe_step=4,
    lag_scale=0.0,
    no_trail=False,
    two_handed=False,
    compare=False,
    impact_points=[],
)

# Conservative browser MP4: H.264 Baseline, no B frames, yuv420p, AAC silent
# audio, fast-start. This avoids the 0:00 player issue seen in chat previews.
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-i", str(RAW),
    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
    "-map", "0:v:0", "-map", "1:a:0",
    "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
    "-pix_fmt", "yuv420p", "-bf", "0", "-r", str(fps),
    "-preset", "medium", "-crf", "19",
    "-c:a", "aac", "-b:a", "96k", "-shortest",
    "-movflags", "+faststart", "-video_track_timescale", "90000",
    str(FINAL),
], check=True)

# Decode entire output; any damaged frame makes the workflow fail.
subprocess.run([
    "ffmpeg", "-v", "error", "-i", str(FINAL), "-f", "null", "-"
], check=True)

probe = subprocess.run([
    "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
    "-show_entries", "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_read_frames:format=duration,size",
    "-of", "json", str(FINAL),
], check=True, capture_output=True, text=True)

result = {
    "repository_version": main.VERSION,
    "cache_frames": len(frames),
    "cache_fps": fps,
    "cache_source_dimensions": [orig_w, orig_h],
    "rendered_frames": encoded,
    "output_dimensions": [width, height],
    "racket_direction_sources": sources,
    "equipment": equipment,
    "uploaded_draw_racket_signature": original_signature,
    "test_only_racket_signature_patch_applied": patched_draw_racket,
    "ffprobe": json.loads(probe.stdout),
}
DIAG.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
