# Tennis Stickman v12.4 Upgrade — 2026-09-10
"""테니스 영상 → 스틱맨. 실행 예제와 검증 결과는 함께 제공한 README.md 참고.

v12.4의 드로잉을 유지하고 캐시 검증, 배속 오디오, 구간 선택, 추적 누락,
왼손 양손그립, 인코더 오류 처리 및 분석 보고서를 개선한 독립 배포본.
"""
import cv2
import numpy as np
import subprocess
import os
import math
import urllib.request
import argparse
from types import SimpleNamespace
import sys
import shutil
import tempfile
import json
from pathlib import Path
from functools import lru_cache
from stickman_runtime import (validate_options, video_identity, load_pose_cache,
                              save_pose_cache, atomic_write_json,
                              FFmpegVideoWriter, mux_audio)
from stickman_racket import grip_direction, RacketDirectionTracker

VERSION = "12.4-upgrade.2"

# ─────────────────────────────────────────
# CLI 인자 파싱
# ─────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="테니스 스틱맨 v12.4 업그레이드")
    parser.add_argument("url", help="로컬 영상 파일 또는 영상 URL")
    parser.add_argument("name", help="결과 파일 이름 (폴더 경로 제외)")
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--left", action="store_true", help="왼손잡이")
    parser.add_argument("--two-handed", action="store_true", help="양손 백핸드")
    parser.add_argument("--stroke", choices=["auto", "forehand", "backhand", "serve"], default="auto",
                        help="동작 종류. auto는 파일 이름의 serve/서브로 판단")
    parser.add_argument("--label", help="영상 제목 (한글 지원)")
    parser.add_argument("--desc", help="제목 아래 설명")
    parser.add_argument("--speed", type=float, default=1.0, help="배속 0.05~8 (오디오도 함께 변경)")
    parser.add_argument("--strobe", action="store_true", help="다중 잔상")
    parser.add_argument("--strobe-frames", type=int, default=32)
    parser.add_argument("--strobe-step", type=int, default=4)
    parser.add_argument("--lag-scale", type=float, default=0.0)
    parser.add_argument("--no-trail", action="store_true", help="스윙 궤적 숨김")
    parser.add_argument("--audio-impact", action="store_true", help="오디오와 자세를 함께 검증")
    parser.add_argument("--ensemble", action="store_true", help="오디오/자세 앙상블 임팩트 추정")
    parser.add_argument("--ball-track", action="store_true", help="TrackNet 공 추적 (별도 가중치 필요)")
    parser.add_argument("--tracknet-model", default=None)
    parser.add_argument("--impact-frame", type=int, nargs="+", help="선택 구간 내 0부터 시작하는 수동 임팩트 프레임")
    parser.add_argument("--min-impact-gap", type=float, default=1.0, help="자동 임팩트 최소 간격 (초)")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "renders")
    parser.add_argument("--model", type=Path, default=None, help="MediaPipe .task 모델 파일")
    parser.add_argument("--height", type=int, default=720, help="출력 높이: 짝수 240~2160")
    parser.add_argument("--ssaa", type=int, default=2, help="슈퍼샘플링 1~3")
    parser.add_argument("--start", type=float, default=0.0, help="시작 시각 (초)")
    parser.add_argument("--duration", type=float, help="처리할 길이 (초)")
    parser.add_argument("--compare", action="store_true", help="원본과 스틱맨을 나란히 표시")
    parser.add_argument("--no-audio", action="store_true", help="무음 출력")
    parser.add_argument("--no-cache", action="store_true", help="자세 캐시 읽기/쓰기 생략")
    parser.add_argument("--overwrite", action="store_true", help="기존 결과 교체")
    parser.add_argument("--legacy-leg-fix", action="store_true", help="v12.4의 강제 다리 방향/간격 보정 적용")
    args = parser.parse_args(argv)
    try:
        validate_options(args.name, args.speed, args.strobe_frames, args.strobe_step,
                         args.height, args.ssaa, args.start, args.duration, args.min_impact_gap)
        if not math.isfinite(args.lag_scale) or not 0 <= args.lag_scale <= 5:
            raise ValueError("lag-scale은 0~5 사이여야 합니다.")
        if args.impact_frame is not None and any(f < 0 for f in args.impact_frame):
            raise ValueError("임팩트 프레임은 0 이상이어야 합니다.")
    except ValueError as exc:
        parser.error(str(exc))
    return args


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = str(Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task")

SSAA  = 2      # 슈퍼샘플링 배수 (렌더 = 출력 × SSAA)
OUT_H = 720    # 출력 세로 해상도
LW = SSAA

# 전역 손목 각도 및 라켓 상태 캐시 (프레임 간 스무딩)
_racket_offset_prev = None
_racket_face_prev = None

def configure_thickness(render_h):
    """렌더 높이에 맞춰 선 두께 단위와 명명된 두께 상수를 설정."""
    global LW, LIMB_THICKNESS, NECK_THICKNESS, HEAD_OUTLINE_THICKNESS, TORSO_OUTLINE_THICKNESS
    # render_h already includes supersampling; applying SSAA again doubles widths.
    LW = render_h / 720.0
    LIMB_THICKNESS          = max(int(6 * LW), 2)
    NECK_THICKNESS          = max(int(7 * LW), 2)
    HEAD_OUTLINE_THICKNESS  = max(int(8 * LW), 2)
    TORSO_OUTLINE_THICKNESS = max(int(6 * LW), 2)

BODY_COLOR        = (20, 20, 20)
LIMB_THICKNESS    = 6 * SSAA
NECK_THICKNESS    = 7 * SSAA
HEAD_OUTLINE_THICKNESS  = 8 * SSAA
TORSO_OUTLINE_THICKNESS = 6 * SSAA
HEAD_FILL_COLOR   = (255, 255, 255)
UPPER_FILL_COLOR  = (255, 255, 255)  # 흰색 상의 (BGR)
SHORTS_FILL_COLOR = (40, 35, 180)    # 빨간색 반바지 (BGR)
OUTLINE_COLOR     = (20, 20, 20)
SHOE_FILL_COLOR   = (155, 155, 155)
SHOE_OUTLINE_COLOR = (20, 20, 20)
HAND_FILL_COLOR   = (60, 60, 60)
HAND_OUTLINE_COLOR = (20, 20, 20)

RACKET_FRAME_COLOR  = (30, 30, 220)
RACKET_STRING_COLOR = (210, 210, 215)
RACKET_GRIP_COLOR   = (40, 40, 40)

COURT_GREEN         = (78, 115, 76)
SKY_GRADIENT_START  = (215, 215, 215)
SKY_GRADIENT_END    = (238, 238, 238)

ONE_EURO_MIN_CUTOFF = 0.2
ONE_EURO_BETA       = 0.02
ONE_EURO_D_CUTOFF   = 1.0

SCALE_FACTOR = 0.9
OFFSET_X     = 0
OFFSET_Y     = 10

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"  # 한글 자막용

NOSE = 0
L_SHOULDER = 11; R_SHOULDER = 12
L_ELBOW = 13;    R_ELBOW = 14
L_WRIST = 15;    R_WRIST = 16
L_HIP = 23;      R_HIP = 24
L_KNEE = 25;     R_KNEE = 26
L_ANKLE = 27;    R_ANKLE = 28
L_HEEL = 29;     R_HEEL = 30
L_FOOT_INDEX = 31; R_FOOT_INDEX = 32


# ─────────────────────────────────────────
# One-Euro 필터 (모션 스무딩)
# ─────────────────────────────────────────

class OneEuroFilter:
    def __init__(self, freq, min_cutoff, beta, d_cutoff):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        edx = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = edx
        return x_hat


class PoseSmoother:
    def __init__(self, freq, n=33):
        mk = lambda: OneEuroFilter(freq, ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.fx = [mk() for _ in range(n)]
        self.fy = [mk() for _ in range(n)]
        self.fz = [mk() for _ in range(n)]

    def apply(self, raw):
        return [SimpleNamespace(
            x=self.fx[i](raw[i].x),
            y=self.fy[i](raw[i].y),
            z=self.fz[i](raw[i].z),
            v=getattr(raw[i], "v", getattr(raw[i], "visibility", 1.0)),
        ) for i in range(len(raw))]


# ─────────────────────────────────────────
# 텍스트/자막 레이어
# ─────────────────────────────────────────

@lru_cache(maxsize=32)
def load_font(size):
    from PIL import ImageFont
    candidates = [FONT_PATH, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "/System/Library/Fonts/Supplemental/Arial.ttf"]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def draw_label(frame_bgr, label, desc=None):
    from PIL import Image, ImageDraw, ImageFont
    h, w = frame_bgr.shape[:2]
    pad = max(int(h * 0.025), 8)
    f_label = load_font(max(int(h / 13), 18))
    f_desc  = load_font(max(int(h / 26), 12)) if desc else None

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")

    # Portrait panels need width-aware typography as well as height scaling.
    available = max(20, w - 4 * pad)
    def fit(text, font):
        size = getattr(font, "size", 18)
        while size > 8 and draw.textbbox((0, 0), text, font=font)[2] > available:
            size -= 1
            font = load_font(size)
        while text and draw.textbbox((0, 0), text, font=font)[2] > available:
            text = text[:-2] + "…" if len(text) > 2 else ""
        return text, font
    label, f_label = fit(label, f_label)
    if desc:
        desc, f_desc = fit(desc, f_desc)

    lb = draw.textbbox((0, 0), label, font=f_label)
    lw_, lh_ = lb[2] - lb[0], lb[3] - lb[1]
    dw_ = dh_ = 0
    if desc:
        db = draw.textbbox((0, 0), desc, font=f_desc)
        dw_, dh_ = db[2] - db[0], db[3] - db[1]

    bar_w = max(lw_, dw_) + pad * 2
    bar_h = lh_ + (dh_ + pad // 2 if desc else 0) + pad * 2
    x0, y0 = pad, pad
    draw.rounded_rectangle([x0, y0, x0 + bar_w, y0 + bar_h],
                           radius=pad, fill=(20, 20, 20, 150))
    draw.text((x0 + pad, y0 + pad - lb[1]), label, font=f_label, fill=(255, 255, 255, 255))
    if desc:
        draw.text((x0 + pad, y0 + pad + lh_ + pad // 2 - db[1]), desc,
                  font=f_desc, fill=(210, 210, 210, 255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────
# 다운로드 및 파일 핸들링
# ─────────────────────────────────────────

def download_video(video_url, input_path):
    if not video_url.startswith("http"):
        if os.path.exists(video_url):
            print(f"[✓] Using local file: {video_url}")
            return video_url
        print(f"[✗] Local file not found: {video_url}")
        return None

    if os.path.exists(input_path):
        print(f"[✓] Input video already exists: {input_path}")
        return input_path
    print(f"[↓] Downloading video from {video_url} ...")
    try:
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", input_path,
            video_url,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[✓] Video downloaded: {input_path}")
        return input_path
    except Exception as e:
        print(f"[✗] Failed to download video: {e}")
        return None


def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"[✓] MediaPipe model already exists: {MODEL_PATH}")
        return True
    print(f"[↓] Downloading MediaPipe model ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[✓] Model downloaded: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[✗] Failed to download model: {e}")
        return False


# ─────────────────────────────────────────
# 코트 배경 그리기
# ─────────────────────────────────────────

def get_projected_pt(cx, cy, w, h):
    horizon_y = int(h * 0.65)
    screen_y = horizon_y + int(cy * (h - horizon_y))
    perspective_scale = 0.3 + 0.7 * cy
    screen_x = int(w / 2 + cx * (w / 2) * perspective_scale)
    return screen_x, screen_y


def draw_court_background(w, h):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    horizon_y = int(h * 0.65)

    for y in range(horizon_y):
        t = y / max(horizon_y - 1, 1)
        color = tuple(
            int(SKY_GRADIENT_START[c] * (1 - t) + SKY_GRADIENT_END[c] * t)
            for c in range(3)
        )
        bg[y, :] = color

    court_top = np.array(COURT_GREEN, dtype=np.float32)
    court_bot = np.array(COURT_GREEN, dtype=np.float32) * 0.75
    for y in range(horizon_y, h):
        t = (y - horizon_y) / max(h - horizon_y - 1, 1)
        color = tuple(int(court_top[c] * (1 - t) + court_bot[c] * t) for c in range(3))
        bg[y, :] = color

    line_color = (200, 200, 200)
    line_thick = max(1, int(round(LW)))

    cv2.line(bg, get_projected_pt(-0.9, 0.95, w, h), get_projected_pt(0.9, 0.95, w, h), line_color, line_thick, cv2.LINE_AA)
    cv2.line(bg, get_projected_pt(-0.9, 0.55, w, h), get_projected_pt(0.9, 0.55, w, h), line_color, line_thick, cv2.LINE_AA)
    cv2.line(bg, get_projected_pt(0.0, 0.0, w, h),   get_projected_pt(0.0, 0.55, w, h), line_color, line_thick, cv2.LINE_AA)

    for x_norm in [-0.7, 0.7]:
        cv2.line(bg, get_projected_pt(x_norm, 0.0, w, h), get_projected_pt(x_norm, 0.95, w, h), line_color, line_thick, cv2.LINE_AA)
    for x_norm in [-0.9, 0.9]:
        cv2.line(bg, get_projected_pt(x_norm, 0.0, w, h), get_projected_pt(x_norm, 0.95, w, h), line_color, line_thick, cv2.LINE_AA)

    return bg


# ─────────────────────────────────────────
# 스틱맨 부위별 렌더링 함수
# ─────────────────────────────────────────

def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    cx, cy = w / 2, h / 2
    x = cx + (lm.x * w - cx) * SCALE_FACTOR + OFFSET_X * LW
    y = cy + (lm.y * h - cy) * SCALE_FACTOR + OFFSET_Y * LW
    return (int(x), int(y))


def draw_head(canvas, cx, cy, radius):
    cv2.circle(canvas, (cx, cy), radius, HEAD_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius, OUTLINE_COLOR, HEAD_OUTLINE_THICKNESS, cv2.LINE_AA)


def draw_body_line(canvas, p1, p2, thickness=None):
    cv2.line(canvas, p1, p2, BODY_COLOR, thickness or LIMB_THICKNESS, cv2.LINE_AA)


def draw_pentagon_torso(canvas, neck, l_shoulder, r_shoulder, l_hip, r_hip):
    pts = np.array([neck, r_shoulder, r_hip, l_hip, l_shoulder], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], UPPER_FILL_COLOR, cv2.LINE_AA)
    cv2.polylines(canvas, [pts], isClosed=True, color=OUTLINE_COLOR,
                  thickness=TORSO_OUTLINE_THICKNESS, lineType=cv2.LINE_AA)


def draw_shoulder_depth_arc(canvas, l_shoulder, r_shoulder, l_z, r_z, head_r):
    """어깨 z-깊이 타원으로 몸통 앞/뒤 방향 표시.

    MediaPipe z: 값이 작을수록 카메라에 가까움.
    가까운 어깨(near) = 흰 점, 먼 어깨(far) = 어두운 점.
    어깨선 위에 타원 호를 그려 몸통 회전각(깊이감) 표현.
    """
    ls = np.array(l_shoulder, dtype=np.float64)
    rs = np.array(r_shoulder, dtype=np.float64)
    s_vec = rs - ls
    s_len = np.linalg.norm(s_vec)
    if s_len < 4:
        return

    angle_deg  = math.degrees(math.atan2(s_vec[1], s_vec[0]))
    center_pt  = tuple(((ls + rs) / 2).astype(np.int32))
    semi_major = max(int(s_len / 2), 4)

    # z_diff < 0 → L이 가까움(카메라 쪽), z_diff > 0 → R이 가까움
    z_diff = l_z - r_z
    max_depth  = max(s_len * 0.42, 5.0)
    semi_minor = int(np.clip(abs(z_diff) * 2.0 * s_len, 4, max_depth))

    if semi_minor < 4:
        return

    # 타원 호: 위쪽 절반(0→180) 얇게, 아래쪽 절반(180→360) 진하게
    cv2.ellipse(canvas, center_pt, (semi_major, semi_minor),
                angle_deg, 0, 180,
                (160, 160, 160), max(int(1.5 * LW), 1), cv2.LINE_AA)
    cv2.ellipse(canvas, center_pt, (semi_major, semi_minor),
                angle_deg, 180, 360,
                OUTLINE_COLOR, max(int(3 * LW), 3), cv2.LINE_AA)

    # 어깨 끝 점: 가까운 쪽 = 밝은 흰 점, 먼 쪽 = 어두운 회색 점
    dot_r   = max(int(head_r * 0.27), 5)
    if z_diff < 0:   # L이 가까움
        near_pt = tuple(ls.astype(np.int32))
        far_pt  = tuple(rs.astype(np.int32))
    else:            # R이 가까움
        near_pt = tuple(rs.astype(np.int32))
        far_pt  = tuple(ls.astype(np.int32))

    cv2.circle(canvas, far_pt,  dot_r, (65, 65, 65),    -1, cv2.LINE_AA)
    cv2.circle(canvas, far_pt,  dot_r, OUTLINE_COLOR, max(int(2 * LW), 2), cv2.LINE_AA)
    cv2.circle(canvas, near_pt, dot_r, (230, 230, 230), -1, cv2.LINE_AA)
    cv2.circle(canvas, near_pt, dot_r, OUTLINE_COLOR, max(int(2 * LW), 2), cv2.LINE_AA)


def draw_hand(canvas, wrist, radius):
    cv2.circle(canvas, wrist, radius, HAND_FILL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, wrist, radius, HAND_OUTLINE_COLOR, max(int(3 * LW), 3), cv2.LINE_AA)


# 전역 신발 상태 캐시 (왼발/오른발 구분)
_shoe_blend_cache = {"left": None, "right": None}

def _stable_shoe_basis(foot, knee_to_ankle, size, side_key):
    """Filter shoe rotation without moving the ankle or delaying a foot lift.

    A screen-vertical foot has no reliable left/right side profile. Blend it
    into the symmetric front profile before the side normal changes sign;
    otherwise subpixel heel/toe noise can mirror the entire shoe in one frame.
    """
    length = float(np.linalg.norm(foot))
    size = max(float(size), 1e-6)
    previous = _shoe_blend_cache.get(side_key)
    previous = previous if isinstance(previous, dict) else None
    if length > size * 0.08:
        angle = math.atan2(float(foot[1]), float(foot[0]))
    elif previous is not None:
        angle = previous["angle"]
    else:
        angle = math.atan2(float(knee_to_ankle[1]), float(knee_to_ankle[0]))
        if float(np.linalg.norm(knee_to_ankle)) < 1e-6:
            angle = math.pi / 2
    if previous is not None:
        delta = math.atan2(math.sin(angle - previous["angle"]),
                           math.cos(angle - previous["angle"]))
        alpha = 0.35 + 0.45 * min(abs(delta) / math.radians(45), 1.0)
        angle = previous["angle"] + alpha * delta
    direction = np.array([math.cos(angle), math.sin(angle)])
    # The normal reversal is invisible in the symmetric front view.
    side_weight = float(np.clip((abs(direction[0]) - 0.12) / 0.30, 0.0, 1.0))
    side_weight = side_weight * side_weight * (3.0 - 2.0 * side_weight)
    length_weight = float(np.clip((length / size - 0.20) / 0.45, 0.0, 1.0))
    blend = side_weight * length_weight
    if previous is not None:
        blend = previous["blend"] + 0.30 * (blend - previous["blend"])
    # Do not leave a fading side profile visible during a normal reversal or
    # when heel and toe collapse to the same point.
    blend = min(blend, side_weight, length_weight)
    normal = np.array([-direction[1], direction[0]])
    if normal[1] < 0:
        normal = -normal
    _shoe_blend_cache[side_key] = {"angle": angle, "blend": blend}
    return direction, normal, blend


def draw_shoe(canvas, ankle, heel, toe, knee, size, is_back_view, side_key="left"):
    global _shoe_blend_cache

    # 왼발(밝은 회색)·오른발(어두운 회색)로 구분
    shoe_fill = (210, 210, 210) if side_key == "left" else (90, 90, 90)

    ankle = np.array(ankle, dtype=np.float64)
    heel  = np.array(heel,  dtype=np.float64)
    toe   = np.array(toe,   dtype=np.float64)
    knee  = np.array(knee,  dtype=np.float64)

    foot = toe - heel
    direction, side_normal, blend_val = _stable_shoe_basis(
        foot, ankle - knee, size, side_key)

    # 헬퍼 렌더러 정의
    def render_front_view(target_canvas):
        d_dir = direction
        w_dir = np.array([-d_dir[1], d_dir[0]]) * 0.70

        if is_back_view:
            profile = [
                (-0.22, 0.00), (-0.32, 0.20), (-0.40, 0.60), (-0.20, 0.65),
                ( 0.20, 0.65), ( 0.40, 0.60), ( 0.32, 0.20), ( 0.22, 0.00),
            ]
            pts = np.array([
                (ankle + w_dir * fx * size + d_dir * fy * size)
                for fx, fy in profile
            ], dtype=np.int32)

            cv2.fillPoly(target_canvas, [pts], shoe_fill, cv2.LINE_AA)
            cv2.polylines(target_canvas, [pts], isClosed=True, color=SHOE_OUTLINE_COLOR,
                          thickness=max(int(3 * LW), 2), lineType=cv2.LINE_AA)

            sole = np.array([
                (ankle - w_dir * (size * 0.40) + d_dir * (size * 0.60)),
                (ankle - w_dir * (size * 0.20) + d_dir * (size * 0.65)),
                (ankle + w_dir * (size * 0.20) + d_dir * (size * 0.65)),
                (ankle + w_dir * (size * 0.40) + d_dir * (size * 0.60))
            ], dtype=np.int32)
            cv2.polylines(target_canvas, [sole], isClosed=False, color=(110, 110, 110),
                          thickness=max(int(2 * LW), 2), lineType=cv2.LINE_AA)

            heel_strip_start = (ankle + d_dir * (size * 0.05)).astype(np.int32)
            heel_strip_end = (ankle + d_dir * (size * 0.22)).astype(np.int32)
            cv2.line(target_canvas, heel_strip_start, heel_strip_end, SHOE_OUTLINE_COLOR,
                     max(int(2.5 * LW), 2), cv2.LINE_AA)
        else:
            profile = [
                (-0.22, 0.00), (-0.35, 0.25), (-0.42, 0.70), (-0.20, 0.76),
                ( 0.20, 0.76), ( 0.42, 0.70), ( 0.35, 0.25), ( 0.22, 0.00),
            ]
            pts = np.array([
                (ankle + w_dir * fx * size + d_dir * fy * size)
                for fx, fy in profile
            ], dtype=np.int32)

            cv2.fillPoly(target_canvas, [pts], shoe_fill, cv2.LINE_AA)
            cv2.polylines(target_canvas, [pts], isClosed=True, color=SHOE_OUTLINE_COLOR,
                          thickness=max(int(3 * LW), 2), lineType=cv2.LINE_AA)

            sole = np.array([
                (ankle - w_dir * (size * 0.42) + d_dir * (size * 0.70)),
                (ankle - w_dir * (size * 0.20) + d_dir * (size * 0.76)),
                (ankle + w_dir * (size * 0.20) + d_dir * (size * 0.76)),
                (ankle + w_dir * (size * 0.42) + d_dir * (size * 0.70))
            ], dtype=np.int32)
            cv2.polylines(target_canvas, [sole], isClosed=False, color=(110, 110, 110),
                          thickness=max(int(2 * LW), 2), lineType=cv2.LINE_AA)

            lace_start = (ankle + d_dir * (size * 0.12))
            lace_end = (ankle + d_dir * (size * 0.45))
            cv2.line(target_canvas, lace_start.astype(np.int32), lace_end.astype(np.int32),
                     SHOE_OUTLINE_COLOR, max(int(1 * LW), 1), cv2.LINE_AA)

            for frac in [0.20, 0.30, 0.40]:
                bar_center = ankle + d_dir * (size * frac)
                bar_left = (bar_center - w_dir * (size * 0.12)).astype(np.int32)
                bar_right = (bar_center + w_dir * (size * 0.12)).astype(np.int32)
                cv2.line(target_canvas, bar_left, bar_right, (255, 255, 255),
                         max(int(1 * LW), 1), cv2.LINE_AA)

            toe_cap_center = ankle + d_dir * (size * 0.52)
            toe_cap_left = (toe_cap_center - w_dir * (size * 0.32)).astype(np.int32)
            toe_cap_right = (toe_cap_center + w_dir * (size * 0.32)).astype(np.int32)
            cv2.line(target_canvas, toe_cap_left, toe_cap_right, SHOE_OUTLINE_COLOR,
                     max(int(1.2 * LW), 2), cv2.LINE_AA)

    def render_side_view(target_canvas):
        u = direction
        perp = side_normal

        length = size * 1.15
        height = size * 0.62

        profile = [
            (-0.12, 0.00), (-0.22, 0.20), (-0.24, 0.50), (-0.18, 0.78),
            (-0.06, 0.94), ( 0.18, 1.00), ( 0.48, 1.00), ( 0.72, 0.97),
            ( 0.90, 0.88), ( 1.00, 0.72), ( 1.02, 0.54), ( 0.96, 0.36),
            ( 0.82, 0.24), ( 0.60, 0.16), ( 0.38, 0.11), ( 0.18, 0.06),
        ]
        pts = np.array([
            (ankle + u * fx * length + perp * fy * height)
            for fx, fy in profile
        ], dtype=np.int32)

        cv2.fillPoly(target_canvas, [pts], SHOE_FILL_COLOR, cv2.LINE_AA)
        cv2.polylines(target_canvas, [pts], isClosed=True, color=SHOE_OUTLINE_COLOR,
                      thickness=max(int(3 * LW), 2), lineType=cv2.LINE_AA)

        sole = np.array([
            (ankle + u * fx * length + perp * fy * height)
            for fx, fy in [(-0.06, 0.94), (0.18, 1.00), (0.48, 1.00), (0.72, 0.97)]
        ], dtype=np.int32)
        cv2.polylines(target_canvas, [sole], isClosed=False, color=(110, 110, 110),
                      thickness=max(int(2 * LW), 2), lineType=cv2.LINE_AA)

    # 블렌딩 렌더링 적용 (임시 캔버스 활용)
    if blend_val <= 0.001:
        render_front_view(canvas)
    elif blend_val >= 0.999:
        render_side_view(canvas)
    else:
        canvas_front = canvas.copy()
        canvas_side = canvas.copy()
        render_front_view(canvas_front)
        render_side_view(canvas_side)
        cv2.addWeighted(canvas_front, 1.0 - blend_val, canvas_side, blend_val, 0, dst=canvas)


def draw_shadow(canvas, l_ankle, r_ankle):
    cx = (l_ankle[0] + r_ankle[0]) // 2
    cy = max(l_ankle[1], r_ankle[1]) + int(6 * LW)
    spread = max(abs(l_ankle[0] - r_ankle[0]), int(40 * LW))
    overlay = canvas.copy()
    cv2.ellipse(overlay, (cx, cy), (int(spread * 0.7), max(int(spread * 0.1), int(6 * LW))),
                0, 0, 360, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0, canvas)


def draw_shorts(canvas, l_hip, r_hip, l_knee, r_knee, head_r):
    """
    PGNC 반바지 실제 외곽선 실루엣 기반 렌더링 (v11)
    - 정규화 좌표: 반바지 이미지에서 추출한 10-point 외곽선
    - 허리선(l_hip~r_hip) + 밑단(무릎 45% 지점) 기준으로 변환
    """
    l_hip_arr   = np.array(l_hip,   dtype=np.float64)
    r_hip_arr   = np.array(r_hip,   dtype=np.float64)
    l_knee_arr  = np.array(l_knee,  dtype=np.float64)
    r_knee_arr  = np.array(r_knee,  dtype=np.float64)

    # 반바지 밑단: 엉덩이~무릎의 45% 지점
    l_bottom = l_hip_arr + 0.45 * (l_knee_arr - l_hip_arr)
    r_bottom = r_hip_arr + 0.45 * (r_knee_arr - r_hip_arr)

    # 반바지 박스의 4개 꼭짓점 정의
    # 왼쪽 허리 → 오른쪽 허리 → 오른쪽 밑단 → 왼쪽 밑단
    # 정규화 좌표계: x(0=왼쪽, 1=오른쪽), y(0=허리, 1=밑단)
    # 이미지에서 추출한 10-point 외곽선 (PGNC 반바지)
    # 순서: 좌상단 → 좌하단 → 가랑이 → 우하단 → 우상단 → 허리 중앙
    SHORTS_NORM_PTS = np.array([
        [0.2029, 0.003 ],  # 0: 왼쪽 허리선 안쪽
        [0.0692, 0.4012],  # 1: 왼쪽 옆선 중간
        [0.0048, 0.8743],  # 2: 왼쪽 밑단 끝
        [0.4511, 0.997 ],  # 3: 왼쪽 가랑이 밑단
        [0.5036, 0.7904],  # 4: 가랑이 중앙 (오목)
        [0.5561, 0.988 ],  # 5: 오른쪽 가랑이 밑단
        [0.9976, 0.8743],  # 6: 오른쪽 밑단 끝
        [0.9379, 0.4521],  # 7: 오른쪽 옆선 중간
        [0.7947, 0.0   ],  # 8: 오른쪽 허리선 안쪽
        [0.5012, 0.0778],  # 9: 허리 중앙 (고무밴드)
    ], dtype=np.float64)

    # 좌표 변환: 정규화(0~1) → 스크린 픽셀
    # x축: l_hip(x=0) ~ r_hip(x=1) 방향 벡터
    # y축: hip(y=0) ~ bottom(y=1) 방향 벡터
    hip_vec   = r_hip_arr - l_hip_arr          # 허리 방향 벡터 (x축)
    # 왼/오른 각각 다리 방향이 다를 수 있어 평균 사용
    l_leg_vec = l_bottom - l_hip_arr
    r_leg_vec = r_bottom - r_hip_arr

    pts_screen = []
    for nx, ny in SHORTS_NORM_PTS:
        # 허리선 위의 점: l_hip + nx * (r_hip - l_hip)
        waist_pt = l_hip_arr + nx * hip_vec
        # 다리 방향: 왼쪽(nx<0.5)은 l_leg_vec, 오른쪽은 r_leg_vec 가중 블렌딩
        leg_vec  = (1.0 - nx) * l_leg_vec + nx * r_leg_vec
        # 최종 스크린 좌표
        pt = waist_pt + ny * leg_vec
        pts_screen.append(pt)

    pts_screen = np.array(pts_screen, dtype=np.int32)

    # 1. 빨간색 반바지 채우기
    cv2.fillPoly(canvas, [pts_screen], SHORTS_FILL_COLOR, cv2.LINE_AA)

    # 2. 허리 밴드 (약간 밝은 선)
    waist_left  = pts_screen[0]
    waist_right = pts_screen[8]
    waist_mid   = pts_screen[9]
    band_color  = (55, 50, 205)
    band_thick  = max(int(LW * 1.5), 2)
    cv2.line(canvas, tuple(waist_left), tuple(waist_mid),   band_color, band_thick, cv2.LINE_AA)
    cv2.line(canvas, tuple(waist_mid),  tuple(waist_right), band_color, band_thick, cv2.LINE_AA)

    # 3. 가운데 주름선 (가랑이 위 → 허리 중앙)
    crease_top    = pts_screen[9]                      # 허리 중앙
    crease_bottom = pts_screen[4]                      # 가랑이 오목점
    crease_color  = (25, 20, 130)
    crease_thick  = max(1, int(round(LW * 0.8)))
    cv2.line(canvas, tuple(crease_top), tuple(crease_bottom), crease_color, crease_thick, cv2.LINE_AA)

    # 4. 외곽선
    cv2.polylines(canvas, [pts_screen], isClosed=True,
                  color=OUTLINE_COLOR, thickness=TORSO_OUTLINE_THICKNESS, lineType=cv2.LINE_AA)


def draw_racket(canvas, wrist, elbow, head_r, nx_racket=None, ny_racket=None, racket_face_ratio=1.0):
    wx, wy = wrist
    ex, ey = elbow
    
    if nx_racket is None or ny_racket is None:
        dx, dy = wx - ex, wy - ey
        arm_len = math.sqrt(dx * dx + dy * dy) + 1e-6
        nx_racket, ny_racket = dx / arm_len, dy / arm_len

    grip_length = int(head_r * 1.2)
    frame_rx = int(head_r * 1.15 * racket_face_ratio)
    frame_ry = int(head_r * 1.5)

    grip_end_x = int(wx + nx_racket * grip_length)
    grip_end_y = int(wy + ny_racket * grip_length)
    cv2.line(canvas, (wx, wy), (grip_end_x, grip_end_y),
             RACKET_GRIP_COLOR, max(int(8 * LW), 6), cv2.LINE_AA)

    head_cx = int(grip_end_x + nx_racket * frame_ry)
    head_cy = int(grip_end_y + ny_racket * frame_ry)
    angle = math.degrees(math.atan2(ny_racket, nx_racket))

    # 90도 회전 버그 수정: frame_ry를 장축, frame_rx를 단축으로 대입
    cv2.ellipse(canvas, (head_cx, head_cy), (frame_ry, frame_rx),
                angle, 0, 360, RACKET_FRAME_COLOR, max(int(7 * LW), 5), cv2.LINE_AA)

    string_thick = max(1, int(round(LW)))
    perp_x, perp_y = -ny_racket, nx_racket
    for frac in [-0.3, 0.0, 0.3]:
        sx = int(head_cx + perp_x * frame_rx * frac * 0.8)
        sy = int(head_cy + perp_y * frame_rx * frac * 0.8)
        cv2.line(canvas,
                 (int(sx - nx_racket * frame_ry * 0.6), int(sy - ny_racket * frame_ry * 0.6)),
                 (int(sx + nx_racket * frame_ry * 0.6), int(sy + ny_racket * frame_ry * 0.6)),
                 RACKET_STRING_COLOR, string_thick, cv2.LINE_AA)
    for frac in [-0.3, 0.0, 0.3]:
        sx = int(head_cx + nx_racket * frame_ry * frac * 0.8)
        sy = int(head_cy + ny_racket * frame_ry * frac * 0.8)
        cv2.line(canvas,
                 (int(sx - perp_x * frame_rx * 0.6), int(sy - perp_y * frame_rx * 0.6)),
                 (int(sx + perp_x * frame_rx * 0.6), int(sy + perp_y * frame_rx * 0.6)),
                 RACKET_STRING_COLOR, string_thick, cv2.LINE_AA)


# ─────────────────────────────────────────
# 생체역학 및 잔상 지원 렌더러
# ─────────────────────────────────────────

class PhaseSmoother:
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.history = []
        
    def add_and_get(self, phase):
        self.history.append(phase)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return max(set(self.history), key=self.history.count)


def calculate_angle_2d(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def draw_angle_overlay(canvas, joint, p1, p2, color):
    angle_val = calculate_angle_2d(p1, joint, p2)
    v1 = np.array(p1) - np.array(joint)
    v2 = np.array(p2) - np.array(joint)
    ang1 = math.atan2(v1[1], v1[0])
    ang2 = math.atan2(v2[1], v2[0])
    deg1 = int(np.degrees(ang1))
    deg2 = int(np.degrees(ang2))
    
    diff = (deg2 - deg1) % 360
    if diff > 180:
        start_angle = deg2
        end_angle = deg1 + 360
    else:
        start_angle = deg1
        end_angle = deg2
        
    overlay = canvas.copy()
    radius = int(13 * LW)
    
    cv2.ellipse(overlay, joint, (radius, radius), 0, start_angle, end_angle, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
    
    cv2.ellipse(canvas, joint, (radius, radius), 0, start_angle, end_angle, color, max(1, int(1.2 * LW)), cv2.LINE_AA)
    
    bisector_deg = (start_angle + end_angle) / 2
    bisector_rad = np.radians(bisector_deg)
    
    text_dist = radius + int(8 * LW)
    tx = int(joint[0] + text_dist * np.cos(bisector_rad))
    ty = int(joint[1] + text_dist * np.sin(bisector_rad))
    
    text = f"{int(round(angle_val))}°"
    rgb_color = (color[2], color[1], color[0])
    return (text, (tx, ty), rgb_color)


def draw_texts_pil(canvas, tasks, font_size):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.truetype(FONT_PATH, int(round(font_size)))
    
    for text, (x, y), color in tasks:
        tb = draw.textbbox((0, 0), text, font=font)
        outline_color = (20, 20, 20, 255)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy - tb[1]), text, font=font, fill=outline_color)
        draw.text((x, y - tb[1]), text, font=font, fill=color + (255,))
        
    canvas_new = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    np.copyto(canvas, canvas_new)


def draw_glowing_trail(canvas, trail, color):
    if len(trail) < 2:
        return
    n = len(trail)
    overlay = canvas.copy()
    for i in range(1, n):
        t = i / (n - 1)
        thick = max(1, int(10 * LW * t))
        cv2.line(overlay, trail[i-1], trail[i], color, thick, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    
    overlay_core = canvas.copy()
    for i in range(1, n):
        t = i / (n - 1)
        thick = max(1, int(3 * LW * t))
        cv2.line(overlay_core, trail[i-1], trail[i], (255, 255, 255), thick, cv2.LINE_AA)
    cv2.addWeighted(overlay_core, 0.65, canvas, 0.35, 0, canvas)


def detect_serve_phase(lm, is_right_handed, elbow_angle):
    h_wrist = lm[R_WRIST] if is_right_handed else lm[L_WRIST]
    h_shoulder = lm[R_SHOULDER] if is_right_handed else lm[L_SHOULDER]
    h_elbow = lm[R_ELBOW] if is_right_handed else lm[L_ELBOW]
    nh_wrist = lm[L_WRIST] if is_right_handed else lm[R_WRIST]
    nh_shoulder = lm[L_SHOULDER] if is_right_handed else lm[R_SHOULDER]
    nose = lm[NOSE]
    
    if h_wrist.y < nose.y and elbow_angle > 155:
        return "Impact"
    if h_elbow.y < h_shoulder.y + 0.05 and elbow_angle < 95 and h_wrist.y > h_elbow.y:
        return "Racket Drop"
    if nh_wrist.y < nh_shoulder.y - 0.05 and elbow_angle < 130 and elbow_angle > 60:
        return "Trophy Pose"
    if is_right_handed:
        if h_wrist.y > h_shoulder.y and h_wrist.x < lm[L_SHOULDER].x:
            return "Follow Through"
    else:
        if h_wrist.y > h_shoulder.y and h_wrist.x > lm[R_SHOULDER].x:
            return "Follow Through"
    return "Preparation"


def detect_groundstroke_phase(lm, is_right_handed, elbow_angle):
    h_wrist = lm[R_WRIST] if is_right_handed else lm[L_WRIST]
    h_shoulder = lm[R_SHOULDER] if is_right_handed else lm[L_SHOULDER]
    nh_shoulder = lm[L_SHOULDER] if is_right_handed else lm[R_SHOULDER]
    h_hip = lm[R_HIP] if is_right_handed else lm[L_HIP]
    
    def dist_2d(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
    d_opp_shoulder = dist_2d(h_wrist, nh_shoulder)
    
    if d_opp_shoulder < 0.18 and h_wrist.y < h_shoulder.y + 0.05:
        return "Follow Through"
    if elbow_angle > 135 and h_wrist.y > h_shoulder.y - 0.05 and h_wrist.y < h_hip.y + 0.1:
        return "Impact"
    if d_opp_shoulder > 0.35:
        return "Take Back"
    return "Preparation"


def draw_phase_badge(canvas, phase_name, w, h):
    if not phase_name:
        return canvas
    from PIL import Image, ImageDraw, ImageFont
    pad = max(int(h * 0.02), 6)
    font = ImageFont.truetype(FONT_PATH, max(int(h / 24), 14))
    
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    
    text = f"Phase: {phase_name}"
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    
    x1 = w - tw - pad * 3
    y1 = pad
    x2 = w - pad
    y2 = y1 + th + pad * 2
    
    border_color = (200, 200, 200, 255)
    if "Trophy" in phase_name:
        border_color = (255, 165, 0, 255)
    elif "Drop" in phase_name or "Take" in phase_name:
        border_color = (230, 50, 255, 255)
    elif "Impact" in phase_name:
        border_color = (50, 255, 50, 255)
    elif "Follow" in phase_name:
        border_color = (50, 180, 255, 255)
    elif "Preparation" in phase_name:
        border_color = (180, 180, 180, 255)
        
    draw.rounded_rectangle([x1, y1, x2, y2], radius=pad, fill=(20, 20, 20, 180),
                           outline=border_color, width=max(1, int(1.5 * LW)))
    draw.text((x1 + pad * 1.5, y1 + pad - tb[1]), text, font=font, fill=(255, 255, 255, 255))
    
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────
# 잔상 그리기 헬퍼 함수
# ─────────────────────────────────────────

def draw_ghost_figure(canvas, state, is_right_handed):
    # 각 관절 좌표 추출
    l_shoulder = state["l_shoulder"]
    r_shoulder = state["r_shoulder"]
    l_elbow    = state["l_elbow"]
    r_elbow    = state["r_elbow"]
    l_wrist    = state["l_wrist"]
    r_wrist    = state["r_wrist"]
    
    nx_racket  = state["nx_racket"]
    ny_racket  = state["ny_racket"]
    racket_face_ratio = state["racket_face_ratio"]
    head_r     = state["head_r"]
    hand_r     = state["hand_r"]
    
    joint_r = max(LIMB_THICKNESS // 2, int(2 * LW))
    
    # 오른손잡이이면 오른팔과 라켓, 왼손잡이이면 왼팔과 라켓만 잔상으로 그림
    h_shoulder = r_shoulder if is_right_handed else l_shoulder
    h_elbow = r_elbow if is_right_handed else l_elbow
    h_wrist = r_wrist if is_right_handed else l_wrist
    
    # 팔 그리기 (어깨-팔꿈치-손목)
    draw_body_line(canvas, h_shoulder, h_elbow)
    cv2.circle(canvas, h_elbow, joint_r, BODY_COLOR, -1, cv2.LINE_AA)
    draw_body_line(canvas, h_elbow, h_wrist)
    draw_hand(canvas, h_wrist, hand_r)
    
    # 라켓 그리기
    draw_racket(canvas, h_wrist, h_elbow, head_r, nx_racket, ny_racket, racket_face_ratio)


# ─────────────────────────────────────────
# 메인 스틱맨 그리기 함수
# ─────────────────────────────────────────

def get_grip_points(l_wrist, r_wrist, nx_racket, ny_racket, grip_length,
                    is_right_handed=True, two_handed=False):
    """Place the non-dominant hand on the grip for either handedness."""
    if not two_handed:
        return l_wrist, r_wrist
    dominant = r_wrist if is_right_handed else l_wrist
    other = (int(dominant[0] + nx_racket * grip_length * 0.65),
             int(dominant[1] + ny_racket * grip_length * 0.65))
    return (other, r_wrist) if is_right_handed else (l_wrist, other)


def draw_stickman(canvas, landmarks, w, h, is_right_handed=True, racket_trail=None, hand_trail=None,
                  phase_smoother=None, is_serve=True, k=9999, strobe_history=None, strobe_frames=32, strobe_step=4, lag_scale=1.0, two_handed=False,
                  racket_direction=None):
    lm = landmarks
    l_shoulder = get_point(lm, L_SHOULDER, w, h)
    r_shoulder = get_point(lm, R_SHOULDER, w, h)
    l_elbow    = get_point(lm, L_ELBOW, w, h)
    r_elbow    = get_point(lm, R_ELBOW, w, h)
    l_wrist    = get_point(lm, L_WRIST, w, h)
    r_wrist    = get_point(lm, R_WRIST, w, h)
    l_hip      = get_point(lm, L_HIP, w, h)
    r_hip      = get_point(lm, R_HIP, w, h)
    l_knee     = get_point(lm, L_KNEE, w, h)
    r_knee     = get_point(lm, R_KNEE, w, h)
    l_ankle    = get_point(lm, L_ANKLE, w, h)
    r_ankle    = get_point(lm, R_ANKLE, w, h)
    l_heel     = get_point(lm, L_HEEL, w, h)
    r_heel     = get_point(lm, R_HEEL, w, h)
    l_foot_idx = get_point(lm, L_FOOT_INDEX, w, h)
    r_foot_idx = get_point(lm, R_FOOT_INDEX, w, h)

    neck    = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)
    mid_hip = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)

    shoulder_width = math.sqrt(
        (l_shoulder[0] - r_shoulder[0]) ** 2 + (l_shoulder[1] - r_shoulder[1]) ** 2
    )
    head_r  = max(int(shoulder_width * 0.55), int(16 * LW))
    hand_r  = max(int(head_r * 0.26), int(6 * LW))

    neck_v = np.array(neck, dtype=np.float64)
    spine  = neck_v - np.array(mid_hip, dtype=np.float64)
    sn = np.linalg.norm(spine)
    up = spine / sn if sn > 1e-3 else np.array([0.0, -1.0])
    neck_len = int(head_r * 0.5)
    head_center = neck_v + up * (neck_len + head_r)
    head_cx, head_cy = int(head_center[0]), int(head_center[1])
    head_bottom = (head_center - up * head_r).astype(int)

    is_back_view = lm[R_SHOULDER].x > lm[L_SHOULDER].x

    # ── 생체역학 손목 각도 및 라켓 방향 계산 ──
    wx, wy = r_wrist if is_right_handed else l_wrist
    ex, ey = r_elbow if is_right_handed else l_elbow
    dx, dy = wx - ex, wy - ey
    arm_len = math.sqrt(dx * dx + dy * dy) + 1e-6
    nx, ny = dx / arm_len, dy / arm_len
    forearm_angle = math.atan2(ny, nx)
    if racket_direction is None:
        racket_direction = grip_direction(lm, w, h, is_right_handed)
    grip_angle = (math.atan2(racket_direction[1], racket_direction[0])
                  if racket_direction is not None else forearm_angle)

    # 손목 래그(Wrist Lag) 및 와이퍼 프로네이션 각도 오프셋 정의
    global _racket_offset_prev
    if is_serve:
        target_offset = 0.0
    else:
        if -45 < k < 40:
            if k < -15:
                # 준비 자세: 오프셋 없음
                target_offset = 0.0
            elif k < -10:
                # 레이백 진입
                frac = (k + 15) / 5.0
                target_offset = -1.2 * frac
            elif k < -2:
                # 테이크백 및 드롭 (최대 레이백)
                frac = (k + 10) / 8.0
                target_offset = -1.2 * (1.0 - frac) + -2.8 * frac
            elif k <= 0:
                # 임팩트 전 스냅 스윙 (가속 단계)
                frac = (k + 2) / 2.0
                target_offset = -2.8 * (1.0 - frac) + -0.6 * frac
            elif k <= 6:
                # 임팩트 후 와이퍼 프로네이션 감아올리기
                frac = (k / 6.0)
                target_offset = -0.6 * (1.0 - frac) + 1.0 * frac
            elif k < 30:
                # 천천히 감쇠
                frac = (k - 6) / 24.0
                target_offset = 1.0 * (1.0 - frac)
            else:
                target_offset = 0.0
        else:
            target_offset = 0.0

    # Multiply target_offset by lag_scale
    target_offset_scaled = target_offset * lag_scale

    if _racket_offset_prev is None:
        _racket_offset_prev = target_offset_scaled
    else:
        # 가속 및 임팩트 스냅 구간(-2~8프레임)에는 스냅 응답성을 최대화하기 위해 필터 지연 최소화
        alpha_offset = 0.75 if -2 <= k <= 8 else 0.25
        _racket_offset_prev = _racket_offset_prev + alpha_offset * (target_offset_scaled - _racket_offset_prev)

    pronation_angle = grip_angle + (_racket_offset_prev if is_right_handed else -_racket_offset_prev)
    nx_racket = math.cos(pronation_angle)
    ny_racket = math.sin(pronation_angle)

    # ── 3D 라켓 헤드 모핑 제어 (가로세로 비율 조절) ──
    global _racket_face_prev
    if is_serve:
        target_factor = 1.0
    else:
        if -45 < k < 40:
            if k < -6:
                target_factor = 1.0
            elif k < 0:
                # 임팩트 직전 닫힘 (Drop 엣지온)
                frac = (k + 6) / 6.0
                target_factor = 1.0 * (1.0 - frac) + (0.20 / 1.15) * frac
            elif k < 6:
                # 임팩트 후 프로네이션 회전 (엣지온으로 전개)
                target_factor = (0.20 + 0.35 * (k / 6.0)) / 1.15
            elif k < 30:
                target_factor = 0.55 / 1.15
            else:
                # 다시 중립 준비 상태로 완만하게 회복
                frac = (k - 30) / 10.0
                target_factor = (0.55 / 1.15) * (1.0 - frac) + 1.0 * frac
        else:
            target_factor = 1.0

    if _racket_face_prev is None:
        _racket_face_prev = target_factor
    else:
        alpha_face = 0.12
        _racket_face_prev = _racket_face_prev + alpha_face * (target_factor - _racket_face_prev)

    # ── 다중 잔상(Stroboscopic Ghosting) 렌더링 ──
    current_state = {
        "l_shoulder": l_shoulder, "r_shoulder": r_shoulder,
        "l_elbow": l_elbow, "r_elbow": r_elbow,
        "l_wrist": l_wrist, "r_wrist": r_wrist,
        "l_hip": l_hip, "r_hip": r_hip,
        "l_knee": l_knee, "r_knee": r_knee,
        "l_ankle": l_ankle, "r_ankle": r_ankle,
        "l_heel": l_heel, "r_heel": r_heel,
        "l_foot_idx": l_foot_idx, "r_foot_idx": r_foot_idx,
        "nx_racket": nx_racket, "ny_racket": ny_racket,
        "racket_face_ratio": _racket_face_prev,
        "head_r": head_r, "hand_r": hand_r,
        "is_back_view": is_back_view
    }

    if strobe_history is not None:
        # 임팩트 이후(k > 0)에는 이전 잔상이 나타나지 않도록 히스토리를 초기화
        if 0 < k < 9999:
            strobe_history.clear()
        n_hist = len(strobe_history)
        if n_hist >= strobe_step:
            alpha_base = 0.25  # 가장 최신 잔상의 투명도
            for idx in range(0, n_hist, strobe_step):
                # 오래될수록 흐려지는 그라데이션
                alpha = 0.05 + (alpha_base - 0.05) * (idx / max(n_hist - 1, 1))
                overlay = canvas.copy()
                draw_ghost_figure(overlay, strobe_history[idx], is_right_handed)
                cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

        strobe_history.append(current_state)
        if len(strobe_history) > strobe_frames:
            strobe_history.pop(0)

    # ── 네온 스윙 궤적 계산 및 그리기 ──
    racket_wrist = r_wrist if is_right_handed else l_wrist
    grip_length = int(head_r * 1.2)
    frame_ry = int(head_r * 1.5)
    grip_end_x = int(racket_wrist[0] + nx_racket * grip_length)
    grip_end_y = int(racket_wrist[1] + ny_racket * grip_length)
    head_cx_racket = int(grip_end_x + nx_racket * frame_ry)
    head_cy_racket = int(grip_end_y + ny_racket * frame_ry)
    racket_center = (head_cx_racket, head_cy_racket)

    if racket_trail is not None:
        racket_trail.append(racket_center)
        if len(racket_trail) > 20:
            racket_trail.pop(0)
        draw_glowing_trail(canvas, racket_trail, color=(0, 220, 255))  # 네온 옐로우 (BGR)
        
    if hand_trail is not None:
        hand_trail.append(racket_wrist)
        if len(hand_trail) > 20:
            hand_trail.pop(0)
        draw_glowing_trail(canvas, hand_trail, color=(255, 150, 50))   # 네온 시안 (BGR)

    # ── 1. 맨 밑바탕: 그림자 그리기 ──
    draw_shadow(canvas, l_ankle, r_ankle)

    # ── 2. 다리 그리기 ──
    draw_body_line(canvas, l_hip, l_knee)
    draw_body_line(canvas, l_knee, l_ankle)
    draw_body_line(canvas, r_hip, r_knee)
    draw_body_line(canvas, r_knee, r_ankle)
    joint_r = max(LIMB_THICKNESS // 2, int(2 * LW))
    cv2.circle(canvas, l_knee, joint_r, BODY_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, r_knee, joint_r, BODY_COLOR, -1, cv2.LINE_AA)

    # ── 3. Z-depth 기준 레이어 렌더링 ──
    # 양손 그립 모드: 왼손을 라켓 그립 상단(오른손 위)에 고정
    grip_length_px = int(head_r * 1.2)
    l_wrist_render, r_wrist_render = get_grip_points(
        l_wrist, r_wrist, nx_racket, ny_racket, grip_length_px,
        is_right_handed=is_right_handed, two_handed=two_handed,
    )

    def draw_l_upper_arm():
        draw_body_line(canvas, l_shoulder, l_elbow)
        cv2.circle(canvas, l_elbow, joint_r, BODY_COLOR, -1, cv2.LINE_AA)

    def draw_l_forearm():
        draw_body_line(canvas, l_elbow, l_wrist_render)
        draw_hand(canvas, l_wrist_render, hand_r)
        if not is_right_handed:
            draw_racket(canvas, l_wrist_render, l_elbow, head_r, nx_racket, ny_racket, _racket_face_prev)

    def draw_r_upper_arm():
        draw_body_line(canvas, r_shoulder, r_elbow)
        cv2.circle(canvas, r_elbow, joint_r, BODY_COLOR, -1, cv2.LINE_AA)

    def draw_r_forearm():
        draw_body_line(canvas, r_elbow, r_wrist_render)
        draw_hand(canvas, r_wrist_render, hand_r)
        if is_right_handed:
            draw_racket(canvas, r_wrist_render, r_elbow, head_r, nx_racket, ny_racket, _racket_face_prev)

    def draw_trunk():
        draw_shorts(canvas, l_hip, r_hip, l_knee, r_knee, head_r)
        draw_pentagon_torso(canvas, neck, l_shoulder, r_shoulder, l_hip, r_hip)
        draw_shoulder_depth_arc(canvas, l_shoulder, r_shoulder,
                                lm[L_SHOULDER].z, lm[R_SHOULDER].z, head_r)
        draw_body_line(canvas, neck, tuple(head_bottom), thickness=NECK_THICKNESS)
        draw_head(canvas, head_cx, head_cy, head_r)

    l_upper_z = max(lm[L_SHOULDER].z, lm[L_ELBOW].z)
    l_forearm_z = lm[L_ELBOW].z
    r_upper_z = max(lm[R_SHOULDER].z, lm[R_ELBOW].z)
    r_forearm_z = lm[R_ELBOW].z
    
    if is_back_view:
        trunk_z = min(l_upper_z, l_forearm_z, r_upper_z, r_forearm_z) - 0.1
    else:
        trunk_z = max(l_upper_z, l_forearm_z, r_upper_z, r_forearm_z) + 0.1

    # 동작 단계(Phase) 감지를 미리 수행하여 Z-depth 보정에 사용
    elbow_joint = r_elbow if is_right_handed else l_elbow
    elbow_p1 = r_shoulder if is_right_handed else l_shoulder
    elbow_p2 = r_wrist if is_right_handed else l_wrist
    
    h_elbow_angle = calculate_angle_2d(elbow_p1, elbow_joint, elbow_p2)
    if is_serve:
        raw_phase = detect_serve_phase(lm, is_right_handed, h_elbow_angle)
    else:
        raw_phase = detect_groundstroke_phase(lm, is_right_handed, h_elbow_angle)
        
    if phase_smoother is not None:
        current_phase = phase_smoother.add_and_get(raw_phase)
    else:
        current_phase = raw_phase

    # 타격 팔이 몸/머리 뒤로 넘어갔는지 검사
    # 방법 1: 기하학적 조건 (서브 팔로우스루 등, 팔이 목 반대편으로 넘어갈 때)
    is_arm_behind_geom = False
    if is_right_handed:
        if r_wrist[0] < neck[0] and r_elbow[1] < r_shoulder[1] + int(30 * LW):
            is_arm_behind_geom = True
    else:
        if l_wrist[0] > neck[0] and l_elbow[1] < l_shoulder[1] + int(30 * LW):
            is_arm_behind_geom = True

    # 방법 2: MediaPipe z-depth 기반 감지 (백핸드 팔로우스루 후 몸 뒤로 넘어가는 경우)
    # z가 양수일수록 카메라에서 멀리 있음. 팔 평균 z가 상체 z보다 유의미하게 크면 뒤에 있는 것
    torso_z_mid = (lm[L_SHOULDER].z + lm[R_SHOULDER].z) / 2.0
    if is_right_handed:
        arm_z_avg = (lm[R_WRIST].z + lm[R_ELBOW].z) / 2.0
    else:
        arm_z_avg = (lm[L_WRIST].z + lm[L_ELBOW].z) / 2.0
    is_arm_behind_z = arm_z_avg > torso_z_mid + 0.10

    is_arm_behind = is_arm_behind_geom or is_arm_behind_z

    h_wrist_y = r_wrist[1] if is_right_handed else l_wrist[1]
    is_occluded = False
    if (is_serve and current_phase == "Racket Drop") or (is_arm_behind_geom and is_arm_behind_z):
        if h_wrist_y > head_cy:
            is_occluded = True
    elif is_arm_behind_z:
        # z-depth 감지의 경우: 손목 높이와 무관하게 오클루전 적용 (백핸드 팔로우스루 포함)
        is_occluded = True

    if is_occluded:
        # 양손 백핸드처럼 양팔이 함께 뒤로 넘어가는 경우를 위해 두 팔 전체 최솟값 기준으로 몸통 우선 렌더링
        trunk_z = min(l_upper_z, l_forearm_z, r_upper_z, r_forearm_z) - 0.1

    draw_tasks = [
        (l_upper_z, draw_l_upper_arm),
        (l_forearm_z, draw_l_forearm),
        (r_upper_z, draw_r_upper_arm),
        (r_forearm_z, draw_r_forearm),
        (trunk_z, draw_trunk)
    ]
    draw_tasks.sort(key=lambda x: x[0], reverse=True)
    for depth, draw_func in draw_tasks:
        draw_func()

    # ── 4. 신발 그리기 ──
    draw_shoe(canvas, l_ankle, l_heel, l_foot_idx, l_knee, head_r, is_back_view, side_key="left")
    draw_shoe(canvas, r_ankle, r_heel, r_foot_idx, r_knee, head_r, is_back_view, side_key="right")

    # ── 5. 관절 각도 계산 및 오버레이 그리기 ──
    draw_angle_tasks = []
    
    elbow_z = lm[R_ELBOW].z if is_right_handed else lm[L_ELBOW].z
    
    if elbow_z <= trunk_z:
        task1 = draw_angle_overlay(canvas, elbow_joint, elbow_p1, elbow_p2, color=(0, 165, 255))
        draw_angle_tasks.append(task1)
    
    task2 = draw_angle_overlay(canvas, l_knee, l_hip, l_ankle, color=(50, 220, 50))
    task3 = draw_angle_overlay(canvas, r_knee, r_hip, r_ankle, color=(50, 220, 50))
    draw_angle_tasks.extend([task2, task3])

    draw_texts_pil(canvas, draw_angle_tasks, font_size=max(int(10 * LW), 10))

    return current_phase


def correct_leg_swaps(landmarks, prev_landmarks):
    if prev_landmarks is None:
        return landmarks
    pairs = [
        (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
    ]
    dist_no_swap = 0.0
    for l_idx, r_idx in pairs:
        pl = prev_landmarks[l_idx]
        pr = prev_landmarks[r_idx]
        cl = landmarks[l_idx]
        cr = landmarks[r_idx]
        dist_no_swap += (cl.x - pl.x)**2 + (cl.y - pl.y)**2
        dist_no_swap += (cr.x - pr.x)**2 + (cr.y - pr.y)**2

    dist_swap = 0.0
    for l_idx, r_idx in pairs:
        pl = prev_landmarks[l_idx]
        pr = prev_landmarks[r_idx]
        cl = landmarks[l_idx]
        cr = landmarks[r_idx]
        dist_swap += (cl.x - pr.x)**2 + (cl.y - pr.y)**2
        dist_swap += (cr.x - pl.x)**2 + (cr.y - pl.y)**2

    if dist_swap < dist_no_swap and (dist_no_swap - dist_swap) > 0.015:
        for l_idx, r_idx in pairs:
            landmarks[l_idx], landmarks[r_idx] = landmarks[r_idx], landmarks[l_idx]

    return landmarks


def correct_leg_orientation_global(all_smoothed_landmarks):
    """발목 방향 다수결로 다리 랜드마크 좌우를 교정.

    발목 L<R 비율로 기준 방향 결정 후, 기준과 다르고 차이가 임계값 이상인
    프레임만 다리 쌍(23-32) 좌우 교환.
    THRESHOLD: 두 발목이 충분히 벌어진 경우에만 교정해 점프 중 미세 교차 오탐 방지.
    경계 전후 스무딩은 이후 ±LEG_WIN 이동평균 안정화 단계에서 처리.
    """
    L_ANKLE, R_ANKLE = 27, 28
    LEG_PAIRS = [(23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
    THRESHOLD = 0.03  # 발목 차이 최소 임계값 (이하는 교정 안 함)

    valid = [lms for lms in all_smoothed_landmarks if lms is not None]
    if not valid:
        return all_smoothed_landmarks
    back_count = sum(1 for lms in valid if lms[L_ANKLE].x < lms[R_ANKLE].x)
    expected_back = back_count / len(valid) > 0.5

    corrected = 0
    for lms in all_smoothed_landmarks:
        if lms is None:
            continue
        diff = lms[L_ANKLE].x - lms[R_ANKLE].x
        ankle_back = diff < 0
        if ankle_back != expected_back and abs(diff) >= THRESHOLD:
            for l_idx, r_idx in LEG_PAIRS:
                l, r = lms[l_idx], lms[r_idx]
                lms[l_idx] = SimpleNamespace(x=r.x, y=r.y, z=r.z, v=getattr(r, 'v', 1.0))
                lms[r_idx] = SimpleNamespace(x=l.x, y=l.y, z=l.z, v=getattr(l, 'v', 1.0))
            corrected += 1

    direction = "등향(L<R)" if expected_back else "정면향(L>R)"
    print(f"[다리 방향 교정] 기준={direction}, {corrected}프레임 교정 (임계값 {THRESHOLD})")
    return all_smoothed_landmarks


# ─────────────────────────────────────────
# TrackNet 딥러닝 공 추적 (v10 신규)
# ─────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _ConvBlock(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        import torch.nn as nn
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch)
        )
    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module if _TORCH_AVAILABLE else object):
    """TrackNet — 3프레임 9ch 입력 → 공 위치 히트맵"""
    def __init__(self, out_channels=256):
        super().__init__()
        import torch.nn as nn
        self.out_channels = out_channels
        self.conv1  = _ConvBlock(9, 64);   self.conv2  = _ConvBlock(64, 64)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.conv3  = _ConvBlock(64, 128); self.conv4  = _ConvBlock(128, 128)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.conv5  = _ConvBlock(128, 256); self.conv6 = _ConvBlock(256, 256); self.conv7 = _ConvBlock(256, 256)
        self.pool3  = nn.MaxPool2d(2, 2)
        self.conv8  = _ConvBlock(256, 512); self.conv9 = _ConvBlock(512, 512); self.conv10 = _ConvBlock(512, 512)
        self.ups1   = nn.Upsample(scale_factor=2)
        self.conv11 = _ConvBlock(512, 256); self.conv12 = _ConvBlock(256, 256); self.conv13 = _ConvBlock(256, 256)
        self.ups2   = nn.Upsample(scale_factor=2)
        self.conv14 = _ConvBlock(256, 128); self.conv15 = _ConvBlock(128, 128)
        self.ups3   = nn.Upsample(scale_factor=2)
        self.conv16 = _ConvBlock(128, 64);  self.conv17 = _ConvBlock(64, 64)
        self.conv18 = _ConvBlock(64, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, testing=False):
        b = x.size(0)
        x = self.pool1(self.conv2(self.conv1(x)))
        x = self.pool2(self.conv4(self.conv3(x)))
        x = self.pool3(self.conv7(self.conv6(self.conv5(x))))
        x = self.conv10(self.conv9(self.conv8(x)))
        x = self.conv13(self.conv12(self.conv11(self.ups1(x))))
        x = self.conv15(self.conv14(self.ups2(x)))
        x = self.conv18(self.conv17(self.conv16(self.ups3(x))))
        out = x.reshape(b, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out


def _tracknet_postprocess(feature_map, w=640):
    """히트맵 argmax 배열 → (x, y) 0~1 정규화 좌표"""
    h = feature_map.shape[0] // w
    values = np.asarray(feature_map)
    if np.issubdtype(values.dtype, np.floating) and values.size and values.max() <= 1.0:
        values = values * 255.0
    fm = np.clip(values, 0, 255).reshape(h, w).astype(np.uint8)
    _, hm = cv2.threshold(fm, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(hm, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                               param1=50, param2=2, minRadius=2, maxRadius=7)
    if circles is not None and len(circles) == 1:
        return circles[0][0][0] / w, circles[0][0][1] / h  # 정규화 좌표
    return None, None


def detect_impacts_from_tracknet(video_path, fps, n_frames, model_path, min_gap_sec=1.5):
    """TrackNet으로 공 추적 → x방향 속도 반전 = 임팩트"""
    if not _TORCH_AVAILABLE:
        print("[!] PyTorch 미설치 — pip install torch 실행 후 재시도")
        return []

    import torch
    print(f"[i] TrackNet 공 추적 분석 중... (model: {model_path})")

    device = 'cpu'
    model = BallTrackerNet()
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    except Exception as e:
        print(f"[!] 모델 로드 실패: {e}")
        return []
    model.to(device).eval()

    from collections import deque
    cap = cv2.VideoCapture(video_path)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Exact training aspect, dimensions divisible by all three pooling stages.
    TW, TH = 640, 360
    recent = deque(maxlen=3)
    ball_track = []
    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                recent.append(cv2.resize(frame, (TW, TH)))
                if len(recent) < 3:
                    ball_track.append((None, None))
                    continue
                imgs = np.concatenate(list(reversed(recent)), axis=2).astype(np.float32) / 255.0
                inp = torch.from_numpy(np.rollaxis(imgs, 2, 0)[None]).float()
                out = model(inp, testing=True)
                feat = out.argmax(dim=1).detach().cpu().numpy()[0]
                xn, yn = _tracknet_postprocess(feat, w=TW)
                ball_track.append((xn * actual_w, yn * actual_h) if xn is not None else (None, None))
                if len(ball_track) % 100 == 0:
                    print(f"  TrackNet: {len(ball_track)}/{n_frames}", flush=True)
    finally:
        cap.release()
    n_frames = len(ball_track)
    if n_frames < 3:
        return []

    det = sum(1 for p in ball_track if p[0] is not None)
    print(f"[i] 공 감지 프레임: {det}/{n_frames} ({det/n_frames*100:.1f}%)")

    if det < n_frames * 0.03:
        print("[!] 공 감지율 3% 미만")
        return []

    # x방향 속도 (window=3)
    win = 3
    vx = [None] * len(ball_track)
    for i in range(win, len(ball_track) - win):
        pa, pb = ball_track[i - win], ball_track[i + win]
        if pa[0] is not None and pb[0] is not None:
            vx[i] = pb[0] - pa[0]

    # 속도 반전 = 임팩트
    min_gap = int(min_gap_sec * fps)
    impact_points = []
    for i in range(win, len(vx) - 1):
        if vx[i] is None or vx[i + 1] is None:
            continue
        if vx[i] * vx[i + 1] < 0 and (abs(vx[i]) + abs(vx[i + 1])) > 15:
            if not impact_points or i - impact_points[-1] >= min_gap:
                impact_points.append(i)
                print(f"[✓] TrackNet impact: f{i} "
                      f"(t={i/fps:.2f}s, vx: {vx[i]:+.1f}→{vx[i+1]:+.1f}, "
                      f"pos={ball_track[i]})")

    return impact_points


# ─────────────────────────────────────────
# 공 추적 기반 임팩트 감지 (HSV 색상 필터)
# ─────────────────────────────────────────

def detect_impacts_from_ball(video_path, fps, n_frames, min_gap_sec=1.5):
    """HSV로 테니스 공 추적 → x방향 속도 반전 = 임팩트"""
    print("[i] 공 추적 분석 중...")

    lower_ball = np.array([20, 60, 60])
    upper_ball = np.array([50, 255, 255])

    cap = cv2.VideoCapture(video_path)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 상단 20% 영역은 자막/로고 오탐 방지를 위해 마스크에서 제외
    roi_top = int(frame_h * 0.20)

    ball_positions = []  # (cx, cy) 또는 None
    kernel = np.ones((3, 3), np.uint8)
    prev_pos = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_ball, upper_ball)
        # 상단 ROI 제외
        mask[:roi_top, :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_pos, best_circ = None, 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 8 or area > 1500:
                continue
            peri = cv2.arcLength(c, True)
            if peri < 1:
                continue
            circ = 4 * math.pi * area / (peri * peri)
            if circ <= 0.45:
                continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            # 이전 공 위치에서 120픽셀 이내인 후보만 유효 (연속성)
            if prev_pos is not None:
                dist = math.sqrt((cx - prev_pos[0])**2 + (cy - prev_pos[1])**2)
                if dist > 120:
                    continue
            if circ > best_circ:
                best_circ = circ
                best_pos = (cx, cy)

        # 이전 위치 없을 때는 연속성 무시하고 원형도만으로 선택 (첫 감지용)
        if best_pos is None and prev_pos is None:
            best_circ2 = 0.0
            for c in contours:
                area = cv2.contourArea(c)
                if area < 8 or area > 1500:
                    continue
                peri = cv2.arcLength(c, True)
                if peri < 1:
                    continue
                circ = 4 * math.pi * area / (peri * peri)
                if circ > best_circ2:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        best_circ2 = circ
                        best_pos = (M["m10"] / M["m00"], M["m01"] / M["m00"])
            if best_circ2 < 0.45:
                best_pos = None

        ball_positions.append(best_pos)
        prev_pos = best_pos if best_pos else prev_pos  # 미감지 시 이전 위치 유지 (연속성용)
        frame_idx += 1
        if frame_idx % 100 == 0:
            detected = sum(1 for p in ball_positions if p is not None)
            print(f"  Ball tracking: frame {frame_idx}/{n_frames} (감지: {detected}프레임)")

    cap.release()

    detected_count = sum(1 for p in ball_positions if p is not None)
    print(f"[i] 공 감지 프레임: {detected_count}/{n_frames} ({detected_count/n_frames*100:.1f}%)")

    if detected_count < n_frames * 0.03:
        print("[!] 공 감지율 3% 미만 — HSV 범위가 맞지 않을 수 있음")
        return []

    # x방향 속도 계산 (window=2프레임)
    win = 2
    vx = [None] * len(ball_positions)
    for i in range(win, len(ball_positions) - win):
        pa = ball_positions[i - win]
        pb = ball_positions[i + win]
        if pa and pb:
            vx[i] = pb[0] - pa[0]

    # x방향 속도 반전 = 임팩트 (임계: 총 변화량 30픽셀 이상)
    min_gap = int(min_gap_sec * fps)
    impact_points = []
    for i in range(win, len(vx) - 1):
        if vx[i] is None or vx[i + 1] is None:
            continue
        if vx[i] * vx[i + 1] < 0 and (abs(vx[i]) + abs(vx[i + 1])) > 30:
            if not impact_points or i - impact_points[-1] >= min_gap:
                impact_points.append(i)
                print(f"[✓] Ball impact at Frame {i} "
                      f"(t={i/fps:.2f}s, vx: {vx[i]:+.1f}→{vx[i+1]:+.1f}, "
                      f"pos={ball_positions[i]})")

    return impact_points


# ─────────────────────────────────────────
# 오디오 타구음 기반 임팩트 감지
# ─────────────────────────────────────────

def detect_impacts_from_audio(video_path, fps, n_frames, min_gap_sec=1.0, threshold_ratio=0.20):
    import tempfile
    import wave
    if n_frames <= 0 or not math.isfinite(fps) or fps <= 0:
        return [], []
    if not math.isfinite(min_gap_sec) or min_gap_sec <= 0:
        raise ValueError("min_gap_sec must be positive and finite")
    if not math.isfinite(threshold_ratio) or not 0 < threshold_ratio <= 1:
        raise ValueError("threshold_ratio must be in (0, 1]")
    print("[i] 오디오 타구음 분석 중...")
    try:
        # A private temporary directory is safe on Windows and cleans up failures too.
        with tempfile.TemporaryDirectory(prefix="tennis_audio_") as tmp_dir:
            tmp_wav = os.path.join(tmp_dir, "audio.wav")
            subprocess.run([
                'ffmpeg', '-y', '-nostdin', '-i', str(video_path),
                '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '44100',
                '-c:a', 'pcm_s16le', tmp_wav
            ], capture_output=True, check=True)
            with wave.open(tmp_wav, 'rb') as wf:
                sample_rate = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
    except Exception as e:
        print(f"[!] 오디오 추출 실패: {e}")
        return [], []

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # 단시간 에너지 계산 (20ms 윈도우, 5ms 홉)
    win = max(1, int(sample_rate * 0.020))
    hop = max(1, int(sample_rate * 0.005))
    n_wins = max(0, (len(samples) - win) // hop + 1)
    if n_wins < 2:
        return [], []
    energy = np.array([
        np.sum(samples[i*hop:i*hop+win] ** 2) / win
        for i in range(n_wins)
    ])

    # 에너지 상승률 (onset 강도)
    onset = np.diff(energy)
    onset = np.clip(onset, 0, None)
    peak_onset = float(onset.max())
    if peak_onset <= 0:
        return [], []
    onset /= peak_onset

    # 피크 탐색 (최소 간격: min_gap_sec)
    min_gap_wins = max(1, round(min_gap_sec * sample_rate / hop))
    peaks = []
    i = 0
    while i < len(onset):
        if onset[i] >= threshold_ratio:
            j = i
            while j < len(onset) and onset[j] >= threshold_ratio * 0.4:
                j += 1
            pk = i + int(np.argmax(onset[i:j]))
            if not peaks or pk - peaks[-1] > min_gap_wins:
                peaks.append(pk)
            i = j
        else:
            i += 1

    # 오디오 피크 → 비디오 프레임 + onset 값 변환
    video_frames = []
    onset_values = []
    for pk in peaks:
        t = pk * hop / sample_rate
        f = min(int(t * fps), n_frames - 1)
        video_frames.append(f)
        onset_values.append(float(onset[pk]))
        print(f"[✓] Audio impact at Frame {f} (t={t:.3f}s, onset={onset[pk]:.3f})")

    return video_frames, onset_values


# ─────────────────────────────────────────
# 팔신장 임팩트 감지 (독립 함수 — 앙상블용)
# ─────────────────────────────────────────

def detect_impacts_arm_extension(all_smoothed_landmarks, is_right_handed, fps=30.0, min_gap_sec=1.0):
    """어깨-손목 신장도 기반 임팩트 감지. (impact_points, extensions) 반환."""
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(min_gap_sec) or min_gap_sec <= 0:
        raise ValueError("fps and min_gap_sec must be positive and finite")
    h_shoulder_idx = R_SHOULDER if is_right_handed else L_SHOULDER
    h_wrist_idx    = R_WRIST    if is_right_handed else L_WRIST
    n = len(all_smoothed_landmarks)
    win = 3

    extensions = []
    for fi in range(n):
        sm = all_smoothed_landmarks[fi]
        if sm is None:
            extensions.append(0.0)
            continue
        s = sm[h_shoulder_idx]
        w = sm[h_wrist_idx]
        extensions.append(math.sqrt((w.x - s.x)**2 + (w.y - s.y)**2))

    abs_velocities = []
    for fi in range(n):
        a, b = fi - win, fi + win
        if a < 0 or b >= n or all_smoothed_landmarks[a] is None or all_smoothed_landmarks[b] is None:
            abs_velocities.append(0.0)
            continue
        wa = all_smoothed_landmarks[a][h_wrist_idx]
        wb = all_smoothed_landmarks[b][h_wrist_idx]
        abs_velocities.append(math.sqrt((wb.x - wa.x)**2 + (wb.y - wa.y)**2) / (2 * win))

    max_abs = max(abs_velocities) if abs_velocities else 0
    if max_abs <= 1e-8:
        return [], extensions
    vel_threshold = max_abs * 0.12

    in_swing, swing_start = False, 0
    swing_windows = []
    for fi in range(n):
        if not in_swing and abs_velocities[fi] >= vel_threshold:
            in_swing, swing_start = True, fi
        elif in_swing and abs_velocities[fi] < vel_threshold * 0.25:
            if fi - swing_start >= 6:
                swing_windows.append((swing_start, fi))
            in_swing = False
    if in_swing:
        swing_windows.append((swing_start, n - 1))

    merged = []
    for ws, we in swing_windows:
        if merged and ws - merged[-1][1] <= 20:
            merged[-1] = (merged[-1][0], we)
        else:
            merged.append((ws, we))

    impact_points = []
    for ws, we in merged:
        ws_ext = max(0, ws - 15)
        window_exts = extensions[ws_ext:we + 1]
        max_ext = max(window_exts) if window_exts else 0
        threshold_ext = max_ext * 0.50

        impact_f = we
        for fi in range(ws_ext, we + 1):
            if extensions[fi] >= threshold_ext:
                impact_f = fi
                break

        sm = all_smoothed_landmarks[impact_f]
        if sm is not None:
            wrist_y    = sm[h_wrist_idx].y
            shoulder_y = sm[h_shoulder_idx].y
            if wrist_y < shoulder_y - 0.04:
                print(f"[i] Skipped frame {impact_f}: wrist above shoulder")
                continue

        impact_points.append(impact_f)
        print(f"[✓] Arm-extension impact at Frame {impact_f} "
              f"(ext={extensions[impact_f]:.4f}, window={ws_ext}-{we})")

    min_gap = max(1, round(fps * min_gap_sec))
    deduped = []
    for imp in sorted(impact_points):
        if deduped and imp - deduped[-1] < min_gap:
            if extensions[imp] > extensions[deduped[-1]]:
                print(f"[i] Replaced f{deduped[-1]} with f{imp} (higher extension)")
                deduped[-1] = imp
            else:
                print(f"[i] Removed f{imp}: too close to f{deduped[-1]} (lower extension)")
        else:
            deduped.append(imp)

    return deduped, extensions


def detect_impacts_audio_validated(video_path, fps, n_frames, all_smoothed_landmarks, is_right_handed, min_gap_sec=1.0):
    """오디오 타구음 + 랜드마크 이중검증.
    - 오디오 피크를 먼저 클러스터링 (4초 이내 = 같은 샷 묶음)
    - 클러스터당 최고 점수 프레임 1개 선택
    - 검색 창: [-20, +5] 비대칭
    - 점수: xoff + 손목속도 가중치 + 팔꿈치 증가 조건
    """
    raw_frames, onsets = detect_impacts_from_audio(video_path, fps, n_frames, min_gap_sec=min_gap_sec)
    if not raw_frames or not all_smoothed_landmarks:
        return []

    # ── 상대 임계값 필터: max_onset * 0.25 이상만 사용 ──
    # 약한 에코/잡음 제거, 강한 타구음만 남김
    max_onset = max(onsets) if onsets else 0
    rel_threshold = max(0.20, max_onset * 0.25)
    audio_candidates = [f for f, o in zip(raw_frames, onsets) if o >= rel_threshold]
    print(f"[i] 오디오 필터: max_onset={max_onset:.3f}, threshold={rel_threshold:.3f} → {len(audio_candidates)}/{len(raw_frames)}개 통과: {audio_candidates}")

    if not audio_candidates:
        return []

    # ── 오디오 피크 클러스터링: fps*4(4초) 이내는 같은 샷으로 묶음 ──
    CLUSTER_GAP = max(1, round(fps * min_gap_sec))
    clusters = []
    for af in sorted(audio_candidates):
        if clusters and af - clusters[-1][-1] < CLUSTER_GAP:
            clusters[-1].append(af)
        else:
            clusters.append([af])
    print(f"[i] 오디오 클러스터 {len(clusters)}개: {clusters}")

    h_s = R_SHOULDER if is_right_handed else L_SHOULDER
    h_w = R_WRIST    if is_right_handed else L_WRIST
    h_e = R_ELBOW    if is_right_handed else L_ELBOW
    VEL_WIN  = 3
    LOOK_BACK = 5

    def best_frame_for_audio(af):
        """오디오 피크 af 주변에서 최적 임팩트 프레임과 점수 반환."""
        best_f, best_s = af, -1.0
        for fi in range(max(0, af - 20), min(n_frames, af + 3)):
            sm = all_smoothed_landmarks[fi]
            if sm is None:
                continue
            s, w, e = sm[h_s], sm[h_w], sm[h_e]
            if abs(w.x - s.x) <= 0.03:
                continue
            if not (s.y - 0.05 <= w.y <= s.y + 0.30):
                continue
            ea = calculate_angle_2d((s.x, s.y), (e.x, e.y), (w.x, w.y))
            if ea < 100.0:
                continue
            # 팔꿈치 증가 조건: 이전 5프레임보다 현재 각도가 높아야 (팔로우스루 제거)
            prev_angles = [
                calculate_angle_2d(
                    (all_smoothed_landmarks[p][h_s].x, all_smoothed_landmarks[p][h_s].y),
                    (all_smoothed_landmarks[p][h_e].x, all_smoothed_landmarks[p][h_e].y),
                    (all_smoothed_landmarks[p][h_w].x, all_smoothed_landmarks[p][h_w].y),
                )
                for p in range(max(0, fi - 5), fi)
                if all_smoothed_landmarks[p] is not None
            ]
            if prev_angles and ea <= max(prev_angles):
                continue
            x_off = abs(w.x - s.x)
            a, b = fi - VEL_WIN, fi + VEL_WIN
            if 0 <= a and b < n_frames and all_smoothed_landmarks[a] and all_smoothed_landmarks[b]:
                wa, wb = all_smoothed_landmarks[a][h_w], all_smoothed_landmarks[b][h_w]
                vel = math.sqrt((wb.x - wa.x)**2 + (wb.y - wa.y)**2) / (2 * VEL_WIN)
            else:
                vel = 0.0
            # elbow(153°→높을수록) + xoff + vel 복합 점수
            score = ea * 0.001 + x_off + vel * 7.5
            if score > best_s:
                best_s, best_f = score, fi
        return best_f, best_s

    # ── 클러스터당 최고 점수 프레임 1개 선택 ──
    result = []
    for cluster in clusters:
        cluster_best_f, cluster_best_s = None, -1.0
        for af in cluster:
            bf, bs = best_frame_for_audio(af)
            if bs > cluster_best_s:
                cluster_best_s, cluster_best_f = bs, bf
        if cluster_best_s > 0:
            result.append(cluster_best_f)
            print(f"[✓] 오디오+랜드마크 검증: f{cluster_best_f} "
                  f"(cluster={cluster}, score={cluster_best_s:.3f})")
        else:
            print(f"[i] 클러스터 {cluster}: 랜드마크 검증 실패")

    return result


def _elbow_angle_at(lms, fi, is_right_handed):
    """fi 프레임의 팔꿈치 각도(도) 반환. 랜드마크 없으면 0."""
    sm = lms[fi]
    if sm is None:
        return 0.0
    h_s = R_SHOULDER if is_right_handed else L_SHOULDER
    h_e = R_ELBOW    if is_right_handed else L_ELBOW
    h_w = R_WRIST    if is_right_handed else L_WRIST
    s, e, w = sm[h_s], sm[h_e], sm[h_w]
    v1 = (s.x - e.x, s.y - e.y)
    v2 = (w.x - e.x, w.y - e.y)
    denom = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2) + 1e-6
    cos_t = (v1[0]*v2[0] + v1[1]*v2[1]) / denom
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))


def _wrist_speed_at(lms, fi, is_right_handed, win=3):
    """fi 프레임의 손목 속도 반환 (±win 프레임 차분)."""
    n = len(lms)
    a, b = fi - win, fi + win
    if a < 0 or b >= n or lms[a] is None or lms[b] is None:
        return 0.0
    h_w = R_WRIST if is_right_handed else L_WRIST
    wa, wb = lms[a][h_w], lms[b][h_w]
    return math.sqrt((wb.x - wa.x)**2 + (wb.y - wa.y)**2) / (2 * win)


def detect_impacts_ensemble(video_path, fps, n_frames, all_smoothed_landmarks, is_right_handed, min_gap_sec=1.0):
    """통합 앙상블: 오디오+랜드마크 이중검증(1순위) → 팔꿈치조건 팔신장(폴백)."""
    print("[i] 통합 앙상블 감지...")

    # 1순위: 오디오 + 랜드마크 이중검증 (v12.1 방식)
    result = detect_impacts_audio_validated(
        video_path, fps, n_frames, all_smoothed_landmarks, is_right_handed, min_gap_sec=min_gap_sec
    )
    if result:
        print(f"[✓] 오디오+랜드마크 이중검증 성공 → {result}")
        return result

    # 2순위: 팔꿈치 조건(≥140°+감소중) + 복합점수 팔신장 (v12.3 방식)
    print("[i] 오디오+랜드마크 실패 → 팔꿈치 조건 팔신장 폴백...")
    h_s = R_SHOULDER if is_right_handed else L_SHOULDER
    h_w = R_WRIST    if is_right_handed else L_WRIST
    n = len(all_smoothed_landmarks)
    win = 3

    extensions, abs_vels = [], []
    for fi in range(n):
        sm = all_smoothed_landmarks[fi]
        if sm is None:
            extensions.append(0.0); abs_vels.append(0.0); continue
        s, w = sm[h_s], sm[h_w]
        extensions.append(math.sqrt((w.x - s.x)**2 + (w.y - s.y)**2))
        a, b = fi - win, fi + win
        if a < 0 or b >= n or all_smoothed_landmarks[a] is None or all_smoothed_landmarks[b] is None:
            abs_vels.append(0.0)
        else:
            wa, wb = all_smoothed_landmarks[a][h_w], all_smoothed_landmarks[b][h_w]
            abs_vels.append(math.sqrt((wb.x - wa.x)**2 + (wb.y - wa.y)**2) / (2 * win))

    max_v = max(abs_vels) if abs_vels else 0
    if max_v <= 1e-8:
        return []
    vel_thr = max_v * 0.12
    in_sw, sw_start = False, 0
    windows = []
    for fi in range(n):
        if not in_sw and abs_vels[fi] >= vel_thr:
            in_sw, sw_start = True, fi
        elif in_sw and abs_vels[fi] < vel_thr * 0.25:
            if fi - sw_start >= 6:
                windows.append((sw_start, fi))
            in_sw = False
    if in_sw:
        windows.append((sw_start, n - 1))
    merged = []
    for ws, we in windows:
        if merged and ws - merged[-1][1] <= 20:
            merged[-1] = (merged[-1][0], we)
        else:
            merged.append((ws, we))

    ELBOW_MIN, LOOK_BACK = 140.0, 5
    fallback = []
    for ws, we in merged:
        ws_ext = max(0, ws - 15)
        max_ext = max(extensions[ws_ext:we + 1]) if extensions[ws_ext:we + 1] else 0
        thr = max_ext * 0.50
        best_f, best_score = we, -1.0
        for fi in range(ws_ext, we + 1):
            if extensions[fi] < thr:
                continue
            sm = all_smoothed_landmarks[fi]
            if sm is None:
                continue
            if sm[h_w].y < sm[h_s].y - 0.04:
                continue
            ea = _elbow_angle_at(all_smoothed_landmarks, fi, is_right_handed)
            if ea < ELBOW_MIN:
                continue
            prev_max = max(
                (_elbow_angle_at(all_smoothed_landmarks, pfi, is_right_handed)
                 for pfi in range(max(0, fi - LOOK_BACK), fi)),
                default=0.0
            )
            if ea >= prev_max:
                continue
            score = extensions[fi] * abs_vels[fi]
            if score > best_score:
                best_score, best_f = score, fi
        if best_score > 0:
            fallback.append(best_f)
            print(f"[✓] 팔꿈치조건 폴백: f{best_f} "
                  f"(elbow={_elbow_angle_at(all_smoothed_landmarks, best_f, is_right_handed):.1f}°, score={best_score:.5f})")

    deduped = []
    for f in sorted(fallback):
        if not deduped or f - deduped[-1] >= max(1, round(fps * min_gap_sec)):
            deduped.append(f)

    print(f"[✓] 앙상블 최종 임팩트: {deduped}")
    return deduped


# ─────────────────────────────────────────
# 2-Pass 비디오 처리 및 렌더링 파이프라인
# ─────────────────────────────────────────

def legacy_leg_correction(all_smoothed_landmarks):
    if not any(lm is not None for lm in all_smoothed_landmarks):
        return all_smoothed_landmarks
    # 어깨 방향 기반 전역 다리 교정 (캐시 로드·신규 추출 모두 적용)
    all_smoothed_landmarks = correct_leg_orientation_global(all_smoothed_landmarks)

    # ── 발 추적 손실 구간 동결·보간 보정 ──────────────────────────────────────────
    # 문제: MediaPipe가 서브 follow-through 회전 중 L/R 발목 레이블을 점진적으로 교환.
    #       visibility=1.0 으로 자신있게 틀리므로 기존 visibility 기반 동결로는 감지 불가.
    # 해결: 발목 gap이 _CONV_RATE 이상 속도로 N프레임 연속 감소하면 추적 손실로 판정,
    #       마지막 안정 위치를 동결 유지 → 해제 후 교정된 위치까지 선형 보간.
    _CONV_RATE     = 0.012  # 프레임당 gap 감소 임계값 (정규화 좌표)
    _CONV_CONFIRM  = 3      # 연속 N프레임 이상 감소해야 동결 시작
    _STAB_FRAMES   = 5      # gap 재증가 N프레임 연속 → 안정 판정(동결 해제)
    _INTERP_FRAMES = 12     # 동결 해제 후 교정 위치까지 보간할 프레임 수
    _LEG_ALL_IDX   = list(range(23, 33))   # hip ~ foot_index

    _b_cnt   = sum(1 for _lm in all_smoothed_landmarks if _lm is not None and _lm[27].x < _lm[28].x)
    _t_cnt   = sum(1 for _lm in all_smoothed_landmarks if _lm is not None)
    _exp_bk  = (_b_cnt / _t_cnt > 0.5) if _t_cnt else True

    def _ankle_gap(_lm):
        return (_lm[28].x - _lm[27].x) if _exp_bk else (_lm[27].x - _lm[28].x)

    _orig_gaps = [_ankle_gap(_lm) if _lm is not None else None
                  for _lm in all_smoothed_landmarks]

    _freeze_zones = []
    _frozen, _zone_start, _ref_fi = False, None, None
    _consec_dec, _consec_stab = 0, 0

    for _fi in range(len(all_smoothed_landmarks)):
        if _orig_gaps[_fi] is None:
            continue
        _g  = _orig_gaps[_fi]
        _gp = next((_orig_gaps[_pf] for _pf in range(_fi - 1, -1, -1)
                    if _orig_gaps[_pf] is not None), None)
        if _gp is None:
            continue
        _dg = _g - _gp

        if not _frozen:
            if _dg < -_CONV_RATE:
                _consec_dec += 1
            else:
                _consec_dec = 0
            if _consec_dec >= _CONV_CONFIRM:
                _frozen     = True
                _zone_start = _fi - _CONV_CONFIRM
                _ref_fi     = _zone_start - 1
                while _ref_fi >= 0 and all_smoothed_landmarks[_ref_fi] is None:
                    _ref_fi -= 1
                _consec_dec = _consec_stab = 0
        else:
            _consec_stab = (_consec_stab + 1) if _dg > 0 else 0
            if _consec_stab >= _STAB_FRAMES:
                _freeze_zones.append((_zone_start, _fi - _STAB_FRAMES, _ref_fi))
                _frozen = False
                _zone_start = _ref_fi = None
                _consec_stab = _consec_dec = 0

    if _frozen and _zone_start is not None:
        _freeze_zones.append((_zone_start, len(all_smoothed_landmarks) - 1, _ref_fi))

    for _zs, _ze, _rfi in _freeze_zones:
        if _rfi is None or _rfi < 0:
            continue
        _ref_lm = all_smoothed_landmarks[_rfi]
        if _ref_lm is None:
            continue
        for _fi in range(_zs, _ze + 1):
            _lm = all_smoothed_landmarks[_fi]
            if _lm is None:
                continue
            for _idx in _LEG_ALL_IDX:
                _src = _ref_lm[_idx]
                _lm[_idx] = SimpleNamespace(x=_src.x, y=_src.y, z=_src.z,
                                            v=getattr(_src, 'v', 1.0))
        _iend = min(_ze + _INTERP_FRAMES, len(all_smoothed_landmarks) - 1)
        for _fi in range(_ze + 1, _iend + 1):
            _lm = all_smoothed_landmarks[_fi]
            if _lm is None:
                continue
            _t = (_fi - _ze) / _INTERP_FRAMES
            for _idx in _LEG_ALL_IDX:
                _s = _ref_lm[_idx]
                _d = _lm[_idx]
                _lm[_idx] = SimpleNamespace(
                    x=_s.x * (1 - _t) + _d.x * _t,
                    y=_s.y * (1 - _t) + _d.y * _t,
                    z=_s.z * (1 - _t) + _d.z * _t,
                    v=getattr(_d, 'v', 1.0))

    if _freeze_zones:
        print(f"[발 추적 손실 보정] {len(_freeze_zones)}구간 동결·보간: "
              + ", ".join(f"f{zs}-f{ze}(ref f{rfi})" for zs, ze, rfi in _freeze_zones))

    # 다리 랜드마크 안정화 (hip~foot_index 인덱스 23~32)
    # 1단계: visibility < 0.5 프레임은 마지막 신뢰 값으로 대체
    # 2단계: ±3프레임 이동평균으로 잔여 노이즈 제거
    LEG_IDX = list(range(23, 33))
    VIS_THRESH = 0.5
    LEG_WIN = 3
    n_frames_total = len(all_smoothed_landmarks)

    for idx in LEG_IDX:
        # 1단계: visibility 기반 대체
        last_x, last_y = None, None
        for lm in all_smoothed_landmarks:
            if lm is None:
                continue
            vis = getattr(lm[idx], 'v', 1.0)
            if vis >= VIS_THRESH:
                last_x, last_y = lm[idx].x, lm[idx].y
            elif last_x is not None:
                lm[idx].x, lm[idx].y = last_x, last_y

        # 2단계: 이동평균 스무딩
        xs = [lm[idx].x if lm is not None else None for lm in all_smoothed_landmarks]
        ys = [lm[idx].y if lm is not None else None for lm in all_smoothed_landmarks]
        for i, lm in enumerate(all_smoothed_landmarks):
            if lm is None:
                continue
            lo, hi = max(0, i - LEG_WIN), min(n_frames_total, i + LEG_WIN + 1)
            wx = [v for v in xs[lo:hi] if v is not None]
            wy = [v for v in ys[lo:hi] if v is not None]
            if wx:
                lm[idx].x = sum(wx) / len(wx)
            if wy:
                lm[idx].y = sum(wy) / len(wy)

    # 다리 최소 간격 강제: 발목(27,28) x 차이가 MIN_FOOT_SEP 미만이면 강제 이격
    # 서브 점프 시 두 발이 2D 투영상 겹쳐 보이는 현상 방지
    L_ANKLE_IDX, R_ANKLE_IDX = 27, 28
    MIN_FOOT_SEP = 0.06   # 정규화 좌표 기준 (약 24px / 404px 캔버스)
    # 기준 방향: 과반수 프레임에서 L_ANKLE < R_ANKLE 이면 등향(L이 왼쪽)
    back_frames = sum(1 for lm in all_smoothed_landmarks
                      if lm is not None and lm[L_ANKLE_IDX].x < lm[R_ANKLE_IDX].x)
    total_valid  = sum(1 for lm in all_smoothed_landmarks if lm is not None)
    ankle_expected_back = back_frames / total_valid > 0.5

    for lm in all_smoothed_landmarks:
        if lm is None:
            continue
        la, ra = lm[L_ANKLE_IDX], lm[R_ANKLE_IDX]
        if ankle_expected_back:
            # 등향: la.x < ra.x 이어야 함
            gap = ra.x - la.x
            if gap < MIN_FOOT_SEP:
                push = (MIN_FOOT_SEP - gap) / 2
                lm[L_ANKLE_IDX].x = la.x - push
                lm[R_ANKLE_IDX].x = ra.x + push
        else:
            # 정면향: la.x > ra.x 이어야 함
            gap = la.x - ra.x
            if gap < MIN_FOOT_SEP:
                push = (MIN_FOOT_SEP - gap) / 2
                lm[L_ANKLE_IDX].x = la.x + push
                lm[R_ANKLE_IDX].x = ra.x - push

    return all_smoothed_landmarks


def resolve_model(model_path=None):
    """Resolve relative to the script, never a particular user's home directory."""
    candidate = Path(model_path) if model_path else Path(MODEL_PATH)
    candidate = candidate.expanduser().resolve()
    if candidate.is_file():
        return candidate
    if model_path:
        raise FileNotFoundError(f"MediaPipe 모델 파일이 없습니다: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    print("[i] MediaPipe 모델 다운로드 중...")
    fd, temp_name = tempfile.mkstemp(suffix=".part", dir=candidate.parent)
    os.close(fd)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=90) as response, open(temp_name, "wb") as dst:
            shutil.copyfileobj(response, dst)
        if os.path.getsize(temp_name) < 1_000_000:
            raise RuntimeError("모델 다운로드가 불완전합니다.")
        os.replace(temp_name, candidate)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)
    return candidate


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {path}")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if w <= 0 or h <= 0 or not math.isfinite(fps) or not 0 < fps <= 240 or count <= 0:
            raise ValueError(f"영상 메타데이터가 올바르지 않습니다: {w}x{h}, {fps}fps, {count}frames")
        return w, h, fps, count
    finally:
        cap.release()


def prepare_source(source, temporary, start, duration):
    if str(source).lower().startswith(("http://", "https://")):
        downloaded = Path(temporary) / "download.mp4"
        # Each request downloads anew: a previous URL can never alias a new source.
        command = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress",
                   "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                   "--merge-output-format", "mp4", "-o", str(downloaded), "--", str(source)]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError("영상 다운로드 실패 (yt-dlp 설치 여부 확인): " + result.stderr[-1600:])
        actual = downloaded
    else:
        actual = Path(source).expanduser().resolve()
        if not actual.is_file():
            raise FileNotFoundError(f"입력 영상이 없습니다: {actual}")
    identity = video_identity(actual)
    w, h, fps, count = video_info(actual)
    if start >= count / fps:
        raise ValueError("시작 시각이 영상 길이 이상입니다.")
    if start > 0 or duration is not None:
        trimmed = Path(temporary) / "selected_clip.mp4"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-ss", str(start), "-i", str(actual)]
        if duration is not None:
            cmd += ["-t", str(duration)]
        cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "fast",
                "-crf", "16", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:a", "aac", str(trimmed)]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError("영상 구간 추출 실패: " + result.stderr[-1600:])
        actual = trimmed
    return actual, identity


def extract_poses(video_path, model_path, fps, expected_frames):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("MediaPipe를 사용할 수 없습니다. Python 3.13에서 requirements.txt를 설치하세요.") from exc
    # model_asset_buffer avoids native Windows file APIs failing on Korean paths.
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
        running_mode=vision.RunningMode.VIDEO, num_poses=2,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(video_path))
    frames, prev_raw = [], None
    smoother = PoseSmoother(freq=fps)
    missing = 0
    try:
        with vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = detector.detect_for_video(image, round(len(frames) * 1000 / fps))
                poses = [p for p in result.pose_landmarks
                         if (p[L_HIP].y + p[R_HIP].y) / 2 >= 0.35]
                smoothed = None
                if poses:
                    if prev_raw is None:
                        pose = max(poses, key=lambda p: (p[L_HIP].y + p[R_HIP].y) / 2)
                    else:
                        pose = min(poses, key=lambda p: sum(
                            (p[j].x - prev_raw[j].x)**2 + (p[j].y - prev_raw[j].y)**2
                            for j in (L_HIP, R_HIP, L_SHOULDER, R_SHOULDER)))
                    raw = [SimpleNamespace(x=p.x, y=p.y, z=p.z,
                                           v=float(p.visibility if p.visibility is not None else 1.0)) for p in pose]
                    raw = correct_leg_swaps(raw, prev_raw)
                    prev_raw = raw
                    smoothed = smoother.apply(raw)
                    missing = 0
                else:
                    missing += 1
                    if missing > max(1, round(fps * 0.2)):
                        prev_raw = None
                        smoother = PoseSmoother(freq=fps)
                frames.append(smoothed)
                if len(frames) % 50 == 0:
                    print(f"[자세 추출] {len(frames)}/{expected_frames}", flush=True)
    finally:
        cap.release()
    if len(frames) != expected_frames:
        raise RuntimeError(f"영상 디코딩이 중단되었습니다: 예상 {expected_frames}, 실제 {len(frames)}프레임")
    return frames


def fill_short_pose_gaps(frames, max_gap):
    """Only fill bounded short gaps; do not extrapolate missing starts/ends or cuts."""
    filled = 0
    i = 0
    while i < len(frames):
        if frames[i] is not None:
            i += 1
            continue
        start = i
        while i < len(frames) and frames[i] is None:
            i += 1
        gap = i - start
        if start == 0 or i == len(frames) or gap > max_gap:
            continue
        before, after = frames[start - 1], frames[i]
        if max(math.hypot(before[j].x - after[j].x, before[j].y - after[j].y)
               for j in (L_HIP, R_HIP, L_SHOULDER, R_SHOULDER)) > 0.2:
            continue
        for offset in range(gap):
            alpha = (offset + 1) / (gap + 1)
            frames[start + offset] = [SimpleNamespace(
                x=a.x * (1-alpha) + b.x * alpha,
                y=a.y * (1-alpha) + b.y * alpha,
                z=a.z * (1-alpha) + b.z * alpha,
                v=min(getattr(a, "v", 1.0), getattr(b, "v", 1.0))) for a, b in zip(before, after)]
            filled += 1
    return filled


def choose_impacts(video_path, fps, frames, is_right_handed, is_serve,
                   manual=None, ball_track=False, tracknet_model=None,
                   audio_impact=False, ensemble=False, min_gap=1.0):
    if manual is not None:
        if any(f < 0 or f >= len(frames) for f in manual):
            raise ValueError(f"임팩트 프레임은 0~{len(frames)-1} 범위여야 합니다.")
        return sorted(set(manual)), "manual"
    points, method = [], "none"
    if ball_track:
        if not tracknet_model or not Path(tracknet_model).is_file():
            raise ValueError("--ball-track에는 유효한 --tracknet-model 파일이 필요합니다.")
        points = detect_impacts_from_tracknet(str(video_path), fps, len(frames), str(tracknet_model), min_gap)
        method = "tracknet_estimate"
    if not points and (audio_impact or ensemble) and not is_serve:
        if ensemble:
            points = detect_impacts_ensemble(str(video_path), fps, len(frames), frames, is_right_handed,
                                             min_gap_sec=min_gap)
            method = "ensemble_estimate"
        else:
            points = detect_impacts_audio_validated(str(video_path), fps, len(frames), frames, is_right_handed,
                                                   min_gap_sec=min_gap)
            method = "audio_pose_estimate"
    if not points and is_serve:
        candidates = []
        for i, lm in enumerate(frames):
            if lm is None:
                continue
            s, e, w = (R_SHOULDER, R_ELBOW, R_WRIST) if is_right_handed else (L_SHOULDER, L_ELBOW, L_WRIST)
            angle = calculate_angle_2d((lm[s].x, lm[s].y), (lm[e].x, lm[e].y), (lm[w].x, lm[w].y))
            if detect_serve_phase(lm, is_right_handed, angle) == "Impact":
                candidates.append(i)
        groups = []
        for i in candidates:
            if groups and i == groups[-1][-1] + 1:
                groups[-1].append(i)
            else:
                groups.append([i])
        points = [g[len(g)//2] for g in groups]
        method = "serve_pose_estimate"
    if not points and not is_serve:
        points, _ = detect_impacts_arm_extension(frames, is_right_handed, fps=fps, min_gap_sec=min_gap)
        method = "arm_extension_estimate"
    deduped = []
    for point in sorted(set(points)):
        if 0 <= point < len(frames) and (not deduped or point - deduped[-1] >= max(1, round(fps * min_gap))):
            deduped.append(point)
    return deduped, method if deduped else "none"


def render_frames(actual_input, output_path, frames, fps, orig_w, orig_h, *,
                  height, ssaa, speed, is_right_handed, is_serve, label, desc,
                  strobe, strobe_frames, strobe_step, lag_scale, no_trail, two_handed,
                  compare, impact_points):
    global SSAA, _racket_offset_prev, _racket_face_prev, _shoe_blend_cache
    SSAA = ssaa
    _racket_offset_prev = _racket_face_prev = None
    _shoe_blend_cache = {"left": None, "right": None}
    out_h, out_w = height, max(2, round(orig_w * height / orig_h / 2) * 2)
    render_w, render_h = out_w * ssaa, out_h * ssaa
    if render_w * render_h > 40_000_000:
        raise ValueError("렌더 크기가 너무 큽니다. --height 또는 --ssaa를 줄이세요.")
    configure_thickness(render_h)
    background = draw_court_background(render_w, render_h)
    phase = PhaseSmoother()
    strobe_history = [] if strobe else None
    racket_trail, hand_trail = (None, None) if no_trail else ([], [])
    racket_tracker = RacketDirectionTracker(fps)
    racket_sources = {}
    cap = cv2.VideoCapture(str(actual_input)) if compare else None
    written = 0
    width = out_w * (2 if compare else 1)
    try:
        with FFmpegVideoWriter("ffmpeg", output_path, width, out_h, fps) as writer:
            for i, lm in enumerate(frames):
                racket_direction, racket_source = racket_tracker.update(lm, orig_w, orig_h, is_right_handed)
                racket_sources[racket_source] = racket_sources.get(racket_source, 0) + 1
                canvas = background.copy()
                nearest = min(impact_points, key=lambda p: abs(i-p)) if impact_points else None
                k = i - nearest if nearest is not None else 9999
                if lm is not None:
                    draw_stickman(canvas, lm, render_w, render_h, is_right_handed,
                                  racket_trail=racket_trail, hand_trail=hand_trail, phase_smoother=phase,
                                  is_serve=is_serve, k=k, strobe_history=strobe_history,
                                  strobe_frames=strobe_frames, strobe_step=strobe_step,
                                  lag_scale=0.0 if two_handed else lag_scale, two_handed=two_handed,
                                  racket_direction=racket_direction)
                else:
                    for history in (racket_trail, hand_trail, strobe_history):
                        if history is not None:
                            history.clear()
                    phase = PhaseSmoother()
                    _racket_offset_prev = _racket_face_prev = None
                    _shoe_blend_cache = {"left": None, "right": None}
                final = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)
                if lm is not None and not no_trail:
                    look = min(i + max(1, round(fps * 0.1)), len(frames)-1)
                    nxt = frames[look]
                    if nxt is not None:
                        wi = R_WRIST if is_right_handed else L_WRIST
                        p0 = np.array(get_point(lm, wi, render_w, render_h), dtype=float) / ssaa
                        p1 = np.array(get_point(nxt, wi, render_w, render_h), dtype=float) / ssaa
                        velocity = float(np.linalg.norm(p1-p0))
                        if velocity > 3:
                            scale = min(6.0, max(1.5, velocity/4.0))
                            tip = p0 + (p1-p0) * scale
                            intensity = min(1.0, velocity/28.0)
                            cv2.arrowedLine(final, tuple(p0.astype(int)), tuple(tip.astype(int)),
                                            (0, int(255*(1-intensity)), int(255*intensity)),
                                            max(2, int(2+2*intensity)), cv2.LINE_AA, tipLength=0.25)
                if nearest is not None and 0 <= k <= max(1, round(fps*0.12)):
                    alpha = 0.35 * (1 - k/max(1, round(fps*0.12)))
                    cv2.addWeighted(np.full_like(final, 255), alpha, final, 1-alpha, 0, final)
                if nearest is not None and -round(fps*0.05) <= k < round(fps*0.5):
                    text_scale = max(0.5, min(out_w / 300, out_h / 400))
                    size = cv2.getTextSize("IMPACT", cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)[0]
                    origin = ((out_w-size[0])//2, out_h-max(20, out_h//12))
                    cv2.putText(final, "IMPACT", origin, cv2.FONT_HERSHEY_SIMPLEX, text_scale, (20,20,20), 5, cv2.LINE_AA)
                    cv2.putText(final, "IMPACT", origin, cv2.FONT_HERSHEY_SIMPLEX, text_scale, (50,255,50), 2, cv2.LINE_AA)
                if lm is None:
                    cv2.putText(final, "Pose unavailable", (10, out_h-20), cv2.FONT_HERSHEY_SIMPLEX,
                                max(0.4, out_h/1200), (30,30,180), 1, cv2.LINE_AA)
                if label:
                    final = draw_label(final, label, desc)
                if compare:
                    ret, source_frame = cap.read()
                    if not ret:
                        raise RuntimeError(f"비교 영상 읽기 실패: {i}프레임")
                    original = cv2.resize(source_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    final = np.hstack((original, final))
                # Cumulative rounding prevents float drift and reports actual encoded frames.
                target_count = max(1, int(math.floor((i + 1) / speed + 0.5)))
                while written < target_count:
                    writer.write(final)
                    written += 1
                if (i+1) % 50 == 0:
                    print(f"[렌더링] {i+1}/{len(frames)} ({(i+1)/len(frames):.0%})", flush=True)
    finally:
        if cap is not None:
            cap.release()
    return width, out_h, written, racket_sources


def process_video(video_url, name, is_right_handed=True, label=None, desc=None, speed=1.0,
                  strobe=False, strobe_frames=32, strobe_step=4, lag_scale=0.0, no_trail=False,
                  two_handed=False, audio_impact=False, ball_track=False, impact_frames=None,
                  tracknet_model=None, ensemble=False, output_dir=None, model=None, height=720,
                  ssaa=2, start=0.0, duration=None, compare=False, no_audio=False, no_cache=False,
                  overwrite=False, stroke="auto", min_impact_gap=1.0, legacy_leg_fix=False):
    validate_options(name, speed, strobe_frames, strobe_step, height, ssaa, start, duration, min_impact_gap)
    if not math.isfinite(lag_scale) or not 0 <= lag_scale <= 5:
        raise ValueError("lag-scale은 0~5 사이여야 합니다.")
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise RuntimeError(f"{command}를 찾을 수 없습니다. FFmpeg를 설치하고 PATH에 추가하세요.")
    output_dir = Path(output_dir or Path.cwd() / "renders").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}_stickman.mp4"
    report_path = output_dir / f"{name}_analysis.json"
    if not overwrite and (output_path.exists() or report_path.exists()):
        raise FileExistsError(f"기존 결과가 있습니다: {output_path.name}. 다른 이름 또는 --overwrite를 사용하세요.")
    if not str(video_url).lower().startswith(("http://", "https://")):
        if Path(video_url).expanduser().resolve() in (output_path, report_path):
            raise ValueError("입력 영상과 결과 경로가 같습니다.")
    with tempfile.TemporaryDirectory(prefix=".stickman_", dir=output_dir) as temporary:
        actual_input, identity = prepare_source(video_url, temporary, start, duration)
        orig_w, orig_h, fps, total = video_info(actual_input)
        is_serve = stroke == "serve" or (stroke == "auto" and ("serve" in name.lower() or "서브" in name))
        model_candidate = Path(model) if model else Path(MODEL_PATH)
        model_hash = video_identity(model_candidate)["sha256"] if model_candidate.is_file() else None
        metadata = {"source": identity, "start": start, "duration": duration,
                    "fps": fps, "frame_count": total, "width": orig_w, "height": orig_h,
                    "extraction": "v12.4-upgrade.1-visibility-continuity", "model_sha256": model_hash}
        cache_path = output_dir / f"{name}_pose_cache.json"
        frames = None if no_cache else load_pose_cache(cache_path, metadata)
        cache_used = frames is not None
        if frames is None:
            model_path = resolve_model(model)
            metadata["model_sha256"] = video_identity(model_path)["sha256"]
            print(f"[i] 자세 추출: {total}프레임, {fps:.3f}fps", flush=True)
            frames = extract_poses(actual_input, model_path, fps, total)
            if not any(lm is not None for lm in frames):
                raise RuntimeError("선수 자세를 감지하지 못했습니다. 선수가 크게 보이는 영상을 사용하세요.")
            if not no_cache:
                save_pose_cache(cache_path, metadata, frames)
        else:
            print(f"[i] 입력 영상과 설정이 일치하는 자세 캐시 사용: {total}프레임", flush=True)
        if len(frames) != total or not any(lm is not None for lm in frames):
            raise RuntimeError("유효한 자세 데이터가 없습니다.")
        missing_before = sum(lm is None for lm in frames)
        filled = fill_short_pose_gaps(frames, max(1, round(fps*0.15)))
        if legacy_leg_fix:
            frames = legacy_leg_correction(frames)
        impacts, method = choose_impacts(actual_input, fps, frames, is_right_handed, is_serve,
                                        manual=impact_frames, ball_track=ball_track,
                                        tracknet_model=tracknet_model, audio_impact=audio_impact,
                                        ensemble=ensemble, min_gap=min_impact_gap)
        print(f"[i] 임팩트 {len(impacts)}개 ({method}): {impacts}", flush=True)
        if method == "none":
            print("[i] 임팩트를 확인하지 못해 임팩트 효과를 생략합니다.")
        silent = Path(temporary) / "silent.mp4"
        width, out_height, encoded, racket_sources = render_frames(
            actual_input, silent, frames, fps, orig_w, orig_h, height=height, ssaa=ssaa,
            speed=speed, is_right_handed=is_right_handed, is_serve=is_serve,
            label=label, desc=desc, strobe=strobe, strobe_frames=strobe_frames,
            strobe_step=strobe_step, lag_scale=lag_scale, no_trail=no_trail,
            two_handed=two_handed, compare=compare, impact_points=impacts)
        ready = Path(temporary) / "complete.mp4"
        if no_audio:
            shutil.copyfile(silent, ready)
        else:
            mux_audio("ffmpeg", "ffprobe", silent, actual_input, ready, speed=speed, duration=encoded/fps)
        report = {"version": VERSION, "source": str(video_url), "source_identity": identity,
                  "output": str(output_path), "width": width, "height": out_height,
                  "fps": fps, "input_frames": total, "output_frames": encoded,
                  "duration_seconds": encoded/fps, "speed": speed, "source_start_seconds": start,
                  "cache_used": cache_used, "pose_detected_frames": total-missing_before,
                  "pose_gap_filled_frames": filled, "pose_missing_frames": missing_before-filled,
                  "impact_method": method, "impact_accuracy": "manual" if method == "manual" else "heuristic_estimate",
                  "racket_direction_sources": racket_sources,
                  "impacts": [{"clip_frame": f, "source_time_seconds": start+f/fps,
                               "output_time_seconds": f/fps/speed} for f in impacts],
                  "options": {"stroke": stroke, "is_right_handed": is_right_handed, "two_handed": two_handed,
                              "compare": compare, "ssaa": ssaa, "legacy_leg_fix": legacy_leg_fix}}
        # All encoding and audio work must succeed before replacing the user-facing video.
        staged_report = Path(temporary) / "analysis.json"
        atomic_write_json(staged_report, report)
        os.replace(ready, output_path)
        os.replace(staged_report, report_path)
    print(f"[완료] {output_path}\n       {width}x{out_height}, {encoded}프레임, {encoded/fps:.3f}초", flush=True)
    return output_path


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    try:
        values = vars(args).copy()
        values["video_url"] = values.pop("url")
        values["is_right_handed"] = not values.pop("left")
        values["impact_frames"] = values.pop("impact_frame")
        process_video(**values)
        return 0
    except KeyboardInterrupt:
        print("[중단] 작업이 취소되었습니다.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
