'''Render a synthesized stream's pose as MP4 with subtitle overlay.

Renders ONLY the 77 keypoints the StreamSLST model actually consumes (defined in
config.ALL_SELECTED_IDS = BODY_IDS + LEFT_HAND_IDS + RIGHT_HAND_IDS + MOUTH_IDS + FACE_IDS).
The 56 unused COCO-WholeBody keypoints (ears, hips, knees, ankles, feet, most of face) are
dropped so the visualization shows what the model actually sees - and so noisy out-of-frame
HRNet predictions for the lower body do not corrupt the auto-fit window.

Auto-fit window is computed from the 77 model-input keypoints with confidence > 0.4 across the
whole stream, percentile-clipped (1, 99) to drop residual HRNet outliers.

Subtitle text uses PIL with a CJK-capable font for Chinese support.
Usage:
    DATASET=PHOENIX python -m data_synth.visualize_stream \
        --pose data/synth/phoenix/poses/test_00000.npy \
        --vtt  data/synth/phoenix/vtt/test_00000.vtt --out examples/phoenix_test_00000.mp4

    DATASET=CSL python -m data_synth.visualize_stream \
        --pose data/synth/csl/poses/test_00000.npy \
        --vtt  data/synth/csl/vtt/test_00000.vtt --out examples/csl_test_00000.mp4
'''
import os, sys, argparse, cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import parse_vtt
from config import (
    BODY_IDS, LEFT_HAND_IDS, RIGHT_HAND_IDS, MOUTH_IDS, FACE_IDS,
    ALL_SELECTED_IDS, FPS, DATASET,
)

# Body skeleton edges expressed as RAW COCO-WholeBody indices (subset within BODY_IDS only).
# BODY_IDS = [0(nose), 1(L_eye), 2(R_eye), 5(L_shoulder), 6(R_shoulder), 7(L_elbow), 8(R_elbow), 9(L_wrist), 10(R_wrist)]
BODY_EDGES_RAW = [
    (5, 7), (7, 9),     # left arm
    (6, 8), (8, 10),    # right arm
    (5, 6),             # shoulder line
    (0, 1), (0, 2),     # nose-eyes
]
HAND_EDGES_LOCAL = [(0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20)]


def find_cjk_font():
    candidates = [
        'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/simsun.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/System/Library/Fonts/PingFang.ttc',
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return None


def load_font(font_path, size):
    if font_path and os.path.exists(font_path):
        try: return ImageFont.truetype(font_path, size)
        except Exception: pass
    return ImageFont.load_default()


def auto_fit_77(poses: np.ndarray, conf_thr: float = 0.4, pad_frac: float = 0.10):
    # Bounding box from the 77 model-input keypoints only (high-conf), percentile-clipped
    sub = poses[:, ALL_SELECTED_IDS, :]
    mask = sub[..., 2] > conf_thr
    if not mask.any(): return 0.0, 0.0, 1.0
    xs = sub[..., 0][mask]
    ys = sub[..., 1][mask]
    xmin, xmax = float(np.percentile(xs, 1)), float(np.percentile(xs, 99))
    ymin, ymax = float(np.percentile(ys, 1)), float(np.percentile(ys, 99))
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    side = max(xmax - xmin, ymax - ymin) * (1.0 + 2.0 * pad_frac)
    return cx - side / 2.0, cy - side / 2.0, max(side, 1.0)


def draw_77(canvas, kpts, x0, y0, side, out_size, conf_thr=0.3):
    pts, conf = kpts[..., :2], kpts[..., 2]

    def to_xy(idx):
        x = int((pts[idx, 0] - x0) / side * out_size)
        y = int((pts[idx, 1] - y0) / side * out_size)
        return x, y

    # Body skeleton (upper body only - matches BODY_IDS)
    for a, b in BODY_EDGES_RAW:
        if conf[a] > conf_thr and conf[b] > conf_thr:
            cv2.line(canvas, to_xy(a), to_xy(b), (255, 255, 255), 2)
    for i in BODY_IDS:
        if conf[i] > conf_thr:
            cv2.circle(canvas, to_xy(i), 3, (0, 255, 0), -1)

    # Face (FACE_IDS = 23..39 + 53)  - dots only
    for i in FACE_IDS:
        if conf[i] > conf_thr * 0.6:
            cv2.circle(canvas, to_xy(i), 1, (180, 180, 180), -1)

    # Mouth (MOUTH_IDS = 83..90)
    for i in MOUTH_IDS:
        if conf[i] > conf_thr * 0.6:
            cv2.circle(canvas, to_xy(i), 1, (0, 255, 255), -1)

    # Hands - LEFT_HAND_IDS = 91..111, RIGHT_HAND_IDS = 112..132
    for a, b in HAND_EDGES_LOCAL:
        ai, bi = LEFT_HAND_IDS[a], LEFT_HAND_IDS[b]
        if conf[ai] > 0.2 and conf[bi] > 0.2:
            cv2.line(canvas, to_xy(ai), to_xy(bi), (0, 200, 255), 1)
    for i in LEFT_HAND_IDS:
        if conf[i] > 0.2:
            cv2.circle(canvas, to_xy(i), 2, (0, 200, 255), -1)
    for a, b in HAND_EDGES_LOCAL:
        ai, bi = RIGHT_HAND_IDS[a], RIGHT_HAND_IDS[b]
        if conf[ai] > 0.2 and conf[bi] > 0.2:
            cv2.line(canvas, to_xy(ai), to_xy(bi), (255, 200, 0), 1)
    for i in RIGHT_HAND_IDS:
        if conf[i] > 0.2:
            cv2.circle(canvas, to_xy(i), 2, (255, 200, 0), -1)
    return canvas


def overlay_text(canvas_bgr, top_label, bottom_text, header_color, font_path):
    img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    H, W = canvas_bgr.shape[:2]
    font_big = load_font(font_path, max(14, H // 26))
    font_small = load_font(font_path, max(11, H // 36))
    draw.text((8, 8), top_label, font=font_big, fill=header_color)
    if bottom_text:
        max_chars = max(20, W // 14)
        if len(bottom_text) > max_chars: bottom_text = bottom_text[:max_chars] + '...'
        draw.text((8, H - max(20, H // 22)), bottom_text, font=font_small, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pose', required=True)
    p.add_argument('--vtt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--canvas', type=int, default=512)
    p.add_argument('--fps', type=float, default=FPS)
    p.add_argument('--font', default=None)
    args = p.parse_args()

    poses = np.load(args.pose)
    cues = parse_vtt(args.vtt)
    T = poses.shape[0]
    font_path = args.font or find_cjk_font()
    x0, y0, side = auto_fit_77(poses)
    print(f'Pose T={T}, cues={len(cues)}, DATASET={DATASET}, font={font_path}')
    print(f'window: x0={x0:.1f} y0={y0:.1f} side={side:.1f}')

    out_size = args.canvas
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (out_size, out_size))

    for t in range(T):
        canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
        draw_77(canvas, poses[t], x0, y0, side, out_size)
        time_s = t / args.fps
        active = next((c for c in cues if c['start'] <= time_s <= c['end']), None)
        if active:
            top = f'[SIGN]  t={time_s:5.2f}s  f={t}/{T}'
            color = (0, 255, 0)
            bottom = active['text']
        else:
            top = f'[BG/PAUSE]  t={time_s:5.2f}s  f={t}/{T}'
            color = (180, 180, 180)
            bottom = ''
        canvas = overlay_text(canvas, top, bottom, color, font_path)
        writer.write(canvas)

    writer.release()
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
