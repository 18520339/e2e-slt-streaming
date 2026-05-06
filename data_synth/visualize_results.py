'''Render MP4s with pose, GT timeline (top), and predicted timeline (bottom).

Reads `by_stream.json` produced by `evaluation.dump_predictions` plus the original pose
.npy and GT .vtt for each stream. The predicted timeline shows two rows:

    [top row]   NMS-pruned predictions (opaque, paper-quality view)
    [bottom row] ALL raw predictions across overlapping windows (translucent stacked
                 bands, readable even when ~10 events overlap at one instant)

Both views are kept so the professor can compare side-by-side what the raw model emits
vs the deduped output. Captions of currently-active cues are overlaid below each
timeline (top: GT, bottom: predictions). PIL is used for CJK-capable text.

Each row is labeled in a left margin and a legend strip at the bottom names every
color so the video is self-explanatory without a separate caption.

By default this renders ALL streams in `by_stream.json`. Pass `--max_samples N` to
render a random subset of size N (seeded for reproducibility).

CJK note: on Colab/Linux, install a CJK-capable font once before rendering CSL/H2S:
    apt-get install -y fonts-noto-cjk
This script auto-detects the installed font; without one, Chinese characters render
as missing-glyph boxes (the script prints a warning when this happens).

Usage:
    DATASET=PHOENIX python -m data_synth.visualize_results \
        --pose_root data/synth/phoenix/poses --vtt_root data/synth/phoenix/vtt \
        --predictions data_synth/eval_dumps/phoenix/test/by_stream.json \
        --out_dir data_synth/eval_videos/phoenix/test
        # add  --max_samples 20  to render only 20 random streams

    DATASET=CSL  ...
    DATASET=H2S  ...
'''
import os, sys, argparse, json, random, cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_synth.visualize_stream import find_cjk_font, load_font, auto_fit_77, draw_77
from utils import parse_vtt
from config import FPS, DATASET


WIDTH = 640
LEFT_MARGIN = 100          # space for row labels "GT" / "Pred (NMS)" / "Pred (raw)"
PLOT_W = WIDTH - LEFT_MARGIN
LAYOUT = {
    'header_h':     30,
    'pose_h':       380,
    'gt_text_h':    34,    # one tall line of GT caption
    'gt_strip_h':   24,
    'pred_strip_h': 40,    # 2 rows: 22 (NMS) + 4 gap + 26 (raw stack)
    'pred_text_h':  34,    # up to 2 lines of pred caption
}
def hex_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


COLOR_GT       = hex_to_bgr((80, 200, 120))   # green
COLOR_NMS      = hex_to_bgr((255, 200, 0))    # amber
COLOR_RAW      = hex_to_bgr((200, 120, 255))  # violet
COLOR_CURSOR   = hex_to_bgr((255, 255, 255))  # white
COLOR_BG       = hex_to_bgr((22, 22, 28))
COLOR_LABEL    = (220, 220, 220)              # PIL RGB for left-margin labels
COLOR_TXT_GT   = (170, 230, 180)
COLOR_TXT_PRED = (255, 220, 150)
COLOR_TXT_DIM  = (160, 160, 160)
CJK_PROBE = '中文'  # if width collapses, font lacks CJK glyphs


def font_supports_cjk(font_path: str) -> bool:
    if not font_path or not os.path.exists(font_path): return False
    try: f = ImageFont.truetype(font_path, 18)
    except Exception: return False
    try:
        if hasattr(f, 'getbbox'):
            bb = f.getbbox(CJK_PROBE)
            return (bb[2] - bb[0]) > 6
        w, _ = f.getsize(CJK_PROBE)
        return w > 6
    except Exception: return False


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, 'textbbox'):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    return draw.textsize(text, font=font)


# ---------- Drawing primitives ----------

def time_to_x(t: float, total_dur: float) -> int:
    if total_dur <= 0: return LEFT_MARGIN
    return LEFT_MARGIN + int(t / total_dur * (PLOT_W - 1))


def draw_timeline_row(canvas, y_top: int, y_bot: int, total_dur: float, cues: List[Dict],
                      color, alpha: float = 1.0, outline: bool = True):
    if total_dur <= 0 or y_bot <= y_top: return
    overlay = canvas.copy() if alpha < 1.0 else canvas
    # Backing rail (dark) so the band area is obvious even when no cues land there.
    cv2.rectangle(overlay, (LEFT_MARGIN, y_top), (LEFT_MARGIN + PLOT_W - 1, y_bot), (50, 50, 56), thickness=-1)
    for c in cues:
        x0 = time_to_x(c['start'], total_dur)
        x1 = time_to_x(c['end'],   total_dur)
        x0 = max(LEFT_MARGIN, min(LEFT_MARGIN + PLOT_W - 1, x0))
        x1 = max(x0 + 1,      min(LEFT_MARGIN + PLOT_W - 1, x1))
        cv2.rectangle(overlay, (x0, y_top), (x1, y_bot), color, thickness=-1)
        if outline: cv2.rectangle(overlay, (x0, y_top), (x1, y_bot), (0, 0, 0), thickness=1)
    if alpha < 1.0: cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)
    cv2.line(canvas, (LEFT_MARGIN, y_bot), (LEFT_MARGIN + PLOT_W - 1, y_bot), (90, 90, 90), 1)


def pil_text(canvas_bgr, items: List[Tuple[Tuple[int, int], str, Tuple[int, int, int], int]], font_path: str):
    # Batch-draw multiple PIL text snippets on a single BGR canvas in one PIL pass.
    # items: list of ((x, y), text, rgb_color, font_size)
    if not items: return canvas_bgr
    img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for (x, y), text, color, size in items:
        font = load_font(font_path, size)
        draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def wrap_text_by_width(text: str, font: ImageFont.ImageFont, draw: ImageDraw.ImageDraw, 
                       max_w: int, max_lines: int = 2) -> List[str]:
    if not text: return []
    # Try whole-string first.
    tw, _ = _text_size(draw, text, font)
    if tw <= max_w: return [text]
    # Word wrap; for CJK (no spaces) fall through to char wrap.
    words = text.split(' ') if ' ' in text else list(text)
    sep = ' ' if ' ' in text else ''
    lines: List[str] = []
    cur = ''
    for w in words:
        trial = (cur + sep + w) if cur else w
        tw, _ = _text_size(draw, trial, font)
        if tw <= max_w: cur = trial
        else:
            if cur: lines.append(cur)
            cur = w
            if len(lines) == max_lines - 1:
                # Truncate remainder on last line.
                cur = (cur + sep + sep.join(words[words.index(w) + 1:])) if ' ' in text else (cur + ''.join(words[words.index(w) + 1:]))
                # add ellipsis if it overflows
                while cur and _text_size(draw, cur + '...', font)[0] > max_w: cur = cur[:-1]
                cur = cur + '...'
                break
    if cur: lines.append(cur)
    return lines[:max_lines]


def draw_caption_strip(canvas_bgr, y_top: int, y_bot: int, prefix: str, prefix_color,
                       text: str, text_color, font_path: str, dim_when_empty: bool = True):
    # Render "<prefix>: <text>" wrapped to fit PLOT_W; supports CJK. Returns updated canvas (PIL pass)
    if y_bot <= y_top: return canvas_bgr
    h = y_bot - y_top
    fsize = max(8, min(22, h - 8 if h >= 30 else h - 4))
    img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = load_font(font_path, fsize)
    label_font = load_font(font_path, max(8, fsize - 4))
    draw.text((6, y_top + 4), prefix, font=label_font, fill=prefix_color) # Prefix label at left
    # Wrap text into up to 2 lines starting at LEFT_MARGIN
    if not text and dim_when_empty: draw.text((LEFT_MARGIN, y_top + 4), '—', font=font, fill=COLOR_TXT_DIM)
    else:
        lines = wrap_text_by_width(text, font, draw, max_w=PLOT_W - 8, max_lines=2)
        line_h = fsize + 4
        for i, ln in enumerate(lines[:max(1, h // line_h)]):
            draw.text((LEFT_MARGIN, y_top + 2 + i * line_h), ln, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_row_label(canvas_bgr, y_top: int, y_bot: int, label: str, font_path: str):
    img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fsize = max(10, min(15, (y_bot - y_top) - 6))
    font = load_font(font_path, fsize)
    tw, th = _text_size(draw, label, font)
    x = max(2, LEFT_MARGIN - tw - 8)
    y = y_top + ((y_bot - y_top) - th) // 2
    draw.text((x, y), label, font=font, fill=COLOR_LABEL)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def active_at(cues: List[Dict], t: float) -> List[Dict]:
    return [c for c in cues if c['start'] <= t <= c['end']]


# ---------- Per-stream renderer ----------

def render_stream(sid: str, pose_path: Path, gt_cues_full: List[Dict],
                  pred_all: List[Dict], pred_nms: List[Dict],
                  out_path: Path, font_path: str, fps: float):
    poses = np.load(pose_path)
    T = poses.shape[0]
    total_dur = T / fps

    HEIGHT = sum(LAYOUT.values())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (WIDTH, HEIGHT))
    x0, y0, side = auto_fit_77(poses)
    pose_dim = min(LAYOUT['pose_h'], WIDTH)

    y = 0 # Y offsets
    y_header_top = y;  y += LAYOUT['header_h']    ;  y_header_bot = y
    y_pose_top   = y;  y += LAYOUT['pose_h']      ;  y_pose_bot   = y
    y_gtxt_top   = y;  y += LAYOUT['gt_text_h']   ;  y_gtxt_bot   = y
    y_gstr_top   = y;  y += LAYOUT['gt_strip_h']  ;  y_gstr_bot   = y
    y_pstr_top   = y;  y += LAYOUT['pred_strip_h'];  y_pstr_bot   = y
    y_ptxt_top   = y;  y += LAYOUT['pred_text_h'] ;  y_ptxt_bot   = y

    # Pred strip split into NMS row (top half) and raw row (bottom half)
    half = LAYOUT['pred_strip_h'] // 2
    pred_nms_top = y_pstr_top + 1
    pred_nms_bot = y_pstr_top + half - 2
    pred_raw_top = y_pstr_top + half + 2
    pred_raw_bot = y_pstr_bot - 1

    # Build STATIC layer (everything that doesn't change frame-to-frame).
    static = np.full((HEIGHT, WIDTH, 3), COLOR_BG, dtype=np.uint8)
    # Timelines
    draw_timeline_row(static, y_gstr_top + 2, y_gstr_bot - 2, total_dur, gt_cues_full, COLOR_GT, alpha=1.0, outline=True)
    draw_timeline_row(static, pred_nms_top, pred_nms_bot, total_dur, pred_nms, COLOR_NMS, alpha=1.0, outline=True)
    draw_timeline_row(static, pred_raw_top, pred_raw_bot, total_dur, pred_all, COLOR_RAW, alpha=0.45, outline=False)
    # Row labels (in left margin)
    static = draw_row_label(static, y_gstr_top, y_gstr_bot, 'GT', font_path)
    static = draw_row_label(static, pred_nms_top, pred_nms_bot, 'Pred (NMS)', font_path)
    static = draw_row_label(static, pred_raw_top, pred_raw_bot, 'Pred (raw)', font_path)
    # Header
    header_label = (f'[{DATASET}] {sid}   T={T}f / {total_dur:.1f}s   '
                    f'GT={len(gt_cues_full)}   pred(NMS)={len(pred_nms)}   pred(all)={len(pred_all)}')
    static = pil_text(static, [((6, y_header_top + 4), header_label, COLOR_LABEL, max(8, LAYOUT['header_h'] - 10))], font_path)

    # Per-frame loop
    for t in range(T):
        canvas = static.copy()
        time_s = t / fps

        # Pose centered horizontally
        pose_canvas = np.zeros((pose_dim, pose_dim, 3), dtype=np.uint8)
        draw_77(pose_canvas, poses[t], x0, y0, side, pose_dim)
        x_left = (WIDTH - pose_dim) // 2
        canvas[y_pose_top:y_pose_top + pose_dim, x_left:x_left + pose_dim] = pose_canvas

        # Cursors over both timelines
        cx = LEFT_MARGIN + int(time_s / max(total_dur, 1e-6) * (PLOT_W - 1))
        cv2.line(canvas, (cx, y_gstr_top), (cx, y_gstr_bot), COLOR_CURSOR, 1)
        cv2.line(canvas, (cx, y_pstr_top), (cx, y_pstr_bot), COLOR_CURSOR, 1)

        # Active GT caption
        gt_active = active_at(gt_cues_full, time_s)
        gt_text = '  |  '.join(c['text'] for c in gt_active)
        canvas = draw_caption_strip(canvas, y_gtxt_top, y_gtxt_bot, 'GT:', COLOR_TXT_GT, gt_text, COLOR_TXT_GT, font_path)

        # Active pred caption (NMS view drives the text; raw count appended)
        pr_active = active_at(pred_nms, time_s)
        pr_all_active = active_at(pred_all, time_s)
        if pr_active: pred_text = '  |  '.join(f"{c['text']}  ({c['score']:+.1f})" for c in pr_active)
        else: pred_text = ''
        if pr_all_active:
            extra = len(pr_all_active) - len(pr_active)
            if extra > 0:
                suffix = f'   (+{extra} raw band' + ('s' if extra != 1 else '') + ')'
                pred_text = (pred_text + suffix) if pred_text else f'(no NMS event;{suffix.lstrip()})'
        canvas = draw_caption_strip(canvas, y_ptxt_top, y_ptxt_bot, 'Pred:', COLOR_TXT_PRED, pred_text, COLOR_TXT_PRED, font_path)
        writer.write(canvas)
    writer.release()


def detect_cjk_in_streams(by_stream: Dict[str, Dict]) -> bool:
    for sid, info in by_stream.items():
        for cue_set in (info.get('gt', []), info.get('pred_all', []), info.get('pred_nms', [])):
            for c in cue_set:
                txt = c.get('text', '')
                for ch in txt:
                    if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿': return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose_root', required=True, help='dir with <sid>.npy files')
    ap.add_argument('--vtt_root',  required=True, help='dir with <sid>.vtt GT files (fallback to predictions json gt)')
    ap.add_argument('--predictions', required=True, help='by_stream.json from dump_predictions')
    ap.add_argument('--out_dir', required=True, help='dir to write MP4s')
    ap.add_argument('--max_samples', type=int, default=None, help='If set, render N random streams. Otherwise render all.')
    ap.add_argument('--fps', type=float, default=FPS)
    ap.add_argument('--font', default=None, help='Override font path. If unset, auto-detects a CJK-capable font on the system.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    pose_root = Path(args.pose_root)
    vtt_root  = Path(args.vtt_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font_path = args.font or find_cjk_font()
    print(f'font: {font_path or "(none — using PIL bitmap default)"}')

    by_stream: Dict[str, Dict] = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    needs_cjk = detect_cjk_in_streams(by_stream)
    if needs_cjk:
        ok = font_path is not None and font_supports_cjk(font_path)
        if not ok:
            print('WARNING: CJK characters detected in subtitles, but the resolved font lacks CJK glyphs.')
            print('         Chinese/Japanese characters will render as missing-glyph boxes.')
            print('         On Colab/Linux: apt-get install -y fonts-noto-cjk    (then re-run)')
            print('         Or pass --font /path/to/NotoSansCJK-Regular.ttc explicitly.')

    sids = sorted(by_stream.keys())
    if args.max_samples is not None and args.max_samples < len(sids):
        rng = random.Random(args.seed)
        sids = rng.sample(sids, args.max_samples)
        sids.sort()
        print(f'rendering {len(sids)} random samples (seed={args.seed})')
    else: print(f'rendering all {len(sids)} streams')

    n_done, n_skip = 0, 0
    for sid in sids:
        info = by_stream[sid]
        pose_path = pose_root / f'{sid}.npy'
        if not pose_path.exists():
            print(f'  skip {sid}: pose missing ({pose_path})')
            n_skip += 1
            continue
        # Prefer canonical VTT; fall back to GT in by_stream.json.
        vtt_path = vtt_root / f'{sid}.vtt'
        if vtt_path.exists():
            try: gt_cues = [{'start': float(c['start']), 'end': float(c['end']), 'text': str(c['text'])} for c in parse_vtt(vtt_path)]
            except Exception: gt_cues = info.get('gt', [])
        else: gt_cues = info.get('gt', [])

        out_path = out_dir / f'{sid}.mp4'
        try:
            render_stream(sid, pose_path, gt_cues, info.get('pred_all', []), info.get('pred_nms', []), out_path, font_path, args.fps)
            n_done += 1
            if n_done % 25 == 0: print(f'  rendered {n_done}/{len(sids)}')
        except Exception as e: print(f'  ERROR rendering {sid}: {e}'); n_skip += 1
    print(f'Done. wrote {n_done} videos to {out_dir} (skipped {n_skip})')


if __name__ == '__main__':
    main()
