'''Synthesize streaming sign-language datasets from offline pre-segmented pose pickles.

Co-articulated streams via four minimal-knob mechanisms:
  1. **trim_rest** strips leading/trailing rest frames (wrists below shoulders) from each
     clip. Removes the isolated-recording "preparation" and "retraction" phases that would
     otherwise produce an obviously detectable hands-down boundary at every join.
  2. **Empirical pause sampling** draws inter-clip gaps directly from the BOBSL manual
     annotations (~74% are exactly 0 -> co-articulated boundary, ~26% are positive with a
     heavy right tail).
  3. **BG_pre / BG_post sampled from positives only**: the silent broadcast lead-in and
     lead-out are conceptually distinct from inter-sentence joins, and must always be
     visibly present in the stream. Sampling from the positive-only subset of BOBSL gaps
     enforces this without introducing a min-duration knob.
  4. **K-per-stream from a 60s sliding window** (= 4x model training window) so synthesized
     streams span multiple windows and the streaming inference behaviour actually fires at
     evaluation time rather than collapsing to single-window decoding.
     
A short Hermite bridge (>= MIN_BRIDGE_FRAMES) connects every seam smoothly even when the
sampled pause is 0; tangent magnitude is damped inversely with bridge length so long BG
segments do not extrapolate clip-end momentum into visible overshoot.

Drop-in BOBSL-style outputs:
    POSE_ROOT/<stream_id>.npy               # (T, 133, 3) float32 at 12.5fps in NATIVE pixel coords
    VTT_DIR/<stream_id>.vtt                 # WEBVTT, one sentence per cue (single-line text)
    SUBSET_JSON                             # {"train": [...], "val": [...], "test": [...]}
    manifest.json                           # provenance: signer/clip ids per stream + seed

Run:
    python -m data_synth.synthesize_streams --dataset PHOENIX --out_root data/synth/phoenix
    python -m data_synth.synthesize_streams --dataset CSL     --out_root data/synth/csl
'''
import argparse, json, pickle, re
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_META, WINDOW_DURATION_SECONDS, FPS

# Minimum signer-pure pool size to be included in synthesis 
#   (otherwise the signer is dropped entirely since we can't form a multi-clip stream from them). 
# This is a tuning knob to trade off diversity (# signers) vs co-articulation realism (multi-clip streams per signer). 
# Setting it to 2 is a minimal requirement for co-articulation, 
#   and already excludes some very small pools with only one clip. 
# Setting it higher would exclude more signers and reduce diversity, 
#   but might improve co-articulation if those signers had many single-clip pools. 
# Setting it to 1 would include all signers but allow some single-clip streams with no co-articulation.
MIN_POOL_SIZE = 2  
MIN_BRIDGE_FRAMES = 2  # Numerical floor for Hermite interp to define a transition (not a tuning knob)

# Trim-rest geometry: COCO-WholeBody-133 indices for shoulders and wrists. A frame is "rest" when
# both wrists sit BELOW (image y-down: larger y) the shoulder line. Pure geometric rule, scales
# automatically across canvases by being relative to each signer's own shoulders.
SHOULDER_IDS = [5, 6]
WRIST_IDS = [9, 10]

# --- Loading & signer grouping ---------------------------------------------

def signer_id(dataset: str, clip_name: str) -> str:
    if dataset == 'PHOENIX': return re.sub(r'-\d+$', '', clip_name.split('/', 1)[-1])
    if dataset == 'CSL':
        m = re.search(r'_P(\d+)_', clip_name)
        return f'P{m.group(1)}' if m else clip_name
    raise ValueError(dataset)


def group_by_signer(clips: Dict[str, dict], dataset: str) -> Dict[str, List[str]]:
    g: Dict[str, List[str]] = defaultdict(list)
    for name in clips.keys():
        g[signer_id(dataset, name)].append(name)
    return g


# --- Pose handling: native pixel space throughout --------------------------

def to_numpy_kpts(rec: dict) -> np.ndarray:
    kp = rec['keypoint']
    if hasattr(kp, 'cpu'): kp = kp.cpu().numpy()
    kp = np.asarray(kp, dtype=np.float32)
    if kp.ndim != 3 or kp.shape[1] != 133 or kp.shape[2] != 3:
        raise ValueError(f'Unexpected keypoint shape {kp.shape}')
    kp[..., 2] = np.clip(kp[..., 2], 0.0, 1.0)
    return kp


def trim_rest(kpts: np.ndarray) -> np.ndarray:
    '''Drop contiguous leading/trailing "rest" frames at clip boundaries.

    Real continuous signing exhibits **co-articulation**: signers do not return to a full
    rest pose between adjacent sentences in the same utterance. Each pose clip is recorded
    in isolation, so it begins with a "preparation" phase (hands rising from lap to first
    sign) and ends with a "retraction" phase (hands falling back to lap). When clips are
    concatenated naively, these isolation artefacts produce an obvious "hands-down -> long
    pause -> hands-up" boundary that the localization head can latch onto trivially. By
    trimming them, the synthesized stream looks like continuous broadcast signing where
    sentence boundaries are *kinematically continuous* and must be inferred from sign
    content, not from a free pause cue.

    Rule (zero hyperparameters): a frame is "rest" iff both wrists sit BELOW the shoulder
    line in image-y coordinates. Pure geometry, signer-relative, scales automatically across
    canvases. Trimming is contiguous-only (we never cut mid-clip) and respects a 3-frame
    floor so the residual clip remains usable for Hermite endpoint-velocity estimation.
    '''
    T = kpts.shape[0]
    if T < 3: return kpts
    sh_y = kpts[:, SHOULDER_IDS, 1].mean(axis=1)
    wr_y = kpts[:, WRIST_IDS, 1].mean(axis=1)
    sh_c = kpts[:, SHOULDER_IDS, 2].min(axis=1)
    wr_c = kpts[:, WRIST_IDS, 2].min(axis=1)
    valid = (sh_c > 0) & (wr_c > 0)
    is_rest = (wr_y > sh_y) & valid  # image y-down: larger y == lower in image == hands below shoulders
    start = 0
    while start < T and is_rest[start]: start += 1
    end = T
    while end > start and is_rest[end - 1]: end -= 1
    if end - start < 3: return kpts  # safeguard: don't reduce clip below 3 frames
    return kpts[start:end]


def resample_to_fps(kpts: np.ndarray, src_fps: float, tgt_fps: float = FPS) -> np.ndarray:
    '''Vectorized linear resample on (x, y), nearest-neighbour on confidence.

    Replaces a 133-joint Python loop (~266 per-clip np.interp calls) with a single set of
    NumPy index/broadcast ops. Result is mathematically identical to the prior per-joint
    implementation (linear interp on x/y with side='left' nearest-neighbour for confidence).
    '''
    if abs(src_fps - tgt_fps) < 1e-6 or kpts.shape[0] < 2: return kpts
    T_src = kpts.shape[0]
    duration = T_src / src_fps
    T_tgt = max(1, int(round(duration * tgt_fps)))
    src_t = np.arange(T_src, dtype=np.float64) / src_fps
    tgt_t = np.clip(np.arange(T_tgt, dtype=np.float64) / tgt_fps, src_t[0], src_t[-1])

    # Linear interp on x, y: locate bracketing source indices and per-target weight.
    lo = np.clip(np.searchsorted(src_t, tgt_t, side='right') - 1, 0, T_src - 2)
    hi = lo + 1
    span = src_t[hi] - src_t[lo]
    span[span == 0] = 1.0
    frac = ((tgt_t - src_t[lo]) / span).astype(np.float32)[:, None, None]
    out = np.empty((T_tgt, 133, 3), dtype=np.float32)
    out[..., :2] = (1.0 - frac) * kpts[lo][..., :2] + frac * kpts[hi][..., :2]
    # Confidence: nearest-neighbour, matching prior `searchsorted side='left'` semantics.
    near = np.clip(np.searchsorted(src_t, tgt_t, side='left'), 0, T_src - 1)
    out[..., 2] = kpts[near, :, 2]
    return out


# --- Stream synthesis ------------------------------------------------------

def hermite_interp_segment(
    p0: np.ndarray, p1: np.ndarray,
    v0: np.ndarray, v1: np.ndarray,
    n: int,
) -> np.ndarray:
    '''Cubic Hermite spline interpolation per joint on (x, y); confidence = element-wise min.

    Endpoints `p0`, `p1` are real frames from the data (last frame of clip A, first frame of clip
    B). Tangents `v0`, `v1` are the per-frame displacements at those endpoints (clip_A[-1] -
    clip_A[-2] and clip_B[1] - clip_B[0]) so the interpolation matches the signer's actual
    velocity at each side of the seam. This gives C1 continuity at boundaries: the hand motion
    at the end of the spoken sentence flows smoothly into the pause and out into the next sentence.

    Tangents are scaled to the spline parameter s in [0, 1] (so segment length-aware) and clamped
    to at most 2 * |p1 - p0| per joint to prevent overshoot when endpoint velocities are large.

    Zero free parameters: everything is derived from the data.
    '''
    if n <= 0: return np.zeros((0, 133, 3), dtype=np.float32)
    span = (n + 1.0)  # parameter step is 1/span between frames

    # Damp endpoint-velocity contribution for LONG bridges. Short bridges (n <= MIN_BRIDGE_FRAMES)
    # keep full clip-end momentum -- this IS the co-articulation continuity we want when the
    # sampled pause was 0. For longer bridges (e.g. multi-second BG segments) the full per-frame
    # velocity propagated across many frames extrapolates the clip's last motion far beyond where
    # the hand should physically end up, producing visible overshoot. Damping inversely with n
    # makes the spline smoothly converge to a position-only interpolation as the bridge grows.
    v_scale = float(MIN_BRIDGE_FRAMES) / max(n, MIN_BRIDGE_FRAMES)
    m0 = (v0[:, :2] * span * v_scale).astype(np.float32)
    m1 = (v1[:, :2] * span * v_scale).astype(np.float32)

    # Clamp tangent magnitude to 2 * |p1 - p0| per joint to prevent overshoot
    delta = (p1[:, :2] - p0[:, :2]).astype(np.float32)
    max_mag = 2.0 * np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-3
    for m in (m0, m1):
        mag = np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9
        scale = np.minimum(1.0, max_mag / mag)
        m *= scale  # in-place clamp

    out = np.zeros((n, 133, 3), dtype=np.float32)
    conf = np.minimum(p0[:, 2], p1[:, 2]).astype(np.float32)
    for t in range(n):
        s = (t + 1.0) / span  # s in (0, 1)
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        out[t, :, :2] = h00 * p0[:, :2] + h10 * m0 + h01 * p1[:, :2] + h11 * m1
        out[t, :, 2] = conf
    return out


def endpoint_velocity(clip: np.ndarray, end: str) -> np.ndarray:
    '''Per-frame (x, y, conf) velocity at a clip endpoint. `end` in {'last', 'first'}.

    Returns a (133, 3) array with v[:, :2] = displacement, v[:, 2] = 0 (unused).
    '''
    if clip.shape[0] < 2: return np.zeros((133, 3), dtype=np.float32)
    if end == 'last': v = (clip[-1] - clip[-2]).astype(np.float32)
    elif end == 'first': v = (clip[1] - clip[0]).astype(np.float32)
    else: raise ValueError(end)
    v[:, 2] = 0.0
    return v


def sample_duration_s(rng: np.random.Generator, pause: dict) -> float:
    '''Direct empirical sample from the BOBSL inter-subtitle gap array.

    No LogNormal fit, no min/max clamp, no `pause_min_s`. The empirical array already encodes
    BOBSL's natural shape: ~74% of inter-subtitle gaps are exactly 0 (touching/overlapping
    subtitles = continuous co-articulated signing) and ~26% are positive with a heavy right
    tail. Sampling directly preserves both modes by construction.
    '''
    return float(rng.choice(pause['samples_s']))


def sample_bg_duration_s(rng: np.random.Generator, pause: dict) -> float:
    '''Sample a BG_pre/BG_post duration from the positive-only subset of BOBSL gaps.

    Inter-clip pauses can legitimately be 0s (continuous co-articulated signing), but BG segments
    represent the silent broadcast preamble where there is no caption -- they should always be
    visibly present in the stream. Sampling from positives only enforces this by construction
    without introducing a min-duration knob: the floor is whatever the smallest positive BOBSL
    inter-subtitle gap is.
    '''
    return float(rng.choice(pause['bg_samples_s']))


def synth_one_stream(
    rng: np.random.Generator, signer_pool: List[str],
    clips: Dict[str, dict], src_fps: float, pause: dict, k_range: Tuple[int, int],
) -> Tuple[np.ndarray, List[Tuple[float, float, str]], dict]:
    k_lo, k_hi = k_range
    K = int(rng.integers(min(k_lo, len(signer_pool)), min(k_hi, len(signer_pool)) + 1))
    chosen = list(rng.choice(signer_pool, size=K, replace=False))

    # Resample + trim_rest each chosen clip. Trim removes the isolated-clip preparation/retraction
    # frames so concatenated streams expose co-articulated boundaries instead of obvious "hands-down
    # -> long pause -> hands-up" artefacts.
    resampled, texts, keep_names = [], [], []
    for name in chosen:
        rec = clips[name]
        text = str(rec.get('text', '')).strip()
        if not text: continue
        kp = trim_rest(resample_to_fps(to_numpy_kpts(rec), src_fps))
        if kp.shape[0] < int(1.0 * FPS): continue # drop sub-1s clips (matches loader MIN_SUB_DURATION)
        resampled.append(kp)
        texts.append(text)
        keep_names.append(name)

    if not resampled: return np.zeros((0, 133, 3), dtype=np.float32), [], {'clips': [], 'K': 0}

    # Phantom clips for BG_pre and BG_post: random clips from this signer's pool that are NOT in `chosen`.
    # We deliberately do NOT trim_rest the phantoms -- BG segments represent the broadcast lead-in /
    # lead-out where the signer naturally holds a rest pose, so retaining the rest frames at the
    # phantom endpoint is realistic. If the pool has no spare clips, fall back to using the first/last
    # selected clip itself (the seam will still be Hermite-interpolated rather than a held frame).
    spare = [n for n in signer_pool if n not in chosen]
    if spare:
        phantom_left_name = str(rng.choice(spare))
        phantom_right_name = str(rng.choice(spare))
        phantom_left = resample_to_fps(to_numpy_kpts(clips[phantom_left_name]), src_fps)
        phantom_right = resample_to_fps(to_numpy_kpts(clips[phantom_right_name]), src_fps)
    else:
        phantom_left_name = phantom_right_name = '<self>'
        phantom_left = resampled[-1] if len(resampled) > 1 else resampled[0]
        phantom_right = resampled[0] if len(resampled) > 1 else resampled[-1]

    segments: List[np.ndarray] = []
    cues: List[Tuple[float, float, str]] = []
    durations: dict = {'pre_s': 0.0, 'pauses_s': [], 'post_s': 0.0}
    cur = 0

    # BG_pre = Hermite interp from phantom_left[-1] -> resampled[0][0] (signer "transitioning into"
    # the first sentence). Animated, biomechanically grounded in real signer poses. Sampled from
    # the POSITIVE-only subset of BOBSL gaps so BG segments are always visibly present in the
    # stream -- representing the silent broadcast lead-in, not an inter-sentence join.
    L_pre_s = sample_bg_duration_s(rng, pause)
    n_pre = max(MIN_BRIDGE_FRAMES, int(round(L_pre_s * FPS)))
    bg_pre = hermite_interp_segment(
        phantom_left[-1], resampled[0][0],
        endpoint_velocity(phantom_left, 'last'),
        endpoint_velocity(resampled[0], 'first'),
        n_pre,
    )
    segments.append(bg_pre)
    durations['pre_s'] = L_pre_s
    cur += bg_pre.shape[0]

    for i, (clip, text) in enumerate(zip(resampled, texts)):
        segments.append(clip)
        cues.append((cur / FPS, (cur + clip.shape[0]) / FPS, text))
        cur += clip.shape[0]
        if i < len(resampled) - 1:
            L_pause_s = sample_duration_s(rng, pause)
            n_pause = max(MIN_BRIDGE_FRAMES, int(round(L_pause_s * FPS)))
            pause_seg = hermite_interp_segment(
                clip[-1], resampled[i + 1][0],
                endpoint_velocity(clip, 'last'),
                endpoint_velocity(resampled[i + 1], 'first'),
                n_pause,
            )
            segments.append(pause_seg)
            cur += pause_seg.shape[0]
            durations['pauses_s'].append(L_pause_s)

    # BG_post = Hermite interp from resampled[-1][-1] -> phantom_right[0] (broadcast lead-out;
    # same positive-only sampling as BG_pre).
    L_post_s = sample_bg_duration_s(rng, pause)
    n_post = max(MIN_BRIDGE_FRAMES, int(round(L_post_s * FPS)))
    bg_post = hermite_interp_segment(
        resampled[-1][-1], phantom_right[0],
        endpoint_velocity(resampled[-1], 'last'),
        endpoint_velocity(phantom_right, 'first'),
        n_post,
    )
    segments.append(bg_post)
    cur += bg_post.shape[0]
    durations['post_s'] = L_post_s

    poses = np.concatenate(segments, axis=0).astype(np.float32)
    prov = {
        'clips': keep_names, 'K': len(cues),
        'phantom_left': phantom_left_name, 'phantom_right': phantom_right_name,
        'duration_s': float(poses.shape[0] / FPS), **durations,
    }
    return poses, cues, prov


# --- Output writers --------------------------------------------------------

def fmt_vtt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'


def write_vtt(path: Path, cues: List[Tuple[float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['WEBVTT', '']
    for s, e, text in cues:
        lines.append(f'{fmt_vtt_time(s)} --> {fmt_vtt_time(e)}')
        lines.append(' '.join(text.split()))
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_pose(out_dir: Path, stream_id: str, poses: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f'{stream_id}.npy', poses.astype(np.float32))


# --- Per-split orchestration -----------------------------------------------

def load_pause_dist(stats_path: str) -> dict:
    '''Load BOBSL empirical inter-subtitle gap samples (replaces the old LogNormal parameters).

    Looks for `bobsl_gap_samples.npy` next to the stats JSON. If absent, falls back to an
    analytic emulation that reproduces the real BOBSL shape (~74% zero + LogNormal positive
    tail) so the pipeline still runs without the optional samples file.

    Returned dict: {'samples_s': np.ndarray, 'frac_zero': float, 'median_s': float, 'p90_s': float}
    '''
    samples_path = Path(stats_path).with_name('bobsl_gap_samples.npy')
    if samples_path.exists():
        samples = np.load(samples_path).astype(np.float32)
        samples = np.clip(samples, 0.0, None)  # negative gaps (overlapping subs) = co-articulated, treat as 0
        src = f'BOBSL empirical n={samples.size}'
    else:
        rng = np.random.default_rng(0)
        n = 10000
        is_pos = rng.random(n) > 0.74  # match observed BOBSL ~26% positive-gap fraction
        samples = np.where(is_pos, rng.lognormal(np.log(2.0), 0.83), 0.0).astype(np.float32)
        src = f'fallback (no {samples_path.name}; emulating 74% zero + LogNormal positive tail)'
    # Positive-only subset: used for BG_pre/BG_post (broadcast lead-in / lead-out). Inter-clip
    # pauses use the full empirical (which is ~74% zeros, capturing co-articulation), but BG
    # segments are conceptually different -- they represent the silent broadcast preamble where
    # there is no caption, not an inter-sentence join. Sampling them from the full empirical
    # collapses ~74% of streams to a 2-frame BG (invisible). Sampling from positives only keeps
    # them broadcast-realistic (median ~2s, heavy tail) without adding new knobs.
    bg_samples = samples[samples > 0]
    if bg_samples.size == 0: bg_samples = np.array([2.0], dtype=np.float32)  # degenerate fallback
    info = {
        'samples_s': samples,
        'bg_samples_s': bg_samples,
        'frac_zero': float((samples == 0).mean()),
        'median_s': float(np.median(samples)),
        'p90_s': float(np.percentile(samples, 90)),
        'bg_median_s': float(np.median(bg_samples)),
        'bg_p90_s': float(np.percentile(bg_samples, 90)),
    }
    print(f"(pause samples: {src}; frac_zero={info['frac_zero']:.3f}, "
          f"median={info['median_s']:.2f}s, p90={info['p90_s']:.2f}s; "
          f"BG (positive only): median={info['bg_median_s']:.2f}s, p90={info['bg_p90_s']:.2f}s)")
    return info


def load_k_range(stats_path: str) -> Tuple[int, int]:
    '''Sentences-per-stream range derived from BOBSL "subs per 15s sliding window" stats.

    If `bobsl_gap_stats.json` does not contain `subs_per_window_p10` / `subs_per_window_p90`
    (older versions), fall back to deriving K from `sub_dur_median_s` + `gap_median_s`:
        K_typical = WINDOW_DURATION_SECONDS / (sub_dur_median + max(gap_median_pos, 0))
        K_lo = max(1, K_typical // 2);  K_hi = K_typical * 2
    Otherwise use the empirical p10..p90 range from the BOBSL distribution.
    '''
    p = Path(stats_path)
    if p.exists():
        s = json.loads(p.read_text())
        # Prefer subs_per_STREAM_window (60s) — anchors K to a stream length that spans multiple
        # 15s training windows so the streaming inference behaviour actually fires at eval time.
        if 'subs_per_stream_window_p10' in s and 'subs_per_stream_window_p90' in s:
            lo = max(2, int(s['subs_per_stream_window_p10']))
            hi = max(lo + 1, int(s['subs_per_stream_window_p90']))
            sw = float(s.get('stream_window_seconds', 60.0))
            print(f'(BOBSL K range: [{lo}, {hi}] from subs_per_{sw:.0f}s_window p10..p90)')
            return lo, hi
        if 'subs_per_window_p10' in s and 'subs_per_window_p90' in s:
            lo = max(1, int(s['subs_per_window_p10']))
            hi = max(lo + 1, int(s['subs_per_window_p90']))
            print(f'(BOBSL K range: [{lo}, {hi}] from subs_per_15s_window p10..p90 [no 60s stats found])')
            return lo, hi
        sub_med = float(s.get('sub_dur_median_s', 4.0))
        # gap_median_s is often 0 in BOBSL because most subs are adjacent; use exp(log_pos_mean) as positive median
        gap_med_pos = float(np.exp(s.get('log_pos_mean', np.log(2.0))))
        cycle = max(2.0, sub_med + gap_med_pos)
        k_typ = max(2, int(round(WINDOW_DURATION_SECONDS / cycle)))
        lo, hi = max(1, k_typ // 2), max(k_typ + 1, k_typ * 2)
        print(f'(K range derived from sub_dur_median={sub_med:.1f}s + gap_median_pos={gap_med_pos:.1f}s '
              f'-> K_typical={k_typ}, range [{lo}, {hi}])')
        return lo, hi
    print('(no BOBSL stats; K range fallback [2, 10])')
    return 2, 10


def synthesize_split(
    dataset: str, split: str, out_pose_dir: Path, out_vtt_dir: Path,
    base_seed: int, pause: dict, k_range: Tuple[int, int],
) -> Tuple[List[str], List[dict]]:
    src_fps = DATASET_META[dataset]['src_fps']
    load_split = 'dev' if split == 'val' else split
    meta = DATASET_META[dataset]
    pickle_path = meta['pickle_dir'] / f"{meta['pickle_prefix']}.{load_split}"
    
    print(f'[{dataset}/{split}] loading {pickle_path}')
    with open(pickle_path, 'rb') as f:
        clips = pickle.load(f)
        
    min_src_frames = int(1.0 * src_fps)
    clips = {k: v for k, v in clips.items() if str(v.get('text', '')).strip() and v.get('num_frames', 0) >= min_src_frames}
    groups = group_by_signer(clips, dataset)
    groups = {sid: lst for sid, lst in groups.items() if len(lst) >= MIN_POOL_SIZE}
    n_usable = sum(len(v) for v in groups.values())
    k_avg = (k_range[0] + k_range[1]) / 2.0
    # Each clip used ~once on average across the split (n_usable / K_avg). Preserves the offline
    # split's contract: no clip duplication beyond what's statistically inevitable. We deliberately
    # do NOT inflate this with a permutation-multiplier knob -- on small signer pools (PHOENIX
    # broadcasts have only 2-3 clips each) any multiplier above 1 saturates the K! permutation
    # space and biases training toward duplicated streams.
    n_streams = max(1, int(round(n_usable / k_avg)))
    print(f'[{dataset}/{split}] usable clips: {n_usable}, signer-pure groups: {len(groups)} -> {n_streams} streams')
    if not groups: raise RuntimeError(f'No signer groups for {dataset}/{split}')

    rng = np.random.default_rng(base_seed)
    signer_ids = list(groups.keys())
    stream_ids: List[str] = []
    manifest: List[dict] = []
    
    for i in tqdm(range(n_streams), desc=f'synth {split}'):
        stream_rng = np.random.default_rng([base_seed, i])
        sid = signer_ids[int(rng.integers(0, len(signer_ids)))]
        poses, cues, prov = synth_one_stream(stream_rng, groups[sid], clips, src_fps, pause, k_range)
        if not cues: continue
        stream_id = f'{split}_{i:05d}'
        write_pose(out_pose_dir, stream_id, poses)
        write_vtt(out_vtt_dir / f'{stream_id}.vtt', cues)
        prov.update({'stream_id': stream_id, 'signer': sid, 'split': split})
        manifest.append(prov)
        stream_ids.append(stream_id)
    return stream_ids, manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['PHOENIX', 'CSL'])
    p.add_argument('--out_root', required=True)
    p.add_argument('--bobsl_gap_stats', default='data_synth/stats/bobsl_gap_stats.json',
                   help='JSON of BOBSL manual-aligned pause statistics; if missing, use fallback.')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    
    out_root = Path(args.out_root)
    out_pose_dir = out_root / 'poses'
    out_vtt_dir = out_root / 'vtt'
    out_pose_dir.mkdir(parents=True, exist_ok=True)
    out_vtt_dir.mkdir(parents=True, exist_ok=True)

    pause = load_pause_dist(args.bobsl_gap_stats)
    k_range = load_k_range(args.bobsl_gap_stats)
    splits_cfg = [('train', args.seed), ('val', args.seed + 1), ('test', args.seed + 2)]
    subset: Dict[str, List[str]] = {}
    full_manifest: Dict[str, List[dict]] = {}
    for split, seed in splits_cfg:
        ids, m = synthesize_split(args.dataset, split, out_pose_dir, out_vtt_dir, seed, pause, k_range)
        subset[split] = ids
        full_manifest[split] = m

    (out_root / 'subset2episode.json').write_text(json.dumps(subset, indent=2))
    (out_root / 'manifest.json').write_text(json.dumps({
        'dataset': args.dataset,
        'src_meta': {k: (str(v) if isinstance(v, Path) else v) for k, v in DATASET_META[args.dataset].items()},
        'target_fps': FPS,
        # Drop the raw samples arrays from manifest -- store only the summary statistics that
        # describe the empirical pause distribution actually used during synthesis.
        'pause': {k: v for k, v in pause.items() if k not in ('samples_s', 'bg_samples_s')},
        'k_range': list(k_range), 'min_pool_size': MIN_POOL_SIZE, 'window_seconds': WINDOW_DURATION_SECONDS,
        'min_bridge_frames': MIN_BRIDGE_FRAMES, 'trim_rest': True,
        'streams': full_manifest,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nDone. Wrote {sum(len(v) for v in subset.values())} streams to {out_root}")


if __name__ == '__main__':
    main()
