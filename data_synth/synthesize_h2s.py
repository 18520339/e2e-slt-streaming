'''Synthesize streaming How2Sign benchmark from OpenPose JSONs + realigned CSV.

Differs from PHOENIX/CSL synth in synthesize_streams.py:
  - source = pre-extracted OpenPose-137 JSONs (one folder per sentence clip), CSV with
    realigned start/end timestamps in the original video timeline
  - per-stream grouping = VIDEO_ID (not signer)
  - timeline = REAL: clip i is placed at frame round(START_REALIGNED[i] * FPS); gaps between
    sentences are the natural inter-sentence silences from the source video
  - bridges = LINEAR (not Hermite). Per-design: avoid synthetic co-articulation since H2S
    already has real signing-rhythm gaps. Linear interp keeps the seam C0-continuous (no
    teleport) without inventing fake clip-end momentum.
  - trim_rest still applied: H2S clips are trimmed-per-sentence and end with rest poses,
    which would otherwise create a "hands-down = boundary" trivial cue.

Outputs match BOBSL/synth contract:
    data/synth/h2s/{poses,vtt,subset2episode.json,manifest.json}

Run:
    python -m data_synth.synthesize_h2s --split val --out_root data/synth/h2s
    python -m data_synth.synthesize_h2s --split train --out_root data/synth/h2s
    python -m data_synth.synthesize_h2s --split test --out_root data/synth/h2s
'''
import sys, argparse, csv, json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FPS  # 12.5 Hz target
from data_synth.op2coco import load_clip_pose
from data_synth.synthesize_streams import trim_rest, resample_to_fps
H2S_NATIVE_FPS = 30.0   # How2Sign realigned videos
MIN_BRIDGE_FRAMES = 2   # numerical floor (matches PHOENIX/CSL synth)


def linear_bridge(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    '''Linear (C0) interpolation between two (133, 3) endpoints, n intermediate frames.

    No tangent matching — H2S synth deliberately skips co-articulation simulation per
    project decision. Confidence is element-wise min between endpoints. Returns (n, 133, 3).
    '''
    if n <= 0: return np.zeros((0, 133, 3), dtype=np.float32)
    out = np.zeros((n, 133, 3), dtype=np.float32)
    conf = np.minimum(p0[:, 2], p1[:, 2]).astype(np.float32)
    for t in range(n):
        s = (t + 1.0) / (n + 1.0)
        out[t, :, :2] = (1.0 - s) * p0[:, :2] + s * p1[:, :2]
        out[t, :, 2] = conf
    return out


def write_vtt(out_vtt: Path, cues): # Write WEBVTT with 1 cue/sentence. cues = [(start_s, end_s, text), ...]
    def fmt_ts(s: float) -> str:
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"
    out_vtt.write_text( "WEBVTT\n\n" + "\n\n".join(
        f"{fmt_ts(a)} --> {fmt_ts(b)}\n{txt.replace(chr(10), ' ').strip()}"
        for a, b, txt in cues
    ) + "\n", encoding="utf-8")


def synthesize_video(rows, json_root: Path, out_pose: Path, out_vtt: Path):
    '''Build one stream from all rows belonging to a single VIDEO_ID. 
    Returns (n_cues, dur_s) or (0, 0.0) if no usable clips.

    rows: list of dict (CSV rows, sorted by START_REALIGNED).
    json_root: parent of <SENTENCE_NAME>/ folders.
    '''
    # 1. Load each clip from its json subfolder, resample to target FPS, trim_rest.
    prepared = []  # list of (start_s, end_s, sentence, pose: (T, 133, 3))
    for r in rows:
        clip_dir = json_root / r['SENTENCE_NAME']
        kp = load_clip_pose(str(clip_dir))
        if kp.shape[0] < 3: continue
        kp = resample_to_fps(kp, src_fps=H2S_NATIVE_FPS, tgt_fps=FPS)
        kp = trim_rest(kp)
        
        if kp.shape[0] < 3: continue
        start_s = float(r['START_REALIGNED'])
        end_s = float(r['END_REALIGNED'])
        if end_s <= start_s: continue
        prepared.append((start_s, end_s, r['SENTENCE'].strip(), kp))
    if not prepared: return 0, 0.0

    # 2. Build stream timeline. Stream local time t=0 corresponds to original-video t=START_REALIGNED[0].
    t0 = prepared[0][0]
    segments = []   # list of (133, 3) frame arrays in stream order
    cues = []       # (start_s, end_s, text) in stream-local seconds
    cur_frame = 0   # running frame counter on stream timeline

    for i, (start_s, end_s, text, pose) in enumerate(prepared):
        # Target stream-local placement of this clip:
        target_frame = int(round((start_s - t0) * FPS))
        # Pad with bridge frames to hit target placement (gap between previous clip and this one).
        if i == 0: assert target_frame == 0 # No leading BG: stream begins at first clip onset.
        else:
            gap_frames = max(0, target_frame - cur_frame)
            prev_pose = prepared[i - 1][3]
            if gap_frames > 0:
                # Linear bridge from last frame of prev clip → first frame of this clip.
                # If gap below MIN_BRIDGE_FRAMES, still emit linear (n>=1) — no co-articulation
                # cleanup needed since trim_rest already removed clip-end rest poses.
                bridge = linear_bridge(prev_pose[-1], pose[0], gap_frames)
                segments.append(bridge)
                cur_frame += gap_frames

        # Emit clip frames; record cue at this position.
        cue_start_s = cur_frame / FPS
        segments.append(pose)
        cur_frame += pose.shape[0]
        cue_end_s = cur_frame / FPS
        cues.append((cue_start_s, cue_end_s, text))

    # No trailing BG (no phantom): H2S streams end at last clip offset. Reviewer-friendly:
    # the stream's silent regions are exactly the original-video silences between sentences.
    if not segments: return 0, 0.0
    pose_full = np.concatenate(segments, axis=0).astype(np.float32)
    out_pose.parent.mkdir(parents=True, exist_ok=True)
    out_vtt.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_pose, pose_full)
    write_vtt(out_vtt, cues)
    return len(cues), pose_full.shape[0] / FPS


def run_split(args):
    src_root = Path(args.src_root)
    csv_path = src_root / f'how2sign_realigned_{args.split}.csv'
    json_root = src_root / 'json'
    out_root = Path(args.out_root)
    poses_dir = out_root / 'poses'
    vtt_dir = out_root / 'vtt'

    if not csv_path.exists() : raise FileNotFoundError(f'CSV not found: {csv_path}. Have you placed H2S {args.split} split at {src_root}?')
    if not json_root.exists(): raise FileNotFoundError(f'JSON dir not found: {json_root}')

    # Load + group CSV by VIDEO_ID.
    groups: dict = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            groups[row['VIDEO_ID']].append(row)

    # Drop single-sentence videos (no streaming structure to learn).
    streamable = {vid: rows for vid, rows in groups.items() if len(rows) >= 2}
    if args.min_cues > 2: streamable = {vid: rows for vid, rows in streamable.items() if len(rows) >= args.min_cues}
    manifest = {'split': args.split, 'n_input_videos': len(groups), 'n_streams_kept': 0,
                'min_cues': args.min_cues, 'src_fps_native': H2S_NATIVE_FPS, 'fps_target': FPS, 'streams': {}}
    kept_ids = []

    for vid, rows in tqdm(sorted(streamable.items()), desc=f'H2S/{args.split}'):
        rows.sort(key=lambda r: float(r['START_REALIGNED']))
        out_pose = poses_dir / f'{vid}.npy'
        out_vtt = vtt_dir / f'{vid}.vtt'
        n_cues, dur_s = synthesize_video(rows, json_root, out_pose, out_vtt)
        if n_cues == 0: continue
        manifest['streams'][vid] = {'n_cues': n_cues, 'duration_s': dur_s, 'sentence_ids': [r['SENTENCE_ID'] for r in rows]}
        kept_ids.append(vid)

    manifest['n_streams_kept'] = len(kept_ids)
    return kept_ids, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', default='data/How2Sign/train_2D_keypoints',
                    help='Folder containing how2sign_realigned_<split>.csv and json/<clip>/*.json')
    ap.add_argument('--out_root', default='data/synth/h2s')
    ap.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    ap.add_argument('--min_cues', type=int, default=2,
                    help='Drop streams with fewer than this many sentences after filtering')
    args = ap.parse_args()

    kept_ids, manifest = run_split(args)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    subset_path = out_root / 'subset2episode.json'
    if subset_path.exists(): existing = json.loads(subset_path.read_text(encoding='utf-8'))
    else: existing = {'train': [], 'val': [], 'test': []}
    
    # Map H2S split names to BOBSL convention used by our loader.
    split_map = {'train': 'train', 'val': 'val', 'test': 'test'}
    existing[split_map[args.split]] = sorted(kept_ids)
    subset_path.write_text(json.dumps(existing, indent=2), encoding='utf-8')
    manifest_path = out_root / f'manifest_{args.split}.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print(f'[H2S/{args.split}] {len(kept_ids)} streams written to {out_root}')
    print(f'  poses: {out_root / 'poses'}/<video_id>.npy')
    print(f'  vtt:   {out_root / 'vtt'}/<video_id>.vtt')
    print(f'  subset2episode.json updated → split={split_map[args.split]}')
    print(f'  manifest: {manifest_path}')


if __name__ == "__main__":
    main()