'''Dump per-stream predictions from a trained StreamSLST checkpoint, alongside GT.

Two-stage workflow:
    Stage 1 (this script): load checkpoint, iterate sliding windows over val/test split,
    forward through model, decode top-K events per window, map (start, end) back from
    window-relative [0, 1] to absolute seconds on the original stream timeline. Write:

        <out_dir>/<split>/predictions.jsonl    # per-window rows (raw, untouched)
        <out_dir>/<split>/by_stream.json       # grouped per stream_id, w/ NMS variant
        <out_dir>/<split>/samples_txt/<sid>.txt  # human-readable GT-vs-pred per stream

    Stage 2: data_synth/visualize_results.py reads by_stream.json + pose npy + GT VTT
    and renders MP4s with two timelines (GT on top, predictions on bottom).

This is independent of the HF Trainer — we run inference manually so we can keep the
stream id and absolute window offset that Trainer's collate fn discards.

Usage:
    DATASET=PHOENIX python -m evaluation.dump_predictions \
        --checkpoint_path checkpoints/phoenix_cosign --split test \
        --out_dir data_synth/eval_dumps/phoenix
'''
import os, sys, json, argparse, gc
from pathlib import Path
from typing import List, Dict, Tuple
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DeformableDetrConfig
from tqdm import tqdm

from loader import DVCDataset
from pdvc import DeformableDetrForObjectDetection
from captioners import LSTMCaptioner, MBartDecoderCaptioner
from postprocess import post_process_object_detection
from evaluation.helpers import select_topN_per_window, extract_gt_per_window
from utils import parse_vtt
from config import TGT_LANG, TRIMMED_TOKENIZER_DIR, FPS, WINDOW_DURATION_SECONDS, VTT_DIR, DATASET


# ---------- Model construction (mirrors eval.py defaults) ----------

def build_model(checkpoint_path: str, tokenizer, device: str,
                d_model=1024, encoder_layers=2, decoder_layers=2,
                encoder_attention_heads=8, decoder_attention_heads=8,
                encoder_n_points=4, decoder_n_points=4,
                num_feature_levels=4, num_queries=30, num_labels=1,
                auxiliary_loss=True, with_box_refine=True,
                num_cap_layers=3, cap_dropout_rate=0.1,
                captioner_type='mbart',
                max_event_tokens=40, max_events=10):
    config = DeformableDetrConfig(
        d_model=d_model, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        encoder_attention_heads=encoder_attention_heads, decoder_attention_heads=decoder_attention_heads,
        encoder_n_points=encoder_n_points, decoder_n_points=decoder_n_points,
        activation_function='gelu', num_feature_levels=num_feature_levels,
        num_queries=num_queries, num_labels=num_labels,
        auxiliary_loss=auxiliary_loss, with_box_refine=with_box_refine,
    )
    model = DeformableDetrForObjectDetection(
        config=config,
        captioner_class=MBartDecoderCaptioner if captioner_type == 'mbart' else LSTMCaptioner,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.lang_code_to_id[TGT_LANG],
        num_cap_layers=num_cap_layers, cap_dropout_rate=cap_dropout_rate,
        max_event_tokens=max_event_tokens, max_events=max_events,
        use_gt_boxes_for_caption=not with_box_refine,
        weight_dict={'loss_ce': 2.0, 'loss_bbox': 0.0, 'loss_giou': 4.0,
                     'loss_counter': 2.0, 'loss_caption': 2.0},
    ).to(device)
    state = torch.load(os.path.join(checkpoint_path, 'pytorch_model.bin'), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def collate_for_dump(batch): # Preserve video_id and window offsets — unlike trainer_collate_fn which drops them.
    vids, starts, ends, poses, masks, labels = zip(*batch)
    return {
        'video_ids': list(vids),
        'window_start_frames': list(starts),
        'window_end_frames': list(ends),
        'pixel_values': torch.stack(poses),
        'pixel_mask': torch.stack(masks),
        'labels': list(labels),
    }


@torch.no_grad()
def run_inference(model, loader, tokenizer, device, ranking_temperature: float, alpha: float, top_k: int) -> List[Dict]:
    rows: List[Dict] = []
    window_size_frames = int(WINDOW_DURATION_SECONDS * FPS)
    window_dur_s = float(WINDOW_DURATION_SECONDS)

    for batch in tqdm(loader, desc='inference'):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=None)

        post = post_process_object_detection(
            outputs=outputs, top_k=top_k, threshold=0.0,
            target_lengths=None, tokenizer=tokenizer,
        )
        pred_counts = (
            outputs.pred_counts.argmax(dim=-1).clamp(min=0).tolist()
            if outputs.pred_counts is not None else None
        )
        b_pred_events_rel, b_pred_caps, b_pred_cap_scores = select_topN_per_window(
            post, pred_counts, tokenizer,
            ranking_temperature=ranking_temperature, alpha=alpha, top_k=top_k,
        )
        b_gt_events_rel, b_gt_caps = extract_gt_per_window(batch['labels'], tokenizer)

        # Per-window detection scores (top of joint reranking score) from `post`. For NMS later
        # we want a comparable scalar per kept event. Use event_scores corresponding to the
        # selected indices — but select_topN_per_window doesn't return those. We re-derive
        # by looking up cap_scores; localization-only NMS works fine on cap-rerank scores.
        for w_idx in range(len(post)):
            video_id = batch['video_ids'][w_idx]
            wsf = int(batch['window_start_frames'][w_idx])
            wef = int(batch['window_end_frames'][w_idx])
            t_offset = wsf / FPS
            # Convert window-relative [0, 1] → absolute seconds on the stream timeline.
            pred_events_abs = [
                (t_offset + s * window_dur_s, t_offset + e * window_dur_s)
                for (s, e) in b_pred_events_rel[w_idx]
            ]
            gt_events_abs = [
                (t_offset + s * window_dur_s, t_offset + e * window_dur_s)
                for (s, e) in b_gt_events_rel[w_idx]
            ]
            rows.append({
                'video_id': video_id,
                'window_start_frame': wsf, 'window_end_frame': wef,
                'window_start_s': t_offset, 'window_end_s': wef / FPS,
                'pred_events_s': pred_events_abs,
                'pred_captions': list(b_pred_caps[w_idx]),
                'pred_cap_scores': [float(x) for x in b_pred_cap_scores[w_idx]],
                'gt_events_s': gt_events_abs,
                'gt_captions': list(b_gt_caps[w_idx]),
            })
    return rows


def nms_events(events_s: List[Tuple[float, float]], captions: List[str], scores: List[float], iou_thr: float = 0.5) -> List[int]:
    # Greedy NMS: keep highest-score event, drop overlaps with IoU >= iou_thr. Returns indices
    order = sorted(range(len(events_s)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    for i in order:
        s_i, e_i = events_s[i]
        ok = True
        for j in keep:
            s_j, e_j = events_s[j]
            inter = max(0.0, min(e_i, e_j) - max(s_i, s_j))
            union = max(e_i, e_j) - min(s_i, s_j)
            iou = inter / max(union, 1e-9)
            if iou >= iou_thr:
                ok = False
                break
        if ok: keep.append(i)
    return sorted(keep, key=lambda i: events_s[i][0])


# ---------- Aggregation ----------

def gather_gt_from_vtt(video_id: str) -> List[Dict]:
    # Return canonical GT cues from the source VTT for the stream (full list, not per-window)
    p = Path(VTT_DIR) / f'{video_id}.vtt'
    try: cues = parse_vtt(p)
    except Exception: return []
    return [{'start': float(c['start']), 'end': float(c['end']), 'text': str(c['text'])} for c in cues]


def group_by_stream(rows: List[Dict], nms_iou: float) -> Dict[str, Dict]:
    by: Dict[str, Dict] = {}
    for r in rows:
        sid = r['video_id']
        if sid not in by: by[sid] = {'pred_all': [], 'pred_scores_all': []}
        for (s, e), cap, cs in zip(r['pred_events_s'], r['pred_captions'], r['pred_cap_scores']):
            by[sid]['pred_all'].append({'start': float(s), 'end': float(e), 'text': cap, 'score': float(cs)})
            by[sid]['pred_scores_all'].append(float(cs))

    out: Dict[str, Dict] = {}
    for sid, info in by.items():
        preds = info['pred_all']
        preds.sort(key=lambda x: x['start'])
        events = [(p['start'], p['end']) for p in preds]
        caps = [p['text'] for p in preds]
        scores = [p['score'] for p in preds]
        keep_idx = nms_events(events, caps, scores, iou_thr=nms_iou)
        pred_nms = [preds[i] for i in keep_idx]
        gt = gather_gt_from_vtt(sid)
        gt.sort(key=lambda x: x['start'])
        out[sid] = {'gt': gt, 'pred_all': preds, 'pred_nms': pred_nms, 'nms_iou': nms_iou,}
    return out


# ---------- Human-readable per-stream TXT ----------

def fmt_t(s: float) -> str:
    m, sec = divmod(s, 60.0)
    return f'{int(m):02d}:{sec:05.2f}'


def write_sample_txt(out_path: Path, sid: str, info: Dict):
    lines: List[str] = []
    lines.append(f'STREAM: {sid}')
    lines.append(f'  GT cues:    {len(info["gt"])}')
    lines.append(f'  Pred (raw): {len(info["pred_all"])}')
    lines.append(f'  Pred (NMS@{info["nms_iou"]:.2f}): {len(info["pred_nms"])}')
    lines.append('')
    lines.append('---- GROUND TRUTH ----')
    for c in info['gt']: lines.append(f'  [{fmt_t(c["start"])} - {fmt_t(c["end"])}]  {c["text"]}')
    lines.append('')
    lines.append('---- PRED (NMS) ----')
    for c in info['pred_nms']: lines.append(f'  [{fmt_t(c["start"])} - {fmt_t(c["end"])}]  ({c["score"]:+.2f})  {c["text"]}')
    lines.append('')
    lines.append('---- PRED (ALL, before NMS) ----')
    for c in info['pred_all']: lines.append(f'  [{fmt_t(c["start"])} - {fmt_t(c["end"])}]  ({c["score"]:+.2f})  {c["text"]}')
    out_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint_path', required=True)
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    ap.add_argument('--out_dir', required=True, help='Output root. Files go to <out_dir>/<split>/...')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--top_k', type=int, default=20)
    ap.add_argument('--alpha', type=float, default=0.3)
    ap.add_argument('--ranking_temperature', type=float, default=2.0)
    ap.add_argument('--stride_ratio', type=float, default=0.9)
    ap.add_argument('--max_event_tokens', type=int, default=40)
    ap.add_argument('--max_window_tokens', type=int, default=128)
    ap.add_argument('--max_events', type=int, default=10)
    ap.add_argument('--min_events', type=int, default=1)
    ap.add_argument('--captioner_type', default='mbart', choices=['mbart', 'lstm'])
    ap.add_argument('--nms_iou', type=float, default=0.5, help='IoU threshold for greedy NMS over absolute-time events.')
    ap.add_argument('--device', default=None, help='Override device (cuda/cpu).')
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    out_split_dir = Path(args.out_dir) # / args.split
    out_split_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_split_dir / 'samples_txt'
    samples_dir.mkdir(parents=True, exist_ok=True)

    print(f'DATASET={DATASET}  split={args.split}  device={device}')
    print(f'checkpoint: {args.checkpoint_path}')
    print(f'output:     {out_split_dir}')

    tokenizer = AutoTokenizer.from_pretrained(TRIMMED_TOKENIZER_DIR)
    model = build_model(args.checkpoint_path, tokenizer, device,
                        max_event_tokens=args.max_event_tokens,
                        max_events=args.max_events,
                        captioner_type=args.captioner_type)

    dataset = DVCDataset(
        split=args.split, tokenizer=tokenizer, pose_augment=False,
        stride_ratio=args.stride_ratio,
        min_events=args.min_events, max_events=args.max_events,
        max_event_tokens=args.max_event_tokens, max_window_tokens=args.max_window_tokens,
        load_by='window', seed=42,
    )
    print(f'eval windows: {len(dataset)}')

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_for_dump,
    )
    rows = run_inference(
        model, loader, tokenizer, device,
        ranking_temperature=args.ranking_temperature, alpha=args.alpha, top_k=args.top_k,
    )

    # Stage A: per-window JSONL
    jsonl_path = out_split_dir / 'predictions.jsonl'
    with jsonl_path.open('w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'wrote {jsonl_path}  ({len(rows)} window rows)')

    # Stage B: grouped by stream + NMS
    by_stream = group_by_stream(rows, nms_iou=args.nms_iou)
    by_path = out_split_dir / 'by_stream.json'
    by_path.write_text(json.dumps(by_stream, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {by_path}  ({len(by_stream)} streams)')

    # Stage C: human-readable per-stream TXTs
    for sid, info in by_stream.items(): write_sample_txt(samples_dir / f'{sid}.txt', sid, info)
    print(f'wrote {len(by_stream)} per-stream TXTs to {samples_dir}')

    # Cleanup
    del model, tokenizer, dataset, loader
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
