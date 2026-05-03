'''Compute the inter-subtitle pause-length distribution from BOBSL's *manual* annotations.

Why manual not auto: the reviewer's criticism of the original paper was specifically about
auto-aligned subtitle quality. Borrowing pause statistics from the same noisy source defeats the
purpose. Manual-aligned BOBSL is small but trustworthy; we use it once to derive the pause
distribution that the synthesizer then samples from directly (no parametric fit).

Outputs:
    bobsl_gap_stats.json   - summary statistics (mean/median/p90, plus subs-per-15s-window for K)
    bobsl_gap_samples.npy  - the FULL empirical inter-subtitle gap array (negatives clipped to 0).
                             The synthesizer prefers this file over any LogNormal fit because the
                             real distribution has ~74% of gaps at exactly 0 (co-articulated
                             continuous signing) and a heavy positive tail -- a shape no
                             parametric distribution captures cleanly.

Usage:
    python -m data_synth.analyze_bobsl_gaps \
        --vtt_dir data/BOBSL/manual_annotations/signing_aligned_subtitles \
        --out data_synth/stats/bobsl_gap_stats.json
'''
import json
import argparse
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import parse_vtt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vtt_dir', required=True, help='BOBSL manual-aligned VTT directory.')
    p.add_argument('--out', default='data_synth/stats/bobsl_gap_stats.json')
    p.add_argument('--max_files', type=int, default=0, help='0=all')
    args = p.parse_args()

    vtt_dir = Path(args.vtt_dir)
    files = sorted(vtt_dir.glob('*.vtt'))
    if args.max_files > 0: files = files[: args.max_files]
    if not files: raise SystemExit(f'No .vtt found under {vtt_dir}')
    print(f'Scanning {len(files)} VTT files (manual-aligned)')

    WINDOW_S = 15.0   # match StreamSLST sliding window length (used for sanity stats)
    STREAM_S = 60.0   # synthetic stream length target = 4x training window so the eval signal
                      # actually exercises cross-window streaming inference, not single-window decoding
    STEP_S = 1.0      # step for sliding K-per-window count
    gaps, sub_durations = [], []
    subs_per_window = []  # subtitle counts per 15s sliding window across all files
    subs_per_stream_window = []  # subtitle counts per 60s sliding window (for stream K derivation)

    for fp in files:
        try: subs = parse_vtt(fp)
        except Exception as e:
            print(f'skip {fp.name}: {e}')
            continue
        for s in subs: sub_durations.append(s['duration'])
        for a, b in zip(subs[:-1], subs[1:]):
            gap = b['start'] - a['end']
            if gap >= 0: gaps.append(gap)
        if subs:
            t_max = max(s['end'] for s in subs)
            for win_s, bucket in [(WINDOW_S, subs_per_window), (STREAM_S, subs_per_stream_window)]:
                starts = np.arange(0.0, max(0.0, t_max - win_s) + 1e-6, STEP_S)
                for t0 in starts:
                    t1 = t0 + win_s
                    k = sum(1 for s in subs if s['start'] >= t0 and s['end'] <= t1)
                    bucket.append(k)

    gaps = np.asarray(gaps, dtype=np.float64)
    sub_durations = np.asarray(sub_durations, dtype=np.float64)
    spw = np.asarray(subs_per_window, dtype=np.int32)
    spsw = np.asarray(subs_per_stream_window, dtype=np.int32)
    if gaps.size == 0: raise SystemExit('No gaps found.')

    pos = gaps[gaps > 0]
    log_pos = np.log(pos.clip(min=1e-3))
    p05, p95 = float(np.percentile(pos, 5)), float(np.percentile(pos, 95))
    stats = {
        'source': 'bobsl_manual_aligned',
        'n_gaps': int(gaps.size),
        'n_positive_gaps': int(pos.size),
        'gap_mean_s': float(gaps.mean()),
        'gap_median_s': float(np.median(gaps)),
        'gap_p90_s': float(np.percentile(gaps, 90)),
        'gap_p95_s': p95,
        'gap_p99_s': float(np.percentile(gaps, 99)),
        'log_pos_mean': float(log_pos.mean()),
        'log_pos_sigma': float(log_pos.std()),
        'sub_dur_mean_s': float(sub_durations.mean()),
        'sub_dur_median_s': float(np.median(sub_durations)),
        # Synthesizer reads these:
        'recommended_pause_log_mean': float(log_pos.mean()),
        'recommended_pause_log_sigma': float(log_pos.std()),
        'pause_min_s': max(0.1, p05),
        'pause_max_s': p95,
        # Sentences-per-15s-sliding-window distribution (matches model training window)
        'window_seconds': WINDOW_S,
        'subs_per_window_n': int(spw.size),
        'subs_per_window_mean': float(spw.mean()) if spw.size else 0.0,
        'subs_per_window_p10': float(np.percentile(spw, 10)) if spw.size else 1.0,
        'subs_per_window_p50': float(np.percentile(spw, 50)) if spw.size else 2.0,
        'subs_per_window_p90': float(np.percentile(spw, 90)) if spw.size else 6.0,
        'subs_per_window_max': int(spw.max()) if spw.size else 10,
        # Sentences-per-60s-sliding-window distribution (used for K-per-STREAM sampling). Anchoring K
        # to a 4x model window guarantees synthesized streams span multiple training windows so the
        # streaming inference behaviour the model is supposed to demonstrate actually triggers.
        'stream_window_seconds': STREAM_S,
        'subs_per_stream_window_n': int(spsw.size),
        'subs_per_stream_window_mean': float(spsw.mean()) if spsw.size else 0.0,
        'subs_per_stream_window_p10': float(np.percentile(spsw, 10)) if spsw.size else 4.0,
        'subs_per_stream_window_p50': float(np.percentile(spsw, 50)) if spsw.size else 8.0,
        'subs_per_stream_window_p90': float(np.percentile(spsw, 90)) if spsw.size else 24.0,
        'subs_per_stream_window_max': int(spsw.max()) if spsw.size else 40,
    }
    print(json.dumps(stats, indent=2))
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2))
    print(f'Wrote {out_json}')

    # Also dump full empirical gap array (negatives clipped to 0). synthesize_streams reads this
    # in preference to any parametric fit because the real BOBSL distribution is bimodal: a large
    # spike at 0 (co-articulated boundaries) plus a heavy positive tail.
    out_samples = out_json.with_name('bobsl_gap_samples.npy')
    np.save(out_samples, np.clip(gaps, 0.0, None).astype(np.float32))
    print(f'Wrote {out_samples}  (n={gaps.size}, frac_zero={(gaps <= 0).mean():.3f})')


if __name__ == '__main__':
    main()
