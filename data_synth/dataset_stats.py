'''Compute BOBSL-paper-style statistics for the synthesized streaming benchmark.

Reference: Albanie et al., "BOBSL: BBC-Oxford British Sign Language Dataset" (arXiv:2111.03635) Table 1.

Reports per split:
    - #streams (videos)
    - total duration (h)
    - mean / median / p90 stream length (s)
    - #signers (signer-pure pools)
    - #subtitles (cues), mean / median subtitle duration (s)
    - signing density: total signing-frame fraction
    - inter-cue pause distribution: mean / median / p90 (s)
    - vocabulary size (tokens), avg #tokens / cue
    - mean K (sentences per stream)

Usage:
    DATASET=PHOENIX python -m data_synth.dataset_stats --root data/synth/phoenix --out data_synth/stats/phoenix_stats.json
    DATASET=CSL     python -m data_synth.dataset_stats --root data/synth/csl     --out data_synth/stats/csl_stats.json

If both manifest.json and per-stream pose .npy files exist, the script also reports the BOBSL-style
"signing density" (frac of frames that fall inside any cue) computed exactly from the cues.
'''
import sys, argparse, json, re
import numpy as np
from collections import Counter
from typing import Dict, List
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import parse_vtt
from config import DATASET, TRIMMED_TOKENIZER_DIR
WORD_RE = re.compile(r"\w+", re.UNICODE)

def percentile(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p)) if len(xs) else 0.0

def tokenize_for_vocab(text: str) -> List[str]:
    if DATASET == 'CSL': # Chinese: per-character tokens (matches sacrebleu zh tokenize and CSL-Daily convention)
        return [c for c in text if c.strip() and c not in '。，、；：！？·']
    return WORD_RE.findall(text.lower())


def maybe_load_bpe_tokenizer():
    '''Load the trimmed mBART tokenizer for the active DATASET if it exists; else None.

    The vocab number reported by `vocab_size` (word/char level) and the BPE-subword vocab
    used by the model are different metrics by definition: BPE splits words into multiple
    subword pieces (typically 1.5-2x more units than whole words). When this tokenizer is
    available we additionally report the count of distinct BPE subwords actually used in
    training subtitles (`bpe_vocab_used`) and the trimmed-vocab size kept by the tokenizer
    (`bpe_vocab_size`), so the paper can show both.
    '''
    try:
        from transformers import AutoTokenizer
        tok_dir = Path(TRIMMED_TOKENIZER_DIR)
        if not tok_dir.exists():
            return None
        return AutoTokenizer.from_pretrained(str(tok_dir))
    except Exception as e:
        print(f'(bpe tokenizer not loaded: {e})')
        return None


def stats_for_split(root: Path, split: str, ids: List[str], bpe_tokenizer=None) -> dict:
    stream_durations, signing_durations, cue_durations = [], [], []
    pauses, cues_per_stream = [], []
    total_tokens, cue_count = 0, 0
    vocab = Counter()
    bpe_seen: set = set()
    bpe_total_tokens = 0

    for sid in ids: # Stream length: support both flat layout (poses/<sid>.npy) and BOBSL dir layout (poses/<sid>/000.npy)
        flat = root / 'poses' / f'{sid}.npy'
        nested = root / 'poses' / sid / '000.npy'
        pose_path = flat if flat.exists() else nested
        try: T = int(np.load(pose_path, mmap_mode='r').shape[0])
        except Exception: continue
        
        from config import FPS as TARGET_FPS
        stream_dur = T / float(TARGET_FPS)
        stream_durations.append(stream_dur)

        # Cues from VTT (single source of truth for signing-vs-bg)
        vtt_path = root / 'vtt' / f'{sid}.vtt'
        try: subs = parse_vtt(vtt_path)
        except Exception: subs = []
        cues_per_stream.append(len(subs))
        signing_durations.append(sum(s['duration'] for s in subs))
        
        for s in subs:
            cue_durations.append(s['duration'])
            cue_count += 1
            toks = tokenize_for_vocab(s['text'])
            vocab.update(toks)
            total_tokens += len(toks)
            if bpe_tokenizer is not None:
                bpe_ids = bpe_tokenizer.encode(s['text'], add_special_tokens=False)
                bpe_seen.update(bpe_ids)
                bpe_total_tokens += len(bpe_ids)
            
        for a, b in zip(subs[:-1], subs[1:]):
            gap = b['start'] - a['end']
            if gap >= 0: pauses.append(gap)

    total_dur = sum(stream_durations)
    total_signing = sum(signing_durations)
    out = {
        'n_streams': len(stream_durations),
        'total_duration_h': total_dur / 3600.0,
        'stream_dur_mean_s': float(np.mean(stream_durations)) if stream_durations else 0.0,
        'stream_dur_median_s': float(np.median(stream_durations)) if stream_durations else 0.0,
        'stream_dur_p90_s': percentile(stream_durations, 90),
        'n_cues': cue_count,
        'cues_per_stream_mean': float(np.mean(cues_per_stream)) if cues_per_stream else 0.0,
        'cue_dur_mean_s': float(np.mean(cue_durations)) if cue_durations else 0.0,
        'cue_dur_median_s': float(np.median(cue_durations)) if cue_durations else 0.0,
        'cue_dur_p90_s': percentile(cue_durations, 90),
        'pause_count': len(pauses),
        'pause_mean_s': float(np.mean(pauses)) if pauses else 0.0,
        'pause_median_s': float(np.median(pauses)) if pauses else 0.0,
        'pause_p90_s': percentile(pauses, 90),
        'signing_density': total_signing / max(total_dur, 1e-6),
        # Word/char-level vocab (for human-readable stats; comparable to BOBSL paper's vocab numbers)
        'word_vocab_size': len(vocab),
        'word_total_tokens': total_tokens,
        'word_avg_tokens_per_cue': total_tokens / max(cue_count, 1),
        'top10_word_tokens': vocab.most_common(10),
    }
    if bpe_tokenizer is not None:
        out['bpe_vocab_used'] = len(bpe_seen)
        out['bpe_vocab_size'] = int(getattr(bpe_tokenizer, 'vocab_size', 0))
        out['bpe_total_tokens'] = bpe_total_tokens
        out['bpe_avg_tokens_per_cue'] = bpe_total_tokens / max(cue_count, 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Synth dataset root (data/synth/phoenix or data/synth/csl)')
    p.add_argument('--out', required=True, help='Output JSON path')
    args = p.parse_args()

    root = Path(args.root)
    sub = json.loads((root / 'subset2episode.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())

    # signers per split (from manifest provenance)
    signers_per_split: Dict[str, set] = {'train': set(), 'val': set(), 'test': set()}
    if 'streams' in manifest:
        for split, lst in manifest['streams'].items():
            for s in lst: signers_per_split[split].add(s.get('signer'))

    bpe_tokenizer = maybe_load_bpe_tokenizer()
    out = {'dataset': DATASET, 'splits': {}}
    print(f'==== {DATASET} synthetic streaming benchmark ====')
    if bpe_tokenizer is None:
        print('(no trimmed-mBART tokenizer found; reporting only word/char-level vocab)')
    for split in ['train', 'val', 'test']:
        ids = sub.get(split, [])
        s = stats_for_split(root, split, ids, bpe_tokenizer=bpe_tokenizer)
        s['n_signers'] = len(signers_per_split.get(split, set()))
        out['splits'][split] = s
        print(f'\n[{split}]  streams={s["n_streams"]}  signers={s["n_signers"]}  hours={s["total_duration_h"]:.2f}')
        print(f'  stream dur (s)  mean={s["stream_dur_mean_s"]:.1f}  median={s["stream_dur_median_s"]:.1f}  p90={s["stream_dur_p90_s"]:.1f}')
        print(f'  cues            total={s["n_cues"]}  per-stream={s["cues_per_stream_mean"]:.1f}')
        print(f'  cue dur  (s)    mean={s["cue_dur_mean_s"]:.2f}  median={s["cue_dur_median_s"]:.2f}  p90={s["cue_dur_p90_s"]:.2f}')
        print(f'  pause    (s)    n={s["pause_count"]}  mean={s["pause_mean_s"]:.2f}  median={s["pause_median_s"]:.2f}  p90={s["pause_p90_s"]:.2f}')
        print(f'  signing density {s["signing_density"]:.3f}')
        print(f'  word vocab      |V_word|={s["word_vocab_size"]}  total={s["word_total_tokens"]}  avg/cue={s["word_avg_tokens_per_cue"]:.2f}')
        if 'bpe_vocab_used' in s:
            print(f'  bpe  vocab      |V_bpe_used|={s["bpe_vocab_used"]}  trimmed_size={s["bpe_vocab_size"]}  '
                  f'avg/cue={s["bpe_avg_tokens_per_cue"]:.2f}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()