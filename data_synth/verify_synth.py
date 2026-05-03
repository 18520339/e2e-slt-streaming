'''Sanity check on synthesized streams: count splits, sample one stream per dataset, verify it round-trips 
through the existing BOBSL preprocessing pipeline using EACH dataset's own WIDTH/HEIGHT (no BOBSL borrowing).
'''
import os, sys, importlib, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def reload_config_for(ds: str):
    os.environ['DATASET'] = ds
    import config as _config
    importlib.reload(_config)
    return _config


if __name__ == '__main__':
    for ds_env, ds_dir in [('PHOENIX', 'phoenix'), ('CSL', 'csl')]:
        cfg = reload_config_for(ds_env)
        
        # Re-import preprocessing so it picks up the freshly-reloaded WIDTH/HEIGHT
        import poses.preprocessing as pp
        importlib.reload(pp)
        from utils import parse_vtt
        root = Path(f'data/synth/{ds_dir}')
        if not root.exists():
            print(f'{ds_env}: missing {root}')
            continue
        
        sub = json.loads((root / 'subset2episode.json').read_text())
        n_pose = len(list((root / 'poses').iterdir()))
        n_vtt = len(list((root / 'vtt').iterdir()))
        print(f'== {ds_env} (W,H = {cfg.WIDTH},{cfg.HEIGHT}) ==')
        print(f'  splits  train={len(sub["train"])}, val={len(sub["val"])}, test={len(sub["test"])}')
        print(f'  files   pose dirs={n_pose}, vtt files={n_vtt}')
        
        sample_id = sub['test'][0]
        arr = np.load(root / 'poses' / sample_id / '000.npy')
        cues = parse_vtt(root / 'vtt' / f'{sample_id}.vtt')
        print(f'  raw native coords  x [{arr[...,0].min():.1f}, {arr[...,0].max():.1f}]'
              f'  y [{arr[...,1].min():.1f}, {arr[...,1].max():.1f}]'
              f'  conf [{arr[...,2].min():.2f}, {arr[...,2].max():.2f}]')
        
        norm = pp.normalize_keypoints(arr.astype(np.float32))
        thr = pp.threshold_confidence(norm)
        print(f'  sample  {sample_id}  T={arr.shape[0]} ({arr.shape[0]/12.5:.1f}s)  {len(cues)} cues  '
              f'post-norm shape={norm.shape}  finite={bool(np.isfinite(thr).all())}  '
              f'post-norm range x [{norm[...,0].min():.2f}, {norm[...,0].max():.2f}]')
        print(f'  first cue  {cues[0]["start"]:.2f}-{cues[0]["end"]:.2f}s  duration {cues[0]["duration"]:.2f}s')
        print(f'  manifest  {(root / "manifest.json").stat().st_size} bytes')