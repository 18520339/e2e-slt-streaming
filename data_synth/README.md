# Synthetic streaming SL benchmarks for StreamSLST

Synthesize streaming sign-language datasets from offline pre-segmented MSKA pickles for **PHOENIX-2014T** (DGS, German) and **CSL-Daily** (CSL, Chinese), in BOBSL-compatible layout so the same training and evaluation code that runs on BOBSL runs on these without modification.

## Why synthetic streams

The original paper trains/evaluates on BOBSL only, whose **auto-aligned** subtitles introduce annotation noise. Synthetic streams give us:

- two more sign languages (DGS, CSL) covering cross-language generalization;
- **oracle event boundaries** by construction (we *know* exactly when each sentence starts and ends), letting us re-run the alignment / learned-vs-GT ablations to disentangle "model can't localize" from "BOBSL labels are noisy";
- knobs we set ourselves — but kept to a **minimum** for reproducibility.

## Drop-in compatibility

Outputs match BOBSL's `loader.DVCDataset` directory contract:

```
data/synth/<lang>/
├── poses/<stream_id>.npy           # (T, 133, 3) float32 at 12.5 fps, NATIVE pixel coords
├── vtt/<stream_id>.vtt             # WEBVTT, one sentence per cue (single-line text)
├── subset2episode.json             # {"train": [...], "val": [...], "test": [...]}
└── manifest.json                   # provenance: clip ids per stream + seed + pause/k_range used
```

`DVCDataset._build_video_metadata` was extended to support both layouts:
- **BOBSL**: `POSE_ROOT/<video_id>/*.npy` (multi-segment)
- **synth**: `POSE_ROOT/<stream_id>.npy` (flat single file)

Switch dataset via the `DATASET` env var: `BOBSL` (default) | `PHOENIX` | `CSL`. `config.py` then resolves all paths, the mBART backbone, the target language code (`en_XX` / `de_DE` / `zh_CN`), and the per-dataset frame canvas (`(W,H) = (444,444) / (210,260) / (512,512)`) automatically. We use **`facebook/mbart-large-cc25`** for all three languages — its 25-language vocabulary covers all three target codes natively.

## Minimal-knob synthesis design

A stream is `[BG_pre] + clip_1 + [pause_1] + clip_2 + ... + clip_K + [BG_post]`. Every join — including the two BG segments — uses the **same** Hermite-bridge mechanism with the **same** empirical duration distribution drawn from BOBSL. No special-case code paths.

### 1. trim_rest — co-articulation by construction

Each MSKA clip is recorded in isolation, so it begins with a *preparation* phase (signer raises hands from lap to first sign) and ends with a *retraction* phase (hands fall back to lap). Concatenating naively creates an obvious "hands-down → long pause → hands-up" boundary that the localization head can latch onto trivially — and that does **not** occur in real continuous broadcast signing, which exhibits **co-articulation** (signers move directly between adjacent sentences without returning to rest).

Rule (zero hyperparameters):

```
is_rest[t] = mean(wrist_y) > mean(shoulder_y)    # image y-down: wrists below shoulder line
```

Trim the contiguous leading run and the contiguous trailing run of `is_rest` frames. Geometry only — signer-relative, scales automatically across PHOENIX (210×260 canvas) and CSL (512×512 canvas) without any per-dataset threshold. Safety floor: never reduce a clip below 3 frames (the minimum needed for endpoint-velocity estimation).

### 2. Empirical pause sampling

Inter-clip gaps are drawn directly from BOBSL's manual-aligned annotations via:

```
gap_s = rng.choice(bobsl_gap_samples)            # samples loaded from bobsl_gap_samples.npy
```

The real BOBSL distribution is **bimodal**: ~74% of inter-subtitle gaps are exactly 0 (touching or overlapping subtitles = continuous co-articulated signing) and ~26% are positive with a heavy right tail (median of positive gaps ≈ 2 s, p90 ≈ 6 s). Direct empirical sampling preserves both modes; any LogNormal fit would either ignore the zero spike or distort the positive tail. The `pause_min_s`, `pause_max_s`, `log_mean`, `log_sigma` parameters of earlier iterations are **gone**.

### 3. Hermite bridge at every seam (with length-aware tangent damping)

Per-joint **cubic Hermite spline** interpolation on `(x, y)` from the last frame of one clip to the first frame of the next. Endpoint tangents are the actual per-frame velocities at those clip boundaries (`clip[-1] - clip[-2]` and `next[1] - next[0]`), giving C1 continuity at the seam. Tangents are clamped to at most twice `|p1 - p0|` per joint to prevent overshoot. Confidence is the element-wise minimum of the two endpoints.

Bridge length: `n_pause = max(MIN_BRIDGE_FRAMES, round(gap_s * FPS))`. `MIN_BRIDGE_FRAMES=2` is a numerical floor (not a tuning knob) so a sampled `gap_s = 0` still produces a smooth two-frame transition rather than a kinematic teleport. With trim_rest in place, those two-frame transitions are exactly the *movement-epenthesis* signal that real continuous signing exhibits.

**Tangent damping for long bridges**: the per-frame endpoint velocity, propagated across many frames of a long BG segment, would extrapolate the clip's last hand motion far past where the hand should physically end up — visible overshoot. Damping `v_scale = MIN_BRIDGE_FRAMES / max(n, MIN_BRIDGE_FRAMES)` makes short bridges keep full clip-end momentum (co-articulation continuity) and long bridges progressively converge to a position-only interpolation (settle into rest). Single line, no new knob, uses the existing `MIN_BRIDGE_FRAMES`.

### 4. BG_pre / BG_post via phantom clips, sampled from positives only

For the leading and trailing background segments, one endpoint of the Hermite bridge is the first/last frame of an *additional clip from the same signer's pool that was NOT chosen* ("phantom clip") so the BG region is an animated transition between two real signer poses, not a frozen frame. Phantom clips are **not** trimmed — BG segments represent the broadcast lead-in/lead-out where holding a real rest pose at the phantom endpoint is realistic.

BG_pre and BG_post durations are sampled from the **positive-only subset** of the BOBSL empirical gaps, not the full distribution. Inter-clip pauses can legitimately be 0 s (continuous co-articulated signing), but BG segments represent the silent broadcast preamble where there is no caption — they should always be visibly present in the stream. Sampling from positives only enforces this by construction without introducing a min-duration knob: the floor is whatever the smallest positive BOBSL inter-subtitle gap is.

If the signer's pool is fully used, the synthesizer falls back to a phantom drawn from the chosen clips themselves (the seam is still Hermite-interpolated, just less varied).

### 5. K-per-stream from a 60-second window of BOBSL

K (sentences per stream) is sampled from `[subs_per_stream_window_p10, subs_per_stream_window_p90]` of BOBSL's manual annotations counted in a **60-s** sliding window — explicitly 4× the model's 15-s training window. Anchoring K to a stream length that spans multiple training windows is what makes the eval signal exercise *streaming* inference (cross-window event handling, latent-state propagation) rather than collapsing to single-window decoding. The earlier 15-s anchor produced K ∈ [1, 5] and 12-s streams that fit inside one training window — not a streaming benchmark in any meaningful sense. The 60-s anchor produces K ∈ [6, 19] and median 40–60 s streams.

### Hyperparameters (all BOBSL-derived or pure geometry)

Read once from `data_synth/stats/bobsl_gap_stats.json` + `bobsl_gap_samples.npy` (produced by `analyze_bobsl_gaps.py` over **manual-aligned** BOBSL VTTs — auto-aligned is *not* used because that's the very source of annotation noise the original paper was criticised for):

| | source | example value |
|---|---|---|
| inter-clip pause | `bobsl_gap_samples.npy` empirical array (full) | n≈30k, ~74% zero, p90 ≈ 2 s |
| BG_pre / BG_post duration | positive-only subset of `bobsl_gap_samples.npy` | n≈8k, median 2 s, p90 ≈ 10 s |
| K per stream | `[subs_per_stream_window_p10, subs_per_stream_window_p90]` from a 1-s sliding **60-s** window over BOBSL | [6, 19] in current run |
| trim_rest threshold | wrists vs shoulder geometry | none — pure boolean |
| Hermite tangent damping | `MIN_BRIDGE_FRAMES / max(n, MIN_BRIDGE_FRAMES)` | reuses existing constant, no new knob |
| `MIN_BRIDGE_FRAMES` | numerical floor for Hermite (≥2 needed to define a transition) | 2 |
| min signer-pool size | constant — must have ≥1 phantom plus chosen clips | 2 |

Stream count per split is data-derived: `n_streams = round(N_usable_clips / K_avg)` where `K_avg = mean(k_range)`. Each clip is used about once on average across the split.

### What was deliberately removed (vs earlier iterations)

`recommended_pause_log_mean`, `recommended_pause_log_sigma`, `pause_min_s`, `pause_max_s`, `per-signer rest pose extraction`, `top-K rest candidate sampling`, `AR(1) idle-drift jitter`, `seam_blend_frames`, `n_rest_candidates`, `rest_edge_frames`, `hold_jitter_px`, `held first/last frame BG`. None of these are needed once pauses are sampled empirically from BOBSL, BG segments are sampled from the positive-only subset, clip endpoints are trimmed to their signing portions, and the Hermite tangent is damped with bridge length.

### Same-signer per stream

By construction, no pseudo data — derived from the MSKA-stored `name` field:
- **PHOENIX**: clip `name` is `<split>/<broadcast>-<sentence_idx>`; one broadcast = one signer in PHOENIX. Group by `name.split('/')[1].rsplit('-', 1)[0]`.
- **CSL**: `Sxxx_P0000_Tyy` — group by `P_id`.

## Workflow

```bash
# 0. (one-time, after BOBSL.zip extracted) compute pause + K stats from BOBSL MANUAL annotations.
python -m data_synth.analyze_bobsl_gaps \
    --vtt_dir data/BOBSL/manual_annotations/signing_aligned_subtitles \
    --out data_synth/stats/bobsl_gap_stats.json
# If skipped, the synthesizer falls back to defaults that approximate the same shape.

# 1. Synthesize PHOENIX streams (no count args; everything derived from data + BOBSL stats)
python -m data_synth.synthesize_streams --dataset PHOENIX --out_root data/synth/phoenix

# 2. Synthesize CSL-Daily streams
python -m data_synth.synthesize_streams --dataset CSL --out_root data/synth/csl

# 3. Visualize a few streams (reads DATASET env to pick canvas; renders the 77 model-input keypoints
#    with auto-fit window so out-of-frame HRNet outliers don't push the body off-screen).
DATASET=PHOENIX python -m data_synth.visualize_stream \
    --pose data/synth/phoenix/poses/test_00000.npy \
    --vtt  data/synth/phoenix/vtt/test_00000.vtt --out data_synth/examples/phoenix_test_00000.mp4
DATASET=CSL python -m data_synth.visualize_stream \
    --pose data/synth/csl/poses/test_00000.npy \
    --vtt  data/synth/csl/vtt/test_00000.vtt --out data_synth/examples/csl_test_00000.mp4

# 4. Trim mBART tokenizer + model per language (writes captioners/trimmed_*_<lang>/)
DATASET=PHOENIX python -m captioners.trim_mbart
DATASET=CSL     python -m captioners.trim_mbart

# 5. Compute BOBSL-paper-style dataset stats (also reports BPE-subword vocab from the trimmed tokenizer)
DATASET=PHOENIX python -m data_synth.dataset_stats --root data/synth/phoenix --out data_synth/stats/phoenix_stats.json
DATASET=CSL     python -m data_synth.dataset_stats --root data/synth/csl     --out data_synth/stats/csl_stats.json
```

## Resulting benchmark sizes

K is now derived from a **60-second** sliding window of BOBSL (= 4× the model's 15-s training window) so synthesized streams span multiple model windows and the streaming inference behaviour the paper claims actually fires at evaluation time. Combined with trim_rest + empirical pauses + positive-only BG, streams are now median 40–60 s long with 7–11 sentences each:

| split | streams | signer-pure pools | hours | cues / stream | stream dur p50 / p90 (s) | pause med / p90 (s) | density |
|---|---:|---:|---:|---:|---:|---:|---:|
| PHOENIX train | 564 | 387 broadcasts | 6.99 | 7.67 | 41.6 / 65.2 | 0.00 / 2.00 | 0.715 |
| PHOENIX val | 27 | 25 | 0.10 | 2.07 | 13.0 / 20.8 | 0.00 / 1.10 | 0.515 |
| PHOENIX test | 38 | 35 | 0.17 | 2.50 | 15.1 / 25.8 | 0.00 / 2.00 | 0.522 |
| CSL train | 1467 | 10 P-ids | 22.10 | 10.61 | 52.3 / 81.1 | 0.00 / 2.00 | 0.715 |
| CSL val | 86 | 10 | 1.29 | 10.30 | 51.6 / 81.6 | 0.00 / 2.00 | 0.718 |
| CSL test | 94 | 10 | 1.57 | 10.77 | 53.6 / 92.2 | 0.00 / 2.00 | 0.717 |

Pause distribution matches the BOBSL empirical sample by construction (~74% of inter-clip joins have `gap_s = 0` and concatenate co-articulated). PHOENIX dev/test streams are shorter than train because per-broadcast clip pools are only ~2–3 clips on those splits — same-signer-per-stream caps K at the pool size; the underlying biological signer count is 9 across all 629 broadcasts. CSL is unaffected (each of the 10 P-ids has hundreds of clips per split).

## Files

- `synthesize_streams.py` — main pipeline (trim_rest, Hermite bridge, empirical pause)
- `analyze_bobsl_gaps.py` — writes `bobsl_gap_stats.json` (summary + K-range stats) and `bobsl_gap_samples.npy` (full empirical gap array consumed by the synthesizer)
- `bobsl_gap_stats.json` / `bobsl_gap_samples.npy` — produced by the above; both auto-detected by `synthesize_streams.py`
- `visualize_stream.py` — renders the 77 model-input keypoints as MP4 with PIL CJK subtitle overlay
- `dataset_stats.py` — BOBSL-paper-style stats; reports word vocab + BPE vocab side by side
- `verify_synth.py` — round-trip sanity check (load → normalize → threshold → parse_vtt)
