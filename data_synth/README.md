# Synthetic streaming SL benchmarks for StreamSLST

Synthesize streaming sign-language datasets from offline pre-segmented pose pickles for **PHOENIX-2014T** (DGS, German) and **CSL-Daily** (CSL, Chinese), in BOBSL-compatible layout.

## Why synthetic streams

BOBSL's **auto-aligned** subtitles introduce annotation noise. Synthetic streams give us:

- two more sign languages (DGS, CSL) covering cross-language generalization;
- **oracle event boundaries** by construction (we *know* exactly when each sentence starts and ends), letting us re-run the alignment / learned-vs-GT ablations to disentangle "model can't localize" from "BOBSL labels are noisy";

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

Switch dataset via the `DATASET` env var: `BOBSL` (default) | `PHOENIX` | `CSL`. `config.py` then resolves all paths, the mBART backbone, the target language code (`en_XX` / `de_DE` / `zh_CN`), and the per-dataset frame canvas (`(W,H) = (444,444) / (210,260) / (512,512)`) automatically. We use **`facebook/mbart-large-cc25`** for all 3 languages — its 25-language vocabulary covers all 3 target codes natively.

---

# 1. Problem statement

Each input dataset is a pickle that maps a clip identifier to one isolated, sentence-aligned record:

```
clip_record  := {
  keypoint   : float32 (T, 133, 3)     # COCO-WholeBody-133, NATIVE pixel coordinates
  text       : str                     # German subtitle (PHOENIX) | Chinese subtitle (CSL)
  num_frames : int                     # = T at the source frame rate
  name       : str                     # canonical clip identifier (encodes signer ID)
}
```

The downstream streaming model expects an **untrimmed** pose stream that contains multiple sentences interleaved with non-signing background. Naive concatenation of these isolated clips introduces several artifacts that an event-localization head learns to exploit instead of learning real sign-content boundaries:

| Artifact | Cause | Trivial cue model can latch onto |
|---|---|---|
| Hands-down → pause → hands-up at every join | Each clip recorded in isolation: signer rests before/after every utterance | Boundary = neutral-pose detector |
| Long pause between every pair of clips | Naive synth inserts a sampled pause between every clip | Pause length = localization signal |
| Frozen BG_pre / BG_post | Held first/last frame as background | BG = "no motion" Boolean |
| Discontinuous velocity at the seam | Last frame of A and first of B chosen independently | C0 discontinuity in keypoint trajectory |

The algorithm below removes all 4 artifacts. Each rule is **either pure geometry or sampled directly from BOBSL manual annotations**.

---

# 2. Algorithm overview

A synthetic stream is a temporal sequence assembled from $K$ sentence-level pose clips drawn from a single signer. The intuition behind every design choice is "*reproduce the joint distribution of co-articulation, pause length, and stream length that real continuous broadcast signing exhibits, while preserving oracle event boundaries that the offline corpus already gives us for free*".

Concretely, a stream $S$ has the schematic structure

```
S  =  [ B_pre ][ c_1 ][ Δ_1 ][ c_2 ][ Δ_2 ] ... [ c_K ][ B_post ]
       \_____/  \___/  \___/  \___/  \___/       \___/  \______/
         BG      clip  bridge  clip  bridge       clip     BG
```

where each $c_k$ is a real signer-pose clip from the offline pickle (after rest-trimming, §4.4) and each connector $\Delta_k, B_\text{pre}, B_\text{post}$ is a Hermite-interpolated bridge whose duration is sampled directly from the empirical BOBSL inter-subtitle gap distribution. Most $\Delta_k$ collapse to a 2-frame "movement-epenthesis" transition because BOBSL's empirical gap distribution puts ~74% of its mass at zero — i.e. real broadcast signing concatenates sentences co-articulated, not separated by neutral rest.

---

# 3. Notation

| Symbol | Definition | Source / value |
|---|---|---|
| $C$ | Pose clip pool, $C = \\{c_i = (P_i, t_i, s_i, n_i)\\}$ | Input pickle |
| $P_i \in \mathbb{R}^{T_i \times 133 \times 3}$ | Keypoint tensor for clip $i$ — $T_i$ frames, 133 joints, $(x, y, c)$ per joint | clip_record["keypoint"] |
| $t_i$ | Subtitle text for clip $i$ | clip_record["text"] |
| $s_i$ | Signer ID for clip $i$ | parsed from clip_record["name"] (§4.2) |
| $\mathcal{P}_s$ | Same-signer clip subset, $\mathcal{P}_s = \\{c_i \in C : s_i = s\\}$ | derived |
| $f$ | Target frame rate | $12.5$ Hz (matches StreamSLST `config.FPS`) |
| $W$ | Model training window length | $15$ s (matches StreamSLST `config.WINDOW_DURATION_SECONDS`) |
| $W_\text{stream}$ | Stream design target window | $4 W = 60$ s (see §4.9) |
| $G \in \mathbb{R}^{30552}$ | BOBSL empirical inter-subtitle gap array (negatives clipped to 0) | `bobsl_gap_samples.npy` |
| $G_+ \subset G$ | Positive-only subset of $G$ | $\lvert G_+ \rvert = 7,997$ |
| $K$ | Number of sentence clips per stream | sampled from $[K_\text{lo}, K_\text{hi}]$ |
| $K_\text{lo}, K_\text{hi}$ | $p_{10}, p_{90}$ of `subs_per_60s_window`(BOBSL) | $[6, 19]$ |
| $n_\text{min}$ | Numerical floor for the Hermite bridge | $2$ frames (§4.5) |
| $\rho_\text{min}$ | Minimum signer-pool size (need ≥1 phantom + ≥1 chosen) | $2$ |

---

# 4. Algorithm components

This section walks through every step of the synthesis pipeline. Each subsection states **why** the rule exists, gives a **mathematical** definition, and shows the **code** implementation. The complete end-to-end algorithm is assembled in §5.

## 4.1 Stream skeleton

A stream is the temporal concatenation of $2K + 1$ segments:

$$S \ =\  B_\text{pre} \ \Vert\  c_1 \ \Vert\  \Delta_1 \ \Vert\  c_2 \ \Vert\  \Delta_2 \ \Vert\  \cdots \ \Vert\  c_K \ \Vert\  B_\text{post}$$

where $\Vert$ denotes frame-axis concatenation. The **clip segments** $c_k$ contribute the actual subtitle events; the **bridge segments** $\Delta_k$ and the **background segments** $B_\text{pre}, B_\text{post}$ are non-event and contribute only background frames. The cue annotation list is computed deterministically from the segment lengths:

$$A = \\{(s_k, e_k, t_k)\\}_{k=1..K}, \quad s_k = \frac{1}{f}(\lvert B_\text{pre}\rvert + \sum_{j<k}(\lvert c_j\rvert + \lvert \Delta_j\rvert)), \quad e_k = s_k + \frac{\lvert c_k\rvert}{f}$$

Because $A$ is computed from segment lengths chosen by the synthesizer, the boundaries are **oracle** — exact to the frame.

## 4.2 Same-signer clip pool

Before any stream is synthesized, the clip pool is partitioned by signer:

$$\mathcal{P}_s = \\{c_i \in C : s_i = s\\}, \quad \text{drop pools with } \lvert \mathcal{P}_s \rvert < \rho_\text{min}$$

A stream draws all of its $K$ clips from a single $\mathcal{P}_s$. Cross-signer concatenation is avoided because it introduces two leakage hazards:

1. **Skeleton-scale jumps** at clip boundaries (different signers have different limb lengths and camera-relative scales) → free signal that the localization head can use as a boundary detector ("learn the seam, not the sign").
2. **Signer-identity change** becomes a confound for the boundary cue.

The signer ID is parsed deterministically from each clip's `name` field (no external metadata, no pseudo mapping):

- **PHOENIX**: clip name is `<split>/<broadcast>-<sentence_idx>` (e.g. `train/11August_2010_Wednesday_tagesschau-1`). One broadcast = one signer in PHOENIX. We group by `name.split('/')[1].rsplit('-', 1)[0]`.
- **CSL**: clip name is `Sxxx_P<id>_Tyy`. We group by the `P<id>` token.

```python
def signer_id(dataset, clip_name):
    if dataset == 'PHOENIX': return re.sub(r'-\d+$', '', clip_name.split('/', 1)[-1])
    if dataset == 'CSL':     return f"P{re.search(r'_P(\\d+)_', clip_name).group(1)}"
```

## 4.3 Frame-rate resampling

Source clips arrive at the source dataset's native frame rate ($25$ Hz for PHOENIX, $30$ Hz for CSL). They are resampled to the StreamSLST training rate $f = 12.5$ Hz by per-joint linear interpolation on $(x, y)$ and nearest-neighbour on the confidence channel:

$$P^{(f)}_t \ =\  \text{interp}(P^{(f_\text{src})}, t / f, \  t \in 0 .. \lfloor f \cdot T_\text{src} / f_\text{src} \rfloor)$$

Confidence is taken nearest-neighbour rather than averaged because averaging confidences is semantically meaningless (a 50%-confident average of two predictions does not mean "half-confident at the average position").

## 4.4 `trim_rest` — geometric removal of preparation / retraction

Each isolated clip begins with a *preparation* phase (signer raises hands from lap to first sign location) and ends with a *retraction* phase (hands fall back to lap). These artifacts are absent in real continuous broadcast signing — signers move directly between adjacent sentences without returning to rest. Concatenating non-trimmed clips creates an obvious "hands-down → long pause → hands-up" boundary that the localization head can latch onto trivially.

A frame is "rest" if both wrists sit BELOW the shoulder line in image coordinates (where larger $y$ = lower in the image):

$$\text{rest}(t) \ =\  \mathbb{1}\ \left[\ \frac{1}{|W|}\ \sum_{j \in W} P^{j,y}_t \ >\  \frac{1}{|S|}\ \sum_{j \in S} P^{j,y}_t\ \right]$$

where $W = \\{9, 10\\}$ (wrist indices in COCO-WholeBody-133) and $S = \\{5, 6\\}$ (shoulder indices). The rule is **signer-relative** — it compares the signer's own wrists against the signer's own shoulders — so it auto-scales across PHOENIX (210 × 260 canvas) and CSL (512 × 512 canvas) without any per-dataset threshold.

The trim interval is the contiguous non-rest core:

$$t_\text{start} = \min\\{t : \text{rest}(t) = 0\\}, \qquad t_\text{end} = \max\\{t : \text{rest}(t) = 0\\} + 1$$

Trim is applied only to the *contiguous prefix and suffix* runs of rest frames — we never cut mid-clip. A safety floor preserves the original clip if trimming would leave fewer than 3 frames (the minimum needed for endpoint-velocity estimation in the Hermite bridge).

```python
def trim_rest(P):
    sh_y = P[:, [5, 6],  1].mean(axis=1)
    wr_y = P[:, [9, 10], 1].mean(axis=1)
    valid = (P[:, [5, 6, 9, 10], 2].min(axis=1) > 0)        # all 4 anchors confidently estimated
    is_rest = (wr_y > sh_y) & valid
    start = next(t for t in range(len(P)) if not is_rest[t])
    end   = next(t for t in range(len(P)-1, -1, -1) if not is_rest[t]) + 1
    return P if end - start < 3 else P[start:end]
```

After this transformation, each clip starts and ends *during signing activity*. Concatenating two trimmed clips produces a kinematically continuous signing-to-signing transition rather than a rest-pose-mediated one — exactly the **co-articulation** phenomenon real signers exhibit (also called *movement epenthesis* in the sign-linguistics literature).

## 4.5 Hermite bridge — C¹-continuous seam

Even after trimming, the last frame of clip $A$ and the first frame of clip $B$ are kinematically independent — joining them directly would create a position discontinuity (teleport) and a velocity discontinuity at the seam. A linear interpolation between $\mathbf{p}_0 = c_A[-1]$ and $\mathbf{p}_1 = c_B[0]$ would remove the position teleport but still has $C^0$ continuity (velocity discontinuity remains). A cubic Hermite spline achieves $C^1$ continuity by additionally matching the per-frame velocity at each endpoint.

Per joint, on $(x, y)$ only (the confidence channel is propagated separately, see below), the spline is parameterized by $s \in [0, 1]$:

$$\mathbf{h}(s) \ =\  h_{00}(s)\ \mathbf{p}_0 \ +\  h_{10}(s)\ \mathbf{m}_0 \ +\  h_{01}(s)\ \mathbf{p}_1 \ +\  h_{11}(s)\ \mathbf{m}_1$$

with the standard cubic Hermite basis polynomials

$$h_{00}(s) = 2s^3 - 3s^2 + 1, \qquad h_{10}(s) = s^3 - 2s^2 + s$$

$$h_{01}(s) = -2s^3 + 3s^2, \qquad h_{11}(s) = s^3 - s^2$$

The **endpoint tangents** $\mathbf{m}_0, \mathbf{m}_1$ are derived from the actual per-frame velocities at the seam, scaled to per-unit-$s$ via $\text{span} = n + 1$:

$$\mathbf{v}_0 = c_A[-1] - c_A[-2], \qquad \mathbf{v}_1 = c_B[1]  - c_B[0]$$

$$\mathbf{m}_i = \mathbf{v}_i \cdot (n+1) \cdot \alpha_n$$

where $\alpha_n$ is the **length-aware tangent damping** factor

$$\boxed{\ \alpha_n = \frac{n_\text{min}}{\max(n, n_\text{min})}\ }$$

with $n_\text{min} = 2$. This factor is the key to making the spline behave correctly across both ends of the bridge-length spectrum:

- **Short bridges** ($n \leq n_\text{min}$): $\alpha_n = 1$ → full clip-end momentum is preserved → the spline reproduces the natural movement-epenthesis trajectory between two signing poses (co-articulation).
- **Long bridges** ($n \gg n_\text{min}$): $\alpha_n \to 0$ → tangent contribution vanishes → the spline degenerates to a position-only interpolation between $\mathbf{p}_0$ and $\mathbf{p}_1$ → the hand smoothly settles into rest, no overshoot.

Tangent magnitude is additionally clamped per joint to prevent overshoot when endpoint velocities happen to be locally large:

$$\lVert\mathbf{m}_i\rVert \ \leftarrow\  \min\ (\lVert\mathbf{m}_i\rVert,\  2\ \lVert\mathbf{p}_1 - \mathbf{p}_0\rVert)$$

Confidence is the element-wise minimum of the two endpoint confidences, $c_\text{out}^{(j)} = \min(c_0^{(j)}, c_1^{(j)})$, marking interpolated frames as no more confident than the worst of the two real endpoints.

The bridge length itself is sampled in seconds and converted to integer frames with the floor:

$$n \ =\  \max\ (n_\text{min},\  \lfloor \ell \cdot f \rceil)$$

The floor $n_\text{min} = 2$ is a **numerical** floor, not a tuning knob — a Hermite spline needs at least two interior frames to define a transition (one frame is just a point).

## 4.6 Inter-clip pause sampling

One attempt is to fit a LogNormal $\ell \sim \text{LogNormal}(\mu, \sigma)$ to BOBSL inter-subtitle gaps and sample from it. This fails badly because the real distribution is *bimodal*: a large point-mass at $\ell = 0$ (touching or overlapping subtitles = continuous co-articulated signing) plus a heavy-tailed positive component (real sentence-break pauses). No symmetric or unimodal parametric distribution captures the zero spike. We instead sample directly from the empirical CDF of BOBSL's manual-aligned inter-subtitle gaps:

$$\hat{\mathbb{P}}(\ell) \ =\  \frac{1}{|G|} \sum_{i=1}^{|G|} \delta(\ell - G_i), \qquad \ell \sim \hat{\mathbb{P}}$$

```python
def sample_pause_s(rng, G):
    return float(rng.choice(G))
```

Where the array $G$ comes from $4{,}468$ BOBSL manual VTT files: for every adjacent subtitle pair $(a, b)$ in every file we compute $g = b.\text{start} - a.\text{end}$, giving $|G| = 30{,}552$ gaps. Negatives are clipped to zero (overlapping subtitles cannot be a "pause"; they are continuous signing). After clipping,

$$\mathbb{P}(\ell = 0) \ =\  \frac{|\\{G_i : G_i = 0\\}|}{|G|} \ =\  \frac{22\,555}{30\,552} \ \approx\  0.738$$

So **73.8% of inter-clip joins in any synthesized stream concatenate co-articulated** by construction — without any LogNormal fit, any `pause_min_s` knob, or any `pause_max_s` clamp. The empirical CDF *is* the model.

The remaining 26.2% of joins receive positive bridge durations sampled from the right tail of $G$, with median $\approx 2.0$ s and $p_{90} \approx 6.0$ s — matching what real BOBSL exhibits.

## 4.7 BG_pre / BG_post sampling

**Why a different sampler for BG**. The two background segments $B_\text{pre}$ and $B_\text{post}$ are conceptually distinct from inter-clip pauses: they represent the silent broadcast lead-in/lead-out, when there is no caption and the signer may not yet (or no longer) be signing. Sampling them from the full empirical $G$ would collapse ~74% of streams to a 2-frame BG (invisible). BG must always be **visibly present** so that the model gets exposure to genuine no-signing regions.

We therefore sample BG durations from the **conditional distribution** $\hat{\mathbb{P}}(\ell \mid \ell > 0)$, equivalently from the positive-only subset

$$G_+ = \\{ G_i \in G : G_i > 0 \\}, \quad |G_+| = 7\,997$$

```python
def sample_bg_s(rng, G_plus):
    return float(rng.choice(G_plus))
```

This enforces "BG is always positive" by construction without introducing a min-duration knob: the floor is whatever the smallest positive BOBSL inter-subtitle gap is.

**Phantom clips.** For the bridge endpoint of the BG segment that is not adjacent to a chosen clip (i.e. the "outside" endpoint of $B_\text{pre}$ or $B_\text{post}$), we sample one additional clip from the same signer's spare pool — the **phantom clip** $\varphi$. This animates the BG region as a real signer-pose-to-real-signer-pose transition rather than a frozen frame. Phantom clips are **not** trimmed: BG segments represent the broadcast lead-in / lead-out, where retaining the phantom's natural rest pose is realistic. Formally:

$$\varphi_L, \varphi_R \ \sim\  \text{Uniform}(\mathcal{P}_s \setminus \\{c_1, \ldots, c_K\\})$$

If the spare pool is empty, the synthesizer falls back to using the first/last selected clip itself; the seam is still Hermite-interpolated, just less varied.

## 4.8 K-per-stream from a 60-second BOBSL window

**Why 60 s, not 15 s.** The model's training window is $W = 15$ s. If the synthesized streams were also $\sim 15$ s long, then evaluation would amount to single-window decoding — not streaming inference. To genuinely exercise streaming behaviour (cross-window event handling, state propagation across windows), streams must span *multiple* training windows. We choose the design ratio

$$W_\text{stream} = 4\ W = 60 \text{ s}$$

as a deliberate compromise: long enough that streaming inference must fire, short enough that signer pools (especially PHOENIX's tight ones) can fill them.

**K range from BOBSL.** For each BOBSL manual VTT we slide a window of length $W_\text{stream}$ with a $1$-s step and count the number of subtitles that fall entirely within each window position. Let $\\{N_w\\}$ denote the resulting count distribution (over $\sim 161$k window positions across all VTT files). Then

$$K \ \sim\  \text{Uniform}\ \left(\\{K_\text{lo}, K_\text{lo}+1, \ldots, K_\text{hi}\\}\right)$$

with

$$K_\text{lo} = Q_{10}(\\{N_w\\}), \qquad K_\text{hi} = Q_{90}(\\{N_w\\})$$

In our current run, $K_\text{lo} = 6, K_\text{hi} = 19$. So a synthesized stream contains 6 to 19 sentences, mirroring how densely real BOBSL packs them in a 60-s window. For comparison, the same computation on a 15-s window gives $[Q_{10}, Q_{90}] = [0, 5]$ — confirming that the 60-s anchor produces $\sim 4 \times$ as many sentences per stream, exactly the design intent.

The per-stream $K$ is additionally clamped to the signer's pool size, $K \leftarrow \min(K, |\mathcal{P}_s|)$, since we sample without replacement.

## 4.9 Stream count

Each split contains a target number of streams chosen so that every clip is used about *once* on average:

$$N_S \ =\  \text{round}\ \left(\frac{|\bigcup_s \mathcal{P}_s|}{\bar{K}}\right), \qquad \bar{K} = \frac{K_\text{lo} + K_\text{hi}}{2}$$

This preserves the offline split's contract: no clip duplication beyond what's statistically inevitable. We deliberately do **not** inflate this with a permutation-multiplier knob — on small signer pools (PHOENIX broadcasts have only 2-3 clips each) any multiplier above 1 saturates the $K!$ permutation space and biases training toward duplicated streams. (`rng.choice(..., replace=False)` already returns the chosen elements in a random order, so each stream is implicitly a random permutation of its chosen subset — but we let the permutation-space exploration arise naturally from the per-stream sampling, not from explicit replication.)

---

# 5. Putting it all together — full pseudocode

The two algorithms below assemble the components into the end-to-end pipeline.

## Algorithm 1 — synthesize one stream

```
Inputs:  Pool[s], G, G_+, K_lo, K_hi, FPS, MIN_BRIDGE_FRAMES, rng
Output:  pose tensor P (T, 133, 3),  cue list A = [(start_s, end_s, text)]

 1: K         ← rng.uniform_int(K_lo, min(K_hi, |Pool[s]|))
 2: chosen    ← rng.choice(Pool[s], K, replace=False)         # ORDER IS RANDOM (= permutation)
 3: resampled ← [trim_rest(resample_to_FPS(P_i, src_fps)) for c_i in chosen]
 4: drop entries shorter than 1·FPS frames; if resampled = ∅ return empty stream
 5: spare ← Pool[s] \ chosen
 6: if |spare| ≥ 2:  φ_L, φ_R ← rng.choice(spare, 2)          # phantom clips for BG
    else: φ_L, φ_R ← (resampled[-1], resampled[0])            # fall back to chosen

 7: ℓ_pre  ← rng.choice(G_+)                                  # BG_pre  duration (positive-only)
 8: ℓ_post ← rng.choice(G_+)                                  # BG_post duration (positive-only)
 9: B_pre  ← Hermite(φ_L[-1], v(φ_L, "last"),
                     resampled[0][0], v(resampled[0], "first"),
                     n = max(MIN_BRIDGE_FRAMES, ⌊ℓ_pre · FPS⌉))

10: segments ← [B_pre]; cues ← []; cur ← |B_pre|
11: for k = 0 ... K-1:
12:     segments.append(resampled[k])
13:     cues.append( (cur/FPS,  (cur+|resampled[k]|)/FPS,  t_k) )
14:     cur ← cur + |resampled[k]|
15:     if k < K-1:
16:         ℓ ← rng.choice(G)                                 # FULL empirical, ~74% zeros
17:         Δ ← Hermite(resampled[k][-1],   v(resampled[k],   "last"),
                        resampled[k+1][0], v(resampled[k+1], "first"),
                        n = max(MIN_BRIDGE_FRAMES, ⌊ℓ · FPS⌉))
18:         segments.append(Δ); cur ← cur + |Δ|

19: B_post ← Hermite(resampled[-1][-1], v(resampled[-1], "last"),
                     φ_R[0],            v(φ_R,           "first"),
                     n = max(MIN_BRIDGE_FRAMES, ⌊ℓ_post · FPS⌉))
20: segments.append(B_post)
21: P ← concat(segments, axis=0);  A ← cues
22: return (P, A)
```

`v(P, end)` returns the per-frame velocity at the requested clip endpoint: $\mathbf{v}(P, \text{last}) = P[-1] - P[-2]$, $\mathbf{v}(P, \text{first}) = P[1] - P[0]$. Confidence channel set to $0$ (unused for tangent computation).

## Algorithm 2 — driver across splits

```
Inputs:  dataset D ∈ {PHOENIX, CSL}, seed σ
Output:  per-split stream files + manifest.json + subset2episode.json

 1: G        ← np.clip( load("bobsl_gap_samples.npy"), 0, ∞ )           # negatives → 0
 2: G_+      ← G[G > 0]
 3: K_lo, K_hi ← p10, p90 of bobsl_stats["subs_per_stream_window"]      # 60-s sliding window
 4: for split ∈ {train, val, test}:
 5:     C_split ← load_pickle(D, split);  drop clips with |P_i| < 1·src_fps or empty text
 6:     Pool ← group C_split by signer (§4.2);  drop pools with size < ρ_min
 7:     N_S ← max(1, round(|⋃ Pool| / mean(K_lo, K_hi)))                # stream count (§4.9)
 8:     for j = 1 ... N_S:
 9:         pick s uniformly from active signer IDs
10:         (P_j, A_j) ← Algorithm-1(Pool[s], G, G_+, K_lo, K_hi, FPS, MIN_BRIDGE_FRAMES, rng_j)
11:         if A_j ≠ ∅: write poses/<j>.npy, vtt/<j>.vtt, manifest entry
```

The construction `rng_j = np.random.default_rng([base_seed, j])` makes every stream's sampling deterministic and independent of stream ordering — re-running with the same seed reproduces every stream byte-for-byte.

---

# 6. Provenance of every parameter

Every value the synthesizer touches is either pure geometry or measured directly from the BOBSL manual annotations (`data/BOBSL/manual_annotations/signing_aligned_subtitles/*.vtt`, $4\,468$ files). Auto-aligned BOBSL is **not** used — that's the very source of annotation noise the original paper was criticised for.

| Parameter | Source | Computed value |
|---|---|---|
| $f$ (FPS) | StreamSLST `config.FPS` | $12.5$ Hz |
| $W$ (training window) | StreamSLST `config.WINDOW_DURATION_SECONDS` | $15$ s |
| $W_\text{stream}$ | Design ratio $4 W$ | $60$ s |
| $K_\text{lo}, K_\text{hi}$ | $p_{10}, p_{90}$ of `subs_per_60s_window`(BOBSL) | $[6, 19]$ |
| $G$ | All inter-subtitle gaps from BOBSL manual VTTs, negatives clipped to 0 | $\lvert G\rvert = 30\,552$, $\mathbb{P}(g=0)=0.738$, $p_{90}=2.0$ s |
| $G_+$ | $G \setminus \\{0\\}$ | $\lvert G_+\rvert = 7\,997$, median $2.0$ s, $p_{90} = 10.0$ s |
| $n_\text{min}$ | Numerical floor for Hermite (1 frame is just a point) | $2$ |
| $\rho_\text{min}$ | Need ≥1 phantom + ≥1 chosen | $2$ |
| trim_rest threshold | Wrists vs shoulder geometry (Boolean per frame, §4.4) | none |
| Tangent damping $\alpha_n$ | $n_\text{min} / \max(n, n_\text{min})$ | reuses existing constant, no new knob |

---

# 7. Design decisions vs alternatives

| Choice we made | Alternative considered | Why we picked ours |
|---|---|---|
| Empirical pause sampling (§4.6) | LogNormal fit to $G_+$ | Real BOBSL is bimodal (74% zero + 26% positive heavy tail); no parametric distribution captures the zero spike. Empirical CDF *is* the truth. |
| Positive-only $G_+$ for BG (§4.7) | Same $G$ as inter-clip | With $G$, ~74% of streams collapse to a 2-frame BG (invisible). $G_+$ enforces visible BG without a min-duration knob. |
| `trim_rest` rule = wrist-below-shoulder (§4.4) | Per-clip 30th-percentile velocity threshold | Velocity threshold needs a per-clip percentile (a knob); geometric rule is signer-relative and parameter-free. |
| Hermite cubic spline (§4.5) | Linear interpolation | Linear has $C^0$ continuity at the seam → visible velocity discontinuity. Hermite achieves $C^1$. |
| Tangent damping by $1/n$ (§4.5) | Fixed tangent magnitude | Fixed tangent overshoots on long bridges. $1/n$ damping makes long bridges relax to position-only interpolation. |
| Same-signer per stream (§4.2) | Cross-signer | Cross-signer introduces scale-jump and identity-change leakage; same-signer matches BOBSL. |
| $K$ from 60-s window (§4.8) | $K$ from 15-s window | 15 s = model window; streams must span ≥ 4× window for streaming inference to actually fire. |
| Stream count = $\lvert C\rvert / \bar K$ (§4.9) | Multiplier-based inflation | On small pools (PHOENIX, ~3 clips/broadcast) any multiplier $>1$ saturates the $K!$ permutation space and biases training toward duplicated streams. Each clip used $\sim$ once is principled. |

---

# 8. Failure modes & safeguards

| Failure mode | Safeguard in code |
|---|---|
| `trim_rest` cuts entire clip | If $t_\text{end} - t_\text{start} < 3$, return original clip unchanged |
| Hermite called with $n \leq 0$ | Returns empty array of correct shape |
| Hermite tangent magnitude explodes | Clamped to $2 \lVert \mathbf{p}_1 - \mathbf{p}_0\rVert$ per joint per side |
| Sampled pause = 0 | Floored to $n_\text{min} = 2$ (smooth co-articulation transition) |
| Signer pool too small for phantoms | Fall back to using first/last chosen clip as phantom (still Hermite-interpolated) |
| Subtitle is empty / clip < 1 s after resample | Drop the clip; skip stream if no clips remain |
| Negative inter-subtitle gap (overlap) | Clipped to 0 in `bobsl_gap_samples.npy` (load-time, see §4.6) |
| Pose confidence values $> 1.0$ | Clipped to $[0, 1]$ at clip-load time |

---

# 9. Workflow

```bash
# 0. (one-time, after BOBSL.zip extracted) compute pause + K stats from BOBSL MANUAL annotations.
python -m data_synth.analyze_bobsl_gaps \
    --vtt_dir data/BOBSL/manual_annotations/signing_aligned_subtitles \
    --out data_synth/stats/bobsl_gap_stats.json
# Writes both bobsl_gap_stats.json (summary) and bobsl_gap_samples.npy (full empirical array).
# If skipped, the synthesizer falls back to an analytic distribution that emulates the same shape.

# 1. Synthesize streams (no count args; everything derived from data + BOBSL stats).
python -m data_synth.synthesize_streams --dataset PHOENIX --out_root data/synth/phoenix --seed 42
python -m data_synth.synthesize_streams --dataset CSL     --out_root data/synth/csl  --seed 43

# 2. Visualize a few streams (reads DATASET env to pick canvas; renders the 77 model-input keypoints
#    with auto-fit window so out-of-frame pose outliers don't push the body off-screen).
DATASET=PHOENIX python -m data_synth.visualize_stream \
    --pose data/synth/phoenix/poses/test_00000.npy \
    --vtt  data/synth/phoenix/vtt/test_00000.vtt --out data_synth/examples/phoenix_test_00000.mp4
DATASET=CSL python -m data_synth.visualize_stream \
    --pose data/synth/csl/poses/test_00000.npy \
    --vtt  data/synth/csl/vtt/test_00000.vtt --out data_synth/examples/csl_test_00000.mp4

# 3. Trim mBART tokenizer + model per language (writes captioners/trimmed_*_<lang>/)
DATASET=PHOENIX python -m captioners.trim_mbart
DATASET=CSL     python -m captioners.trim_mbart

# 4. Compute BOBSL-paper-style dataset stats (also reports BPE-subword vocab from the trimmed tokenizer)
DATASET=PHOENIX python -m data_synth.dataset_stats --root data/synth/phoenix --out data_synth/stats/phoenix_stats.json
DATASET=CSL     python -m data_synth.dataset_stats --root data/synth/csl     --out data_synth/stats/csl_stats.json
```

---

# 10. Resulting benchmark sizes

K is derived from a **60-second** sliding window of BOBSL ($= 4 W$, the model's training window) so synthesized streams span multiple model windows and the streaming inference behaviour the paper claims actually fires at evaluation time. Combined with `trim_rest` + empirical pauses + positive-only BG, streams are median 40–60 s long with 7–11 sentences each:

| split | streams | signer-pure pools | hours | cues / stream | stream dur p50 / p90 (s) | pause med / p90 (s) | density |
|---|---:|---:|---:|---:|---:|---:|---:|
| PHOENIX train | 564 | 387 broadcasts | 6.99 | 7.67 | 41.6 / 65.2 | 0.00 / 2.00 | 0.715 |
| PHOENIX val | 27 | 25 | 0.10 | 2.07 | 13.0 / 20.8 | 0.00 / 1.10 | 0.515 |
| PHOENIX test | 38 | 35 | 0.17 | 2.50 | 15.1 / 25.8 | 0.00 / 2.00 | 0.522 |
| CSL train | 1467 | 10 P-ids | 21.67 | 10.38 | 50.2 / 82.4 | 0.00 / 2.00 | 0.716 |
| CSL val | 86 | 10 | 1.43 | 10.95 | 55.8 / 88.3 | 0.00 / 2.00 | 0.710 |
| CSL test | 94 | 10 | 1.56 | 11.00 | 58.4 / 90.4 | 0.00 / 2.00 | 0.745 |

Pause distribution matches the BOBSL empirical sample by construction (~74% of inter-clip joins have $\ell = 0$ and concatenate co-articulated, see §4.6). PHOENIX dev/test streams are shorter than train because per-broadcast clip pools on those splits are only ~2–3 clips — same-signer-per-stream caps $K$ at the pool size; the underlying biological signer count is 9 across all 629 broadcasts. CSL is unaffected (each of the 10 P-ids has hundreds of clips per split).

---

# 11. Reproducibility

- Per-stream RNG is `np.random.default_rng([base_seed, stream_idx])` so re-running with the same `--seed` reproduces every stream byte-for-byte.
- Default seeds: train=42, val=43, test=44. Override via `--seed`.
- `manifest.json` records, per stream: chosen clip names, phantom clip names, pause durations sampled, signer ID, $K$, total stream duration. Dataset can be reconstructed from clip pickles + manifest alone.

---

# Files

- `synthesize_streams.py` — main pipeline (Algorithms 1 + 2)
- `analyze_bobsl_gaps.py` — writes `bobsl_gap_stats.json` (summary + K-range stats) and `bobsl_gap_samples.npy` (full empirical gap array consumed by the synthesizer)
- `bobsl_gap_stats.json` / `bobsl_gap_samples.npy` — produced by the above; both auto-detected by `synthesize_streams.py`
- `visualize_stream.py` — renders the 77 model-input keypoints as MP4 with PIL CJK subtitle overlay; segments labelled `[BG/PAUSE]` vs the subtitle text so phases are visually distinguishable
- `dataset_stats.py` — BOBSL-paper-style stats; reports word vocab + BPE vocab side by side
- `verify_synth.py` — round-trip sanity check (load → normalize → threshold → parse_vtt)
