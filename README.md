# Streaming Sign Language Translation via Dense Temporal Grounding

This repository contains the official implementation for end-to-end streaming sign language translation (SLT) using a dense temporal grounding framework. The system jointly localizes and translates sign language events from continuous pose streams without requiring gloss annotations.

## Overview

The model processes continuous pose sequences through a sliding window mechanism and performs:

1. **Temporal event localization** — detecting when individual sentences occur within a stream
2. **Dense captioning** — translating each localized event into the target spoken language

The architecture combines:
- A **pose backbone** (CoSign ST-GCN or MSKA) for spatiotemporal feature extraction from skeleton keypoints
- A **Deformable DETR** encoder-decoder for temporal event detection
- A **trimmed mBART** caption decoder for multilingual translation

Training follows a two-stage procedure:
- **Stage 1:** Visual-language contrastive pre-training (InfoNCE with masked pose views)
- **Stage 2:** Joint localization and captioning with Hungarian matching

A cascaded baseline (**GFSLT-VLP**) is also provided for comparison.

## Supported Datasets

| Dataset | Language | Source | Pose Format |
|---|---|---|---|
| BOBSL | English (BSL) | Auto-aligned broadcast subtitles | COCO-WholeBody-133 via DWPose |
| PHOENIX-2014T | German (DGS) | Synthesized streams from per-clip pickles | COCO-WholeBody-133 |
| CSL-Daily | Chinese (CSL) | Synthesized streams from per-clip pickles | COCO-WholeBody-133 |
| How2Sign | English (ASL) | Real signer-aligned timestamps | COCO-WholeBody-133 (from OpenPose) |

Switch datasets via environment variable: `DATASET={BOBSL,PHOENIX,CSL,H2S}`. All paths, target language codes, and preprocessing parameters are resolved automatically in `config.py`.

---

## 1. Environment Setup

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch 2.6+, Transformers 4.57+, accelerate, sacrebleu, pycocoevalcap, BLEURT.

## 2. Data Preparation

### Directory Layout

Each dataset follows a unified directory structure:

```
data/<dataset>/
├── poses/<video_id>.npy           # (T, 133, 3) float32 at target FPS
├── vtt/<video_id>.vtt             # WebVTT subtitles (one sentence per cue)
└── subset2episode.json            # {"train": [...], "val": [...], "test": [...]}
```

### Synthetic Stream Generation (PHOENIX / CSL / H2S)

For datasets other than BOBSL, synthesize streaming benchmarks from pre-segmented data:

```bash
DATASET=PHOENIX python -m data_synth.synthesize_streams --out_root data/synth/phoenix
DATASET=CSL python -m data_synth.synthesize_streams --out_root data/synth/csl
DATASET=H2S python -m data_synth.synthesize_h2s --out_root data/synth/h2s
```

See `data_synth/README.md` for details on the co-articulation synthesis pipeline.

### Tokenizer and mBART Trimming

Before training, trim the mBART vocabulary to the target dataset's subtitle tokens:

```bash
DATASET=BOBSL python captioners/trim_mbart.py
DATASET=PHOENIX python captioners/trim_mbart.py
DATASET=CSL python captioners/trim_mbart.py
DATASET=H2S python captioners/trim_mbart.py
```

## 3. Training (Proposed Model)

All training uses HuggingFace's `Trainer` with `HfArgumentParser`. Key hyperparameters are CLI flags.

### Stage 1: Visual-Language Contrastive Pre-training

- **Frozen:** encoder, decoder, detection heads, caption head
- **Trained:** pose backbone + text encoder
- **Loss:** 3-way InfoNCE (view1↔view2, view1↔text, view2↔text) with temperature τ=0.07

```bash
torchrun --nproc_per_node <NUM_GPUS> main.py \
    --mode 1 \
    --output_dir ./checkpoints/mode1 \
    --max_event_tokens 40 \
    --d_model 1024 \
    --encoder_layers 2 \
    --decoder_layers 2 \
    --num_cap_layers 3 \
    --num_queries 30 \
    --num_train_epochs 50 \
    --learning_rate 5e-4 \
    --per_device_train_batch_size 32
```

### Stage 2: Joint Localization + Captioning

- **Unfrozen:** all parameters
- **Losses:** classification + GIoU + event count + caption (Hungarian matching)

```bash
torchrun --nproc_per_node <NUM_GPUS> main.py \
    --mode 2 \
    --mode1_checkpoint ./checkpoints/mode1/mode1_final \
    --output_dir ./checkpoints/mode2 \
    --max_event_tokens 40 \
    --d_model 1024 \
    --encoder_layers 2 \
    --decoder_layers 2 \
    --num_cap_layers 3 \
    --num_queries 30 \
    --num_train_epochs 100 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 32
```

### Key Training Flags

| Category | Flag | Default | Description |
|---|---|---|---|
| Data | `--max_event_tokens` | 40 | Max tokens per caption |
| Data | `--stride_ratio` | 0.9 | Sliding window stride (val/test) |
| Data | `--noise_rate` | 0.15 | Token masking rate for contrastive learning |
| Data | `--pose_augment` | False | Apply pose augmentation (train only) |
| Model | `--d_model` | 1024 | Hidden dimension |
| Model | `--num_queries` | 30 | Max detected events per window |
| Model | `--encoder_layers` | 2 | Deformable DETR encoder layers |
| Model | `--decoder_layers` | 2 | Deformable DETR decoder layers |
| Model | `--num_cap_layers` | 3 | Caption decoder layers |
| Model | `--captioner_type` | mbart | Caption head type (`mbart` or `lstms`) |
| Loss | `--class_cost` | 2 | Classification loss weight |
| Loss | `--giou_cost` | 4 | GIoU loss weight |
| Loss | `--counter_cost` | 2 | Event count loss weight |
| Loss | `--caption_cost` | 2 | Caption loss weight |
| Backbone | `BACKBONE` env var | cosign | Pose backbone (`cosign` or `mska`) |

## 4. Training (GFSLT Baseline)

The cascaded GFSLT-VLP baseline trains in two stages:

```bash
# Stage 1: CLIP-style VLP + Masked LM
torchrun --nproc_per_node <NUM_GPUS> gfslt_stage1.py \
    --output_dir ./checkpoints/gfslt_stage1 \
    --num_train_epochs 50 \
    --learning_rate 1e-4

# Stage 2: End-to-end translation (encoder initialized from Stage 1)
torchrun --nproc_per_node <NUM_GPUS> gfslt_stage2.py \
    --stage1_checkpoint ./checkpoints/gfslt_stage1 \
    --output_dir ./checkpoints/gfslt_stage2 \
    --num_train_epochs 80 \
    --learning_rate 5e-5
```

## 5. Evaluation

Use `eval.py` for the proposed model. **Run on a single GPU** to avoid distributed overhead:

```bash
# Evaluate on both val and test
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --checkpoint_path checkpoints/mode2/mode2_final

# Evaluate only test set with custom settings
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --checkpoint_path checkpoints/mode2/mode2_final \
    --eval_val False \
    --max_event_tokens 40 \
    --num_queries 30 \
    --per_device_eval_batch_size 32 \
    --ranking_temperature 2.0 \
    --top_k 20 \
    --aggregation_mode video
```

For the cascaded (GFSLT + DETR localization) evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python gfslt_cascaded_eval.py \
    --detr_checkpoint_path checkpoints/mode2/pytorch_model.bin \
    --gfslt_checkpoint_path checkpoints/gfslt_stage2/pytorch_model.bin
```

### Evaluation Metrics

The evaluation computes three levels of quality:

1. **Localization:** Precision / Recall / F1 at IoU thresholds {0.3, 0.5, 0.7, 0.9}
2. **Dense captioning:** BLEU-4, METEOR, ROUGE-L, CIDEr, BLEURT for matched (pred, GT) pairs at each IoU threshold, plus SODA_c storytelling F1
3. **Paragraph-level:** Sort predicted captions by time → join into paragraph → compare against GT paragraph (same translation metrics)

Aggregation modes: `--aggregation_mode {corpus, window, video}`

## 6. Model Smoke Test

Verify the forward/backward pass on a small batch:

```bash
python pdvc.py
```

## Project Structure

```
├── main.py                    # Proposed model training (Stage 1 + 2)
├── eval.py                    # Proposed model evaluation
├── gfslt_stage1.py            # GFSLT baseline Stage 1 (VLP)
├── gfslt_stage2.py            # GFSLT baseline Stage 2 (translation)
├── gfslt_cascaded_eval.py     # Cascaded evaluation (DETR loc + GFSLT cap)
├── gfslt_models.py            # GFSLT model definitions
├── pdvc.py                    # Deformable DETR + caption head model
├── loader.py                  # DVCDataset (sliding window, streaming)
├── config.py                  # Dataset/backbone/path configuration
├── loss.py                    # Hungarian matcher + losses
├── postprocess.py             # NMS and top-k event extraction
├── utils.py                   # VTT parsing, helpers
├── backbones/
│   ├── cosign.py              # CoSign ST-GCN backbone
│   └── mska_backbone.py       # MSKA pose encoder backbone
├── captioners/
│   ├── mbart.py               # Trimmed mBART decoder captioner
│   ├── lstm.py                # LSTM captioner (ablation)
│   └── trim_mbart.py          # Vocabulary trimming script
├── deformable_detr/           # Deformable DETR encoder/decoder
├── evaluation/
│   ├── metrics.py             # Trainer-integrated metric computation
│   ├── helpers.py             # Top-k selection, aggregation utilities
│   └── soda_c.py              # SODA_c implementation
├── data_synth/                # Stream synthesis for PHOENIX/CSL/H2S
│   ├── synthesize_streams.py  # PHOENIX/CSL synthesis
│   ├── synthesize_h2s.py      # How2Sign synthesis
│   └── README.md              # Synthesis documentation
└── poses/
    ├── preprocessing.py       # Keypoint normalization
    └── augmentation.py        # Pose augmentation
```

## Troubleshooting

- **VTT parsing:** `webvtt-py` is used if installed; otherwise a built-in parser is used. Ensure `.vtt` files are under the configured `VTT_DIR`.
- **Poses:** Ensure `POSE_ROOT/<video_id>/*.npy` (BOBSL) or `POSE_ROOT/<stream_id>.npy` (synth) exists for each video in your split JSON.
- **Multi-GPU evaluation:** Always set `CUDA_VISIBLE_DEVICES=0` for `eval.py` to avoid unnecessary DDP initialization.
- **BLEURT:** The BLEURT-20 checkpoint is downloaded automatically to `/tmp/BLEURT-20` on first use.
