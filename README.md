# E2E-SLT-Streaming

### 1. Environment

```bash
git clone https://github.com/18520339/e2e-slt-streaming.git
cd e2e-slt-streaming
pip install -r requirements.txt
```

### 2. Data paths and Tokenizer/MBart preparation

This repo expects BOBSL data laid out under `dataset/BOBSL` (already the default). If your dataset lives elsewhere, update these in `config.py`:

-   `DATA_ROOT` (base dataset dir)
-   `SUBSET_JSON` (train/val/test split JSON)
-   `VTT_DIR` (aligned subtitles in .vtt)
-   `POSE_ROOT` (pose `.npy` folders per video id)

After setting up data paths, run the trimming script to create a smaller tokenizer and MBart model for captioning:

```bash
python captioners/trim_mbart.py
```

### 3. Train (Hugging Face Trainer + CLI)

`main.py` is a single-file training script using `HfArgumentParser` with dataclasses. All key knobs are CLI flags with sensible defaults.

Quick start (CPU or single GPU):

```bash
python main.py --output_dir "./outputs/run1"
```

Multi‑GPU with torchrun (recommended):

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nproc_per_node 6 main.py \
	--output_dir "./outputs/run1" \
	--num_train_epochs 200 \
	--per_device_train_batch_size 64 \
	--per_device_eval_batch_size 128 \
	--weight_decay 1e-4 \
	--learning_rate 5e-4
```

Multi‑GPU with accelerate:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --num_processes 6 main.py \
	--output_dir "./outputs/run1" \
	--num_train_epochs 200 \
	--per_device_train_batch_size 64 \
	--per_device_eval_batch_size 128 \
	--weight_decay 1e-4 \
	--learning_rate 5e-4
```

What it does:

-   Builds train/val datasets from BOBSL poses and VTTs.
-   Initializes a Deformable DETR-based model with a captioning head.
-   Trains with Hugging Face Trainer and saves the final model to the `CHECKPOINT_DIR` defined in `config.py`.

Common flags (subset shown):

-   Data: `--max_event_tokens 64`, `--stride_ratio 0.9`
-   Model: `--d_model 1024`, `--num_queries 100`, `--encoder_layers 2`, `--decoder_layers 2`, `--num_cap_layers 3`
-   Metrics: `--alpha 0.3`, `--ranking_temperature 2.0`, `--top_k 20`, `--temporal_iou_thresholds 0.3 0.5 0.7 0.9`
-   Trainer: `--num_train_epochs 200`, `--per_device_train_batch_size 64`, `--per_device_eval_batch_size 128`, `--output_dir ./outputs/run1`

Tips:

-   To change window length or FPS, edit `config.py`. To change caption length, pass `--max_event_tokens`.
-   Metrics are computed during evaluation using `evaluation/metrics.py` (hooked via `compute_metrics`).
-   Trainer logs/checkpoints go to `--output_dir` (default `/tmp`). The final model is also saved to `CHECKPOINT_DIR` from `config.py`.

### 4. Model smoke test (optional)

Runs a forward/backward and an inference step on a small batch:

```bash
python pdvc.py
```

### Troubleshooting

-   VTT parsing: `webvtt-py` is used if available; otherwise a simple parser runs. Ensure your `.vtt` files are under `VTT_DIR`.
-   Poses: Ensure `POSE_ROOT/<video_id>/*.npy` exists for each listed video id in your split JSON.
-   Trainer outputs: by default `/tmp` (change with `--output_dir`).
