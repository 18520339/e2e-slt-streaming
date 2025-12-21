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
torchrun --nproc_per_node 6 main.py \
	--output_dir "./outputs/run1" \
	--num_train_epochs 100 \
	--per_device_train_batch_size 64 \
	--per_device_eval_batch_size 128 \
	--weight_decay 1e-4 \
	--learning_rate 5e-4
```

Multi‑GPU with accelerate:

```bash
accelerate launch --num_processes 6 main.py \
	--output_dir "./outputs/run1" \
	--num_train_epochs 100 \
	--per_device_train_batch_size 64 \
	--per_device_eval_batch_size 128 \
	--weight_decay 1e-4 \
	--learning_rate 5e-4
```

What it does:

-   Builds train/val datasets from BOBSL poses and VTTs.
-   Initializes a Deformable DETR-based model with a captioning head.
-   Trains with Hugging Face Trainer and saves the final model to the `CHECKPOINT_DIR` defined in `config.py`.
-   **Note:** Training no longer includes evaluation. After training completes, use `eval.py` to evaluate the model (see below).

Common training flags (subset shown):

-   Data: `--max_event_tokens 64`, `--stride_ratio 0.9`, `--noise_rate 0.15`, `--pose_augment`
-   Model: `--d_model 1024`, `--num_queries 100`, `--encoder_layers 2`, `--decoder_layers 2`, `--num_cap_layers 3`
-   Trainer: `--num_train_epochs 200`, `--per_device_train_batch_size 64`, `--output_dir ./outputs/run1`, `--learning_rate 5e-4`, `--weight_decay 1e-4`

Tips:

-   To change window length or FPS, edit `config.py`. To change caption length, pass `--max_event_tokens`.
-   Trainer logs/checkpoints go to `--output_dir` (default `/tmp`). The final model is also saved to `CHECKPOINT_DIR` from `config.py`.

### 4. Evaluate (Single GPU Only)

After training, use `eval.py` to evaluate the trained model on validation and/or test sets. **This script is designed to run on a single GPU only** to avoid distributed training overhead during evaluation.

#### Running on a Single GPU (Important!)

**On multi-GPU machines**, the Trainer API will automatically detect all available GPUs and attempt distributed training by default. To explicitly run evaluation on a single GPU, use `CUDA_VISIBLE_DEVICES`:

```bash
# Force evaluation to use only GPU 0
CUDA_VISIBLE_DEVICES=0 python eval.py --checkpoint_path "./outputs/run1"

# Or use GPU 2
CUDA_VISIBLE_DEVICES=2 python eval.py
```

**Without setting `CUDA_VISIBLE_DEVICES`**, if your machine has multiple GPUs, the Trainer will initialize distributed training which adds unnecessary overhead for evaluation.

#### Basic Usage

```bash
# Evaluate on both val and test sets (uses CHECKPOINT_DIR from config.py by default)
CUDA_VISIBLE_DEVICES=0 python eval.py

# Specify checkpoint path explicitly
CUDA_VISIBLE_DEVICES=0 python eval.py --checkpoint_path "./outputs/run1"

# Evaluate only validation set
CUDA_VISIBLE_DEVICES=0 python eval.py --eval_test False

# Evaluate only test set
CUDA_VISIBLE_DEVICES=0 python eval.py --eval_val False

# Customize batch size and workers
CUDA_VISIBLE_DEVICES=0 python eval.py \
	--per_device_eval_batch_size 64 \
	--dataloader_num_workers 8
```

#### Evaluation Flags

-   **Checkpoint & Output:**
    -   `--checkpoint_path`: Path to trained model checkpoint (default: `CHECKPOINT_DIR` from `config.py`)
    -   `--output_dir`: Directory for evaluation outputs (default: `/tmp/eval`)
-   **Data & Metrics:** Same as training (e.g., `--max_event_tokens`, `--stride_ratio`, `--alpha`, `--ranking_temperature`, `--top_k`, `--temporal_iou_thresholds`)
-   **Hardware:**
    -   `--per_device_eval_batch_size`: Batch size for evaluation (default: 128)
    -   `--dataloader_num_workers`: Data loading workers (default: 4)
    -   `--fp16` / `--bf16`: Mixed precision evaluation
-   **Dataset Selection:**
    -   `--eval_val`: Evaluate on validation set (default: True)
    -   `--eval_test`: Evaluate on test set (default: True)

#### What It Does

-   Loads the trained checkpoint
-   Evaluates on validation and/or test sets with computed metrics (BLEU, METEOR, CIDEr, SODA_c, etc.)
-   Prints detailed metrics for each dataset
-   Uses optimized single-GPU inference without distributed training overhead

### 5. Model smoke test (optional)

Runs a forward/backward and an inference step on a small batch:

```bash
python pdvc.py
```

### Troubleshooting

-   VTT parsing: `webvtt-py` is used if available; otherwise a simple parser runs. Ensure your `.vtt` files are under `VTT_DIR`.
-   Poses: Ensure `POSE_ROOT/<video_id>/*.npy` exists for each listed video id in your split JSON.
-   Trainer outputs: by default `/tmp` (change with `--output_dir`).
