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

#### Mode 1: Train localization (backbone + encoder + decoder + detection heads)

-   Freeze: caption_head
-   Train: backbone (transformer.backbone), encoder, decoder, class_head, bbox_head, count_head
-   Use localization losses only (loss_ce, loss_bbox, loss_giou, loss_counter)

```bash
torchrun --nproc_per_node 6 main.py \
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

#### Mode 2: Train captioning (load mode 1 checkpoint)

-   Freeze: backbone, encoder, decoder, class_head, bbox_head, count_head
-   Train: caption_head only
-   Use loss_caption only
-   Use GT boxes for curriculum caption learning (use_gt_boxes_for_caption=True).

```bash
torchrun --nproc_per_node 6 main.py \
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
	--learning_rate 5e-4 \
	--per_device_train_batch_size 32
```

#### Mode 3: Joint fine-tuning (Default mode)

-   Unfreeze everything
-   Train all parameters (backbone, encoder, decoder, all heads)
-   Use all losses with balanced weights
-   Can optionally load mode 2 checkpoint

```bash
torchrun --nproc_per_node 6 main.py \
  --output_dir ./checkpoints/mode3 \
  --max_event_tokens 40 \
  --d_model 1024 \
  --encoder_layers 2 \
  --decoder_layers 2 \
  --num_cap_layers 3 \
  --num_queries 30 \
  --num_train_epochs 100 \
  --learning_rate 5e-4 \
  --per_device_train_batch_size 32
```

#### What it does

-   Builds train/val datasets from BOBSL poses and VTTs.
-   Initializes a Deformable DETR-based model with a captioning head.
-   Trains with Hugging Face Trainer and saves the final model to the `CHECKPOINT_DIR` defined in `config.py`.
-   **Note:** Training no longer includes evaluation. After training completes, use `eval.py` to evaluate the model (see below).

Common training flags (subset shown):

-   Data: `--max_event_tokens 50`, `--stride_ratio 0.9`, `--noise_rate 0.15`, `--pose_augment`
-   Model: `--d_model 1024`, `--num_queries 30`, `--encoder_layers 2`, `--decoder_layers 2`, `--num_cap_layers 3`
-   Trainer: `--num_train_epochs 100`, `--per_device_train_batch_size 32`, `--output_dir ./checkpoints`, `--learning_rate 5e-4`, `--weight_decay 1e-4`

Tips:

-   To change window length or FPS, edit `config.py`. To change caption length, pass `--max_event_tokens`.
-   Trainer logs/checkpoints go to `--output_dir` (default `/tmp`). The final model is also saved to `CHECKPOINT_DIR` from `config.py`.

### 4. Evaluate (Single GPU Only)

After training, use `eval.py` to evaluate the trained model on validation and/or test sets. **This script is designed to run on a single GPU only** to avoid distributed training overhead during evaluation.

#### Running on a Single GPU (Important!)

**On multi-GPU machines**, the Trainer API will automatically detect all available GPUs and attempt distributed training by default. To explicitly run evaluation on a single GPU, use `CUDA_VISIBLE_DEVICES`:

```bash
# Evaluate on both val and test sets (uses CHECKPOINT_DIR from config.py by default)
CUDA_VISIBLE_DEVICES=0 python eval.py

# Evaluate only validation set
CUDA_VISIBLE_DEVICES=0 python eval.py --eval_test False

# Evaluate only test set
CUDA_VISIBLE_DEVICES=0 python eval.py --eval_val False

# More customized evaluation
CUDA_VISIBLE_DEVICES=0 python eval.py \
	--checkpoint_path checkpoints/mode3/mode3_final \
	--eval_val False \
	--max_event_tokens 40 \
	--encoder_layers 2 \
	--decoder_layers 2 \
	--num_cap_layers 3 \
	--num_queries 30 \
	--per_device_eval_batch_size 32 \
	--ranking_temperature 2.0 \
	--top_k 20
```

**Without setting `CUDA_VISIBLE_DEVICES`**, if your machine has multiple GPUs, the Trainer will initialize distributed training which adds unnecessary overhead for evaluation.

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
