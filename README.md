# E2E-SLT-Streaming

### 1. Environment

```bash
git clone https://github.com/18520339/e2e-slt-streaming.git
cd e2e-slt-streaming
pip install -r requirements.txt
```

### 2. Data paths

This repo expects BOBSL data laid out under `dataset/BOBSL` (already the default). If your dataset lives elsewhere, update these in `config.py`:

-   `DATA_ROOT` (base dataset dir)
-   `SUBSET_JSON` (train/val/test split JSON)
-   `VTT_DIR` (aligned subtitles in .vtt)
-   `POSE_ROOT` (pose `.npy` folders per video id)

### 3. Train

```bash
python main.py
```

What it does:

-   Builds train/val datasets from BOBSL poses and VTTs.
-   Initializes a Deformable DETR-based model with a captioning head.
-   Trains with Hugging Face Trainer and saves the final model to `checkpoints/`.

Tips:

-   To change window length, FPS, or caption length, edit `config.py` and `MAX_CAPTION_LEN` in `main.py`.
-   Metrics are available in `evaluation/metrics.py` (the hook in `main.py` is prepared but commented).
-   You can have a look at [Hugging Face Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer) docs for more training options.

### 4. Model smoke test (optional)

Runs a forward/backward and an inference step on a small batch:

```bash
python pdvc.py
```

### Troubleshooting

-   VTT parsing: `webvtt-py` is used if available; otherwise a simple parser runs. Ensure your `.vtt` files are under `VTT_DIR`.
-   Poses: Ensure `POSE_ROOT/<video_id>/*.npy` exists for each listed video id in your split JSON.
-   Temporary Trainer outputs: by default `/tmp` (in `main.py`), which you can change via `TrainingArguments(output_dir=...)`.
