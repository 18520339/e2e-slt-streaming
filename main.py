import gc
import sys
import torch
from functools import partial
from typing import Optional, Tuple
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer, DeformableDetrConfig,
    HfArgumentParser, TrainingArguments, 
    Trainer, EarlyStoppingCallback,
)
from loader import DVCDataset, trainer_collate_fn
from pdvc import DeformableDetrForObjectDetection
from evaluation import preprocess_logits_for_metrics, compute_metrics
from config import *


@dataclass
class ModelArguments:
    # Model Setup
    d_model: int = field(default=512)
    encoder_layers: int = field(default=6)
    decoder_layers: int = field(default=6)
    encoder_attention_heads: int = field(default=8)
    decoder_attention_heads: int = field(default=8)
    encoder_n_points: int = field(default=4)
    decoder_n_points: int = field(default=4)
    num_feature_levels: int = field(default=4, metadata={"help": "The number of input feature levels"})
    num_queries: int = field(default=10, metadata={"help": "Maximum number of events a window can have"})
    num_labels: int = field(default=1, metadata={"help": "Single foreground class for caption"})
    auxiliary_loss: bool = field(default=False, metadata={"help": "The training step may spend a time in per-layer caption alignment and Hungarian matching"})
    class_cost: float = field(default=1.0, metadata={"help": "Relative weight of the classification error"})
    bbox_cost: float = field(default=5.0, metadata={"help": "Relative weight of the L1 error of the bounding box coordinates"})
    giou_cost: float = field(default=2.0, metadata={"help": "Relative weight of the generalized IoU loss of the bounding box"})
    focal_alpha: float = field(default=0.25)
    with_box_refine: bool = field(default=True, metadata={"help": "Learnt (True) or Ground truth proposals (False)"})

    # Caption head / decoder bits
    rnn_num_layers: int = field(default=1)
    cap_dropout_rate: float = field(default=0.1)


@dataclass
class DataArguments:
    # Data Loading
    tokenizer_name: str = field(default='facebook/bart-base')
    use_fast_tokenizer: bool = field(default=True)
    max_caption_len: int = field(default=64)
    train_max_tries: int = field(default=10)
    min_events: int = field(default=1)
    load_by: str = field(default='window')
    seed: int = field(default=2025)
    val_stride_ratio: float = field(default=0.9)

    # Metrics/Ranking
    ranking_temperature: float = field(default=2.0, metadata={"help": "Exponent T in caption score normalization by length^T"})
    alpha: float = field(default=0.3, metadata={"help": "Ranking policy: joint_score = alpha * (caption_score / len(tokens)^T) + (1 - alpha) * det_score"})
    top_k: int = field(default=10, metadata={"help": "Should be num_queries during training"})
    temporal_iou_thresholds: Tuple[float, float, float, float] = field(default=(0.3, 0.5, 0.7, 0.9))


@dataclass
class CustomTrainingArguments(TrainingArguments):
    # Define training arguments for fine-tuning
    output_dir: str = field(default='/tmp', metadata={"help": "Directory for checkpoints and logs"})
    num_train_epochs: float = field(default=20, metadata={"help": "Total number of training epochs"})
    # auto_find_batch_size=True,          # Find batch size that fit memory via exponential decay, avoiding CUDA OOM
    per_device_train_batch_size: int = field(default=16, metadata={"help": "Effective batch size = per_device_train_batch_size x gradient_accumulation_steps x num_devices"})
    per_device_eval_batch_size: int = field(default=16)
    learning_rate: float = field(default=2e-4, metadata={"help": "Initial learning rate"})
    weight_decay: float = field(default=1e-4, metadata={"help": "Regularization"})
    warmup_ratio: float = field(default=0.05)
    lr_scheduler_type: str = field(default='cosine_with_restarts')
    lr_scheduler_kwargs: Optional[dict] = field(default_factory=lambda: dict(num_cycles=1))
    eval_delay: Optional[float] = field(default=0, metadata={"help": "Number of epochs to wait for before the first evaluation can be performed"})
    evaluation_strategy: str = field(default='epoch', metadata={"help": "Evaluate after each epoch"})
    save_strategy: str = field(default='epoch')
    save_total_limit: Optional[int] = field(default=1)
    logging_strategy: str = field(default='epoch')
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Load the best model based on validation loss/Bleu"})
    metric_for_best_model: Optional[str] = field(default='eval_loss', metadata={"help": "Use validation loss/Bleu for early stopping"})
    greater_is_better: Optional[bool] = field(default=False, metadata={"help": "Lower loss / Higher Bleu is better"})
    fp16: bool = field(default_factory=lambda: torch.cuda.is_available(), metadata={"help": "Enable mixed-precision training if a CUDA GPU is available (faster, less memory)"})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "Simulate a larger effective batch size when GPU cannot fit big batches at once"})
    dataloader_num_workers: int = field(default=2, metadata={"help": "Number of subprocesses to use for data loading"})
    save_safetensors: bool = field(default=False, metadata={"help": "Disable safe serialization to avoid the error"})
    report_to: Optional[str] = field(default='none', metadata={"help": "Whether to report to wandb"})
    early_stopping_patience: int = field(default=10)


def main():
    sys.setrecursionlimit(2000)  # Give Soda_c more time for recursion

    # Parse CLI args
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Data Loading
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name, use_fast=data_args.use_fast_tokenizer)
    train_dataset = DVCDataset(
        split='train', max_tries=data_args.train_max_tries, max_caption_len=data_args.max_caption_len,
        min_events=data_args.min_events, load_by=data_args.load_by, tokenizer=tokenizer, seed=data_args.seed
    )
    val_dataset = DVCDataset(
        split='val', stride_ratio=data_args.val_stride_ratio, max_caption_len=data_args.max_caption_len,
        min_events=data_args.min_events, load_by=data_args.load_by, tokenizer=tokenizer, seed=data_args.seed
    )

    # Only log sizes on the main process to avoid clutter in DDP
    if getattr(training_args, 'local_rank', -1) in (-1, 0):
        print(f'Train dataset: {len(train_dataset)} samples')
        print(f'Val dataset: {len(val_dataset)} samples')

    # Model Setup
    config = DeformableDetrConfig(
        d_model=model_args.d_model,
        encoder_layers=model_args.encoder_layers,
        decoder_layers=model_args.decoder_layers,
        encoder_attention_heads=model_args.encoder_attention_heads,
        decoder_attention_heads=model_args.decoder_attention_heads,
        encoder_n_points=model_args.encoder_n_points,
        decoder_n_points=model_args.decoder_n_points,
        num_feature_levels=model_args.num_feature_levels,  # The number of input feature levels
        num_queries=model_args.num_queries,                # Maximum number of events a window can have
        num_labels=model_args.num_labels,                  # Single foreground class for caption
        auxiliary_loss=model_args.auxiliary_loss,          # The training step may spend a time in per-layer caption alignment and Hungarian matching
        # Loss hyper-params in the Hungarian matching cost
        class_cost=model_args.class_cost,                  # Relative weight of the classification error
        bbox_cost=model_args.bbox_cost,                    # Relative weight of the L1 error of the bounding box coordinates
        giou_cost=model_args.giou_cost,                    # Relative weight of the generalized IoU loss of the bounding box
        focal_alpha=model_args.focal_alpha,
        with_box_refine=model_args.with_box_refine,        # Learnt (True) or Ground truth proposals (False)
    )
    model = DeformableDetrForObjectDetection(
        config=config,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        rnn_num_layers=model_args.rnn_num_layers,
        cap_dropout_rate=model_args.cap_dropout_rate,
        max_caption_len=data_args.max_caption_len,
        weight_dict={'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2, 'loss_counter': 0.5, 'loss_caption': 2}
    )  # IMPORTANT: Do not .to(device); Trainer handles device placement and DDP

    total_params = sum(p.numel() for p in model.parameters())
    if getattr(training_args, 'local_rank', -1) in (-1, 0):
        print(f'Model initialized with {total_params / 1e6:.2f}M parameters')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=trainer_collate_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=partial(
            compute_metrics,
            ranking_temperature=data_args.ranking_temperature,  # Exponent T in caption score normalization by length^T
            alpha=data_args.alpha,  # Ranking policy: joint_score = alpha * (caption_score / len(tokens)^T) + (1 - alpha) * det_score
            top_k=data_args.top_k,  # Should be num_queries during training
            temporal_iou_thresholds=data_args.temporal_iou_thresholds,
            tokenizer=tokenizer,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)],
    )

    trainer.train()
    trainer.save_model(CHECKPOINT_DIR)

    # Cleanup to free memory
    del tokenizer, train_dataset, val_dataset, model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()