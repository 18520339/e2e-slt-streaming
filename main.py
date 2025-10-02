import gc
import torch
from functools import partial
from transformers import (
    AutoTokenizer, DeformableDetrConfig,
    TrainingArguments, Trainer,
    EarlyStoppingCallback,
)
from loader import DVCDataset, trainer_collate_fn
from pdvc import DeformableDetrForObjectDetection
from evaluation import preprocess_logits_for_metrics, compute_metrics
from config import *

MAX_CAPTION_LEN = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Loading
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', use_fast=True)
train_dataset = DVCDataset(
    split='train', max_tries=10, max_caption_len=MAX_CAPTION_LEN,
    min_events=1, load_by='window', tokenizer=tokenizer, seed=2025
)
val_dataset = DVCDataset(
    split='val', stride_ratio=0.9, max_caption_len=MAX_CAPTION_LEN,
    min_events=1, load_by='window', tokenizer=tokenizer, seed=2025
)
print(f'Train loader: {len(train_dataset)} samples')
print(f'Val loader: {len(val_dataset)} samples')

# Model Setup
config = DeformableDetrConfig(
    d_model=512,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    encoder_n_points=4,
    decoder_n_points=4,
    num_feature_levels=4, # The number of input feature levels
    num_queries=10,       # Maximum number of events a window can have
    num_labels=1,         # Single foreground class for caption
    auxiliary_loss=False, # The training step may spend a time in per-layer caption alignment and Hungarian matching
    # Loss hyper-params in the Hungarian matching cost
    class_cost=1.0,       # Relative weight of the classification error
    bbox_cost=5.0,        # Relative weight of the L1 error of the bounding box coordinates
    giou_cost=2.0,        # Relative weight of the generalized IoU loss of the bounding box
    focal_alpha=0.25,
    with_box_refine=True, # Learnt (True) or Ground truth proposals (False) 
)

model = DeformableDetrForObjectDetection(
    config=config,
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    rnn_num_layers=1,
    cap_dropout_rate=0.1,
    max_caption_len=MAX_CAPTION_LEN,
    weight_dict={'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2, 'loss_counter': 0.5, 'loss_caption': 2}
).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'Model initialized with {total_params / 1e6:.2f}M parameters')

# Define training arguments for fine-tuning
training_args = TrainingArguments(      # Find out more at https://huggingface.co/docs/transformers/en/main_classes/trainer
    output_dir='/tmp',                  # Directory for checkpoints and logs
    num_train_epochs=20,                # Total number of training epochs
    auto_find_batch_size=True,          # Find a batch size that will fit into memory automatically through exponential decay, avoiding CUDA OOM
    learning_rate=5e-5,                 # Initial learning rate
    weight_decay=1e-4,                  # Regularization
    warmup_ratio=0.05,
    lr_scheduler_type='cosine_with_restarts',
    lr_scheduler_kwargs=dict(num_cycles=1),
    eval_strategy='epoch',              # Evaluate after each epoch
    save_strategy='epoch',
    save_total_limit=1,
    logging_strategy='epoch',           #
    load_best_model_at_end=True,        # Load the best model based on validation loss/map
    metric_for_best_model='eval_loss',  # Use validation loss/map for early stopping
    greater_is_better=False,            # Lower loss / Higher map is better
    fp16=torch.cuda.is_available(),     # Enable mixed-precision training if a CUDA GPU is available (faster, less memory)
    gradient_accumulation_steps=2,      # Updates steps to accumulate gradients for, before performing backward pass
    dataloader_num_workers=2,           # Number of subprocesses to use for data loading
    # remove_unused_columns=False,        # Whether to automatically remove columns unused by the model forward method
    save_safetensors=False,             # Disable safe serialization to avoid the error
    report_to='none',                   # Whether to report to wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=trainer_collate_fn,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=partial(
        compute_metrics,
        ranking_temperature=2.0, # Exponent T in caption score normalization by length^T
        alpha=0.3, # Ranking policy: joint_score = alpha * (caption_score / len(tokens)^T) + (1 - alpha) * det_score
        top_k=10,  # Should be num_queries during training
        temporal_iou_thresholds=(0.3, 0.5, 0.7, 0.9),
        tokenizer=tokenizer,
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)
trainer.train()
trainer.save_model(CHECKPOINT_DIR)