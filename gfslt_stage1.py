# Stage 1: Visual-Language Pre-training (VLP) using HuggingFace Trainer API.
# This stage performs CLIP-style contrastive learning between pose sequences and text.
# along with masked language modeling for better text understanding.
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from transformers import Trainer, TrainingArguments, AutoTokenizer, HfArgumentParser

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gfslt_models import GFSLTConfig, SLRCLIP, TextDecoder, Wrapper4Trainer
from loader import DVCDataset, trainer_collate_fn

def is_bfloat16_supported(): # Checks if the current device supports bfloat16
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


# ======================== Arguments ========================
@dataclass
class ModelArguments:
    embed_dim: int = field(default=1024, metadata={'help': 'Embedding dimension'})
    hidden_size: int = field(default=1024, metadata={'help': 'Hidden size'})
    temporal_kernel: int = field(default=3, metadata={'help': 'Temporal kernel size for CoSign'})
    mbart_name: str = field(default='./captioners/trimmed_mbart', metadata={'help': 'MBart model name'})
    label_smoothing: float = field(default=0.2, metadata={'help': 'Label smoothing'})
    use_text_decoder: bool = field(default=True, metadata={'help': 'Whether to use text decoder for MLM'})
    mlm_loss_weight: float = field(default=1.0, metadata={'help': 'Weight for masked LM loss'})


@dataclass
class DataArguments:
    max_tries: int = field(default=20, metadata={'help': 'Maximum attempts to find a valid window with at least one event'})
    noise_rate: float = field(default=0.15, metadata={'help': 'Proportion of words to mask for noise injection during non-streaming training'})
    pose_augment: bool = field(default=False, metadata={'help': 'Apply pose augmentation during training'})
    stride_ratio: float = field(default=0.9, metadata={'help': 'Stride ratio for window sampling during validation/testing'})
    min_events: int = field(default=1, metadata={'help': 'Minimum number of events in a window'})
    max_events: int = field(default=10, metadata={'help': 'Maximum number of events in a window'})
    max_event_tokens: int = field(default=40, metadata={'help': 'Maximum number of tokens per event/caption'})
    max_window_tokens: int = field(default=256, metadata={'help': 'Maximum number of tokens in a window for non-streaming input'})
    load_by: str = field(default='window', metadata={'help': "Load data by 'window' or by 'video'"})
    
    
@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default='/tmp', metadata={'help': 'Directory for checkpoints and logs'})
    num_train_epochs: float = field(default=50, metadata={'help': 'Total number of training epochs'})
    save_safetensors: bool = field(default=False, metadata={'help': 'Disable safe serialization to avoid the error'})
    
    # Data processing
    # auto_find_batch_size=True, # Find batch size that fit memory via exponential decay, avoiding CUDA OOM
    per_device_train_batch_size: int = field(default=32, metadata={'help': 'Effective batch size = per_device_train_batch_size x gradient_accumulation_steps x num_devices'})
    per_device_eval_batch_size: int = field(default=32, metadata={'help': 'Can be higher if greedy but should be smaller if using beam search'})
    dataloader_num_workers: int = field(default=4, metadata={'help': 'Number of subprocesses to use for data loading'})

    # Precision & optimization
    optim: str = field(default='adamw_torch_fused', metadata={'help': 'Choose optimizer'})
    weight_decay: float = field(default=1e-4, metadata={'help': 'Low since random windows already provide regularization'})
    fp16: bool = field(default=not is_bfloat16_supported(), metadata={'help': 'Use mixed precision training if supported'})
    bf16: bool = field(default=is_bfloat16_supported(), metadata={'help': 'Use bfloat16 (if supported) instead of fp16 for mixed precision training'})
    learning_rate: float = field(default=5e-4, metadata={'help': 'Linear decay learning rate'})
    lr_scheduler_type: str = field(default='cosine', metadata={'help': 'Learning rate scheduler type'})
    ddp_find_unused_parameters: bool = field(default=False, metadata={'help': 'Avoid DDP overhead if all parameters are used'})
    max_grad_norm: float = field(default=1.0, metadata={'help': 'Gradient clipping to avoid exploding gradients'})
    
    # Reporting and saving
    report_to: Optional[str] = field(default='none', metadata={'help': 'Whether to report to wandb/tensorboard/none'})
    logging_strategy: str = field(default='epoch')
    save_strategy: str = field(default='epoch')
    save_total_limit: Optional[int] = field(default=1)
    

# ======================== Custom Trainer for Stage 1 VLP Training ========================
class Stage1Trainer(Trainer): # Handles CLIP-style contrastive loss + optional masked LM loss.
    def __init__(self, text_decoder: Optional[nn.Module] = None, mlm_loss_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.text_decoder = text_decoder
        self.mlm_loss_weight = mlm_loss_weight
        if text_decoder is not None:
            self.text_decoder = text_decoder.to(self.args.device)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        pixel_values = inputs['pixel_values'].to(device)
        pixel_mask = inputs['pixel_mask'].to(device)
        labels = inputs['labels']
        
        # Extract tokens from labels
        pad_token_id = model.tokenizer.pad_token_id
        paragraph_tokens = torch.stack([l['paragraph_tokens'] for l in labels]).to(device)
        paragraph_attention_mask = (paragraph_tokens != pad_token_id).long()
        
        # Forward pass for contrastive loss
        outputs = model.base_module(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            paragraph_tokens=paragraph_tokens,
            paragraph_attention_mask=paragraph_attention_mask,
        )
        total_loss = outputs['loss']
        
        # Add masked LM loss if text decoder is available
        if self.text_decoder is not None and model.training:
            masked_paragraph_tokens = torch.stack([l['masked_paragraph_tokens'] for l in labels]).to(device)
            masked_paragraph_attention_mask = (masked_paragraph_tokens != pad_token_id).long()
            with torch.no_grad(): # Get encoder hidden states from text encoder
                _, encoder_hidden_states = model.base_module.model_txt(masked_paragraph_tokens, masked_paragraph_attention_mask)
            
            lm_logits = self.text_decoder(
                input_ids=paragraph_tokens,
                attention_mask=paragraph_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=masked_paragraph_attention_mask,
            )
            mlm_loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                paragraph_tokens.view(-1),
                ignore_index=pad_token_id,
                label_smoothing=0.2
            )
            total_loss += self.mlm_loss_weight * mlm_loss
            outputs['mlm_loss'] = mlm_loss
        
        outputs['total_loss'] = total_loss
        if return_outputs: return total_loss, outputs
        return total_loss


# ======================== Main Training Function ========================
def train_stage1(model_args: ModelArguments, data_args: DataArguments, training_args: CustomTrainingArguments,):
    tokenizer = AutoTokenizer.from_pretrained(model_args.mbart_name, src_lang='en_XX', tgt_lang='en_XX')
    train_dataset = DVCDataset(
        split='train', tokenizer=tokenizer, max_tries=data_args.max_tries, noise_rate=data_args.noise_rate, pose_augment=data_args.pose_augment, 
        min_events=data_args.min_events, max_events=data_args.max_events, max_window_tokens=data_args.max_window_tokens, 
        max_event_tokens=data_args.max_event_tokens, load_by=data_args.load_by, seed=training_args.seed
    )
    if getattr(training_args, 'local_rank', -1) in (-1, 0): # Only log sizes on the main process to avoid clutter in DDP
        print(f'Train dataset: {len(train_dataset)} samples')
    
    # Model Setup
    config = GFSLTConfig(
        embed_dim=model_args.embed_dim,
        hidden_size=model_args.hidden_size,
        temporal_kernel=model_args.temporal_kernel,
        mbart_name=model_args.mbart_name,
        label_smoothing=model_args.label_smoothing,
    )
    slrclip = SLRCLIP(config)
    model = Wrapper4Trainer(slrclip, tokenizer, stage=1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {n_params / 1e6:.2f}M')
    
    # Initialize trainer
    trainer = Stage1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=trainer_collate_fn,
        text_decoder=TextDecoder(config) if model_args.use_text_decoder else None,
        mlm_loss_weight=model_args.mlm_loss_weight,
    )
    trainer.train()
    trainer.save_model()


# ======================== Entry Point ========================
if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'): # Parse from config file
        model_args, data_args, training_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set defaults for training args if not specified
    if not hasattr(training_args, 'output_dir') or not training_args.output_dir:
        training_args.output_dir = './outputs/stage1'
    train_stage1(model_args, data_args, training_args)