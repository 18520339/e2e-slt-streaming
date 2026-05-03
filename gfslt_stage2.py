# Stage 2: Gloss-Free Sign Language Translation using HuggingFace Trainer API.
# This stage performs end-to-end translation from pose sequences to text, using weights from Stage 1 (VLP) to initialize the encoder.
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import Trainer, TrainingArguments, AutoTokenizer, HfArgumentParser

import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TGT_LANG, TRIMMED_TOKENIZER_DIR, TRIMMED_MBART_DIR
from loader import DVCDataset, trainer_collate_fn
from evaluation.helpers import compute_text_metrics
from gfslt_models import GFSLTConfig, GFSLT, Wrapper4Trainer

def is_bfloat16_supported(): # Checks if the current device supports bfloat16
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

# ======================== Arguments ========================
@dataclass
class ModelArguments:
    embed_dim: int = field(default=1024, metadata={'help': 'Embedding dimension'})
    hidden_size: int = field(default=1024, metadata={'help': 'Hidden size'})
    temporal_kernel: int = field(default=3, metadata={'help': 'Temporal kernel size for CoSign'})
    mbart_name: str = field(default_factory=lambda: f'./{TRIMMED_MBART_DIR}', metadata={'help': 'MBart model name'})
    tokenizer_name: str = field(default_factory=lambda: f'./{TRIMMED_TOKENIZER_DIR}', metadata={'help': 'Tokenizer name'})
    label_smoothing: float = field(default=0.2, metadata={'help': 'Label smoothing'})
    stage1_checkpoint: Optional[str] = field(
        default=None, 
        metadata={'help': 'Path to Stage 1 checkpoint to initialize encoder weights'}
    )
    freeze_encoder: bool = field(default=False, metadata={'help': 'Whether to freeze encoder weights'})


@dataclass
class DataArguments:
    max_tries: int = field(default=20, metadata={'help': 'Maximum attempts to find a valid window with at least one event'})
    noise_rate: float = field(default=0.15, metadata={'help': 'Proportion of words to mask for noise injection during non-streaming training'})
    pose_augment: bool = field(default=False, metadata={'help': 'Apply pose augmentation during training'})
    stride_ratio: float = field(default=0.9, metadata={'help': 'Stride ratio for window sampling during validation/testing'})
    min_events: int = field(default=1, metadata={'help': 'Minimum number of events in a window'})
    max_events: int = field(default=10, metadata={'help': 'Maximum number of events in a window'})
    max_event_tokens: int = field(default=40, metadata={'help': 'Maximum number of tokens per event/caption'})
    max_window_tokens: int = field(default=128, metadata={'help': 'Maximum number of tokens in a window for non-streaming input'})
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
    

@dataclass
class GenerationArguments:
    max_new_tokens: int = field(default=128, metadata={'help': 'Max tokens to generate'})
    num_beams: int = field(default=1, metadata={'help': 'Beam search width'})
    length_penalty: float = field(default=1.0, metadata={'help': 'Length penalty for beam search'})


# ======================== Custom Trainer for Stage 2 Translation Training ========================
class Stage2Trainer(Trainer):
    def __init__(self, trimmed_tokenizer, generation_args: GenerationArguments = None, **kwargs):
        super().__init__(**kwargs)
        self.trimmed_tokenizer = trimmed_tokenizer
        self.generation_args = generation_args or GenerationArguments()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        pixel_values = inputs['pixel_values'].to(device)
        pixel_mask = inputs['pixel_mask'].to(device)
        labels = inputs['labels']
        
        # Extract tokens from labels
        pad_token_id = self.trimmed_tokenizer.pad_token_id
        paragraph_tokens = torch.stack([l['paragraph_tokens'] for l in labels]).to(device)
        paragraph_attention_mask = (paragraph_tokens != pad_token_id).long()
        
        # Forward pass
        outputs = model.base_module(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            paragraph_tokens=paragraph_tokens,
            paragraph_attention_mask=paragraph_attention_mask,
        )
        loss = outputs['loss']
        if return_outputs: return loss, outputs
        return loss
    
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = 'test'): # Evaluate with BLEU score computation
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix) # Run standard evaluation
        if eval_dataset is not None:
            self.model.eval()
            device = next(self.model.parameters()).device
            predictions, references = [], []
            
            eval_dataloader = self.get_eval_dataloader(eval_dataset) # Create a dataloader for evaluation
            with torch.no_grad():
                for batch in eval_dataloader:
                    pixel_values = batch['pixel_values'].to(device)
                    pixel_mask = batch['pixel_mask'].to(device)
                    labels = batch['labels']
                    
                    generated_ids = self.model.base_module.generate(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        max_new_tokens=self.generation_args.max_new_tokens,
                        num_beams=self.generation_args.num_beams,
                        decoder_start_token_id=self.trimmed_tokenizer.lang_code_to_id.get(TGT_LANG, self.trimmed_tokenizer.bos_token_id),
                    )
                    pred_texts = self.trimmed_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    predictions.extend(pred_texts)
                    
                    for l in labels: # Decode references
                        ref_text = self.trimmed_tokenizer.decode(l['paragraph_tokens'], skip_special_tokens=True)
                        references.append(ref_text)
            
            text_metrics = compute_text_metrics(predictions, references)
            for k, v in text_metrics.items():
                metrics[f'{metric_key_prefix}_para_{k}'] = v
        return metrics
    
    
# ======================== Weight Loading Utilities ========================
def load_stage1_weights(model: GFSLT, checkpoint_path: str) -> GFSLT:
    ''' Load encoder weights from Stage 1 checkpoint.
    
    Args:
        model: GFSLT model
        checkpoint_path: Path to Stage 1 checkpoint
        
    Returns:
        Model with loaded weights
    '''
    print(f'Loading Stage 1 weights from {checkpoint_path}...')
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in state_dict: state_dict = state_dict['model']
    elif 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items(): # Map Stage 1 encoder weights to Stage 2
        if 'base_module.model_images.backbone' in k: # The ImageCLIP backbone maps to GFSLT backbone
            new_k = k.replace('base_module.model_images.backbone', 'backbone')
            new_state_dict[new_k] = v
        elif 'base_module.model_images.trans_encoder' in k: # Map to MBart encoder
            new_k = k.replace('base_module.model_images.trans_encoder', 'mbart.model.encoder')
            new_state_dict[new_k] = v
    
    # Load MBart decoder weights from pretrained
    mbart_state = load_file(
        f'{model.config.mbart_name}/model.safetensors'
    ) if Path(f'{model.config.mbart_name}/model.safetensors').exists() else {}
    
    for k, v in mbart_state.items():
        if 'decoder' in k and k not in new_state_dict:
            new_state_dict['mbart.' + k] = v
    
    # Load with strict=False to allow missing/unexpected keys
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f'Loaded weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}')
    if missing: print(f'Sample missing keys: {missing[:5]}')
    if unexpected: print(f'Sample unexpected keys: {unexpected[:5]}')
    return model


# ======================== Main Training Function ========================
def train_stage2(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments, generation_args: GenerationArguments):
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, src_lang=TGT_LANG, tgt_lang=TGT_LANG)
    train_dataset = DVCDataset(
        split='train', tokenizer=tokenizer, max_tries=data_args.max_tries, noise_rate=data_args.noise_rate, pose_augment=data_args.pose_augment, 
        min_events=data_args.min_events, max_events=data_args.max_events, max_window_tokens=data_args.max_window_tokens, 
        max_event_tokens=data_args.max_event_tokens, load_by=data_args.load_by, seed=training_args.seed
    )
    test_dataset = DVCDataset(
        split='test', tokenizer=tokenizer, pose_augment=False, stride_ratio=data_args.stride_ratio, 
        min_events=data_args.min_events, max_events=data_args.max_events, max_event_tokens=data_args.max_event_tokens, 
        max_window_tokens=data_args.max_window_tokens, load_by=data_args.load_by, seed=training_args.seed
    )
    if getattr(training_args, 'local_rank', -1) in (-1, 0): # Only log sizes on the main process to avoid clutter in DDP
        print(f'Train dataset: {len(train_dataset)} samples')
        print(f'Test dataset: {len(test_dataset)} samples')
        
    # Model Setup
    config = GFSLTConfig(
        embed_dim=model_args.embed_dim,
        hidden_size=model_args.hidden_size,
        temporal_kernel=model_args.temporal_kernel,
        mbart_name=model_args.mbart_name,
        label_smoothing=model_args.label_smoothing,
    )
    gfslt = GFSLT(config)
    
    # Load Stage 1 weights if provided
    if model_args.stage1_checkpoint:
        gfslt = load_stage1_weights(gfslt, model_args.stage1_checkpoint)
    
    # Freeze encoder if specified
    if model_args.freeze_encoder:
        print('Freezing encoder weights...')
        for param in gfslt.backbone.parameters():
            param.requires_grad = False
        # Also freeze MBart encoder
        for param in gfslt.mbart.model.encoder.parameters():
            param.requires_grad = False
    
    model = Wrapper4Trainer(gfslt, tokenizer, stage=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'Trainable parameters: {n_params / 1e6:.2f}M / Total: {n_total / 1e6:.2f}M')
    
    # Initialize trainer
    trainer = Stage2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=trainer_collate_fn,
        trimmed_tokenizer=tokenizer,
        generation_args=generation_args,
    )
    
    # Training
    if training_args.do_train:
        print('Starting training...')
        metrics = trainer.train().metrics
        trainer.save_model()
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
    
    # Evaluation
    if training_args.do_eval:
        print('Evaluating...')
        model.load_state_dict(torch.load(os.path.join(training_args.output_dir, 'pytorch_model.bin')))
        metrics = trainer.evaluate(test_dataset)
        trainer.log_metrics('test', metrics)
        trainer.save_metrics('test', metrics)
    return trainer


# ======================== Entry Point ========================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments, GenerationArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'): # Parse from config file
        model_args, data_args, training_args, generation_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        model_args, data_args, training_args, generation_args = parser.parse_args_into_dataclasses()
    
    # Set defaults for training args if not specified
    if not hasattr(training_args, 'output_dir') or not training_args.output_dir:
        training_args.output_dir = './outputs/stage2'
    train_stage2(model_args, data_args, training_args, generation_args)


if __name__ == '__main__':
    main()