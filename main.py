'''Multi-stage Training for Dense Video Captioning

Stage 1: Train localization (backbone + encoder + decoder + detection heads)
- Freeze: caption_head
- Train: backbone (transformer.backbone), encoder, decoder, class_head, bbox_head, count_head
- Use localization losses only (loss_ce, loss_bbox, loss_giou, loss_counter)
> python main.py --stage 1 --num_train_epochs 50 --output_dir checkpoints/stage1

Stage 2: Train captioning (load stage 1 checkpoint)
- Freeze: backbone, encoder, decoder, class_head, bbox_head, count_head 
- Train: caption_head only
- Use loss_caption only
- Optional: Use GT boxes for curriculum caption learning (use_gt_boxes_for_caption=True).
> python main.py --stage 2 --num_train_epochs 100 --stage1_checkpoint checkpoints/stage1/stage1_final --output_dir checkpoints/stage2

Stage 3: Optional joint fine-tuning (load stage 2 checkpoint)
- Unfreeze everything
- Train all parameters (backbone, encoder, decoder, all heads)
- Use all losses with balanced weights
- Lower learning rate for stability
> python main.py --stage 3 --num_train_epochs 30 --stage2_checkpoint checkpoints/stage2/stage2_final --output_dir checkpoints/stage3
'''
import gc
import os
import torch
from typing import Optional
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer, DeformableDetrConfig,
    HfArgumentParser, TrainingArguments, 
    Trainer, EarlyStoppingCallback,
)
from loader import DVCDataset, trainer_collate_fn
from pdvc import DeformableDetrForObjectDetection
from captioners import MBartDecoderCaptioner
from config import *


def is_bfloat16_supported(): # Checks if the current device supports bfloat16
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


@dataclass
class ModelArguments:
    d_model: int = field(default=1024)
    encoder_layers: int = field(default=2)
    decoder_layers: int = field(default=2)
    encoder_attention_heads: int = field(default=8)
    decoder_attention_heads: int = field(default=8)
    encoder_n_points: int = field(default=4)
    decoder_n_points: int = field(default=4)
    num_feature_levels: int = field(default=4, metadata={'help': 'The number of input feature levels'})
    num_queries: int = field(default=100, metadata={'help': 'Maximum number of events a window can have'})
    num_labels: int = field(default=1, metadata={'help': 'Single foreground class for caption'})
    auxiliary_loss: bool = field(default=True, metadata={'help': 'The training step may spend a time in per-layer caption alignment and Hungarian matching'})
    class_cost: float = field(default=2, metadata={'help': 'Relative weight of the classification error'})
    bbox_cost: float = field(default=0, metadata={'help': 'Relative weight of the L1 error of the bounding box coordinates'})
    giou_cost: float = field(default=4, metadata={'help': 'Relative weight of the generalized IoU loss of the bounding box'})
    counter_cost: float = field(default=2, metadata={'help': 'Relative weight of the event counter loss'})
    caption_cost: float = field(default=2, metadata={'help': 'Relative weight of the captioning loss'})
    focal_alpha: float = field(default=0.25)
    with_box_refine: bool = field(default=True, metadata={'help': 'Learnt (True) or Ground truth proposals (False, all losses except caption loss will be disabled)'})

    # Caption head parameters
    num_cap_layers: int = field(default=3)
    cap_dropout_rate: float = field(default=0.1)
    use_gt_boxes_for_caption: bool = field(default=False, metadata={'help': 'Use ground-truth boxes as reference points for caption generation'})


@dataclass
class DataArguments:
    max_tries: int = field(default=20, metadata={'help': 'Maximum attempts to find a valid window with at least one event'})
    noise_rate: float = field(default=0.15, metadata={'help': 'Proportion of words to mask for noise injection during non-streaming training'})
    pose_augment: bool = field(default=False, metadata={'help': 'Apply pose augmentation during training'})
    stride_ratio: float = field(default=0.9, metadata={'help': 'Stride ratio for window sampling during validation/testing'})
    min_events: int = field(default=1, metadata={'help': 'Minimum number of events in a window'})
    max_events: int = field(default=10, metadata={'help': 'Maximum number of events in a window'})
    max_event_tokens: int = field(default=50, metadata={'help': 'Maximum number of tokens per event/caption'})
    max_window_tokens: int = field(default=128, metadata={'help': 'Maximum number of tokens in a window for non-streaming input'})
    load_by: str = field(default='window', metadata={'help': "Load data by 'window' or by 'video'"})


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default='/tmp', metadata={'help': 'Directory for checkpoints and logs'})
    num_train_epochs: float = field(default=100, metadata={'help': 'Total number of training epochs'})
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
    # early_stopping_patience: int = field(default=10, metadata={'help': 'Early stopping patience by validation loss or Bleu'})
    ddp_find_unused_parameters: bool = field(default=False, metadata={'help': 'Avoid DDP overhead if all parameters are used'})
    max_grad_norm: float = field(default=1.0, metadata={'help': 'Gradient clipping to avoid exploding gradients'})
    
    # Reporting
    report_to: Optional[str] = field(default='none', metadata={'help': 'Whether to report to wandb/tensorboard/none'})
    logging_strategy: str = field(default='epoch')
    # eval_strategy: str = field(default='epoch', metadata={'help': 'Evaluate after each epoch'})
    
    # Saving
    save_strategy: str = field(default='epoch')
    save_total_limit: Optional[int] = field(default=1)
    # metric_for_best_model: Optional[str] = field(default='eval_loss', metadata={'help': 'Use validation loss/Bleu for early stopping'})
    # greater_is_better: Optional[bool] = field(default=False, metadata={'help': 'Lower loss / Higher Bleu is better'})
    # load_best_model_at_end: bool = field(default=True, metadata={'help': 'Load the best model based on validation loss/Bleu'})

    # Two-stage specific arguments
    stage: int = field(default=1, metadata={'help': 'Training stage: 1 for localization, 2 for captioning, 3 for joint fine-tuning'})
    stage1_checkpoint: Optional[str] = field(default=None, metadata={'help': 'Path to stage 1 checkpoint for stage 2 training'})
    stage2_checkpoint: Optional[str] = field(default=None, metadata={'help': 'Path to stage 2 checkpoint for stage 3 fine-tuning'})


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True
        
def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable parameters: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.1f}%)')
    
def handle_key_mismatches(state_dict, model_state):
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape: filtered_state[k] = v
            else: print(f'=> Skipping {k}: shape mismatch {v.shape} vs {model_state[k].shape}')
        else: print(f'=> Skipping {k}: not in model')
    return filtered_state
    

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Validate stage requirements
    if training_args.stage == 1 and model_args.use_gt_boxes_for_caption:
        raise ValueError('Stage 1 cannot use --use_gt_boxes_for_caption=True since caption head is frozen.')
    if training_args.stage == 2 and training_args.stage1_checkpoint is None:
        raise ValueError('Stage 2 requires --stage1_checkpoint to be specified.')
    if training_args.stage == 3 and training_args.stage2_checkpoint is None:
        print('Warning: Found no --stage2_checkpoint for stage 3. The model will be trained from scratch.')
    
    # Data Loading
    tokenizer = AutoTokenizer.from_pretrained('captioners/trimmed_tokenizer')
    train_dataset = DVCDataset(
        split='train', tokenizer=tokenizer, max_tries=data_args.max_tries, noise_rate=data_args.noise_rate, pose_augment=data_args.pose_augment, 
        min_events=data_args.min_events, max_events=data_args.max_events, max_window_tokens=data_args.max_window_tokens, 
        max_event_tokens=data_args.max_event_tokens, load_by=data_args.load_by, seed=training_args.seed
    )
    # val_dataset = DVCDataset(
    #     split='val', tokenizer=tokenizer, pose_augment=False, stride_ratio=data_args.stride_ratio, 
    #     min_events=data_args.min_events, max_events=data_args.max_events, max_event_tokens=data_args.max_event_tokens, 
    #     max_window_tokens=data_args.max_window_tokens, load_by=data_args.load_by, seed=training_args.seed
    # )
    if getattr(training_args, 'local_rank', -1) in (-1, 0): # Only log sizes on the main process to avoid clutter in DDP
        print(f'\nTraining Stage: {training_args.stage}')
        print(f'Train dataset: {len(train_dataset)} samples')
        # print(f'Val dataset: {len(val_dataset)} samples')

    # Build weight dict based on stage
    if training_args.stage == 1: # Stage 1: Only localization losses
        weight_dict = {
            'loss_ce': model_args.class_cost, 
            'loss_bbox': model_args.bbox_cost, 
            'loss_giou': model_args.giou_cost, 
            'loss_counter': model_args.counter_cost, 
            'loss_caption': 0  # No caption loss in stage 1
        }
    elif training_args.stage == 2: # Stage 2: Only caption loss
        weight_dict = {
            'loss_ce': 0, 
            'loss_bbox': 0, 
            'loss_giou': 0, 
            'loss_counter': 0, 
            'loss_caption': model_args.caption_cost
        }
    else: # Stage 3: All losses with balanced weights for joint fine-tuning
        weight_dict = {
            'loss_ce': model_args.class_cost, 
            'loss_bbox': model_args.bbox_cost, 
            'loss_giou': model_args.giou_cost, 
            'loss_counter': model_args.counter_cost, 
            'loss_caption': model_args.caption_cost
        }

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
        captioner_class=MBartDecoderCaptioner,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.lang_code_to_id['en_XX'],
        num_cap_layers=model_args.num_cap_layers,
        cap_dropout_rate=model_args.cap_dropout_rate,
        max_event_tokens=data_args.max_event_tokens,
        max_events=data_args.max_events,
        weight_dict=weight_dict,
        use_gt_boxes_for_caption=model_args.use_gt_boxes_for_caption,
    ) # IMPORTANT: Do not .to(device); Trainer handles device placement and DDP
    
    # Load stage 1 checkpoint for stage 2
    if training_args.stage == 2:
        checkpoint_path = os.path.join(training_args.stage1_checkpoint, 'pytorch_model.bin')
        if os.path.exists(checkpoint_path):
            print(f'Loading stage 1 checkpoint from: {checkpoint_path}')
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle potential key mismatches due to with_box_refine difference
            filtered_state = handle_key_mismatches(state_dict, model.state_dict())
            model.load_state_dict(filtered_state, strict=False)
            print(f'Loaded {len(filtered_state)}/{len(state_dict)} parameters from stage 1')
        else: raise FileNotFoundError(f'Stage 1 checkpoint not found: {checkpoint_path}')
    
    # Load stage 2 checkpoint for stage 3
    if training_args.stage == 3 and training_args.stage2_checkpoint is not None:
        checkpoint_path = os.path.join(training_args.stage2_checkpoint, 'pytorch_model.bin')
        if os.path.exists(checkpoint_path):
            print(f'Loading stage 2 checkpoint from: {checkpoint_path}')
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle potential key mismatches due to with_box_refine difference
            filtered_state = handle_key_mismatches(state_dict, model.state_dict())
            model.load_state_dict(filtered_state, strict=False)
            print(f'Loaded {len(filtered_state)}/{len(state_dict)} parameters from stage 2')
        else: raise FileNotFoundError(f'Stage 2 checkpoint not found: {checkpoint_path}')
    
    # Setup freezing based on stage
    print('\n' + '='*80)
    if training_args.stage == 1: # Train localization, freeze caption head
        print('STAGE 1: Training Localization (caption_head frozen)')
        unfreeze_module(model) # Unfreeze everything first
        if isinstance(model.caption_head, torch.nn.ModuleList): # Freeze caption heads
            for head in model.caption_head: freeze_module(head)
        else: freeze_module(model.caption_head)
        
    elif training_args.stage == 2: # Train captioning, freeze localization
        print('STAGE 2: Training Caption Head (localization frozen)')
        freeze_module(model) # Freeze everything first
        if isinstance(model.caption_head, torch.nn.ModuleList): # Unfreeze caption heads
            for head in model.caption_head: unfreeze_module(head)
        else: unfreeze_module(model.caption_head)
    
    else: # Joint fine-tuning - unfreeze everything
        print('STAGE 3: Joint Fine-tuning (all parameters trainable)')
        unfreeze_module(model) # Unfreeze everything
    print('='*80)
    
    if getattr(training_args, 'local_rank', -1) in (-1, 0): 
        print_trainable_parameters(model)
    
    # Move to device
    if training_args._n_gpu <= 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    # Trainer Setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        data_collator=trainer_collate_fn,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)],
    )
    trainer.train()
    
    # Save final model
    save_path = os.path.join(training_args.output_dir, f'stage{training_args.stage}_final')
    trainer.save_model(save_path)
    
    if getattr(training_args, 'local_rank', -1) in (-1, 0):
        print(f'\nStage {training_args.stage} training complete!')
        print(f'Model saved to: {save_path}')
        
        if training_args.stage == 1:
            print(f'To continue with Stage 2, run:')
            print(f'python main.py --stage 2 --stage1_checkpoint {save_path} --output_dir checkpoints/stage2')
        elif training_args.stage == 2:
            print(f'To continue with Stage 3 (optional joint fine-tuning), run:')
            print(f'python main.py --stage 3 --stage2_checkpoint {save_path} --output_dir checkpoints/stage3 --learning_rate 1e-5 --num_train_epochs 30')
    
    # Cleanup to free memory
    model.to('cpu')
    del tokenizer, train_dataset, model, training_args, trainer
    gc.collect()
    if torch.cuda.is_available():  torch.cuda.empty_cache()


if __name__ == '__main__':
    main()