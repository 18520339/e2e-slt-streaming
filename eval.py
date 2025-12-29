import os
import gc
import torch
from functools import partial
from typing import Optional, Tuple
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer, DeformableDetrConfig,
    HfArgumentParser, TrainingArguments, 
    Trainer,
)
from loader import DVCDataset, trainer_collate_fn
from pdvc import DeformableDetrForObjectDetection
from captioners import MBartDecoderCaptioner
from evaluation import preprocess_logits_for_metrics, compute_metrics
from config import *


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

    # Caption head / decoder bits
    num_cap_layers: int = field(default=3)
    cap_dropout_rate: float = field(default=0.1)


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

    # Metrics/Ranking
    ranking_temperature: float = field(default=2.0, metadata={'help': 'Exponent T in caption score normalization by length^T'})
    alpha: float = field(default=0.3, metadata={'help': 'Ranking policy: joint_score = alpha * (caption_score / len(tokens)^T) + (1 - alpha) * det_score'})
    top_k: int = field(default=20, metadata={'help': 'Keep top k events during evaluation for metrics computation'})
    temporal_iou_thresholds: Tuple[float, float, float, float] = field(default=(0.3, 0.5, 0.7, 0.9))
    soda_recursion_limit: int = field(default=0, metadata={'help': 'Increase recursion limit for SODA_c DP if needed, 0 to disable for faster calculations'})


@dataclass
class EvalArguments:
    checkpoint_path: str = field(default=CHECKPOINT_DIR, metadata={'help': 'Path to the checkpoint directory to evaluate'})
    output_dir: str = field(default='/tmp/eval', metadata={'help': 'Directory for evaluation outputs'})
    per_device_eval_batch_size: int = field(default=32, metadata={'help': 'Can be higher if greedy but should be smaller if using beam search'})
    dataloader_num_workers: int = field(default=4, metadata={'help': 'Number of subprocesses to use for data loading'})
    seed: int = field(default=42, metadata={'help': 'Random seed for reproducibility'})
    fp16: bool = field(default=False, metadata={'help': 'Use mixed precision evaluation'})
    bf16: bool = field(default=False, metadata={'help': 'Use bfloat16 for mixed precision evaluation'})
    eval_val: bool = field(default=True, metadata={'help': 'Evaluate on validation set'})
    eval_test: bool = field(default=True, metadata={'help': 'Evaluate on test set'})


def main():
    # Parse CLI args
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading checkpoint from: {eval_args.checkpoint_path}')

    # Model Setup 
    tokenizer = AutoTokenizer.from_pretrained('captioners/trimmed_tokenizer')
    config = DeformableDetrConfig(
        d_model=model_args.d_model,
        encoder_layers=model_args.encoder_layers,
        decoder_layers=model_args.decoder_layers,
        encoder_attention_heads=model_args.encoder_attention_heads,
        decoder_attention_heads=model_args.decoder_attention_heads,
        encoder_n_points=model_args.encoder_n_points,
        decoder_n_points=model_args.decoder_n_points,
        num_feature_levels=model_args.num_feature_levels,
        num_queries=model_args.num_queries,
        num_labels=model_args.num_labels,
        auxiliary_loss=model_args.auxiliary_loss,
        class_cost=model_args.class_cost,
        bbox_cost=model_args.bbox_cost,
        giou_cost=model_args.giou_cost,
        focal_alpha=model_args.focal_alpha,
        with_box_refine=model_args.with_box_refine,
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
        weight_dict={
            'loss_ce': model_args.class_cost, 'loss_bbox': model_args.bbox_cost, 'loss_giou': model_args.giou_cost, 
            'loss_counter': model_args.counter_cost, 'loss_caption': model_args.caption_cost
        }
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(eval_args.checkpoint_path, 'pytorch_model.bin')))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model loaded with {total_params / 1e6:.2f}M parameters')
    
    # Evaluation
    training_args = TrainingArguments( # Create training args for Trainer (required even for evaluation)
        output_dir=eval_args.output_dir,
        per_device_eval_batch_size=eval_args.per_device_eval_batch_size,
        dataloader_num_workers=eval_args.dataloader_num_workers,
        seed=eval_args.seed,
        fp16=eval_args.fp16,
        bf16=eval_args.bf16,
        report_to='none',
        do_train=False,
        do_eval=True,
    )
    
    if eval_args.eval_val:
        print('\n' + '='*80)
        print('Evaluating on validation set...')
        print('='*80)
        
        val_dataset = DVCDataset(
            split='val', tokenizer=tokenizer, pose_augment=False, stride_ratio=data_args.stride_ratio, 
            min_events=data_args.min_events, max_events=data_args.max_events, max_event_tokens=data_args.max_event_tokens, 
            max_window_tokens=data_args.max_window_tokens, load_by=data_args.load_by, seed=eval_args.seed
        )
        print(f'Val dataset: {len(val_dataset)} samples')
        
        eval_trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=val_dataset,
            data_collator=trainer_collate_fn,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=partial(
                compute_metrics,
                ranking_temperature=data_args.ranking_temperature,
                alpha=data_args.alpha,
                top_k=data_args.top_k,
                temporal_iou_thresholds=data_args.temporal_iou_thresholds,
                tokenizer=tokenizer,
                soda_recursion_limit=data_args.soda_recursion_limit,
            ),
        )
        val_metrics = eval_trainer.evaluate(metric_key_prefix='val')
        print(f'\nValidation metrics: {val_metrics}')
        
        del val_dataset, eval_trainer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    if eval_args.eval_test:
        print('\n' + '='*80)
        print('Evaluating on test set...')
        print('='*80)
        
        test_dataset = DVCDataset(
            split='test', tokenizer=tokenizer, pose_augment=False, stride_ratio=data_args.stride_ratio, 
            max_event_tokens=data_args.max_event_tokens, max_window_tokens=data_args.max_window_tokens,
            min_events=data_args.min_events, load_by=data_args.load_by, seed=eval_args.seed
        )
        print(f'Test dataset: {len(test_dataset)} samples')
        
        eval_trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            data_collator=trainer_collate_fn,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=partial(
                compute_metrics,
                ranking_temperature=data_args.ranking_temperature,
                alpha=data_args.alpha,
                top_k=data_args.top_k,
                temporal_iou_thresholds=data_args.temporal_iou_thresholds,
                tokenizer=tokenizer,
                soda_recursion_limit=data_args.soda_recursion_limit,
            ),
        )
        test_metrics = eval_trainer.evaluate(metric_key_prefix='test')
        print('\nTest metrics:', test_metrics, sep='\n')
        
        del test_dataset, eval_trainer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Cleanup
    model.to('cpu')
    del tokenizer, model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()