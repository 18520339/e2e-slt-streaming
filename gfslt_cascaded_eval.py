''' Multi-Stage Evaluation Pipeline: Deformable DETR + GFSLT

This script evaluates a two-stage approach:
1. Stage 1: Use Deformable DETR for temporal event localization
2. Stage 2: Use GFSLT for captioning each localized event

The pipeline computes:
- Localization metrics: Precision, Recall, F1 at IoU thresholds [0.3, 0.5, 0.7, 0.9]
- Captioning metrics: BLEU-4, BLEURT, ROUGE-L, METEOR, CIDEr for matched pred-GT pairs
- Comparison between GFSLT (multi-stage) and DETR captioner (single-stage)

> python multistage_eval.py \
    --detr_checkpoint_path checkpoints/detr/pytorch_model.bin \
    --gfslt_checkpoint_path checkpoints/gfslt/pytorch_model.bin \
    --eval_test
'''
import gc, os, json
import numpy as np

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DeformableDetrConfig, HfArgumentParser

from loader import DVCDataset, collate_fn
from captioners import MBartDecoderCaptioner
from pdvc import DeformableDetrForObjectDetection
from config import TGT_LANG, TRIMMED_TOKENIZER_DIR, TRIMMED_MBART_DIR
from postprocess import post_process_object_detection

from gfslt_models import GFSLT, GFSLTConfig
from config import WINDOW_DURATION_SECONDS, FPS
from evaluation.helpers import select_topN_per_window, extract_gt_per_window, aggregate_metrics


@dataclass
class ModelArguments: # Arguments for model configuration (matched to training configs)
    # DETR config (from main.py)
    d_model: int = field(default=1024)
    encoder_layers: int = field(default=2)
    decoder_layers: int = field(default=2)
    encoder_attention_heads: int = field(default=8)
    decoder_attention_heads: int = field(default=8)
    encoder_n_points: int = field(default=4)
    decoder_n_points: int = field(default=4)
    num_feature_levels: int = field(default=4)
    num_queries: int = field(default=30)
    num_labels: int = field(default=1)
    auxiliary_loss: bool = field(default=True)
    with_box_refine: bool = field(default=True)
    num_cap_layers: int = field(default=3)
    cap_dropout_rate: float = field(default=0.1)
    
    # GFSLT config (from gfslt_stage2.py)
    gfslt_embed_dim: int = field(default=1024)
    gfslt_hidden_size: int = field(default=1024)
    gfslt_temporal_kernel: int = field(default=3)
    gfslt_mbart_name: str = field(default_factory=lambda: f'./{TRIMMED_MBART_DIR}')


@dataclass
class DataArguments: # Arguments for data loading
    stride_ratio: float = field(default=0.9)
    min_events: int = field(default=1)
    max_events: int = field(default=10)
    max_event_tokens: int = field(default=40)
    max_window_tokens: int = field(default=128)
    load_by: str = field(default='window')


@dataclass
class EvalArguments: # Arguments for evaluation
    detr_checkpoint_path: str = field(default=None, metadata={'help': 'Path to DETR checkpoint'})
    gfslt_checkpoint_path: str = field(default=None, metadata={'help': 'Path to GFSLT checkpoint'})
    output_dir: str = field(default='./experiments')
    per_device_eval_batch_size: int = field(default=16)
    dataloader_num_workers: int = field(default=4)
    seed: int = field(default=42)
    eval_val: bool = field(default=False)
    eval_test: bool = field(default=True)
    detection_threshold: float = field(default=0.0, metadata={'help': 'Confidence threshold for DETR detections'})
    gfslt_max_new_tokens: int = field(default=40)
    gfslt_num_beams: int = field(default=1)
    skip_gfslt: bool = field(default=False, metadata={'help': 'Skip GFSLT captioning for fast localization-only eval'})
    max_events_per_window: int = field(default=10, metadata={'help': 'Max events per window to caption (for speed)'})
    use_fp16: bool = field(default=False, metadata={'help': 'Use FP16 for faster inference'})
    aggregation_mode: str = field(default='video', metadata={
        'help': f"How to aggregate caption pairs per IoU threshold. Same metric keys across modes: "
                f"'corpus' (default; pool all pairs -> sacrebleu corpus). "
                f"'window' (corpus per window, mean across windows; linear). "
                f"'video' (group windows by video_id, corpus per video, mean across videos)."})


# ======================== Padding Utility ========================
def pad_event_to_window_size(event_poses: torch.Tensor, window_size: int) -> torch.Tensor:
    ''' Pad event poses to window size by repeating the last frame.
    
    Args:
        event_poses: (T, K, C) pose tensor for the event
        window_size: Target window size in frames
        
    Returns:
        Padded pose tensor of shape (window_size, K, C)
    '''
    T, K, C = event_poses.shape
    if T >= window_size: return event_poses[:window_size]  # Truncate if needed
    
    # Repeat last frame to pad
    pad_len = window_size - T
    last_frame = event_poses[-1:].expand(pad_len, K, C)
    return torch.cat([event_poses, last_frame], dim=0)


# ======================== Multi-Stage Evaluator ========================
class MultiStageEvaluator:
    ''' Two-stage evaluation: DETR localization + GFSLT captioning.
    
    Pipeline:
    1. Run DETR on each window to get temporal event localizations
    2. For each predicted event, extract pose subsequence and pad to window size
    3. Run GFSLT on each padded event to generate captions
    4. Compute metrics comparing predictions to ground truth
    '''
    
    def __init__(
        self, detr_model: DeformableDetrForObjectDetection,
        gfslt_model: GFSLT, detr_tokenizer: AutoTokenizer,
        gfslt_tokenizer: AutoTokenizer, device: torch.device,
        detection_threshold: float = 0.1, max_new_tokens: int = 64,
        num_beams: int = 1, skip_gfslt: bool = False,
        max_events_per_window: int = 5, use_fp16: bool = True,
        aggregation_mode: str = 'video',
    ):
        self.detr_model = detr_model.eval().to(device)
        self.gfslt_model = gfslt_model.eval().to(device) if not skip_gfslt else None
        
        # Apply FP16 for faster inference
        if use_fp16 and device.type == 'cuda':
            self.detr_model = self.detr_model.half()
            if self.gfslt_model is not None:
                self.gfslt_model = self.gfslt_model.half()
        
        self.detr_tokenizer = detr_tokenizer
        self.gfslt_tokenizer = gfslt_tokenizer
        self.device = device
        self.detection_threshold = detection_threshold
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.skip_gfslt = skip_gfslt
        self.max_events_per_window = max_events_per_window
        self.use_fp16 = use_fp16
        self.aggregation_mode = aggregation_mode
        self.window_size = int(WINDOW_DURATION_SECONDS * FPS)  # 15s * 12.5fps = 187 frames
        
        
    @torch.no_grad()
    def predict_detr_localizations(self, batch: Dict) -> List[Dict]:
        ''' Run DETR to get event localizations for each window in the batch.
        
        Returns:
            List of dicts per window, each containing:
            - 'event_ranges_normalized': List of (start, end) tuples in [0,1] for IoU comparison
            - 'event_ranges_frames': List of (start_frame, end_frame) tuples for GFSLT captioning
            - 'event_scores': Detection confidence scores
            - 'detr_captions': Captions from DETR's built-in captioner (for baseline)
        '''
        pixel_values = batch['pixel_values'].to(self.device)
        pixel_mask = batch['pixel_mask'].to(self.device)
        if self.use_fp16: pixel_values = pixel_values.half() # Convert to half precision if using FP16
        
        # Run DETR forward pass
        outputs = self.detr_model(pixel_values=pixel_values, pixel_mask=pixel_mask, return_dict=True)
        if not hasattr(self, '_debug_printed') or not self._debug_printed: # Debug: print DETR output shapes
            print(f"\n[DEBUG DETR Output]")
            print(f"  logits shape: {outputs.logits.shape}")
            print(f"  pred_boxes shape: {outputs.pred_boxes.shape}")
            print(f"  pred_counts shape: {outputs.pred_counts.shape if outputs.pred_counts is not None else 'None'}")
            self._debug_printed = True
        
        # Post-process with threshold=0 (count head + reranking selects later)
        # Get more events than needed, then filter by counter head
        results_normalized = post_process_object_detection(
            outputs=outputs,
            top_k=self.max_events_per_window,  # match metrics.py top_k -- guarantees identical selection
            threshold=0.0,  # No threshold - use counter head instead
            target_lengths=None,  # Keep in [0,1] for IoU comparison
            tokenizer=self.detr_tokenizer,
        )
        # Count head -> per-window N (same as metrics.py compute_metrics)
        if outputs.pred_counts is not None: pred_counts = outputs.pred_counts.argmax(dim=-1).clamp(min=0).cpu().tolist()
        else: pred_counts = None

        # Shared selection helper -> identical to metrics.py path
        batch_pred_events, batch_detr_captions, _ = select_topN_per_window(
            results_normalized, pred_counts, self.detr_tokenizer,
            ranking_temperature=2.0, alpha=0.3, top_k=self.max_events_per_window,
        )
        # Convert normalized [0,1] ranges -> absolute frame indices for GFSLT captioning
        batch_results = []
        for i, (event_ranges_normalized, detr_captions_selected) in enumerate(zip(batch_pred_events, batch_detr_captions)):
            valid_frames = pixel_mask[i].sum().item()
            event_ranges_frames = [(int(s * valid_frames), int(e * valid_frames)) for s, e in event_ranges_normalized]
            batch_results.append({
                'event_ranges_normalized': event_ranges_normalized,
                'event_ranges_frames': event_ranges_frames,
                'detr_captions': detr_captions_selected,
            })
        return batch_results
    
    
    @torch.no_grad()
    def caption_events_with_gfslt(self, poses: torch.Tensor, events: List[Tuple[int, int]]) -> List[str]:
        ''' Caption localized events using GFSLT with batching for speed.
        
        Args:
            poses: (T, K, C) full window poses
            events: List of (start_frame, end_frame) tuples
            
        Returns:
            List of caption strings for each event
        '''
        if not events: return []
        
        # Prepare all event poses in a batch
        batch_poses, valid_indices = [], []
        for idx, (start_frame, end_frame) in enumerate(events):
            start_frame = max(0, int(start_frame))
            end_frame = min(poses.shape[0], int(end_frame))
            
            if start_frame >= end_frame: continue
            event_poses = poses[start_frame:end_frame]
            padded_poses = pad_event_to_window_size(event_poses, self.window_size)
            batch_poses.append(padded_poses)
            valid_indices.append(idx)
        
        if not batch_poses: return [''] * len(events)
        pixel_values = torch.stack(batch_poses, dim=0).to(self.device) # Stack into batch: (N, T, K, C)
        pixel_mask = torch.ones(len(batch_poses), self.window_size, dtype=torch.bool, device=self.device)
        
        # Generate captions for all events in batch
        batch_captions = self._generate_gfslt_captions_batch(pixel_values, pixel_mask)
        
        # Map back to original indices (fill empty for invalid events)
        captions = [''] * len(events)
        for i, idx in enumerate(valid_indices): captions[idx] = batch_captions[i]
        return captions
    
    
    @torch.no_grad()
    def _generate_gfslt_captions_batch(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor) -> List[str]:
        ''' Batched greedy decoding for GFSLT - much faster than sequential.
        
        Args:
            pixel_values: (N, T, K, C) batch of padded event poses
            pixel_mask: (N, T) attention masks
            
        Returns:
            List of caption strings for each event in batch
        '''
        batch_size = pixel_values.shape[0]
        
        # Prepare encoder inputs
        inputs_embeds, attention_mask = self.gfslt_model._prepare_inputs(pixel_values, pixel_mask)
        
        # Run encoder once for entire batch
        encoder_outputs = self.gfslt_model.mbart.model.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
        
        # Initialize decoder inputs for all samples
        decoder_start_id = self.gfslt_tokenizer.lang_code_to_id.get(TGT_LANG, self.gfslt_tokenizer.bos_token_id)
        decoder_input_ids = torch.full((batch_size, 1), decoder_start_id, dtype=torch.long, device=self.device)
        
        # Track which samples are still generating
        eos_token_id = self.gfslt_tokenizer.eos_token_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        generated_ids = [[decoder_start_id] for _ in range(batch_size)]
        
        # KV-cache for faster generation
        past_key_values = None
        for step in range(self.max_new_tokens): # Run decoder with KV-caching
            decoder_outputs = self.gfslt_model.mbart.model.decoder(
                input_ids=decoder_input_ids,  # Only last token when using cache
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = decoder_outputs.past_key_values # Update cache
            
            # Get logits for last token: (batch_size, vocab_size)
            lm_logits = self.gfslt_model.mbart.lm_head(decoder_outputs.last_hidden_state[:, -1, :])
            
            # Greedy: pick token with highest probability
            next_tokens = lm_logits.argmax(dim=-1)  # (batch_size,)
            
            # Update generated sequences
            for i in range(batch_size):
                if not finished[i]:
                    next_token = next_tokens[i].item()
                    generated_ids[i].append(next_token)
                    if next_token == eos_token_id:
                        finished[i] = True
            
            if finished.all(): break # Stop if all finished
            decoder_input_ids = next_tokens.unsqueeze(1) # Next iteration only needs the new token (cache has history)
        return [self.gfslt_tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids] # Decode all to text
    
    
    def evaluate(self, data_loader: DataLoader, iou_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)) -> Dict[str, float]:
        ''' Full evaluation loop computing metrics at each IoU threshold.
        
        Returns:
            Dict of metrics including:
            - Localization: loc_precision@XX, loc_recall@XX, loc_f1@XX
            - GFSLT captioning: gfslt_bleu4@XX, gfslt_bleurt@XX, etc.
            - DETR captioning (baseline): detr_bleu4@XX, etc.
        '''
        # Collect all predictions and ground truth
        all_pred_events: List[List[Tuple[float, float]]] = []
        all_video_ids: List[str] = []  # one per window, for aggregation_mode='video'
        all_gfslt_captions: List[List[str]] = []
        all_detr_captions: List[List[str]] = []
        all_gt_events: List[List[Tuple[float, float]]] = []
        all_gt_captions: List[List[str]] = []
        print(f'\nRunning multi-stage evaluation on {len(data_loader.dataset)} windows...')
        
        for batch in tqdm(data_loader, desc='Evaluating'):
            pixel_values = batch['pixel_values']  # (B, T, K, C)
            labels = batch['labels']
            
            # Stage 1: DETR localization
            detr_results = self.predict_detr_localizations(batch)
            
            # Stage 2: Collect ALL events from ALL windows for batch GFSLT processing
            # Track per-event validity so captions stay aligned even when some events are
            # degenerate (start_frame >= end_frame) and skipped from GFSLT input.
            all_batch_poses = []                       # padded pose tensors for VALID events only
            window_valid_masks: List[List[bool]] = []  # per-window list of bool, True = caption from GFSLT, False = empty

            for i, (poses, detr_result, label) in enumerate(zip(pixel_values, detr_results, labels)):
                pred_events_frames = detr_result['event_ranges_frames'][:self.max_events_per_window]
                valid_mask = []
                for start_frame, end_frame in pred_events_frames:
                    start_frame = max(0, int(start_frame))
                    end_frame = min(poses.shape[0], int(end_frame))
                    if start_frame < end_frame:
                        event_poses = poses[start_frame:end_frame]
                        all_batch_poses.append(pad_event_to_window_size(event_poses, self.window_size))
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                window_valid_masks.append(valid_mask)
            
            # Run GFSLT on ALL events at once
            if self.skip_gfslt or self.gfslt_model is None or len(all_batch_poses) == 0:
                all_gfslt_captions_flat = [""] * len(all_batch_poses)
            else: # Stack all event poses: (total_events, T, K, C)
                stacked_poses = torch.stack(all_batch_poses, dim=0).to(self.device)
                if self.use_fp16: stacked_poses = stacked_poses.half()
                pixel_mask = torch.ones(len(all_batch_poses), self.window_size, dtype=torch.bool, device=self.device)
                all_gfslt_captions_flat = self._generate_gfslt_captions_batch(stacked_poses, pixel_mask)
            
            # Redistribute GFSLT captions back to per-window lists. Use per-event valid_mask
            # so degenerate (skipped) events get '' instead of pulling the next window's caption.
            caption_idx = 0
            batch_pred_events_local, batch_gfslt_caps_local, batch_detr_caps_local = [], [], []
            for i, detr_result in enumerate(detr_results):
                window_caps = []
                for valid in window_valid_masks[i]:
                    if valid:
                        window_caps.append(all_gfslt_captions_flat[caption_idx])
                        caption_idx += 1
                    else:
                        window_caps.append('')
                batch_pred_events_local.append(detr_result['event_ranges_normalized'])
                batch_gfslt_caps_local.append(window_caps)
                batch_detr_caps_local.append(detr_result['detr_captions'])

            # Shared GT extraction -> identical to metrics.py path
            batch_gt_events_local, batch_gt_caps_local = extract_gt_per_window(labels, self.detr_tokenizer)
            all_pred_events.extend(batch_pred_events_local)
            all_gfslt_captions.extend(batch_gfslt_caps_local)
            all_detr_captions.extend(batch_detr_caps_local)
            all_gt_events.extend(batch_gt_events_local)
            all_gt_captions.extend(batch_gt_caps_local)
            # collate_fn keeps a separate `video_ids` tuple on the batch dict (one per window).
            all_video_ids.extend(batch.get('video_ids', [None] * len(labels)))

        # Compute metrics via shared aggregator. Two caption sources -> two calls
        # with different prefixes; localization computed once on the first call.
        metrics: Dict[str, float] = {}
        metrics.update(aggregate_metrics(
            all_pred_events, all_gfslt_captions, all_gt_events, all_gt_captions,
            temporal_iou_thresholds=iou_thresholds, prefix='gfslt',
            include_localization=True, include_paragraph=True, include_segment=True,
            aggregation_mode=self.aggregation_mode, batch_video_ids=all_video_ids,
        ))
        metrics.update(aggregate_metrics(
            all_pred_events, all_detr_captions, all_gt_events, all_gt_captions,
            temporal_iou_thresholds=iou_thresholds, prefix='detr',
            include_localization=False, include_paragraph=True, include_segment=False,
            aggregation_mode=self.aggregation_mode, batch_video_ids=all_video_ids,
        ))
        return metrics


# ======================== Model Loading ========================
def load_detr_model(model_args: ModelArguments, data_args: DataArguments, checkpoint_path: str, tokenizer: AutoTokenizer, device: torch.device):
    # Load Deformable DETR model with correct configuration from main.py
    config = DeformableDetrConfig(
        d_model=model_args.d_model,
        encoder_layers=model_args.encoder_layers,
        decoder_layers=model_args.decoder_layers,
        encoder_attention_heads=model_args.encoder_attention_heads,
        decoder_attention_heads=model_args.decoder_attention_heads,
        encoder_n_points=model_args.encoder_n_points,
        decoder_n_points=model_args.decoder_n_points,
        activation_function='gelu',
        num_feature_levels=model_args.num_feature_levels,
        num_queries=model_args.num_queries,
        num_labels=model_args.num_labels,
        auxiliary_loss=model_args.auxiliary_loss,
        with_box_refine=model_args.with_box_refine,
    )
    model = DeformableDetrForObjectDetection(
        config=config,
        captioner_class=MBartDecoderCaptioner,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.lang_code_to_id[TGT_LANG],
        num_cap_layers=model_args.num_cap_layers,
        cap_dropout_rate=model_args.cap_dropout_rate,
        max_event_tokens=data_args.max_event_tokens,
        max_events=data_args.max_events,
        use_gt_boxes_for_caption=not model_args.with_box_refine,
    )
    if checkpoint_path: # Handle both direct .bin paths and checkpoint directories (like eval.py)
        if os.path.isdir(checkpoint_path): checkpoint_file = os.path.join(checkpoint_path, 'pytorch_model.bin')
        else: checkpoint_file = checkpoint_path
        
        if os.path.exists(checkpoint_file):
            print(f'Loading DETR checkpoint from: {checkpoint_file}')
            state_dict = torch.load(checkpoint_file, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else: print(f'Warning: DETR checkpoint not found at {checkpoint_file}. Using random weights.')
    else: print(f'Warning: No DETR checkpoint path provided. Using random weights.')
    
    print(f'DETR model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters')
    return model


def load_gfslt_model(model_args: ModelArguments, checkpoint_path: str, device: torch.device):
    # Load GFSLT model with correct configuration from gfslt_stage2.py
    config = GFSLTConfig(
        embed_dim=model_args.gfslt_embed_dim,
        hidden_size=model_args.gfslt_hidden_size,
        temporal_kernel=model_args.gfslt_temporal_kernel,
        mbart_name=model_args.gfslt_mbart_name,
    )
    model = GFSLT(config)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f'Loading GFSLT checkpoint from: {checkpoint_path}')
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats (Trainer saves with 'model' or 'state_dict' key)
        if 'model' in state_dict: state_dict = state_dict['model']
        elif 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        
        # Handle Wrapper4Trainer format where keys have 'base_module.' prefix
        # The Wrapper4Trainer wraps GFSLT as self.base_module
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('base_module.'): # Remove the 'base_module.' prefix to match GFSLT's state dict
                new_k = k.replace('base_module.', '', 1) 
                new_state_dict[new_k] = v
            else: new_state_dict[k] = v
        
        if len(new_state_dict) != len(state_dict): 
            print(f"  Remapped {len(state_dict) - len(new_state_dict)} keys with 'base_module.' prefix")
        state_dict = new_state_dict
        
        # Filter out incompatible keys
        model_state = model.state_dict()
        filtered_state = {}
        missing_keys, shape_mismatch = [], []
        
        for k, v in state_dict.items():
            if k in model_state:
                if model_state[k].shape == v.shape: filtered_state[k] = v
                else: shape_mismatch.append(f"{k}: checkpoint {v.shape} vs model {model_state[k].shape}")
            else: missing_keys.append(k)
        
        if missing_keys: print(f"  Keys not in model ({len(missing_keys)}): {missing_keys[:5]}...")
        if shape_mismatch: print(f"  Shape mismatches ({len(shape_mismatch)}): {shape_mismatch[:3]}...")
        
        model.load_state_dict(filtered_state, strict=False)
        print(f'Loaded {len(filtered_state)}/{len(state_dict)} parameters from GFSLT checkpoint')
    else: print(f'Warning: GFSLT checkpoint not found at {checkpoint_path}. Using random weights.')
    
    print(f'GFSLT model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters')
    return model


# ======================== Main ========================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizers
    detr_tokenizer = AutoTokenizer.from_pretrained(f'./{TRIMMED_TOKENIZER_DIR}')
    gfslt_tokenizer = AutoTokenizer.from_pretrained(f'./{TRIMMED_TOKENIZER_DIR}')
    
    # Load models
    detr_model = load_detr_model(model_args, data_args, eval_args.detr_checkpoint_path, detr_tokenizer, device)
    gfslt_model = load_gfslt_model(model_args, eval_args.gfslt_checkpoint_path, device)
    
    # Create evaluator
    evaluator = MultiStageEvaluator(
        detr_model=detr_model,
        gfslt_model=gfslt_model,
        detr_tokenizer=detr_tokenizer,
        gfslt_tokenizer=gfslt_tokenizer,
        device=device,
        detection_threshold=eval_args.detection_threshold,
        max_new_tokens=eval_args.gfslt_max_new_tokens,
        num_beams=eval_args.gfslt_num_beams,
        skip_gfslt=eval_args.skip_gfslt,
        max_events_per_window=eval_args.max_events_per_window,
        use_fp16=eval_args.use_fp16,
        aggregation_mode=eval_args.aggregation_mode,
    )
    os.makedirs(eval_args.output_dir, exist_ok=True) # Create output directory
    
    # Evaluate on val set
    if eval_args.eval_val:
        print('\n' + '='*80)
        print('Evaluating on VALIDATION set...')
        print('='*80)
        
        val_dataset = DVCDataset(
            split='val', tokenizer=detr_tokenizer, pose_augment=False, stride_ratio=data_args.stride_ratio,
            min_events=data_args.min_events, max_events=data_args.max_events, 
            max_event_tokens=data_args.max_event_tokens, max_window_tokens=data_args.max_window_tokens,
            load_by=data_args.load_by, seed=eval_args.seed
        )
        print(f'Val dataset: {len(val_dataset)} windows')
        
        val_loader = DataLoader(
            val_dataset, batch_size=eval_args.per_device_eval_batch_size,
            shuffle=False, num_workers=eval_args.dataloader_num_workers,
            pin_memory=True, collate_fn=collate_fn
        )
        val_metrics = evaluator.evaluate(val_loader)
        print_metrics(val_metrics, prefix='val')
        save_metrics(val_metrics, os.path.join(eval_args.output_dir, 'cascaded_val.json'))
        del val_dataset, val_loader
    
    # Evaluate on test set
    if eval_args.eval_test:
        print('\n' + '='*80)
        print('Evaluating on TEST set...')
        print('='*80)
        
        test_dataset = DVCDataset(
            split='test', tokenizer=detr_tokenizer, pose_augment=False, stride_ratio=data_args.stride_ratio,
            min_events=data_args.min_events, max_events=data_args.max_events,
            max_event_tokens=data_args.max_event_tokens, max_window_tokens=data_args.max_window_tokens,
            load_by=data_args.load_by, seed=eval_args.seed
        )
        print(f'Test dataset: {len(test_dataset)} windows')
        
        test_loader = DataLoader(
            test_dataset, batch_size=eval_args.per_device_eval_batch_size,
            shuffle=False, num_workers=eval_args.dataloader_num_workers,
            pin_memory=True, collate_fn=collate_fn
        )
        
        test_metrics = evaluator.evaluate(test_loader)
        print_metrics(test_metrics, prefix='test')
        save_metrics(test_metrics, os.path.join(eval_args.output_dir, 'cascaded_test.json'))
        del test_dataset, test_loader
    
    # Cleanup
    detr_model.to('cpu')
    gfslt_model.to('cpu')
    del detr_model, gfslt_model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    print('\n' + '='*80)
    print(f'Multi-stage evaluation complete! Results saved to: {eval_args.output_dir}')
    print('='*80)


def print_metrics(metrics: Dict[str, float], prefix: str = ''):
    print(f"\n{'='*60}")
    print(f"  {prefix.upper()} METRICS")
    print(f"{'='*60}")
    
    # Group metrics
    loc_metrics = {k: v for k, v in metrics.items() if k.startswith('loc_')}
    gfslt_metrics = {k: v for k, v in metrics.items() if k.startswith('gfslt_')}
    detr_metrics = {k: v for k, v in metrics.items() if k.startswith('detr_')}
    
    print('\n[Localization Metrics]')
    for k, v in sorted(loc_metrics.items()): print(f'  {k}: {v:.4f}')
    
    print('\n[GFSLT Captioning Metrics (Multi-Stage)]')
    for k, v in sorted(gfslt_metrics.items()): print(f'  {k}: {v:.4f}')
    
    print('\n[DETR Captioning Metrics (Single-Stage Baseline)]')
    for k, v in sorted(detr_metrics.items()): print(f'  {k}: {v:.4f}')
    print(f"\n{'='*60}")


def save_metrics(metrics: Dict[str, float], path: str):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f'Metrics saved to: {path}')


if __name__ == '__main__':
    main()