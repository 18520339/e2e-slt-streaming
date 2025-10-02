''' Metrics utilities for Hugging Face Trainer integration. It evaluates 3 aspects:

1) Localization quality (temporal detection):
   - Precision/Recall/F1 averaged across IoU thresholds {0.3, 0.5, 0.7, 0.9}.

2) Dense captioning quality:
   - Following ActivityNet DVC style: for each IoU threshold, form matched
	 (pred, gt) caption pairs by temporal overlap, compute BLEU-4, METEOR, and CIDEr 
	 using Hugging Face's `evaluate` where available, then average scores across thresholds.
   - Additionally, compute SODA_c-like overall storytelling F1 using a
	 dynamic-programming assignment over (IoU-masked) caption similarities.

3) Paragraph-level captioning quality:
   - For each window, sort predicted captions by start time and join them into a paragraph; 
	 compare against ground-truth paragraphs aggregated the same way; report BLEU-4, METEOR, and CIDEr.

Expected inputs from Trainer (with eval_do_concat_batches=True, batch_eval_metrics=False):
- evaluation_results.predictions: either a dict-like object with keys
  ['logits','pred_boxes','pred_counts','pred_cap_logits','pred_cap_tokens'] or
  a tuple/list in the same order. Arrays should be shaped similarly to the
  model outputs in `DeformableDetrObjectDetectionOutput`.

- evaluation_results.label_ids: a Python list (len=batch_size) of dicts per window:
  {'boxes': FloatTensor [N_i, 2] (center, width, normalized 0..1),
   'seq_tokens': LongTensor [N_i, L] (token IDs, padded with pad/eos)}

Notes:
- We use `post_process_object_detection` to obtain top-k predictions per
  window, and then perform re-ranking by combining localization and caption
  scores. The number of events kept per window defaults to the model's
  predicted count (argmax over pred_counts), with an upper bound of `top_k`.
'''
import torch
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Union, Dict, List, Tuple
from transformers import AutoTokenizer, EvalPrediction

from utils import cw_to_se
from postprocess import post_process_object_detection
from .helpers import (
    precision_recall_at_tiou,
    pairs_for_threshold,
    compute_text_metrics
)
from .soda_c import compute_soda_at_tiou


@dataclass
class ModelOutput:
    logits: torch.FloatTensor
    pred_boxes: torch.FloatTensor
    pred_counts: torch.FloatTensor
    pred_cap_logits: torch.FloatTensor
    pred_cap_tokens: torch.FloatTensor


def preprocess_logits_for_metrics(logits_tuple, labels):
    # https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/29
    logits = logits_tuple[1].detach().cpu()
    pred_boxes = logits_tuple[2].detach().cpu()
    pred_counts = logits_tuple[3].detach().cpu()
    pred_cap_logits = logits_tuple[4].detach().cpu()
    pred_cap_tokens = logits_tuple[5].detach().cpu()
    return logits, pred_boxes, pred_counts, pred_cap_logits, pred_cap_tokens


def compute_metrics(
    evaluation_results: EvalPrediction, # EvalPrediction will be the whole dataset (a big batch of concatenated batches)
    ranking_temperature: float = 2.0,   # Exponent T in caption score normalization by length^T
	alpha: float = 0.3, # Ranking policy: joint_score = alpha * (caption_score / len(tokens)^T) + (1 - alpha) * det_score
    top_k: int = 10,    # Should be num_queries during training
	temporal_iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
    tokenizer: AutoTokenizer = None,
) -> Dict[str, float]:
    predictions = ModelOutput(
        logits=torch.as_tensor(evaluation_results.predictions[0]),
        pred_boxes=torch.as_tensor(evaluation_results.predictions[1]),
        pred_counts=torch.as_tensor(evaluation_results.predictions[2]),
        pred_cap_logits=torch.as_tensor(evaluation_results.predictions[3]),
        pred_cap_tokens=torch.as_tensor(evaluation_results.predictions[4]),
    )
    # Postprocess to get top-k per window, plus caption texts/scores
    post_processed_outputs = post_process_object_detection(
        outputs=predictions,
        top_k=top_k,
        threshold=0.0,       # We'll select top via count head + reranking
        target_lengths=None, # Keep relative [0, 1] boxes for IoU computation
        tokenizer=tokenizer,
    )
    # Build per-window prediction lists with reranking and topN selection
    batch_pred_events: List[List[Tuple[float, float]]] = []
    batch_pred_captions: List[List[str]] = []

    # Number of events predicted per window from count head
    if predictions.pred_counts is not None:
        topNs = predictions.pred_counts.argmax(dim=-1).clamp(min=0).tolist()
    else:
        topNs = [min(top_k, len(p['scores'])) for p in post_processed_outputs]

    for pred_window_idx, pred_window in enumerate(post_processed_outputs):
        event_scores = pred_window['event_scores'].tolist()
        event_ranges = pred_window['event_ranges'].tolist()  # (start, end)
        event_caption_scores = pred_window.get('event_caption_scores', [0.0] * len(event_scores))
        event_captions = pred_window.get('event_captions', [''] * len(event_scores))

        cap_norm = [ # Normalize caption score by length^T to discourage verbosity
            c / (max(1, len(tokenizer.encode(t))) ** ranking_temperature + 1e-5) 
            for c, t in zip(event_caption_scores, event_captions)
        ]
        joint = [alpha * c + (1 - alpha) * s for c, s in zip(cap_norm, event_scores)]
        order = list(np.argsort(joint)[::-1]) # Descending order

        if pred_window_idx < len(topNs): # If pred_counts provided, use it
            keep = int(topNs[pred_window_idx]) 
        else: # Otherwise, use top_k
            keep = min(top_k, len(order))  
            
        keep = max(0, min(keep, len(order))) # Clamp to valid range
        chosen_event_ids = order[:keep]
        batch_pred_events.append([tuple(event_ranges[i]) for i in chosen_event_ids])
        batch_pred_captions.append([event_captions[i] for i in chosen_event_ids])

    # Extract ground truth from label_ids
    labels = evaluation_results.label_ids # List of {'class_labels': (N_i, ), 'boxes': (N_i, 2), 'seq_tokens': (N_i, L)}
    batch_gt_events: List[List[Tuple[float, float]]] = []
    batch_gt_captions: List[List[str]] = []

    for window in labels:
        gt_boxes_cw = window.get('boxes', [])  # (N, 2)
        gt_boxes_cw = gt_boxes_cw if isinstance(gt_boxes_cw, torch.Tensor) else torch.as_tensor(gt_boxes_cw)
        gt_boxes_se = cw_to_se(gt_boxes_cw) if gt_boxes_cw.numel() else gt_boxes_cw
        gt_events = [tuple(map(float, box.tolist())) for box in gt_boxes_se]
        
        seq_tokens = window.get('seq_tokens', [])
        texts = tokenizer.batch_decode( # Decode all at once
            np.where(seq_tokens == -100, tokenizer.pad_token_id, seq_tokens), # Replace -100 (used by HF) with pad token id
            skip_special_tokens=True, clean_up_tokenization_spaces=True
        ) if len(seq_tokens) else []
        
        # Keep aligned to boxes count (truncate if mismatch)
        m = min(len(gt_events), len(texts))
        batch_gt_events.append(gt_events[:m])
        batch_gt_captions.append(texts[:m])

    # Calculate Localization and Dense captioning metrics across IoU thresholds
    metrics: Dict[str, float] = {}
    precs, recs = [], []
    dense_scores_accum = {'bleu4': [], 'meteor': [], 'cider': [], 'soda_c': []}
    
    for tiou in temporal_iou_thresholds: 
        precisions_at_tiou, recalls_at_tiou = [], []
        all_preds_at_tiou: List[str] = []
        all_refs_at_tiou: List[List[str]] = []
        soda_f1s_at_tiou = [] 
        
        for pred_events, pred_captions, gt_events, gt_captions in zip(
            batch_pred_events, batch_pred_captions, batch_gt_events, batch_gt_captions
        ):
            # Localization metrics (precision/recall per window)
            p, r = precision_recall_at_tiou(pred_events, gt_events, tiou) 
            precisions_at_tiou.append(p)
            recalls_at_tiou.append(r)
            
            # Dense captioning metrics (pred/gt pairs per window)
            preds, refs = pairs_for_threshold(pred_events, pred_captions, gt_events, gt_captions, tiou) 
            all_preds_at_tiou.extend(preds)
            all_refs_at_tiou.extend(refs)
            
            if len(pred_events) == 0 or len(gt_events) == 0:
                soda_f1s_at_tiou.append(0.0)
                continue
            
            # SODA_c-like storytelling score (DP over IoU-masked METEOR similarity)
            f1_tiou = compute_soda_at_tiou(pred_events, pred_captions, gt_events, gt_captions, tiou)
            soda_f1s_at_tiou.append(f1_tiou)
        
        # Localization metrics
        precs.append(float(np.mean(precisions_at_tiou) if precisions_at_tiou else 0.0))
        recs.append(float(np.mean(recalls_at_tiou) if recalls_at_tiou else 0.0))
        metrics[f'loc_precision@{tiou * 100:.0f}'] = precs[-1]
        metrics[f'loc_recall@{tiou * 100:.0f}'] = recs[-1]
        metrics[f'loc_f1@{tiou * 100:.0f}'] = 2 * precs[-1] * recs[-1] / (precs[-1] + recs[-1]) if (precs[-1] + recs[-1]) > 0 else 0.0
        
        # Dense captioning metrics
        text_metrics = compute_text_metrics(all_preds_at_tiou, all_refs_at_tiou)
        for metric_name in dense_scores_accum:
            if metric_name in text_metrics:
                dense_scores_accum[metric_name].append(text_metrics.get(metric_name, 0.0))
            elif metric_name == 'soda_c':
                dense_scores_accum[metric_name].append(float(np.mean(soda_f1s_at_tiou) if soda_f1s_at_tiou else 0.0))
            else:
                dense_scores_accum[metric_name].append(0.0) # Metric not available
        metrics[f'dense_{metric_name}@{tiou * 100:.0f}'] = dense_scores_accum[metric_name][-1]
            
    # Average localization metrics across IoU thresholds
    loc_precision = float(np.mean(precs) if precs else 0.0)
    loc_recall = float(np.mean(recs) if recs else 0.0)
    metrics['loc_precision_avg'] = loc_precision
    metrics['loc_recall_avg'] = loc_recall
    metrics['loc_f1_avg'] = 2 * loc_precision * loc_recall / (loc_precision + loc_recall) if (loc_precision + loc_recall) > 0 else 0.0
    
    # Average Dense captioning metrics across IoU thresholds
    metrics['dense_bleu4_avg'] = float(np.mean(dense_scores_accum['bleu4'])) if dense_scores_accum['bleu4'] else 0.0
    metrics['dense_meteor_avg'] = float(np.mean(dense_scores_accum['meteor'])) if dense_scores_accum['meteor'] else 0.0
    metrics['dense_cider_avg'] = float(np.mean(dense_scores_accum['cider'])) if dense_scores_accum['cider'] else 0.0
    metrics['soda_c_avg'] = float(np.mean(dense_scores_accum['soda_c'])) if dense_scores_accum['soda_c'] else 0.0

    # Paragraph-level metrics
    para_preds: List[str] = []
    para_refs: List[List[str]] = []
    for pred_events, pred_captions, gt_events, gt_captions in zip(
        batch_pred_events, batch_pred_captions, batch_gt_events, batch_gt_captions
    ):
        # Sort by start time
        idx_pred = list(np.argsort([s for s, _ in pred_events])) if pred_events else []
        idx_gt = list(np.argsort([s for s, _ in gt_events])) if gt_events else []
        para_pred = '. '.join([pred_captions[i] for i in idx_pred]).strip()
        para_gt = '. '.join([gt_captions[i] for i in idx_gt]).strip()
        para_preds.append(para_pred if para_pred else '')
        para_refs.append([para_gt if para_gt else ''])  # single reference

    para_scores = compute_text_metrics(para_preds, para_refs)
    metrics['para_bleu4'] = para_scores.get('bleu4', 0.0)
    metrics['para_meteor'] = para_scores.get('meteor', 0.0)
    metrics['para_cider'] = para_scores.get('cider', 0.0)
    return metrics