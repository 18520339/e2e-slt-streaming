''' Metrics utilities for Hugging Face Trainer integration. It evaluates 3 aspects:

1) Localization quality (temporal detection):
   - Precision/Recall/F1 averaged across IoU thresholds {0.3, 0.5, 0.7, 0.9}.

2) Dense captioning quality:
   - Following ActivityNet DVC style: for each IoU threshold, form matched
	 (pred, gt) caption pairs by temporal overlap, compute translation metrics
	 using Hugging Face's `evaluate` where available, then average scores across thresholds.
   - Additionally, compute SODA_c-like overall storytelling F1 using a
	 dynamic-programming assignment over (IoU-masked) caption similarities.

3) Paragraph-level captioning quality:
   - For each window, sort predicted captions by start time and join them into a paragraph;
	 compare against ground-truth paragraphs aggregated the same way; report translation metrics.

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
from dataclasses import dataclass
from typing import Sequence, Dict
from transformers import AutoTokenizer, EvalPrediction
from postprocess import post_process_object_detection
from .helpers import select_topN_per_window, extract_gt_per_window, aggregate_metrics


@dataclass
class ModelOutput:
    logits: torch.FloatTensor
    pred_boxes: torch.FloatTensor
    pred_counts: torch.FloatTensor
    pred_cap_logits: torch.FloatTensor
    pred_cap_tokens: torch.LongTensor


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
    top_k: int = 10,    # Max number of events to keep per window
	temporal_iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
    tokenizer: AutoTokenizer = None,
    soda_recursion_limit: int = 0, # Increase recursion limit for SODA_c DP if needed, 0 to disable for faster calculations
    aggregation_mode: str = 'corpus', # 'corpus' | 'window' | 'video'. Same metric keys regardless.
    eval_windows: list = None,  # dataset.eval_windows list, in dataloader order. Required for aggregation_mode='video'.
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
    pred_counts = predictions.pred_counts.argmax(dim=-1).clamp(min=0).tolist() if predictions.pred_counts is not None else None
    batch_pred_events, batch_pred_captions, _ = select_topN_per_window(
        post_processed_outputs, pred_counts, tokenizer,
        ranking_temperature=ranking_temperature, alpha=alpha, top_k=top_k,
    )
    batch_gt_events, batch_gt_captions = extract_gt_per_window(evaluation_results.label_ids, tokenizer)
    # Only build batch_video_ids when actually needed (aggregation_mode='video'). Corpus and
    # window modes ignore it, so don't risk a length-mismatch crash on those code paths.
    # eval_windows is dataset.eval_windows (sequential dataloader order); align to the actual
    # number of predicted windows, not to label_ids (Trainer's label_ids container length can
    # differ from the per-window prediction count depending on how it concatenated batches).
    batch_video_ids = None
    if aggregation_mode == 'video' and eval_windows is not None:
        n = len(batch_pred_events)
        ews = list(eval_windows)[:n]
        batch_video_ids = [w.get('video_id') if isinstance(w, dict) else None for w in ews]
        if len(batch_video_ids) < n:
            print(f'[compute_metrics] WARNING: only {len(batch_video_ids)} eval_windows for {n} predictions; '
                  f'aggregation_mode=video may misalign — falling back to corpus mode.')
            aggregation_mode = 'corpus'
            batch_video_ids = None
    return aggregate_metrics(
        batch_pred_events, batch_pred_captions, batch_gt_events, batch_gt_captions,
        temporal_iou_thresholds=temporal_iou_thresholds,
        prefix='', include_localization=True, include_paragraph=True, include_segment=True,
        soda_recursion_limit=soda_recursion_limit,
        aggregation_mode=aggregation_mode, batch_video_ids=batch_video_ids,
    )
