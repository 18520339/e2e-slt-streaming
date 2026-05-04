import sys
import random
import string
import evaluate
import torch
import numpy as np
from transformers import AutoTokenizer
from bleurt.score import BleurtScorer
from typing import Dict, List, Tuple, Optional, Sequence
from config import BLEURT_CHECKPOINT_PATH, TGT_LANG
from utils import cw_to_se

# HuggingFace ROUGE/CIDEr tokenize on whitespace. Chinese has no spaces -> entire sentence becomes
# one token, ROUGE-L LCS and CIDEr n-grams collapse to 0 unless exact string match. Split chars
# (insert a space between every CJK char) before passing to those metrics. Latin scripts unchanged.
def _zh_charsplit(s: str) -> str:
    out = []
    for ch in s:
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            if out and out[-1] != ' ': out.append(' ')
            out.append(ch)
            out.append(' ')
        else: out.append(ch)
    return ''.join(out).strip()

# Language-aware sacrebleu tokenization. mBART codes map to sacrebleu tokenize values.
SACREBLEU_TOKENIZE_BY_LANG = {
    'en_XX': '13a',       # default
    'de_DE': 'intl',      # international tokenization for European text
    'zh_CN': 'zh',        # Chinese character segmentation
}
SACREBLEU_TOKENIZE = SACREBLEU_TOKENIZE_BY_LANG.get(TGT_LANG, '13a')

# BLEURT-20 supports 100+ languages including de_DE and zh_CN; the older BLEURT-base is English-only.
# Loading is the same; the user must point BLEURT_CHECKPOINT_PATH at a multilingual checkpoint for non-English.
bleu = evaluate.load('sacrebleu')  # Range: 0-100
bleurt = BleurtScorer(BLEURT_CHECKPOINT_PATH)
rouge = evaluate.load('rouge')
cider = evaluate.load('sunhill/cider')
meteor = evaluate.load('meteor')
chrf = evaluate.load('chrf') # CHRF as a tokenization-free fallback (especially useful for Chinese)


def compute_iou(pred_event: Tuple[float, float], gt_event: Tuple[float, float]) -> float:
	s1, e1 = pred_event
	s2, e2 = gt_event
	inter = max(0.0, min(e1, e2) - max(s1, s2))
	union = min(max(e1, e2) - min(s1, s2), (e1 - s1) + (e2 - s2))
	return float(inter) / (union + 1e-8)


def precision_recall_at_tiou(
	pred_events: List[Tuple[float, float]],
	gt_events: List[Tuple[float, float]],
	tiou: float,
) -> Tuple[Optional[float], Optional[float]]:
	''' Compute precision and recall at tIoU for a single window:
	- Precision = fraction of predictions that overlap any GT with IoU >= tIoU.
	- Recall    = fraction of GT covered by any prediction with IoU >= tIoU.

	Edge cases policy (to avoid inflated scores):
	- If both predictions and GT are empty: return (None, None) so caller can skip this window.
	- If predictions are empty but GT non-empty: (0.0, 0.0).
	- If predictions non-empty but GT is empty: (0.0, 0.0) since all predictions are false positives.
	'''
	if len(pred_events) == 0 and len(gt_events) == 0: return None, None # Undefined; skip in aggregation
	if len(pred_events) == 0 and len(gt_events) > 0: return 0.0, 0.0
	if len(pred_events) > 0 and len(gt_events) == 0: return 0.0, 0.0

	pred_covered = sum(1 for p in pred_events if any(compute_iou(p, g) >= tiou for g in gt_events))
	gt_covered = sum(1 for g in gt_events if any(compute_iou(p, g) >= tiou for p in pred_events))

	precision = pred_covered / len(pred_events)
	recall = gt_covered / len(gt_events)
	return precision, recall


def pairs_for_threshold(
	pred_events: List[Tuple[float, float]],
	pred_captions: List[str],
	gt_events: List[Tuple[float, float]],
	gt_captions: List[str],
	tiou: float,
) -> Tuple[List[str], List[List[str]]]:
	''' Create matched pairs at a tIoU threshold following ActivityNet logic.

	- For each prediction, add one pair per GT whose IoU >= tIoU.
	- If a prediction matches no GT, pair it with a random garbage string.

	Returns predictions, references where references is a list of single-item lists
	to match the expected shape (list[str], list[list[str]]) of HuggingFace's evaluate package.
	'''
	preds: List[str] = []
	refs: List[str] = []
	for i, p_span in enumerate(pred_events):
		matched = False
		for j, g_span in enumerate(gt_events):
			if compute_iou(p_span, g_span) >= tiou:
				preds.append(pred_captions[i].lower() if i < len(pred_captions) else '') # Defensive: if captions missing, use empty string
				refs.append(gt_captions[j].lower() if j < len(gt_captions) else '') # Defensive: if captions missing, use empty string
				matched = True
    
		# if not matched:
		# 	garbage = ' '.join(
       	# 		random.choice(string.ascii_lowercase) 
        #   		for _ in range(random.randint(10, 20))
		# 	)
		# 	preds.append(pred_captions[i])
		# 	refs.append(garbage)
	return preds, refs


def compute_text_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
	# Compute BLEU-4, BLEURT, ROUGE-L, CIDEr, METEOR, CHRF using HuggingFace's evaluate package
	# Language-aware: sacrebleu tokenization is selected from TGT_LANG (config); BLEURT/ROUGE/METEOR/CIDEr operate on
	# untokenized strings. For Chinese, ROUGE/METEOR work on chars by default; CHRF and BLEU(zh) carry the most signal.
	if len(predictions) == 0: return {'bleu4': 0.0, 'bleurt': 0.0, 'rougeL': 0.0, 'cider': 0.0, 'meteor': 0.0, 'chrf': 0.0}
	bleu_score = bleu.compute(
		predictions=predictions,
		references=[[ref] for ref in references],
		tokenize=SACREBLEU_TOKENIZE,
	)['score']
	bleurt_scores = bleurt.score(candidates=predictions, references=references)
	bleurt_score = sum(bleurt_scores) / max(1, len(bleurt_scores))

	# Char-split Chinese for whitespace-tokenizing metrics (ROUGE/CIDEr/METEOR); BLEU/CHRF unaffected.
	# rouge_score's default tokenizer strips non-ASCII via re.split(r'[^a-z0-9]+', ...) -- this deletes
	# Chinese chars entirely even after char-splitting. Pass a whitespace-splitting tokenizer to bypass.
	rc_preds = [_zh_charsplit(p) for p in predictions] if TGT_LANG == 'zh_CN' else predictions
	rc_refs  = [_zh_charsplit(r) for r in references]  if TGT_LANG == 'zh_CN' else references
	rouge_kwargs = {'tokenizer': lambda x: x.split()} if TGT_LANG == 'zh_CN' else {}

	rouge_score = rouge.compute(predictions=rc_preds, references=rc_refs, **rouge_kwargs)['rougeL']
	cider_score = cider.compute(predictions=rc_preds, references=[[ref] for ref in rc_refs])['cider_score']
	meteor_score = meteor.compute(predictions=rc_preds, references=rc_refs)['meteor']
	chrf_score = chrf.compute(predictions=predictions, references=[[ref] for ref in references])['score']
	return {
		'bleu4': float(bleu_score),    # SacreBLEU returns corpus BLEU (%) across n-gram up to 4 by default,
		'bleurt': float(bleurt_score), # Roughly between 0 and 1 (sometimes less than 0, sometimes more than 1)
		'rougeL': float(rouge_score),
		'cider': float(cider_score),
		'meteor': float(meteor_score),
		'chrf': float(chrf_score),
	}


# =============================================================================
# Shared selection / extraction / aggregation helpers used by both
# evaluation/metrics.py (HF Trainer compute_metrics) and gfslt_cascaded_eval.py
# (multi-stage DETR + GFSLT). Single source of truth -> identical localization
# numbers between StreamSLST and cascaded paths.
# =============================================================================
def select_topN_per_window(
	post_processed_outputs: List[Dict],
	pred_counts: Optional[Sequence[int]],
	tokenizer: AutoTokenizer = None,
    ranking_temperature: float = 2.0,   # Exponent T in caption score normalization by length^T
	alpha: float = 0.3, # Ranking policy: joint_score = alpha * (caption_score / len(tokens)^T) + (1 - alpha) * det_score
    top_k: int = 10,    # Max number of events to keep per window
) -> Tuple[List[List[Tuple[float, float]]], List[List[str]], List[List[float]]]:
	''' Joint-score reranking + top-N selection per window.

	Inputs are the per-window outputs of `post_process_object_detection` (already
	containing top_k candidates with event_scores/event_ranges/event_captions/
	event_caption_scores) plus a per-window count from the model's count head.

	Returns three batched lists (one entry per window):
		batch_pred_events    : list of (start, end) in [0, 1]
		batch_pred_captions  : list of decoded caption strings
		batch_pred_cap_scores: list of caption log/score floats
	'''
	batch_pred_events: List[List[Tuple[float, float]]] = []
	batch_pred_captions: List[List[str]] = []
	batch_pred_cap_scores: List[List[float]] = []

	for w_idx, pred_window in enumerate(post_processed_outputs):
		event_scores = pred_window['event_scores'].tolist() if hasattr(pred_window['event_scores'], 'tolist') else list(pred_window['event_scores'])
		event_ranges = pred_window['event_ranges'].tolist() if hasattr(pred_window['event_ranges'], 'tolist') else list(pred_window['event_ranges'])
		event_caption_scores = pred_window.get('event_caption_scores', [0.0] * len(event_scores))
		event_captions = pred_window.get('event_captions', [''] * len(event_scores))

		if len(event_scores) == 0:
			batch_pred_events.append([])
			batch_pred_captions.append([])
			batch_pred_cap_scores.append([])
			continue

		cap_norm = [ # Normalize caption score by length^T to discourage verbosity
			c / (max(1, len(tokenizer.encode(t))) ** ranking_temperature + 1e-5)
			for c, t in zip(event_caption_scores, event_captions)
		]
		joint = [alpha * c + (1 - alpha) * s for c, s in zip(cap_norm, event_scores)]
		order = list(np.argsort(joint)[::-1]) # Descending order

		if pred_counts is not None and w_idx < len(pred_counts): # If pred_counts provided, use it
			keep = int(pred_counts[w_idx])
		else: # Otherwise, use top_k
			keep = min(top_k, len(order))
		keep = max(0, min(keep, top_k, len(order))) # Clamp to valid range
		chosen_event_ids = order[:keep]

		batch_pred_events.append([tuple(event_ranges[i]) for i in chosen_event_ids])
		batch_pred_captions.append([event_captions[i] for i in chosen_event_ids])
		batch_pred_cap_scores.append([float(event_caption_scores[i]) for i in chosen_event_ids])
	return batch_pred_events, batch_pred_captions, batch_pred_cap_scores


def extract_gt_per_window(labels: List[Dict], tokenizer) -> Tuple[List[List[Tuple[float, float]]], List[List[str]]]:
	''' Convert per-window label dicts (boxes in cw + seq_tokens) into batched
	(events_se, captions) lists. Mirrors the ground-truth construction that was
	previously duplicated in metrics.py and gfslt_cascaded_eval.py.
	'''
	batch_gt_events: List[List[Tuple[float, float]]] = []
	batch_gt_captions: List[List[str]] = []

	for window in labels: # List of {'class_labels': (N_i, ), 'boxes': (N_i, 2), 'seq_tokens': (N_i, L)}
		gt_boxes_cw = window.get('boxes', []) # (N, 2)
		gt_boxes_cw = gt_boxes_cw if isinstance(gt_boxes_cw, torch.Tensor) else torch.as_tensor(gt_boxes_cw)
		gt_boxes_se = cw_to_se(gt_boxes_cw) if gt_boxes_cw.numel() else gt_boxes_cw
		gt_events = [tuple(map(float, box.tolist())) for box in gt_boxes_se]

		seq_tokens = window.get('seq_tokens', [])
		if hasattr(seq_tokens, 'numpy'): seq_arr = seq_tokens.numpy()
		else: seq_arr = np.asarray(seq_tokens) if len(seq_tokens) else np.empty((0, 0), dtype=np.int64)

		texts = tokenizer.batch_decode( # Decode all at once
			np.where(seq_arr == -100, tokenizer.pad_token_id, seq_arr), # Replace -100 (used by HF) with pad token id
			skip_special_tokens=True, clean_up_tokenization_spaces=True
		) if seq_arr.size else []

		# Keep aligned to boxes count (truncate if mismatch)
		m = min(len(gt_events), len(texts))
		batch_gt_events.append(gt_events[:m])
		batch_gt_captions.append(texts[:m])
	return batch_gt_events, batch_gt_captions


def aggregate_metrics(
	batch_pred_events: List[List[Tuple[float, float]]],
	batch_pred_captions: List[List[str]],
	batch_gt_events: List[List[Tuple[float, float]]],
	batch_gt_captions: List[List[str]],
	temporal_iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
	prefix: str = '',
	include_localization: bool = True,
	include_paragraph: bool = True,
	include_segment: bool = True,
	soda_recursion_limit: int = 0,
) -> Dict[str, float]:
	''' Compute precision/recall/F1 + dense captioning + paragraph + segment metrics.

	`prefix` controls metric naming. Empty prefix yields `loc_*`, `dense_*`, etc.
	(matches metrics.py output). Non-empty prefix yields e.g. `gfslt_dense_*` for
	the cascaded path so two caption sources can coexist in one results dict.

	Localization metrics are computed once per call. Cascaded path should call this twice with 
	`include_localization=True` only on the first call to avoid duplicate loc_* keys.
	'''
	from .soda_c import compute_soda_at_tiou
	def _name(stem: str) -> str: return f'{prefix}_{stem}' if prefix else stem

	metrics: Dict[str, float] = {}
	precs, recs = [], []
	dense_keys = ['bleu4', 'bleurt', 'rougeL', 'cider', 'meteor', 'chrf']
	dense_scores = {k: [] for k in dense_keys}

	use_soda = soda_recursion_limit > 0 # Python use 1000 by default, increase if needed for SODA_c
	if use_soda:
		if soda_recursion_limit <= sys.getrecursionlimit():
			raise ValueError(f'soda_recursion_limit ({soda_recursion_limit}) must be > current ({sys.getrecursionlimit()})')
		sys.setrecursionlimit(soda_recursion_limit)
		dense_scores['soda_c'] = []

	for tiou in temporal_iou_thresholds:
		precs_at_tiou, recs_at_tiou = [], []
		preds_at_tiou, refs_at_tiou = [], []
		soda_f1s_at_tiou = []

		for pred_events, pred_captions, gt_events, gt_captions in zip(batch_pred_events, batch_pred_captions, batch_gt_events, batch_gt_captions):
			if include_localization: # Localization metrics (precision/recall per window)
				p, r = precision_recall_at_tiou(pred_events, gt_events, tiou)
				if p is not None and r is not None: # Skip windows where both GT and predictions are empty (p, r) = (None, None)
					precs_at_tiou.append(p); recs_at_tiou.append(r)

			# Dense captioning metrics (pred/gt pairs per window)
			pp, rr = pairs_for_threshold(pred_events, pred_captions, gt_events, gt_captions, tiou)
			preds_at_tiou.extend(pp); refs_at_tiou.extend(rr)
			if use_soda: # SODA_c-like storytelling score (DP over IoU-masked METEOR similarity)
				if len(pred_events) == 0 or len(gt_events) == 0: soda_f1s_at_tiou.append(0.0)
				else: soda_f1s_at_tiou.append(compute_soda_at_tiou(pred_events, pred_captions, gt_events, gt_captions, tiou))

		if include_localization: # Localization metrics
			p_avg = float(np.mean(precs_at_tiou)) if precs_at_tiou else 0.0
			r_avg = float(np.mean(recs_at_tiou)) if recs_at_tiou else 0.0
			f1 = 2 * p_avg * r_avg / (p_avg + r_avg) if (p_avg + r_avg) > 0 else 0.0
			precs.append(p_avg); recs.append(r_avg)
			metrics[f'loc_precision@{tiou * 100:.0f}'] = p_avg
			metrics[f'loc_recall@{tiou * 100:.0f}']    = r_avg
			metrics[f'loc_f1@{tiou * 100:.0f}']        = f1

		# Dense captioning metrics
		text_metrics = compute_text_metrics(preds_at_tiou, refs_at_tiou)
		for mname in dense_scores:
			if mname in text_metrics: dense_scores[mname].append(text_metrics[mname])
			elif mname == 'soda_c' and use_soda: dense_scores[mname].append(float(np.mean(soda_f1s_at_tiou)) if soda_f1s_at_tiou else 0.0)
			else: dense_scores[mname].append(0.0) # Metric not available
			metrics[_name(f'dense_{mname}@{tiou * 100:.0f}')] = dense_scores[mname][-1]

	if include_localization: # Average localization metrics across IoU thresholds
		loc_p = float(np.mean(precs)) if precs else 0.0
		loc_r = float(np.mean(recs)) if recs else 0.0
		metrics['loc_precision_avg'] = loc_p
		metrics['loc_recall_avg']    = loc_r
		metrics['loc_f1_avg'] = 2 * loc_p * loc_r / (loc_p + loc_r) if (loc_p + loc_r) > 0 else 0.0

	# Average Dense captioning metrics across IoU thresholds
	for mname, vals in dense_scores.items():
		metrics[_name(f'dense_{mname}_avg')] = float(np.mean(vals)) if vals else 0.0

	if include_segment or include_paragraph: # Overall Segmentation and Paragraph-level metrics
		segment_aligns: List[float] = []     # Segment alignment accuracy or the ratio of predicted segments to ground truth segments
		segment_overlaps: List[float] = []   # Average IoU to indicate the model's ability to capture precise segment boundaries
		para_preds: List[str] = []           # Joined predicted captions per window
		para_refs: List[str] = []            # Joined ground-truth captions per window
		for pred_events, pred_captions, gt_events, gt_captions in zip(batch_pred_events, batch_pred_captions, batch_gt_events, batch_gt_captions):
			if include_segment:
				segment_aligns.append(len(pred_events) / len(gt_events) if len(gt_events) > 0 else 0)
				segment_overlaps.append(
					float(np.mean([max([compute_iou(p, g) for g in gt_events], default=0.0) for p in pred_events]))
					if pred_events and gt_events else 0.0
				)
			if include_paragraph: # Join captions into paragraphs
				idx_p = list(np.argsort([s for s, _ in pred_events])) if pred_events else []
				idx_g = list(np.argsort([s for s, _ in gt_events])) if gt_events else []
				para_preds.append(' '.join(pred_captions[i] for i in idx_p).strip())
				para_refs.append(' '.join(gt_captions[i] for i in idx_g).strip())

		if include_segment and include_localization:
			metrics['segment_alignment'] = float(np.mean(segment_aligns)) 	if segment_aligns else 0.0
			metrics['segment_overlap']   = float(np.mean(segment_overlaps)) if segment_overlaps else 0.0
		if include_paragraph:
			para_scores = compute_text_metrics(para_preds, para_refs)
			for mname, v in para_scores.items(): metrics[_name(f'para_{mname}')] = v
	return metrics