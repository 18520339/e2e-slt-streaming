import random
import string
import evaluate
from typing import Dict, List, Tuple, Optional

bleu = evaluate.load('sacrebleu') # Range: 0-100
bleurt = evaluate.load('bleurt', module_type='metric', checkpoint='BLEURT-20')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')
cider = evaluate.load('Kamichanw/CIDEr')


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
	''' Compute precision and recall at tiou for a single window:
	- Precision = fraction of predictions that overlap any GT with IoU >= tiou.
	- Recall    = fraction of GT covered by any prediction with IoU >= tiou.

	Edge cases policy (to avoid inflated scores):
	- If both predictions and GT are empty: return (None, None) so caller can skip this window.
	- If predictions are empty but GT non-empty: (0.0, 0.0).
	- If predictions non-empty but GT is empty: (0.0, 0.0) since all predictions are false positives.
	'''
	if len(pred_events) == 0 and len(gt_events) == 0: return None, None # Undefined; skip in aggregation
	if len(pred_events) == 0 and len(gt_events) > 0: return 0.0, 0.0
	if len(pred_events) > 0 and len(gt_events) == 0: return 0.0, 0.0

	pred_covered, gt_covered = 0, 0
	for p in pred_events: # Pred coverage
		if any(compute_iou(p, g) >= tiou for g in gt_events):
			pred_covered += 1
	precision = pred_covered / len(pred_events)
	
	for g in gt_events: # GT coverage
		if any(compute_iou(p, g) >= tiou for p in pred_events):
			gt_covered += 1
	recall = gt_covered / len(gt_events)
	return precision, recall


def pairs_for_threshold(
	pred_events: List[Tuple[float, float]],
	pred_captions: List[str],
	gt_events: List[Tuple[float, float]],
	gt_captions: List[str],
	tiou: float,
) -> Tuple[List[str], List[List[str]]]:
	''' Create matched pairs at a tiou threshold following ActivityNet logic.

	- For each prediction, add one pair per GT whose IoU >= tiou.
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
				preds.append(pred_captions[i])
				refs.append(gt_captions[j])
				matched = True
    
		if not matched:
			garbage = ' '.join(
       			random.choice(string.ascii_lowercase) 
          		for _ in range(random.randint(10, 20))
			)
			preds.append(pred_captions[i])
			refs.append(garbage)
	return preds, refs


def compute_text_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
	# Compute BLEU-4, BLEURT, ROUGE-L, METEOR, CIDEr, Exact Match, using HuggingFace's evaluate package for consistency
	if len(predictions) == 0:  return {'bleu4': 0.0, 'bleurt': 0.0, 'rougeL': 0.0, 'meteor': 0.0, 'cider': 0.0, 'exact_match': 0.0}
	bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])['score']
	bleurt_score = bleurt.compute(predictions=predictions, references=references)['scores']
	bleurt_score = sum(bleurt_score) / max(1, len(bleurt_score))
	
	rouge_score = rouge.compute(predictions=predictions, references=references)['rougeL']
	cider_score = cider.compute(predictions=predictions, references=[[ref] for ref in references])['CIDEr']
	meteor_score = meteor.compute(predictions=predictions, references=references)['meteor']
	exact_match = sum(p == g[0] for p, g in zip(predictions, references)) / max(1, len(references))
	return {
		'bleu4': float(bleu_score),    # SacreBLEU returns corpus BLEU (%) across n-gram up to 4 by default,
		'bleurt': float(bleurt_score), # Roughly between 0 and 1 (sometimes less than 0, sometimes more than 1)
		'rougeL': float(rouge_score),  
		'meteor': float(meteor_score), 
		'cider': float(cider_score),   # https://github.com/huggingface/evaluate/pull/613/files
		'exact_match': float(exact_match),
	}