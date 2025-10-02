import evaluate
import numpy as np
from typing import List, Tuple
from .helpers import compute_iou
meteor = evaluate.load('meteor')


def chased_dp_assignment(scores: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
	'''Dynamic programming assignment maximizing total score.
 
	The original implementation's backtracking logic can fail with an IndexError in edge cases 
	(e.g., when there are no matched pairs, it returns an empty list without errors), 
	as demonstrated by testing with negative scores where no pairs are selected. 
 	This version reliably recovers 1 valid set of pairs even in ties or empty scenarios

	Recurrence: dp[i, j] = max(dp[i-1, j], dp[i, j-1], dp[i-1, j-1] + scores[i, j])
	Returns (max_score, pairs)
	'''
	if scores.size == 0: return 0.0, []
	M, N = scores.shape
	dp = -np.ones((M, N), dtype=float)
	path = np.zeros((M, N), dtype=np.int8)

	def transition(i: int, j: int) -> float:
		if dp[i, j] >= 0: return dp[i, j]
		if i == 0 and j == 0: candidates = (-1.0, -1.0, scores[i, j])
		elif i == 0: candidates = (-1.0, transition(i, j - 1), scores[i, j])
		elif j == 0: candidates = (transition(i - 1, j), -1.0, scores[i, j])
		else: candidates = (
			transition(i - 1, j),
			transition(i, j - 1),
			transition(i - 1, j - 1) + scores[i, j],
		)
		dp[i, j] = float(max(candidates))
		path[i, j] = int(np.argmax(candidates))
		return dp[i, j]

	def backtrack(i: int, j: int) -> List[Tuple[int, int]]:
		if i < 0 or j < 0: return []
		move = path[i, j]
		if move == 2: return backtrack(i - 1, j - 1) + [(i, j)] # Diag
		if move == 0: return backtrack(i - 1, j) # Up
		return backtrack(i, j - 1) # Left

	max_score = transition(M - 1, N - 1)
	pairs = backtrack(M - 1, N - 1)
	return max_score, pairs


def meteor_similarity_matrix(predictions: List[str], references: List[str]) -> np.ndarray:
	# meteor.compute expects lists; get pairwise by calling per-pair
	M, N = len(references), len(predictions)
	mat = np.zeros((M, N), dtype=float)
	for i, r in enumerate(references):
		for j, p in enumerate(predictions):
			s = meteor.compute(predictions=[p], references=[[r]]).get('meteor', 0.0)
			mat[i, j] = float(s)
	return mat # |references| x |predictions| matrix of pairwise scores


def compute_soda_at_tiou(
	pred_events: List[Tuple[float, float]], pred_captions: List[str],
	gt_events: List[Tuple[float, float]], gt_captions: List[str],
	temporal_iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
	'''Compute SODA-C F1 score for a single video.
 
	Args:
		pred_events (`List[Tuple[float, float]]`):
			Predicted event time intervals as (start, end) in seconds.
		pred_captions (`List[str]`):
			Predicted captions corresponding to `pred_events`.
		gt_events (`List[Tuple[float, float]]`):
			Ground truth event time intervals as (start, end) in seconds.
		gt_captions (`List[str]`):
			Ground truth captions corresponding to `gt_events`.
		temporal_iou_threshold (`float`, *optional*, defaults to 0.5):
			Temporal IoU threshold to consider a predicted event matching a ground truth event.

	Returns:
		Tuple containing (precision, recall, f1_score).
	'''
	# Sort events and captions by their start timestamps
	idx_pred = list(np.argsort([s for s, _ in pred_events])) if pred_events else []
	idx_gt = list(np.argsort([s for s, _ in gt_events])) if gt_events else []
	pred_events_sorted = [pred_events[i] for i in idx_pred]
	pred_captions_sorted = [pred_captions[i] for i in idx_pred]
	gt_events_sorted = [gt_events[i] for i in idx_gt]
	gt_captions_sorted = [gt_captions[i] for i in idx_gt]
 
	# Compute IoU and similarity matrices
	iou_mat = np.array([[compute_iou(p, g) for p in pred_events_sorted] for g in gt_events_sorted], dtype=float)
	iou_mat[iou_mat < temporal_iou_threshold] = 0.0 # IoU mask
	sim_mat = meteor_similarity_matrix(pred_captions_sorted, gt_captions_sorted) # Shape (G, P)

	score_mat = iou_mat * sim_mat
	max_score, _pairs = chased_dp_assignment(score_mat)
	n_g, n_p = score_mat.shape
	p = max_score / max(1, n_p)
	r = max_score / max(1, n_g)
	return 2 * p * r / (p + r) if (p + r) > 0 else 0.0