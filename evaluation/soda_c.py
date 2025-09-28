import evaluate
import numpy as np
from typing import List, Tuple
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