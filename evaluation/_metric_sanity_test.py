import numpy as np
from typing import List, Tuple
from .helpers import precision_recall_at_tiou


def _agg_mean(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def run_basic_checks():
    tiou = 0.5

    # Case 1: both empty → should be skipped (None, None)
    p, r = precision_recall_at_tiou([], [], tiou)
    assert p is None and r is None, f'Expected (None,None), got {(p,r)}'

    # Case 2: only GT → (0,0)
    p, r = precision_recall_at_tiou([], [(0.2, 0.4)], tiou)
    assert p == 0.0 and r == 0.0

    # Case 3: only predictions → (0,0)
    p, r = precision_recall_at_tiou([(0.1, 0.3)], [], tiou)
    assert p == 0.0 and r == 0.0

    # Case 4: perfect match → (1,1)
    p, r = precision_recall_at_tiou([(0.2, 0.4)], [(0.2, 0.4)], tiou)
    assert p == 1.0 and r == 1.0

    # Case 5: partial overlap below tiou → (0,0)
    p, r = precision_recall_at_tiou([(0.0, 0.1)], [(0.2, 0.3)], tiou)
    assert p == 0.0 and r == 0.0

    print('Metric edge cases: OK')


if __name__ == '__main__':
    run_basic_checks()
