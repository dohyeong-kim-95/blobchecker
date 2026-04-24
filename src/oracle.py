"""Oracle: the only interface the algorithm has to ground truth."""

import numpy as np


class Oracle:
    """
    Wraps truth_full_mask so the algorithm cannot access it directly.
    Each call to query() counts as one iteration.
    """

    def __init__(self, truth_full_mask):
        self._mask = truth_full_mask          # (8, H, W)
        self.n_queries = 0

    def query(self, row: int, col: int) -> np.ndarray:
        """Return the 8-label vector at (row, col). Shape (8,), dtype uint8."""
        self.n_queries += 1
        return self._mask[:, row, col].copy()
