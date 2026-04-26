"""
Pre-planned greedy coverage baseline.

Query strategy
--------------
All 1,500 query coordinates are chosen *before* execution begins by greedily
maximising spatial coverage: each step picks the unselected coordinate whose
(2r+1)×(2r+1) neighbourhood contributes the most new (previously uncovered)
pixels.  This is the standard greedy solution to a monotone submodular
coverage function, which guarantees ≥ (1 − 1/e) ≈ 63 % of the optimal
coverage.  The convolution trick makes one greedy step O(H·W·log(H·W))
instead of O(H²·W²).

Reconstruction strategy
-----------------------
After every query, predict() rebuilds each layer's mask using row-wise
scanline fill:
  • For each row, span from the leftmost to the rightmost observed positive.
  • Interior observed zeros are treated as potential hole centres and a
    neighbourhood around them is cleared.
  • Rows that have no observations yet are filled from the nearest row that
    does have observations.

This is a baseline — it deliberately keeps reconstruction simple to isolate
the effect of the query strategy.  It may fall short of 98 % accuracy on
blobs with large interior holes, because holes are only detected where the
pre-planned queries happen to land inside them.
"""

import numpy as np
from scipy.signal import fftconvolve

from .base import BaseAlgorithm
from .reconstruct import scanline_reconstruct


# ---------------------------------------------------------------------------
# Pre-planning
# ---------------------------------------------------------------------------

def _preplan_greedy(H: int, W: int, budget: int, radius: int) -> list:
    """
    Return a list of (row, col) tuples — the pre-planned query schedule.

    Greedy coverage maximisation with a square neighbourhood of half-width
    `radius`.  Once every pixel is covered the remaining budget is filled with
    a row-by-row raster scan of pixels not yet selected, which ensures every
    pixel is queried at least once if budget ≥ H·W.
    """
    covered = np.zeros((H, W), dtype=np.float32)
    selected = np.zeros((H, W), dtype=bool)
    schedule = []

    ksize = 2 * radius + 1
    kernel = np.ones((ksize, ksize), dtype=np.float32)

    for _ in range(budget):
        uncovered = 1.0 - covered           # float, so fftconvolve works
        gain = fftconvolve(uncovered, kernel, mode="same")
        gain[selected] = -1.0              # already-selected points ineligible

        best = int(gain.argmax())
        r, c = divmod(best, W)
        schedule.append((r, c))
        selected[r, c] = True

        r0 = max(0, r - radius)
        r1 = min(H, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(W, c + radius + 1)
        covered[r0:r1, c0:c1] = 1.0

    return schedule


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

class PreplannedGreedy(BaseAlgorithm):
    """
    Pre-planned greedy coverage algorithm — Phase 0 baseline.

    Parameters
    ----------
    H, W    : grid dimensions
    budget  : number of iterations (= iteration_cap)
    radius  : coverage neighbourhood half-width for greedy preplan
    """

    def __init__(self, H: int, W: int, budget: int, radius: int = 1):
        super().__init__(H, W, budget)
        import time
        print(f"  [PreplannedGreedy] preplanning {budget} queries "
              f"on {H}×{W} grid (radius={radius}) …", end=" ", flush=True)
        t0 = time.perf_counter()
        self._schedule = _preplan_greedy(H, W, budget, radius)
        print(f"done in {time.perf_counter()-t0:.2f}s", flush=True)
        self._step = 0

        # Observation storage
        self._obs_mask = np.zeros((H, W), dtype=bool)
        self._obs_labels = np.zeros((8, H, W), dtype=np.uint8)

        # Cache: only rebuilt when new observations arrive
        self._pred_cache: np.ndarray | None = None
        self._cache_dirty = True

    # ------------------------------------------------------------------
    def next_query(self) -> tuple[int, int]:
        return self._schedule[self._step]

    def update(self, row: int, col: int, labels: np.ndarray) -> None:
        self._step += 1
        if not self._obs_mask[row, col]:   # ignore duplicate queries
            self._obs_mask[row, col] = True
            self._obs_labels[:, row, col] = labels
            self._cache_dirty = True

    def predict(self) -> np.ndarray:
        if not self._cache_dirty and self._pred_cache is not None:
            return self._pred_cache

        predicted = scanline_reconstruct(
            self.H, self.W, self._obs_mask, self._obs_labels
        )
        self._pred_cache = predicted
        self._cache_dirty = False
        return predicted
