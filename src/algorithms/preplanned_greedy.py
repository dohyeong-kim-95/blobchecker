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
from scipy.ndimage import label as ndlabel

from .base import BaseAlgorithm

# Hole-suppression half-width: when an observed 0 sits inside a positive
# scanline span, we clear a ±HOLE_HALF pixels neighbourhood around it.
HOLE_HALF = 4


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
# Reconstruction helpers
# ---------------------------------------------------------------------------

def _reconstruct_layer(
    H: int,
    W: int,
    obs_mask: np.ndarray,      # (H, W) bool
    obs_labels_k: np.ndarray,  # (H, W) uint8, valid only where obs_mask
) -> np.ndarray:
    """
    Row-wise scanline fill for one layer.

    For each row:
      1. Find the leftmost and rightmost observed positive.
      2. Fill [left, right] with 1.
      3. Erase a neighbourhood around any observed zero inside that span.

    Rows with no observations copy the prediction of the nearest observed row.
    """
    pred = np.zeros((H, W), dtype=np.uint8)
    observed_rows = []

    for r in range(H):
        row_mask = obs_mask[r]           # (W,) bool
        if not row_mask.any():
            continue

        queried_cols = np.where(row_mask)[0]
        labels = obs_labels_k[r, queried_cols]

        pos_cols = queried_cols[labels == 1]
        if pos_cols.size == 0:
            observed_rows.append(r)
            continue

        c_left = int(pos_cols.min())
        c_right = int(pos_cols.max())
        pred[r, c_left : c_right + 1] = 1

        # Suppress neighbourhood around interior observed zeros (holes).
        neg_cols = queried_cols[
            (labels == 0) & (queried_cols >= c_left) & (queried_cols <= c_right)
        ]
        for c in neg_cols:
            lo = max(c_left, c - HOLE_HALF)
            hi = min(c_right + 1, c + HOLE_HALF + 1)
            pred[r, lo:hi] = 0

        observed_rows.append(r)

    # Propagate to rows that had no observations.
    if not observed_rows:
        return pred

    observed_rows_arr = np.array(observed_rows)
    for r in range(H):
        if obs_mask[r].any():
            continue
        nearest = observed_rows_arr[np.argmin(np.abs(observed_rows_arr - r))]
        pred[r] = pred[nearest]

    # Keep only the largest 8-connected component to suppress
    # isolated outlier pixels that would otherwise inflate the bbox.
    if pred.any():
        struct8 = np.ones((3, 3), dtype=np.int32)
        labeled, n = ndlabel(pred, structure=struct8)
        if n > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            pred = (labeled == sizes.argmax()).astype(np.uint8)

    return pred


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

        predicted = np.zeros((8, self.H, self.W), dtype=np.uint8)
        for k in range(8):
            predicted[k] = _reconstruct_layer(
                self.H, self.W,
                self._obs_mask,
                self._obs_labels[k],
            )

        self._pred_cache = predicted
        self._cache_dirty = False
        return predicted
