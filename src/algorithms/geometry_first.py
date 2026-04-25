"""
Geometry-first adaptive algorithm.

Query strategy
--------------
Phase 1 — Coarse discovery (pre-planned, ~200 iterations)
  Queries a stride-(5,10) grid to reveal approximate blob locations for all
  8 layers.  After this phase every pixel is within ≤5 rows and ≤10 cols of
  at least one observation, giving a rough bounding-box estimate per layer.

Phase 2 — Adaptive entropy acquisition (remaining budget)
  At each step picks the unqueried pixel with the highest sum-of-entropies
  across all 8 layers:

      score(r,c) = Σ_k  [ 1 − 4·(p_k(r,c) − 0.5)² ]

  where p_k(r,c) is the current belief that pixel (r,c) belongs to blob k.
  This quadratic approximation of binary entropy is 0 for certain pixels and
  1 for maximally uncertain ones.

  After every query, beliefs in a (2R+1)×(2R+1) neighbourhood are updated
  toward the observed labels with Chebyshev-distance-based decay:

      p[:, r, c] ← p[:, r, c]·(1−w) + labels·w,   w = decay^dist

  This spatial propagation means:
  • Interior blob pixels (all neighbours return 1) → belief → 1 → entropy → 0
  • Exterior pixels (all neighbours return 0) → belief → 0 → entropy → 0
  • Boundary pixels (mixed neighbours) → belief ≈ 0.5 → entropy → 1 → queried

  The algorithm therefore naturally concentrates its budget on blob boundaries
  without any explicit boundary detection step.

Reconstruction
--------------
predict() thresholds the belief tensor at 0.5 and keeps the largest
8-connected component per layer.  The CC filter suppresses isolated outlier
pixels that would otherwise inflate the bounding box.
"""

import numpy as np
from scipy.ndimage import label as ndlabel

from .base import BaseAlgorithm

# ── Discovery phase parameters ──────────────────────────────────────────────
COARSE_STEP_R = 5
COARSE_STEP_C = 10

# ── Belief propagation parameters ───────────────────────────────────────────
PROP_RADIUS = 3      # Chebyshev radius for neighbourhood update
PROP_DECAY  = 0.65   # weight at distance d = decay^d

# ── Reconstruction ───────────────────────────────────────────────────────────
BELIEF_THRESHOLD = 0.5
_STRUCT8 = np.ones((3, 3), dtype=np.int32)


def _entropy_approx(p: np.ndarray) -> np.ndarray:
    """Quadratic approximation of binary entropy H(p) ∈ [0,1]."""
    return 1.0 - 4.0 * (p - 0.5) ** 2


def _largest_component(mask2d: np.ndarray) -> np.ndarray:
    """Return a binary mask containing only the largest 8-connected component."""
    labeled, n = ndlabel(mask2d, structure=_STRUCT8)
    if n <= 1:
        return mask2d
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(np.uint8)


class GeometryFirstAdaptive(BaseAlgorithm):
    """
    Geometry-first adaptive algorithm — Phase 0 candidate.

    Parameters
    ----------
    H, W    : grid dimensions
    budget  : iteration cap
    """

    def __init__(self, H: int, W: int, budget: int):
        super().__init__(H, W, budget)

        # Per-layer belief: P(pixel is blob) for each of 8 layers
        self._p = np.full((8, H, W), 0.4, dtype=np.float32)

        # Observation bookkeeping
        self._obs_mask   = np.zeros((H, W), dtype=bool)
        self._obs_labels = np.zeros((8, H, W), dtype=np.uint8)

        # Phase 1 discovery schedule
        self._discovery = [
            (r, c)
            for r in range(0, H, COARSE_STEP_R)
            for c in range(0, W, COARSE_STEP_C)
        ]
        self._disc_idx   = 0
        self._phase      = 1

        # Entropy score cache (invalidated by every update)
        self._score: np.ndarray | None = None
        self._score_dirty = True

        # Prediction cache
        self._pred_cache: np.ndarray | None = None
        self._pred_dirty  = True

    # ── Query selection ──────────────────────────────────────────────────────

    def next_query(self) -> tuple[int, int]:
        # Phase 1: follow the pre-computed discovery grid
        if self._phase == 1:
            if self._disc_idx < len(self._discovery):
                row, col = self._discovery[self._disc_idx]
                self._disc_idx += 1
                if self._disc_idx >= len(self._discovery):
                    self._phase = 2
                return row, col
            self._phase = 2

        # Phase 2: adaptive sum-entropy acquisition
        if self._score_dirty or self._score is None:
            self._score = _entropy_approx(self._p).sum(axis=0)   # (H, W)
            self._score[self._obs_mask] = -1.0
            self._score_dirty = False

        best = int(self._score.argmax())
        return divmod(best, self.W)

    # ── Observation update ───────────────────────────────────────────────────

    def update(self, row: int, col: int, labels: np.ndarray) -> None:
        if self._obs_mask[row, col]:
            return

        self._obs_mask[row, col] = True
        self._obs_labels[:, row, col] = labels.astype(np.uint8)

        # Certainty at the observed pixel
        self._p[:, row, col] = labels.astype(np.float32)

        # Vectorised neighbourhood propagation
        r0 = max(0, row - PROP_RADIUS)
        r1 = min(self.H, row + PROP_RADIUS + 1)
        c0 = max(0, col - PROP_RADIUS)
        c1 = min(self.W, col + PROP_RADIUS + 1)

        dr = np.abs(np.arange(r0, r1) - row)[:, None]   # (rh, 1)
        dc = np.abs(np.arange(c0, c1) - col)[None, :]   # (1, cw)
        dist = np.maximum(dr, dc).astype(np.float32)     # (rh, cw) Chebyshev
        w = (PROP_DECAY ** dist)                          # (rh, cw)

        # Zero out the observed pixel itself and already-observed neighbours
        w[row - r0, col - c0] = 0.0
        w *= (~self._obs_mask[r0:r1, c0:c1]).astype(np.float32)

        # Broadcast update over 8 layers: p = p*(1-w) + labels*w
        w3   = w[None, :, :]                                    # (1, rh, cw)
        lbl3 = labels.astype(np.float32)[:, None, None]        # (8, 1, 1)
        self._p[:, r0:r1, c0:c1] = (
            self._p[:, r0:r1, c0:c1] * (1.0 - w3) + lbl3 * w3
        )

        self._score_dirty = True
        self._pred_dirty  = True

    # ── Reconstruction ───────────────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        if not self._pred_dirty and self._pred_cache is not None:
            return self._pred_cache

        predicted = np.zeros((8, self.H, self.W), dtype=np.uint8)
        for k in range(8):
            mask = (self._p[k] >= BELIEF_THRESHOLD).astype(np.uint8)
            if mask.any():
                mask = _largest_component(mask)
            predicted[k] = mask

        self._pred_cache  = predicted
        self._pred_dirty  = False
        return predicted
