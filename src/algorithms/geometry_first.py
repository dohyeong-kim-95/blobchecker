"""
Geometry-first adaptive algorithm.

Query strategy
--------------
Phase 1 — Coarse discovery (~250 iterations)
  Queries a stride-(5,8) grid.  With PROP_RADIUS=4 every interior pixel is
  within propagation range of at least one Phase 1 observation.

Phase 2 — Boundary row refinement (~200 iterations)
  Scans rows in the zone between the last coarse-grid outside row and the
  first coarse-grid inside row for each layer's detected top/bottom edge.
  Observations here are used by the trim step in predict() to remove
  overshoot rows (confirmed label=0, no label=1) from the prediction.

Phase 3 — Adaptive entropy acquisition (remaining ~1050 iterations)
  Picks the unqueried pixel with the highest sum-of-entropies:
      score(r,c) = Σ_k  [1 − 4·(p_k(r,c) − 0.5)²]
  Beliefs are updated with Chebyshev-decay neighbourhood propagation after
  every query.

Reconstruction
--------------
predict() thresholds the belief tensor at 0.5, keeps the largest
8-connected component, then trims top/bottom rows that are confirmed outside
the blob (observed label=0, no observed label=1 for that layer).  This
directly corrects the systematic h_pred > h_truth overestimation caused by
belief-propagation overshoot.
"""

import numpy as np
from scipy.ndimage import label as ndlabel

from .base import BaseAlgorithm

# ── Discovery phase parameters ──────────────────────────────────────────────
COARSE_STEP_R = 5
COARSE_STEP_C = 8    # was 10; stride 8 + PROP_RADIUS 4 covers every interior pixel

# ── Belief propagation parameters ───────────────────────────────────────────
PROP_RADIUS = 4      # was 3; wider coverage per query
PROP_DECAY  = 0.60   # was 0.65; reduced to limit overshoot at larger radius

# ── Boundary refinement (Phase 2) ───────────────────────────────────────────
# Scans the coarse-step gap above/below detected blob edge.
# Column stride 5 ensures ≥1 sample per PROP_RADIUS=4 neighbourhood.
BOUNDARY_SCAN_STEP = 20   # coarse column stride: 10 queries/row, robust for wide blobs
BOUNDARY_CAP       = 200  # ~200 queries: ~25/layer × 8 layers with round-robin

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


def _trim_boundary_rows(
    mask: np.ndarray,
    obs_mask: np.ndarray,
    obs_labels_k: np.ndarray,
) -> np.ndarray:
    """
    Remove top/bottom rows of `mask` that are confirmed outside the blob.

    A row is "confirmed outside" if it has at least one observed label=0
    AND no observed label=1 for this layer.  Rows with no observations at all
    are kept (uncertain — leave them to entropy acquisition to correct).

    This post-processing step eliminates h_pred > h_truth overshoot caused by
    belief propagation from inside-blob pixels pushing adjacent outside pixels
    above the 0.5 threshold.
    """
    result = mask.copy()

    def confirmed_outside(r: int) -> bool:
        row_obs = obs_mask[r]
        if not row_obs.any():
            return False
        has_one  = bool((row_obs & (obs_labels_k[r] == 1)).any())
        has_zero = bool((row_obs & (obs_labels_k[r] == 0)).any())
        return has_zero and not has_one

    # Trim from top
    while True:
        rows = np.where(result.any(axis=1))[0]
        if rows.size == 0:
            break
        if confirmed_outside(int(rows[0])):
            result[rows[0], :] = 0
        else:
            break

    # Trim from bottom
    while True:
        rows = np.where(result.any(axis=1))[0]
        if rows.size == 0:
            break
        if confirmed_outside(int(rows[-1])):
            result[rows[-1], :] = 0
        else:
            break

    return result


def _build_boundary_queries(
    obs_mask: np.ndarray,
    obs_labels: np.ndarray,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """
    After Phase 1, scan the coarse-step gap above/below each layer's detected
    blob edge.  These observations give the trim step confirmed label=0 rows
    to eliminate from the final prediction.

    Round-robin across all 8 layers (innermost row first) so every layer gets
    boundary coverage before the BOUNDARY_CAP is consumed by early layers.
    Returned list is deduplicated and capped at BOUNDARY_CAP.
    """
    # Collect boundary row ranges per layer (innermost first)
    layer_top_rows: list[list[int]] = []
    layer_bot_rows: list[list[int]] = []

    for k in range(8):
        has_pos = obs_mask & (obs_labels[k] == 1)
        pos_rows = np.where(has_pos.any(axis=1))[0]
        if pos_rows.size == 0:
            layer_top_rows.append([])
            layer_bot_rows.append([])
            continue
        top_r = int(pos_rows[0])
        bot_r = int(pos_rows[-1])
        # Innermost first (top_r-1, top_r-2, ..., top_r-COARSE_STEP_R)
        tops = list(reversed(range(max(0, top_r - COARSE_STEP_R), top_r)))
        # Innermost first (bot_r+1, bot_r+2, ...)
        bots = list(range(bot_r + 1, min(H, bot_r + COARSE_STEP_R + 1)))
        layer_top_rows.append(tops)
        layer_bot_rows.append(bots)

    queries: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add_row(r: int) -> None:
        for c in range(0, W, BOUNDARY_SCAN_STEP):
            key = (r, c)
            if not obs_mask[r, c] and key not in seen:
                queries.append(key)
                seen.add(key)

    # Round-robin: one row per layer per pass, top boundaries first
    for row_idx in range(COARSE_STEP_R):
        for k in range(8):
            if row_idx < len(layer_top_rows[k]):
                add_row(layer_top_rows[k][row_idx])
        if len(queries) >= BOUNDARY_CAP:
            return queries[:BOUNDARY_CAP]

    for row_idx in range(COARSE_STEP_R):
        for k in range(8):
            if row_idx < len(layer_bot_rows[k]):
                add_row(layer_bot_rows[k][row_idx])
        if len(queries) >= BOUNDARY_CAP:
            return queries[:BOUNDARY_CAP]

    return queries[:BOUNDARY_CAP]


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
        self._disc_idx = 0
        self._phase    = 1   # 1=coarse, 2=boundary, 3=entropy

        # Phase 2 (boundary) queries — populated at end of Phase 1
        self._boundary_queries: list[tuple[int, int]] = []
        self._boundary_idx = 0

        # Entropy score cache (invalidated by every update)
        self._score: np.ndarray | None = None
        self._score_dirty = True

        # Prediction cache
        self._pred_cache: np.ndarray | None = None
        self._pred_dirty  = True

    # ── Query selection ──────────────────────────────────────────────────────

    def next_query(self) -> tuple[int, int]:
        # Phase 1: follow the pre-computed coarse discovery grid
        if self._phase == 1:
            if self._disc_idx < len(self._discovery):
                row, col = self._discovery[self._disc_idx]
                self._disc_idx += 1
                if self._disc_idx >= len(self._discovery):
                    self._start_boundary_phase()
                return row, col
            self._start_boundary_phase()

        # Phase 2: boundary row refinement
        if self._phase == 2:
            while self._boundary_idx < len(self._boundary_queries):
                row, col = self._boundary_queries[self._boundary_idx]
                self._boundary_idx += 1
                if not self._obs_mask[row, col]:
                    if self._boundary_idx >= len(self._boundary_queries):
                        self._phase = 3
                    return row, col
            self._phase = 3

        # Phase 3: adaptive sum-entropy acquisition
        if self._score_dirty or self._score is None:
            self._score = _entropy_approx(self._p).sum(axis=0)   # (H, W)
            self._score[self._obs_mask] = -1.0
            self._score_dirty = False

        best = int(self._score.argmax())
        return divmod(best, self.W)

    def _start_boundary_phase(self) -> None:
        self._boundary_queries = _build_boundary_queries(
            self._obs_mask, self._obs_labels, self.H, self.W
        )
        self._boundary_idx = 0
        self._phase = 2

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
                mask = _trim_boundary_rows(
                    mask, self._obs_mask, self._obs_labels[k]
                )
            predicted[k] = mask

        self._pred_cache  = predicted
        self._pred_dirty  = False
        return predicted
