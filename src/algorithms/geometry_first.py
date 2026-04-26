"""
Geometry-first adaptive algorithm.

Query strategy
--------------
Phase 1 — Coarse discovery (~250 iterations)
  Queries a stride-(5,8) grid.  With PROP_RADIUS=4 every interior pixel is
  within propagation range of at least one Phase 1 observation.

Phase 2 — Boundary refinement (~320 iterations)
  Scans just outside the detected top/bottom/left/right edges for each layer.
  Observations here are used by the trim step in predict() to remove overshoot
  rows or columns (confirmed label=0, no label=1) from the prediction.

Phase 3 — Adaptive entropy acquisition (remaining ~1050 iterations)
  Picks the unqueried pixel with the highest sum-of-entropies:
      score(r,c) = Σ_k  [1 − 4·(p_k(r,c) − 0.5)²]
  Beliefs are updated with Chebyshev-decay neighbourhood propagation after
  every query.

Reconstruction
--------------
predict() thresholds the belief tensor at 0.5, keeps the largest
8-connected component, then trims boundary rows/columns that are confirmed
outside the blob (observed label=0, no observed label=1 for that layer).  This
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
# Scans the coarse-step gap outside detected blob edges.
BOUNDARY_ROW_SCAN_STEP = 20  # 10 queries per scanned row on 50×200 Phase 0 grid
BOUNDARY_COL_SCAN_STEP = 5   # 10 queries per scanned column on 50×200 Phase 0 grid
BOUNDARY_CAP = 320           # one outside band for 4 sides × 8 layers, less duplicates

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


def _line_confirmed_outside(
    observed: np.ndarray,
    labels: np.ndarray,
) -> bool:
    """Return True when a row/column has zero evidence and no positive evidence."""
    if not observed.any():
        return False
    has_one = bool((observed & (labels == 1)).any())
    has_zero = bool((observed & (labels == 0)).any())
    return has_zero and not has_one


def _trim_confirmed_boundary(
    mask: np.ndarray,
    obs_mask: np.ndarray,
    obs_labels_k: np.ndarray,
) -> np.ndarray:
    """
    Remove boundary rows/columns of `mask` that are confirmed outside the blob.

    A row or column is "confirmed outside" if it has at least one observed
    label=0 AND no observed label=1 for this layer.  Lines with no observations
    are kept because they remain uncertain.

    This post-processing step eliminates bbox overshoot caused by belief
    propagation from inside-blob pixels pushing adjacent outside pixels above
    the 0.5 threshold.
    """
    result = mask.copy()

    # Trim from top
    while True:
        rows = np.where(result.any(axis=1))[0]
        if rows.size == 0:
            break
        r = int(rows[0])
        if _line_confirmed_outside(obs_mask[r], obs_labels_k[r]):
            result[r, :] = 0
        else:
            break

    # Trim from bottom
    while True:
        rows = np.where(result.any(axis=1))[0]
        if rows.size == 0:
            break
        r = int(rows[-1])
        if _line_confirmed_outside(obs_mask[r], obs_labels_k[r]):
            result[r, :] = 0
        else:
            break

    # Trim from left
    while True:
        cols = np.where(result.any(axis=0))[0]
        if cols.size == 0:
            break
        c = int(cols[0])
        if _line_confirmed_outside(obs_mask[:, c], obs_labels_k[:, c]):
            result[:, c] = 0
        else:
            break

    # Trim from right
    while True:
        cols = np.where(result.any(axis=0))[0]
        if cols.size == 0:
            break
        c = int(cols[-1])
        if _line_confirmed_outside(obs_mask[:, c], obs_labels_k[:, c]):
            result[:, c] = 0
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
    After Phase 1, scan the coarse-step gap outside each layer's detected blob
    edges.  These observations give the trim step confirmed label=0 lines to
    eliminate from the final prediction.

    Round-robin across sides and layers (innermost line first) so every layer
    gets boundary coverage before the BOUNDARY_CAP is consumed by early layers.
    Returned list is deduplicated and capped at BOUNDARY_CAP.
    """
    # Collect boundary line ranges per layer (innermost first)
    layer_top_rows: list[list[int]] = []
    layer_bot_rows: list[list[int]] = []
    layer_left_cols: list[list[int]] = []
    layer_right_cols: list[list[int]] = []

    for k in range(8):
        has_pos = obs_mask & (obs_labels[k] == 1)
        pos_rows = np.where(has_pos.any(axis=1))[0]
        pos_cols = np.where(has_pos.any(axis=0))[0]
        if pos_rows.size == 0 or pos_cols.size == 0:
            layer_top_rows.append([])
            layer_bot_rows.append([])
            layer_left_cols.append([])
            layer_right_cols.append([])
            continue
        top_r = int(pos_rows[0])
        bot_r = int(pos_rows[-1])
        left_c = int(pos_cols[0])
        right_c = int(pos_cols[-1])
        # Innermost first (top_r-1, top_r-2, ..., top_r-COARSE_STEP_R)
        tops = list(reversed(range(max(0, top_r - COARSE_STEP_R), top_r)))
        # Innermost first (bot_r+1, bot_r+2, ...)
        bots = list(range(bot_r + 1, min(H, bot_r + COARSE_STEP_R + 1)))
        # Innermost first (left_c-1, left_c-2, ..., left_c-COARSE_STEP_C)
        lefts = list(reversed(range(max(0, left_c - COARSE_STEP_C), left_c)))
        # Innermost first (right_c+1, right_c+2, ...)
        rights = list(range(right_c + 1, min(W, right_c + COARSE_STEP_C + 1)))
        layer_top_rows.append(tops)
        layer_bot_rows.append(bots)
        layer_left_cols.append(lefts)
        layer_right_cols.append(rights)

    queries: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add_row(r: int) -> None:
        for c in range(0, W, BOUNDARY_ROW_SCAN_STEP):
            key = (r, c)
            if not obs_mask[r, c] and key not in seen:
                queries.append(key)
                seen.add(key)

    def add_col(c: int) -> None:
        for r in range(0, H, BOUNDARY_COL_SCAN_STEP):
            key = (r, c)
            if not obs_mask[r, c] and key not in seen:
                queries.append(key)
                seen.add(key)

    # Round-robin: one outside line per side per layer per pass.
    for line_idx in range(max(COARSE_STEP_R, COARSE_STEP_C)):
        for k in range(8):
            if line_idx < len(layer_top_rows[k]):
                add_row(layer_top_rows[k][line_idx])
            if line_idx < len(layer_bot_rows[k]):
                add_row(layer_bot_rows[k][line_idx])
            if line_idx < len(layer_left_cols[k]):
                add_col(layer_left_cols[k][line_idx])
            if line_idx < len(layer_right_cols[k]):
                add_col(layer_right_cols[k][line_idx])
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
        self._last_query_phase = "coarse"

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
                self._last_query_phase = "coarse"
                self._disc_idx += 1
                if self._disc_idx >= len(self._discovery):
                    self._start_boundary_phase()
                return row, col
            self._start_boundary_phase()

        # Phase 2: boundary row/column refinement
        if self._phase == 2:
            while self._boundary_idx < len(self._boundary_queries):
                row, col = self._boundary_queries[self._boundary_idx]
                self._boundary_idx += 1
                if not self._obs_mask[row, col]:
                    if self._boundary_idx >= len(self._boundary_queries):
                        self._phase = 3
                    self._last_query_phase = "boundary"
                    return row, col
            self._phase = 3

        # Phase 3: adaptive sum-entropy acquisition
        if self._score_dirty or self._score is None:
            self._score = _entropy_approx(self._p).sum(axis=0)   # (H, W)
            self._score[self._obs_mask] = -1.0
            self._score_dirty = False

        best = int(self._score.argmax())
        self._last_query_phase = "entropy"
        return divmod(best, self.W)

    def diagnostic_phase(self) -> str:
        """Return the phase that produced the most recent query."""
        return self._last_query_phase

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
                mask = _trim_confirmed_boundary(
                    mask, self._obs_mask, self._obs_labels[k]
                )
            predicted[k] = mask

        self._pred_cache  = predicted
        self._pred_dirty  = False
        return predicted
