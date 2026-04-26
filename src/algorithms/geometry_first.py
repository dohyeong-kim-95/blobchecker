"""
Geometry-first adaptive algorithm.

Query strategy
--------------
Phase 1 — Coarse discovery (~250 iterations)
  Queries a stride-(5,8) grid.  With PROP_RADIUS=4 every interior pixel is
  within propagation range of at least one Phase 1 observation.

Phase 2 — Boundary refinement (~256-320 iterations)
  Builds per-layer boundary brackets from coarse positive observations and
  outward observed zeros, then adaptively binary-searches top/bottom/left/right
  edges.  Observations here are used by the trim step in predict() to remove
  overshoot rows or columns (confirmed label=0, no label=1) from the prediction.

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

from dataclasses import dataclass

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
# Binary-searches coarse brackets outside detected blob edges.
BOUNDARY_ANCHORS_PER_SIDE = 2
BOUNDARY_BINARY_STEPS = 4
BOUNDARY_CAP = 320

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


@dataclass
class _BoundaryTask:
    """One layer-specific bracketed search along a row or column."""

    layer: int
    axis: int       # 0: row varies at fixed column, 1: column varies at fixed row
    fixed: int
    outside: int
    inside: int
    remaining: int = BOUNDARY_BINARY_STEPS

    def converged(self) -> bool:
        return abs(self.outside - self.inside) <= 1 or self.remaining <= 0

    def point(self) -> tuple[int, int]:
        mid = (self.outside + self.inside) // 2
        if self.axis == 0:
            return mid, self.fixed
        return self.fixed, mid

    def apply(self, label: int) -> None:
        mid = (self.outside + self.inside) // 2
        if label:
            self.inside = mid
        else:
            self.outside = mid
        self.remaining -= 1


def _select_evenly(values: np.ndarray, limit: int) -> list[int]:
    """Select up to `limit` representative values without seed-specific tuning."""
    if values.size <= limit:
        return [int(v) for v in values]
    idxs = np.linspace(0, values.size - 1, num=limit, dtype=int)
    return [int(values[i]) for i in idxs]


def _nearest_outside_zero(
    obs_mask: np.ndarray,
    obs_labels_k: np.ndarray,
    axis: int,
    fixed: int,
    inside: int,
    direction: int,
    limit: int,
) -> int | None:
    """Find the nearest observed zero moving outward from an inside positive."""
    pos = inside + direction
    while 0 <= pos < limit:
        if axis == 0:
            observed = obs_mask[pos, fixed]
            label = obs_labels_k[pos, fixed]
        else:
            observed = obs_mask[fixed, pos]
            label = obs_labels_k[fixed, pos]
        if observed and label == 0:
            return pos
        pos += direction
    return None


def _build_boundary_tasks(
    obs_mask: np.ndarray,
    obs_labels: np.ndarray,
    H: int,
    W: int,
) -> list[_BoundaryTask]:
    """
    After Phase 1, create bracketed binary-search tasks for detected blob edges.

    A valid bracket has one inside positive observation and one outward observed
    zero on the same row/column.  Tasks without such a bracket are skipped
    rather than guessed from public-seed geometry.
    """
    tasks_by_layer: list[list[_BoundaryTask]] = [[] for _ in range(8)]

    for k in range(8):
        has_pos = obs_mask & (obs_labels[k] == 1)
        pos_rows = np.where(has_pos.any(axis=1))[0]
        pos_cols = np.where(has_pos.any(axis=0))[0]
        if pos_rows.size == 0 or pos_cols.size == 0:
            continue

        top_r = int(pos_rows[0])
        bot_r = int(pos_rows[-1])
        left_c = int(pos_cols[0])
        right_c = int(pos_cols[-1])

        # Top/bottom: row varies at representative positive columns.
        top_cols = _select_evenly(np.where(has_pos[top_r])[0], BOUNDARY_ANCHORS_PER_SIDE)
        bot_cols = _select_evenly(np.where(has_pos[bot_r])[0], BOUNDARY_ANCHORS_PER_SIDE)
        for c in top_cols:
            outside = _nearest_outside_zero(
                obs_mask, obs_labels[k], axis=0, fixed=c,
                inside=top_r, direction=-1, limit=H,
            )
            if outside is not None and abs(outside - top_r) > 1:
                tasks_by_layer[k].append(_BoundaryTask(k, 0, c, outside, top_r))
        for c in bot_cols:
            outside = _nearest_outside_zero(
                obs_mask, obs_labels[k], axis=0, fixed=c,
                inside=bot_r, direction=1, limit=H,
            )
            if outside is not None and abs(outside - bot_r) > 1:
                tasks_by_layer[k].append(_BoundaryTask(k, 0, c, outside, bot_r))

        # Left/right: column varies at representative positive rows.
        left_rows = _select_evenly(np.where(has_pos[:, left_c])[0], BOUNDARY_ANCHORS_PER_SIDE)
        right_rows = _select_evenly(np.where(has_pos[:, right_c])[0], BOUNDARY_ANCHORS_PER_SIDE)
        for r in left_rows:
            outside = _nearest_outside_zero(
                obs_mask, obs_labels[k], axis=1, fixed=r,
                inside=left_c, direction=-1, limit=W,
            )
            if outside is not None and abs(outside - left_c) > 1:
                tasks_by_layer[k].append(_BoundaryTask(k, 1, r, outside, left_c))
        for r in right_rows:
            outside = _nearest_outside_zero(
                obs_mask, obs_labels[k], axis=1, fixed=r,
                inside=right_c, direction=1, limit=W,
            )
            if outside is not None and abs(outside - right_c) > 1:
                tasks_by_layer[k].append(_BoundaryTask(k, 1, r, outside, right_c))

    tasks: list[_BoundaryTask] = []
    max_len = max((len(layer_tasks) for layer_tasks in tasks_by_layer), default=0)
    for i in range(max_len):
        for layer_tasks in tasks_by_layer:
            if i < len(layer_tasks):
                tasks.append(layer_tasks[i])
    return tasks


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

        # Phase 2 (boundary) binary-search tasks — populated at end of Phase 1
        self._boundary_tasks: list[_BoundaryTask] = []
        self._boundary_queries_used = 0
        self._pending_boundary_task: _BoundaryTask | None = None

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

        # Phase 2: adaptive bracketed boundary refinement
        if self._phase == 2:
            query = self._next_boundary_query()
            if query is not None:
                return query
            self._phase = 3

        # Phase 3: adaptive sum-entropy acquisition
        if self._score_dirty or self._score is None:
            self._score = _entropy_approx(self._p).sum(axis=0)   # (H, W)
            self._score[self._obs_mask] = -1.0
            self._score_dirty = False

        best = int(self._score.argmax())
        return divmod(best, self.W)

    def _start_boundary_phase(self) -> None:
        self._boundary_tasks = _build_boundary_tasks(
            self._obs_mask, self._obs_labels, self.H, self.W
        )
        self._boundary_queries_used = 0
        self._pending_boundary_task = None
        self._phase = 2

    def _next_boundary_query(self) -> tuple[int, int] | None:
        while self._boundary_tasks and self._boundary_queries_used < BOUNDARY_CAP:
            task = self._boundary_tasks.pop(0)
            if task.converged():
                continue

            row, col = task.point()
            if self._obs_mask[row, col]:
                task.apply(int(self._obs_labels[task.layer, row, col]))
                if not task.converged():
                    self._boundary_tasks.append(task)
                continue

            self._pending_boundary_task = task
            self._boundary_queries_used += 1
            return row, col
        return None

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

        if self._pending_boundary_task is not None:
            task = self._pending_boundary_task
            if row == task.point()[0] and col == task.point()[1]:
                task.apply(int(labels[task.layer]))
                if not task.converged():
                    self._boundary_tasks.append(task)
            self._pending_boundary_task = None

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
