"""
contour/contour.py
Level-Set Estimation via contour-following.  No GP, no surrogate model.

Algorithm
---------
Phase 1 — Scanline binary search (log-spaced rows)
    Find a first interior point quickly; establishes blob's vertical extent.

Phase 2 — Per-row boundary scan (every row)
    Binary-search for left / right boundaries on every row.
    Naturally samples both sides of each boundary (0 just outside, 1 just
    inside), so nearest-neighbour prediction is accurate everywhere.

Phase 3 — Interior grid sampling
    Uniform grid inside the blob bounding box to detect interior holes (0s).

Prediction
----------
Nearest-neighbour labelling via scipy distance_transform_edt:
every unsampled pixel inherits the label of its nearest sampled pixel.
This is exact at sampled pixels and accurate in between because Phase 2
gives dense boundary coverage (every row) and Phase 3 gives interior coverage.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

Oracle       = Callable[[int, int], int]
CheckpointFn = Callable[[int, np.ndarray], None]


# ---------------------------------------------------------------------------
# Abstract base  (mirrors lse/lse.py interface)
# ---------------------------------------------------------------------------

class BaseEstimator(ABC):
    @abstractmethod
    def fit(
        self,
        oracle: Oracle,
        grid_size: tuple[int, int],
        budget: int | None = None,
        checkpoint_fn: CheckpointFn | None = None,
    ) -> np.ndarray:
        """Query oracle sequentially; return predicted (H,W) uint8 binary map."""


# ---------------------------------------------------------------------------
# Contour estimator
# ---------------------------------------------------------------------------

class ContourEstimator(BaseEstimator):
    """Contour-following estimator.

    Parameters
    ----------
    phase1_frac : fraction of budget for Phase 1 (log-spaced scanlines)
    phase2_frac : cumulative fraction used by end of Phase 2
    """

    def __init__(
        self,
        phase1_frac: float = 0.05,
        phase2_frac: float = 0.80,
    ) -> None:
        self.phase1_frac = phase1_frac
        self.phase2_frac = phase2_frac

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(
        self,
        oracle: Oracle,
        grid_size: tuple[int, int],
        budget: int | None = None,
        checkpoint_fn: CheckpointFn | None = None,
    ) -> np.ndarray:
        H, W = grid_size
        total = H * W
        if budget is None:
            budget = int(0.15 * total)

        sampled = np.zeros((H, W), dtype=bool)
        vals    = np.zeros((H, W), dtype=np.uint8)
        n       = [0]
        next_ck = [50]

        # ---- inline helpers ----

        def query(r: int, c: int) -> int:
            r = max(0, min(H - 1, int(r)))
            c = max(0, min(W - 1, int(c)))
            if sampled[r, c]:
                return int(vals[r, c])
            v = oracle(r, c)
            sampled[r, c] = True
            vals[r, c] = v
            n[0] += 1
            # fire checkpoints
            while checkpoint_fn is not None and n[0] >= next_ck[0]:
                checkpoint_fn(next_ck[0], _nn_pred())
                next_ck[0] += 50
            return v

        def _nn_pred() -> np.ndarray:
            """Prediction used at checkpoints (pure NN; fast and sufficient early on)."""
            if not sampled.any():
                return np.zeros((H, W), dtype=np.uint8)
            _, idx = distance_transform_edt(~sampled, return_indices=True)
            return vals[idx[0], idx[1]]

        def _structured_pred() -> np.ndarray:
            """Final prediction.
            Exterior/interior split comes from Phase 2 boundary lines (no NN bleed).
            Phase 3 0-samples are expanded inward via NN to recover holes.
            """
            pred = np.zeros((H, W), dtype=np.uint8)
            # Paint each blob row's span
            for r, (l, rr) in bounds.items():
                pred[r, l:rr + 1] = 1
            # Apply sampled interior 0s (holes) via NN among interior samples only
            interior_sampled = sampled & (pred == 1)
            if interior_sampled.any():
                _, idx = distance_transform_edt(~interior_sampled, return_indices=True)
                interior_nn = vals[idx[0], idx[1]]
                pred[pred == 1] = interior_nn[pred == 1]
            return pred

        def _find_seed(r: int, extra_cols: list[int] | None = None) -> int:
            """Query candidate columns in row r; return first col with 1, else -1."""
            cols = [W // 2, W // 4, 3 * W // 4,
                    W // 8, 3 * W // 8, 5 * W // 8, 7 * W // 8]
            if extra_cols:
                cols = extra_cols + cols   # prefer targeted cols first
            for c in cols:
                if n[0] >= budget:
                    return -1
                if query(r, c) == 1:
                    return c
            return -1

        def _bs_left(r: int, lo: int, hi: int) -> int:
            """Leftmost col with value 1 in [lo, hi]; hi must already be 1."""
            while lo < hi and n[0] < budget:
                m = (lo + hi) // 2
                if query(r, m) == 1:
                    hi = m
                else:
                    lo = m + 1
            return lo

        def _bs_right(r: int, lo: int, hi: int) -> int:
            """Rightmost col with value 1 in [lo, hi]; lo must already be 1."""
            while lo < hi and n[0] < budget:
                m = (lo + hi + 1) // 2
                if query(r, m) == 1:
                    lo = m
                else:
                    hi = m - 1
            return lo

        def _scan_row(r: int, extra_cols: list[int] | None = None) -> tuple[int, int] | None:
            """Return (left_boundary, right_boundary) for row r, or None.

            Handles two tricky cases:
            (a) Blob extends to grid edge (W-1 or 0 is 1) — checked before binary search
                so we never mistake a hole mid-way for the boundary.
            (b) Hole between seed and right edge — after finding putative right
                boundary probe a few columns further right to check.
            """
            seed = _find_seed(r, extra_cols)
            if seed < 0:
                return None

            # ---- left boundary ----
            if query(r, 0) == 1:
                left = 0
            else:
                left = _bs_left(r, 0, seed)
                # Check if a hole stopped search early (probe a few cols left)
                for d in (1, 6, 13):
                    p = left - d
                    if p <= 0 or n[0] >= budget:
                        break
                    if query(r, p) == 1:
                        left = _bs_left(r, 0, p)
                        break

            # ---- right boundary ----
            if query(r, W - 1) == 1:
                right = W - 1          # blob reaches grid edge: no search needed
            else:
                right = _bs_right(r, seed, W - 2)   # W-1=0 is exterior anchor
                # Check if a hole stopped search early (probe a few cols right)
                for d in (1, 6, 13):
                    p = right + d
                    if p >= W - 1 or n[0] >= budget:
                        break
                    if query(r, p) == 1:
                        right = _bs_right(r, p, W - 2)
                        break

            return left, right

        # ================================================================
        # Phase 1: log-spaced scanlines from the vertical midpoint outward
        # ================================================================
        p1_cap = max(20, int(self.phase1_frac * budget))
        p2_cap = int(self.phase2_frac * budget)

        mid_r   = H // 2
        offsets = sorted({0} | {2 ** k for k in range(int(np.log2(max(mid_r, 1))) + 1)}
                         | {mid_r})
        p1_rows = sorted({max(0, min(H - 1, mid_r + s)) for s in offsets}
                         | {max(0, min(H - 1, mid_r - s)) for s in offsets})

        bounds: dict[int, tuple[int, int]] = {}   # row -> (left, right)

        for r in p1_rows:
            if n[0] >= p1_cap:
                break
            b = _scan_row(r)
            if b is not None:
                bounds[r] = b

        # ================================================================
        # Phase 2: binary search for every remaining row
        # ================================================================
        # Build targeted probe columns from Phase 1 findings so that narrow
        # blob segments (not hit by the 7 default probe cols) are not missed.
        if bounds:
            known_cols = sorted({c for l, rr in bounds.values()
                                 for c in range(l, rr + 1, max(1, (rr - l) // 4))})
        else:
            known_cols = []

        for r in range(H):
            if r in bounds:
                continue
            if n[0] >= p2_cap:
                break
            b = _scan_row(r, extra_cols=known_cols)
            if b is not None:
                bounds[r] = b

        # ================================================================
        # Phase 2b: anchor 0s just outside every boundary
        #   Without this, interior 1-samples from row r bleed into adjacent
        #   exterior pixels (row r±1 near the boundary) via NN labelling.
        # ================================================================
        for r, (l, rr) in bounds.items():
            if n[0] >= budget:
                break
            if l > 0:
                query(r, l - 1)
            if rr < W - 1:
                query(r, rr + 1)

        # ================================================================
        # Phase 3: interior grid to detect holes
        # ================================================================
        if bounds:
            row_list = sorted(bounds)
            min_r = row_list[0]
            max_r = row_list[-1]
            min_c = min(l for l, _ in bounds.values())
            max_c = max(rr for _, rr in bounds.values())

            area      = max(1, (max_r - min_r + 1) * (max_c - min_c + 1))
            remaining = max(1, budget - n[0])
            step      = max(1, int(np.sqrt(area / remaining)))

            for r in range(min_r, max_r + 1, step):
                if n[0] >= budget:
                    break
                for c in range(min_c, max_c + 1, step):
                    if n[0] >= budget:
                        break
                    query(r, c)

        # Final checkpoint if we ended between multiples of 50
        if checkpoint_fn is not None and n[0] > 0 and n[0] < next_ck[0]:
            checkpoint_fn(n[0], _structured_pred())

        return _structured_pred()
