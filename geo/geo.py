"""
geo/geo.py
Geometry-driven binary map estimation via direct oracle queries.

Classes
-------
BaseEstimator   abstract interface (identical to lse/lse.py)
GeoEstimator    scanline + contour-tracing estimator (no GP, no surrogate)

Algorithm
---------
Phase 1 – Blob detection
    Binary search each of H scanlines to find one interior blob pixel.
    Each found pixel is validated with a 4-neighbour check to reject
    isolated outlier 1s.  Cost: H * ceil(log2(W)) ≈ 400 samples for 50×200.

Phase 2 – Blob extent per row
    For each seeded row, binary search left and right from the seed to find
    the full horizontal extent of the blob on that row.
    Cost: ≈ H_seeded * 2 * log2(W) ≈ 640 samples.

Phase 3 – Interior fill
    Fill pred[r, left:right+1] = 1 for every seeded row (no queries).
    Rows without a seed remain 0.

Phase 4 – Hole detection
    Sample a sparse grid inside the predicted blob bounding box.
    Each candidate 0-pixel is validated by a 4-neighbour query.
    Cost: O(grid_points + validation_cost).

Phase 5 – Hole fill
    For each confirmed hole seed, BFS-flood the connected 0-region
    (querying adjacent pixels that are currently predicted 1) and set
    pred = 0 throughout.

Unsampled pixels are inferred by fill — no model is used.

Budget  : 15 % of total pixels (1 500 for 50×200).
Tracking: accuracy checkpoint every 50 samples.
Target  : ≤ 10 s, ≤ 100 MB on 50×200 grid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Callable

import numpy as np


Oracle = Callable[[int, int], int]                  # f(row, col) -> {0, 1}
CheckpointFn = Callable[[int, np.ndarray], None]    # f(n_sampled, pred_map)


# ---------------------------------------------------------------------------
# Abstract base  (mirrors lse/lse.py)
# ---------------------------------------------------------------------------

class BaseEstimator(ABC):
    """Interface for level-set estimators."""

    @abstractmethod
    def fit(
        self,
        oracle: Oracle,
        grid_size: tuple[int, int],
        budget: int | None = None,
        checkpoint_fn: CheckpointFn | None = None,
    ) -> np.ndarray:
        """Query oracle sequentially and return predicted binary map.

        Parameters
        ----------
        oracle        : deterministic f(row, col) -> {0, 1}
        grid_size     : (H, W)
        budget        : total oracle calls; defaults to 15 % of H*W
        checkpoint_fn : optional callback(n_sampled, predicted_map) called
                        every checkpoint_interval steps

        Returns
        -------
        (H, W) uint8 binary map
        """


# ---------------------------------------------------------------------------
# GeoEstimator
# ---------------------------------------------------------------------------

class GeoEstimator(BaseEstimator):
    """Geometry-driven estimator using scanline binary search + hole BFS.

    No GP or surrogate model.  The algorithm:
      1. Binary searches each row to find one confirmed blob pixel.
      2. Binary searches left/right from that seed to find row extents.
      3. Fills row extents with 1 (no queries).
      4. Samples a sparse grid inside the blob to find holes.
      5. BFS-floods each hole, correcting pred back to 0.

    Parameters
    ----------
    checkpoint_interval : emit checkpoint every this many oracle calls
    hole_grid_steps     : (step_r, step_c) for hole-detection grid inside blob
    extent_smooth_window: neighbour-row radius used to smooth row extents
    """

    def __init__(
        self,
        checkpoint_interval: int = 50,
        hole_grid_steps: tuple[int, int] = (4, 10),
        extent_smooth_window: int = 2,
    ) -> None:
        self.checkpoint_interval = checkpoint_interval
        self.hole_grid_steps = hole_grid_steps
        self.extent_smooth_window = extent_smooth_window

    def _smooth_row_extents(
        self,
        row_extents: dict[int, tuple[int, int]],
        H: int,
    ) -> dict[int, tuple[int, int]]:
        """Bridge local edge dents caused by interior holes.

        Phase 2 finds per-row extents from one seed. When a row intersects a
        hole, binary-search extents can collapse inward and miss the right-side
        1-region.  We smooth extents using neighbouring rows to recover the
        outer blob envelope before Phase 3 fill.
        """
        if not row_extents:
            return row_extents

        rows = sorted(row_extents)
        min_row, max_row = rows[0], rows[-1]
        w = max(0, self.extent_smooth_window)

        smoothed: dict[int, tuple[int, int]] = {}
        for row in range(min_row, max_row + 1):
            lefts: list[int] = []
            rights: list[int] = []
            for rr in range(max(0, row - w), min(H - 1, row + w) + 1):
                if rr in row_extents:
                    l, r = row_extents[rr]
                    lefts.append(l)
                    rights.append(r)

            if lefts and rights:
                smoothed[row] = (min(lefts), max(rights))

        return smoothed

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

        pred = np.zeros((H, W), dtype=np.uint8)
        known = np.zeros((H, W), dtype=bool)
        n_sampled = 0
        next_ckpt = [self.checkpoint_interval]

        # ---- inner helpers ----

        def query(r: int, c: int) -> int:
            nonlocal n_sampled
            if known[r, c]:
                return int(pred[r, c])
            v = oracle(r, c)
            pred[r, c] = v
            known[r, c] = True
            n_sampled += 1
            _maybe_ckpt()
            return v

        def _maybe_ckpt() -> None:
            if checkpoint_fn is not None and n_sampled >= next_ckpt[0]:
                checkpoint_fn(n_sampled, pred.copy())
                next_ckpt[0] += self.checkpoint_interval

        def budget_left() -> int:
            return budget - n_sampled

        # ----------------------------------------------------------------
        # Phase 1 – Binary search per scanline → one seed per row
        # ----------------------------------------------------------------
        row_seeds: dict[int, int] = {}  # row → confirmed 1-col

        for row in range(H):
            if budget_left() <= 0:
                break

            lo, hi = 0, W - 1
            found_col: int | None = None

            while lo <= hi and budget_left() > 0:
                mid = (lo + hi) // 2
                v = query(row, mid)
                if v == 1:
                    found_col = mid
                    hi = mid - 1      # push left to find earlier 1
                else:
                    lo = mid + 1

            if found_col is None:
                continue

            # Validate: require at least one 4-neighbour that is also 1
            r, c = row, found_col
            confirmed = False
            for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if 0 <= nr < H and 0 <= nc < W and budget_left() > 0:
                    if query(nr, nc) == 1:
                        confirmed = True
                        break

            if confirmed:
                row_seeds[row] = found_col

        # ----------------------------------------------------------------
        # Phase 2 – Binary search for left/right blob edge per seeded row
        # ----------------------------------------------------------------
        row_extents: dict[int, tuple[int, int]] = {}  # row → (left, right)

        for row, seed_col in row_seeds.items():
            if budget_left() <= 0:
                break

            # Left edge: find smallest col in [0, seed_col] that is 1
            lo, hi = 0, seed_col
            left_edge = seed_col
            while lo < hi and budget_left() > 0:
                mid = (lo + hi) // 2
                if query(row, mid) == 1:
                    left_edge = mid
                    hi = mid          # could be even further left
                else:
                    lo = mid + 1
                    left_edge = lo    # transition is at lo

            # Right edge: find largest col in [seed_col, W-1] that is 1
            lo, hi = seed_col, W - 1
            right_edge = seed_col
            while lo < hi and budget_left() > 0:
                mid = (lo + hi + 1) // 2
                if query(row, mid) == 1:
                    right_edge = mid
                    lo = mid          # could be even further right
                else:
                    hi = mid - 1
                    right_edge = lo

            row_extents[row] = (left_edge, right_edge)

        # Hole-crossing rows can get shrunken extents; smooth by neighbours.
        row_extents = self._smooth_row_extents(row_extents, H)

        # ----------------------------------------------------------------
        # Phase 3 – Fill blob interior from row extents (no queries)
        # ----------------------------------------------------------------
        for row, (left, right) in row_extents.items():
            pred[row, left:right + 1] = 1

        # ----------------------------------------------------------------
        # Phase 4 – Hole detection: sparse grid inside blob bounding box
        # ----------------------------------------------------------------
        hole_seeds: list[tuple[int, int]] = []

        if row_extents and budget_left() > 0:
            all_rows = sorted(row_extents)
            min_row, max_row = all_rows[0], all_rows[-1]
            min_col = min(l for l, _ in row_extents.values())
            max_col = max(r for _, r in row_extents.values())

            step_r, step_c = self.hole_grid_steps
            visited_holes: set[tuple[int, int]] = set()

            for r in range(min_row, max_row + 1, step_r):
                if budget_left() <= 0:
                    break
                for c in range(min_col, max_col + 1, step_c):
                    if budget_left() <= 0:
                        break
                    if pred[r, c] != 1:
                        continue           # outside blob fill — skip
                    if (r, c) in visited_holes:
                        continue

                    v = query(r, c)
                    if v != 0:
                        continue

                    # Validate: at least one 4-neighbour must also be 0
                    confirmed_hole = False
                    for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                        if 0 <= nr < H and 0 <= nc < W and budget_left() > 0:
                            if query(nr, nc) == 0:
                                confirmed_hole = True
                                break

                    if confirmed_hole:
                        hole_seeds.append((r, c))
                        visited_holes.add((r, c))

        # ----------------------------------------------------------------
        # Phase 5 – Hole fill: BFS-flood each confirmed hole back to 0
        # ----------------------------------------------------------------
        bfs_seen: set[tuple[int, int]] = set()

        for hr, hc in hole_seeds:
            if pred[hr, hc] != 0:
                continue   # absorbed by a previous BFS
            if (hr, hc) in bfs_seen:
                continue

            queue: deque[tuple[int, int]] = deque()
            queue.append((hr, hc))
            bfs_seen.add((hr, hc))
            pred[hr, hc] = 0

            while queue and budget_left() > 0:
                r, c = queue.popleft()
                for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if (nr, nc) in bfs_seen:
                        continue
                    bfs_seen.add((nr, nc))

                    if pred[nr, nc] == 1:
                        # Might be a hole pixel mis-filled in Phase 3
                        if budget_left() > 0:
                            v = query(nr, nc)
                            if v == 0:
                                pred[nr, nc] = 0
                                queue.append((nr, nc))
                    else:
                        # Already 0 — propagate without querying
                        queue.append((nr, nc))

        # ----------------------------------------------------------------
        # Final checkpoint
        # ----------------------------------------------------------------
        if checkpoint_fn is not None:
            checkpoint_fn(n_sampled, pred.copy())

        return pred
