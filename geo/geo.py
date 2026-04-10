"""
geo/geo.py
Geometry-driven binary map estimation via direct oracle queries.

Classes
-------
BaseEstimator   abstract interface (identical to lse/lse.py)
GeoEstimator    scanline + hole-BFS estimator (no GP, no surrogate)

Algorithm
---------
Phase 1 – Blob detection
    Binary search each scanline for a 1-pixel seed; validate with a
    4-neighbour query to reject isolated outlier 1s.  If validation fails
    (outlier found), retry from the next candidate rightward (up to
    max_retries times per row).
    Cost: H × ceil(log₂ W) ≈ 400 samples.

Phase 2 – Blob extent per row
    For each seeded row, binary search left from the seed to find
    left_edge, and right to find the rightmost 1 in the first contiguous
    run.  Then probe rightward at step probe_step to detect any additional
    1-regions (e.g. a right wing separated by a gap).  For each additional
    region found, binary search for its right extent and update right_edge.

    Per-run fill: if the seed col is at a run boundary (col to its left is 0),
    the left binary-search result is a separate run — fill it independently
    rather than filling the entire [left_edge, right_edge] span, which would
    create large false-positive gap regions.
    Cost: ≈ 700 samples (basic) + ≈ 200 samples (right-probe scan).

Phase 2.5 – Downward extension
    After seeding all rows via Phase 1, probe downward from the last seeded
    row to catch rows that Phase 1 missed due to non-monotone binary search
    failures (e.g. small blob at extreme columns).  For each newly seeded
    row, run the same Phase 2 extent-finding logic.
    Cost: a few queries per extension row.

Phase 3 – Interior fill (0 samples)
    Fill each detected run separately.  No oracle calls.

Phase 4 – Hole / gap detection
    Sample a sparse grid (step_r × step_c) inside the blob bounding box.
    Any pred=1 pixel found to be 0 by the oracle is validated by a
    4-neighbour query.  Confirmed seeds are passed to Phase 5.

Phase 5 – Hole/gap fill via BFS (budget-safe)
    BFS-flood from each confirmed hole seed, querying adjacent pixels that
    are currently pred=1.  Pixels confirmed as 0 are marked and propagated.
    BFS does NOT propagate through unqueried pred=0 pixels (prevents
    background explosion).

Budget  : 15 % of total pixels (1 500 for 50×200).
Tracking: accuracy checkpoint every checkpoint_interval samples.
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
    """Geometry-driven estimator: scanline binary search + hole BFS.

    Parameters
    ----------
    checkpoint_interval : emit checkpoint callback every this many queries
    hole_grid_steps     : (step_r, step_c) for hole-detection grid
    probe_step          : column step for right-wing probing after Phase 2
                          binary search  (smaller → catches narrower gaps)
    max_retries         : per-row retries in Phase 1 when validation fails
                          (helps skip isolated outlier pixels)
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

        # ---- inner helpers ----------------------------------------

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
        # Phase 1 – Binary search per scanline → one confirmed seed per row
        # ----------------------------------------------------------------
        # Retries handle isolated outlier 1-pixels: if validation fails,
        # resume binary search to the right of the rejected candidate.
        # ----------------------------------------------------------------
        row_seeds: dict[int, int] = {}   # row → confirmed 1-col

        for row in range(H):
            if budget_left() <= 0:
                break

            lo = 0
            for _attempt in range(self.max_retries + 1):
                if lo >= W or budget_left() <= 0:
                    break

                hi = W - 1
                found_col: int | None = None

                while lo <= hi and budget_left() > 0:
                    mid = (lo + hi) // 2
                    v = query(row, mid)
                    if v == 1:
                        found_col = mid
                        hi = mid - 1      # push left for leftmost 1
                    else:
                        lo = mid + 1

                if found_col is None:
                    break   # no more 1s on this row

                # Validate: at least one 4-neighbour must also be 1
                r, c = row, found_col
                confirmed = False
                for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                    if 0 <= nr < H and 0 <= nc < W and budget_left() > 0:
                        if query(nr, nc) == 1:
                            confirmed = True
                            break

                if confirmed:
                    row_seeds[row] = found_col
                    break
                else:
                    # Candidate was an isolated outlier; retry right of it
                    lo = found_col + 1

        # ----------------------------------------------------------------
        # Phase 2 – Find blob extents per seeded row
        # ----------------------------------------------------------------
        # Step A: binary search for left_edge and initial right_edge.
        # Step B: probe rightward at probe_step to detect additional runs
        #         (e.g. a right wing separated from the main blob by a gap).
        #         For each probe that finds a 1, binary search for its
        #         right extent and update right_edge.
        #
        # Per-run fill: check if seed col is at a run boundary.  If col
        # to its left is 0, the left binary-search result is a SEPARATE run.
        # Fill each run independently to avoid creating large FP gap regions.
        # ----------------------------------------------------------------
        # row_runs maps row → list of (left, right) spans to fill
        row_runs: dict[int, list[tuple[int, int]]] = {}

        def _find_extents_for_row(row: int, seed_col: int) -> None:
            """Compute runs for `row` using `seed_col` as anchor."""
            if budget_left() <= 0:
                return

            # -- Step A: left binary search --
            lo, hi = 0, seed_col
            left_edge = seed_col
            while lo < hi and budget_left() > 0:
                mid = (lo + hi) // 2
                if query(row, mid) == 1:
                    left_edge = mid
                    hi = mid
                else:
                    lo = mid + 1
                    left_edge = lo

            # -- Step A: right binary search (first contiguous run) --
            lo, hi = seed_col, W - 1
            right_edge = seed_col
            while lo < hi and budget_left() > 0:
                mid = (lo + hi + 1) // 2
                if query(row, mid) == 1:
                    right_edge = mid
                    lo = mid
                else:
                    hi = mid - 1
                    right_edge = lo

            # -- Detect run boundary: is there a gap left of seed? --
            # Check the pixel immediately left of seed_col.
            # If it's 0 (or seed_col == 0), the seed sits at the start of its
            # own run, so left_edge belongs to a SEPARATE left run (if any).
            seed_at_boundary = False
            if seed_col == 0:
                seed_at_boundary = True
            elif left_edge < seed_col and budget_left() > 0:
                left_of_seed = query(row, seed_col - 1)
                if left_of_seed == 0:
                    seed_at_boundary = True

            runs: list[tuple[int, int]] = []

            if seed_at_boundary and left_edge < seed_col:
                # left_edge..seed_col-1 is a separate left run; find its right end.
                lo2, hi2 = left_edge, seed_col - 1
                left_run_right = left_edge
                while lo2 < hi2 and budget_left() > 0:
                    mid2 = (lo2 + hi2 + 1) // 2
                    if query(row, mid2) == 1:
                        left_run_right = mid2
                        lo2 = mid2
                    else:
                        hi2 = mid2 - 1
                        left_run_right = lo2
                runs.append((left_edge, left_run_right))
                # Seed's run starts at seed_col
                cur_right = right_edge
                # Step B: probe rightward for additional 1-runs from cur_right
                pos = cur_right + self.probe_step
                while pos <= W - 1 and budget_left() > 0:
                    v = query(row, pos)
                    if v == 1:
                        lo2, hi2 = pos, W - 1
                        cur_right = pos
                        while lo2 < hi2 and budget_left() > 0:
                            mid2 = (lo2 + hi2 + 1) // 2
                            if query(row, mid2) == 1:
                                cur_right = mid2
                                lo2 = mid2
                            else:
                                hi2 = mid2 - 1
                                cur_right = lo2
                        pos = cur_right + self.probe_step
                    else:
                        pos += self.probe_step
                runs.append((seed_col, cur_right))
            else:
                # Seed is connected to left_edge — one contiguous block so far
                cur_right = right_edge
                # Step B: probe rightward for additional 1-runs
                pos = cur_right + self.probe_step
                while pos <= W - 1 and budget_left() > 0:
                    v = query(row, pos)
                    if v == 1:
                        lo2, hi2 = pos, W - 1
                        cur_right = pos
                        while lo2 < hi2 and budget_left() > 0:
                            mid2 = (lo2 + hi2 + 1) // 2
                            if query(row, mid2) == 1:
                                cur_right = mid2
                                lo2 = mid2
                            else:
                                hi2 = mid2 - 1
                                cur_right = lo2
                        pos = cur_right + self.probe_step
                    else:
                        pos += self.probe_step
                runs.append((left_edge, cur_right))

            row_runs[row] = runs

        # Hole-crossing rows can get shrunken extents; smooth by neighbours.
        row_extents = self._smooth_row_extents(row_extents, H)

        # ----------------------------------------------------------------
        # Phase 1 binary search is non-monotone: if the blob for a row is
        # confined to extreme columns, the midpoint probes may all land in
        # gaps.  Fix: after Phase 2, probe downward from the last seeded row
        # using the midpoint of the previous row's span as a starting probe.
        # ----------------------------------------------------------------
        seeded_rows = sorted(row_runs.keys())
        if seeded_rows:
            last_row = seeded_rows[-1]
            for ext_row in range(last_row + 1, H):
                if budget_left() <= 0:
                    break
                prev_runs = row_runs[ext_row - 1]
                # Use midpoint of overall extent of previous row as probe
                prev_left = min(l for l, _ in prev_runs)
                prev_right = max(r for _, r in prev_runs)
                probe_c = (prev_left + prev_right) // 2
                if query(ext_row, probe_c) == 1:
                    # Validate
                    confirmed = False
                    for nr, nc in [(ext_row - 1, probe_c), (ext_row + 1, probe_c),
                                   (ext_row, probe_c - 1), (ext_row, probe_c + 1)]:
                        if 0 <= nr < H and 0 <= nc < W and budget_left() > 0:
                            if query(nr, nc) == 1:
                                confirmed = True
                                break
                    if confirmed:
                        row_seeds[ext_row] = probe_c
                        _find_extents_for_row(ext_row, probe_c)
                    else:
                        break   # no more blob rows below
                else:
                    break   # row is background — stop extending

        # ----------------------------------------------------------------
        # Phase 3 – Fill blob interior (no queries)
        # ----------------------------------------------------------------
        for row, runs in row_runs.items():
            for left, right in runs:
                pred[row, left:right + 1] = 1

        # ----------------------------------------------------------------
        # Phase 4 – Hole / gap detection: sparse grid inside blob bbox
        # ----------------------------------------------------------------
        hole_seeds: list[tuple[int, int]] = []

        if row_runs and budget_left() > 0:
            all_rows = sorted(row_runs)
            min_row, max_row = all_rows[0], all_rows[-1]
            min_col = min(l for runs in row_runs.values() for l, _ in runs)
            max_col = max(r for runs in row_runs.values() for _, r in runs)

            step_r, step_c = self.hole_grid_steps
            visited_holes: set[tuple[int, int]] = set()

            for r in range(min_row, max_row + 1, step_r):
                if budget_left() <= 0:
                    break
                for c in range(min_col, max_col + 1, step_c):
                    if budget_left() <= 0:
                        break
                    if pred[r, c] != 1:
                        continue        # not in blob fill — skip
                    if (r, c) in visited_holes:
                        continue

                    v = query(r, c)
                    if v != 0:
                        continue

                    # Validate: at least one 4-neighbour is also 0
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
        # Phase 5 – Hole/gap fill: BFS from each confirmed hole seed
        # ----------------------------------------------------------------
        # Each BFS expands by querying adjacent pred=1 pixels.  Pixels
        # confirmed as 0 are marked and added to the queue.  We do NOT
        # propagate through unqueried pred=0 pixels (prevents background
        # explosion that wastes the entire remaining budget).
        # ----------------------------------------------------------------
        bfs_seen: set[tuple[int, int]] = set()

        for hr, hc in hole_seeds:
            if pred[hr, hc] != 0:
                continue    # already absorbed by a previous BFS run
            if (hr, hc) in bfs_seen:
                continue

            queue: deque[tuple[int, int]] = deque()
            queue.append((hr, hc))
            bfs_seen.add((hr, hc))
            pred[hr, hc] = 0     # already 0 from Phase 4 query; explicit

            while queue and budget_left() > 0:
                r, c = queue.popleft()
                for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if (nr, nc) in bfs_seen:
                        continue
                    bfs_seen.add((nr, nc))

                    if pred[nr, nc] == 1:
                        # Query: might be a hole pixel over-filled in Phase 3
                        if budget_left() > 0:
                            v = query(nr, nc)
                            if v == 0:
                                pred[nr, nc] = 0
                                queue.append((nr, nc))
                    # pred == 0: unqueried background — do NOT propagate
                    # (avoids budget-exploding traversal of the background)

        # ----------------------------------------------------------------
        # Final checkpoint
        # ----------------------------------------------------------------
        if checkpoint_fn is not None:
            checkpoint_fn(n_sampled, pred.copy())

        return pred
