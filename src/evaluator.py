"""
Evaluator: separate component that holds ground truth and scores the algorithm.

The algorithm never sees truth_blob_mask or truth_outlier_mask directly.
The evaluator wraps the oracle, runs the algorithm for iteration_cap steps,
and records the per-iteration accuracy curve plus the final pass/fail result.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List

from .oracle import Oracle

PHASE0_SEEDS = list(range(10))   # placeholder — freeze before first official run
ACCURACY_THRESHOLD = 0.98


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _per_layer_accuracy(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """mean(predicted == truth) over H×W, per layer. Returns shape (8,)."""
    return np.mean(predicted == truth, axis=(1, 2))


def _blob_bbox(mask2d: np.ndarray):
    """Return (height, width) of bounding box of a 2-D binary mask."""
    rows = np.any(mask2d, axis=1)
    cols = np.any(mask2d, axis=0)
    if not rows.any():
        return 0, 0
    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return rmax - rmin + 1, cmax - cmin + 1


def _bbox_passes(pred2d, truth2d, phase=0):
    ph, pw = _blob_bbox(pred2d)
    th, tw = _blob_bbox(truth2d)
    if phase == 0:
        tol_h = max(1, int(0.05 * th))
        tol_w = max(1, int(0.05 * tw))
        return abs(ph - th) <= tol_h, abs(pw - tw) <= tol_w
    return ph == th, pw == tw


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    seed: int = 0
    phase: str = "phase0"
    grid_shape: tuple = (50, 200)
    iteration_cap: int = 0
    iterations_used: int = 0

    # Per-layer metrics  (length-8 arrays)
    per_layer_accuracy: np.ndarray = field(default_factory=lambda: np.zeros(8))
    height_truth: np.ndarray = field(default_factory=lambda: np.zeros(8, int))
    height_pred: np.ndarray = field(default_factory=lambda: np.zeros(8, int))
    width_truth: np.ndarray = field(default_factory=lambda: np.zeros(8, int))
    width_pred: np.ndarray = field(default_factory=lambda: np.zeros(8, int))
    height_pass: np.ndarray = field(default_factory=lambda: np.zeros(8, bool))
    width_pass: np.ndarray = field(default_factory=lambda: np.zeros(8, bool))
    accuracy_pass: np.ndarray = field(default_factory=lambda: np.zeros(8, bool))
    overall_pass: bool = False

    # Accuracy curve: shape (iterations_used, 8)
    accuracy_curve: np.ndarray = field(default_factory=lambda: np.empty((0, 8)))

    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== EvalResult  seed={self.seed}  phase={self.phase} ===",
            f"  grid          : {self.grid_shape[0]}×{self.grid_shape[1]}",
            f"  iterations    : {self.iterations_used} / {self.iteration_cap}",
            f"  elapsed       : {self.elapsed_seconds:.3f}s",
            f"  overall_pass  : {self.overall_pass}",
            "",
            f"  {'layer':>5}  {'accuracy':>9}  {'acc_ok':>6}  "
            f"{'h_truth':>7}  {'h_pred':>6}  {'h_ok':>4}  "
            f"{'w_truth':>7}  {'w_pred':>6}  {'w_ok':>4}",
        ]
        for k in range(8):
            lines.append(
                f"  {k:>5}  {self.per_layer_accuracy[k]:>9.4f}  "
                f"{str(self.accuracy_pass[k]):>6}  "
                f"{self.height_truth[k]:>7}  {self.height_pred[k]:>6}  "
                f"{str(self.height_pass[k]):>4}  "
                f"{self.width_truth[k]:>7}  {self.width_pred[k]:>6}  "
                f"{str(self.width_pass[k]):>4}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Holds ground truth, wraps the oracle, and runs the algorithm.

    The algorithm receives only the Oracle interface. It never has access
    to truth_blob_mask or truth_outlier_mask.
    """

    def __init__(self, truth_blob_mask, truth_outlier_mask, truth_full_mask):
        self.truth_blob = truth_blob_mask       # (8, H, W)
        self.truth_out = truth_outlier_mask     # (8, H, W)
        self.truth_full = truth_full_mask       # (8, H, W)
        _, self.H, self.W = truth_full_mask.shape
        self.n_layers = truth_full_mask.shape[0]
        self.iteration_cap = int(0.15 * self.H * self.W)

    def run(self, algorithm, seed: int = 0, phase: int = 0) -> EvalResult:
        """
        Run the algorithm for iteration_cap steps and return an EvalResult.

        The algorithm is called as:
            row, col = algorithm.next_query()
            labels   = oracle.query(row, col)
            algorithm.update(row, col, labels)
            predicted = algorithm.predict()   # (8, H, W) uint8
        """
        oracle = Oracle(self.truth_full)
        curve: List[np.ndarray] = []

        t0 = time.perf_counter()
        for _ in range(self.iteration_cap):
            row, col = algorithm.next_query()
            labels = oracle.query(row, col)
            algorithm.update(row, col, labels)
            predicted = algorithm.predict()
            curve.append(_per_layer_accuracy(predicted, self.truth_blob))
        elapsed = time.perf_counter() - t0

        predicted_final = algorithm.predict()
        final_acc = _per_layer_accuracy(predicted_final, self.truth_blob)

        result = EvalResult(
            seed=seed,
            phase=f"phase{phase}",
            grid_shape=(self.H, self.W),
            iteration_cap=self.iteration_cap,
            iterations_used=self.iteration_cap,
            per_layer_accuracy=final_acc,
            accuracy_curve=np.array(curve),   # (n_iter, 8)
            elapsed_seconds=elapsed,
        )

        h_truth = np.zeros(self.n_layers, int)
        w_truth = np.zeros(self.n_layers, int)
        h_pred = np.zeros(self.n_layers, int)
        w_pred = np.zeros(self.n_layers, int)
        h_pass = np.zeros(self.n_layers, bool)
        w_pass = np.zeros(self.n_layers, bool)

        for k in range(self.n_layers):
            ht, wt = _blob_bbox(self.truth_blob[k])
            hp, wp = _blob_bbox(predicted_final[k])
            h_truth[k], w_truth[k] = ht, wt
            h_pred[k], w_pred[k] = hp, wp
            hok, wok = _bbox_passes(predicted_final[k], self.truth_blob[k], phase)
            h_pass[k], w_pass[k] = hok, wok

        result.height_truth = h_truth
        result.width_truth = w_truth
        result.height_pred = h_pred
        result.width_pred = w_pred
        result.height_pass = h_pass
        result.width_pass = w_pass
        result.accuracy_pass = final_acc >= ACCURACY_THRESHOLD
        result.overall_pass = bool(
            result.accuracy_pass.all()
            and result.height_pass.all()
            and result.width_pass.all()
        )

        return result
