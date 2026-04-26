"""Diagnostic artifact helpers for strategy visualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluator import ACCURACY_THRESHOLD, _bbox_passes, _blob_bbox, _per_layer_accuracy


PHASE_IDS = {
    "preplanned": 0,
    "coarse": 1,
    "boundary": 2,
    "entropy": 3,
    "unknown": 9,
}


@dataclass(frozen=True)
class StrategySummary:
    """Serializable final metrics for one seed/algo run."""

    seed: int
    algo: str
    grid_shape: tuple[int, int]
    iteration_cap: int
    overall_pass: bool
    per_layer_accuracy: list[float]
    min_accuracy: float
    accuracy_pass: list[bool]
    height_truth: list[int]
    height_pred: list[int]
    height_pass: list[bool]
    width_truth: list[int]
    width_pred: list[int]
    width_pass: list[bool]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "algo": self.algo,
            "grid_shape": list(self.grid_shape),
            "iteration_cap": self.iteration_cap,
            "overall_pass": self.overall_pass,
            "per_layer_accuracy": self.per_layer_accuracy,
            "min_accuracy": self.min_accuracy,
            "accuracy_pass": self.accuracy_pass,
            "height_truth": self.height_truth,
            "height_pred": self.height_pred,
            "height_pass": self.height_pass,
            "width_truth": self.width_truth,
            "width_pred": self.width_pred,
            "width_pass": self.width_pass,
        }


def phase_id_for_algorithm(algorithm) -> int:
    """Return a stable phase id for algorithms that expose diagnostics."""
    if hasattr(algorithm, "diagnostic_phase"):
        return PHASE_IDS.get(algorithm.diagnostic_phase(), PHASE_IDS["unknown"])
    return PHASE_IDS["preplanned"]


def make_error_map(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """
    Encode prediction-vs-truth differences.

    Codes:
      0: true negative
      1: true positive
      2: false positive
      3: false negative
    """
    pred = predicted.astype(bool)
    target = truth.astype(bool)
    error = np.zeros(predicted.shape, dtype=np.uint8)
    error[pred & target] = 1
    error[pred & ~target] = 2
    error[~pred & target] = 3
    return error


def summarize_final_result(
    *,
    seed: int,
    algo: str,
    truth_blob: np.ndarray,
    predicted_final: np.ndarray,
    iteration_cap: int,
    phase: int = 0,
) -> StrategySummary:
    """Build final scoring metrics without changing evaluator behavior."""
    _, H, W = truth_blob.shape
    final_acc = _per_layer_accuracy(predicted_final, truth_blob)

    height_truth = []
    width_truth = []
    height_pred = []
    width_pred = []
    height_pass = []
    width_pass = []

    for k in range(truth_blob.shape[0]):
        ht, wt = _blob_bbox(truth_blob[k])
        hp, wp = _blob_bbox(predicted_final[k])
        hok, wok = _bbox_passes(predicted_final[k], truth_blob[k], phase)
        height_truth.append(int(ht))
        width_truth.append(int(wt))
        height_pred.append(int(hp))
        width_pred.append(int(wp))
        height_pass.append(bool(hok))
        width_pass.append(bool(wok))

    accuracy_pass = final_acc >= ACCURACY_THRESHOLD
    overall_pass = bool(accuracy_pass.all() and all(height_pass) and all(width_pass))

    return StrategySummary(
        seed=seed,
        algo=algo,
        grid_shape=(H, W),
        iteration_cap=iteration_cap,
        overall_pass=overall_pass,
        per_layer_accuracy=[float(v) for v in final_acc],
        min_accuracy=float(final_acc.min()),
        accuracy_pass=[bool(v) for v in accuracy_pass],
        height_truth=height_truth,
        height_pred=height_pred,
        height_pass=height_pass,
        width_truth=width_truth,
        width_pred=width_pred,
        width_pass=width_pass,
    )
