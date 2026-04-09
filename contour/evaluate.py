"""
contour/evaluate.py
Run ContourEstimator on a synthetic map and report results.

Outputs (written to --out-dir, default contour/output/)
-------------------------------------------------------
  truth.png          ground truth binary map
  predicted.png      final predicted binary map
  accuracy_curve.png accuracy vs samples (every 50 steps)

Printed
-------
  final pixel accuracy, peak memory, elapsed time

Usage
-----
    python contour/evaluate.py
    python contour/evaluate.py --blob_size 0.4 --n_holes 3 --hole_size 6 \
                               --n_outliers 15 --seed 7
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raw.synth.synthetic_map import generate_map
from contour.contour import ContourEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_binary_png(arr: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=110)
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_accuracy_curve(
    steps: list[int],
    accuracies: list[float],
    budget: int,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    ax.plot(steps, [a * 100 for a in accuracies], marker="o", markersize=3,
            linewidth=1.5, color="seagreen")
    ax.axhline(95, color="crimson", linewidth=1, linestyle="--", label="95% target")
    ax.set_xlabel("Samples queried")
    ax.set_ylabel("Pixel accuracy (%)")
    ax.set_title("Accuracy vs samples  (ContourEstimator)")
    ax.set_xlim(0, budget)
    ax.set_ylim(0, 101)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    blob_size: float = 0.35,
    n_holes: int = 2,
    hole_size: float = 5.0,
    n_outliers: int = 10,
    seed: int = 42,
    out_dir: str = "contour/output",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- ground truth ----
    truth = generate_map(
        blob_size=blob_size, n_holes=n_holes, hole_size=hole_size,
        n_outliers=n_outliers, seed=seed,
    )
    H, W = truth.shape
    budget = int(0.15 * H * W)

    print(f"Grid        : {H}×{W}  ({H*W} pixels)")
    print(f"Budget      : {budget} samples  (15%)")
    print(f"Truth cover : {truth.mean():.2%}")

    _save_binary_png(truth, out / "truth.png",
                     f"Ground truth  (coverage {truth.mean():.2%})")
    print(f"Saved       → {out}/truth.png")

    # ---- checkpoint tracking ----
    steps: list[int] = []
    accuracies: list[float] = []

    def on_checkpoint(n_sampled: int, pred: np.ndarray) -> None:
        accuracies.append(float((pred == truth).mean()))
        steps.append(n_sampled)

    # ---- oracle ----
    def oracle(row: int, col: int) -> int:
        return int(truth[row, col])

    # ---- estimate ----
    est = ContourEstimator()
    t0 = time.perf_counter()
    predicted = est.fit(oracle, (H, W), budget=budget, checkpoint_fn=on_checkpoint)
    elapsed = time.perf_counter() - t0
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    accuracy = float((predicted == truth).mean())

    # ---- save outputs ----
    _save_binary_png(predicted, out / "predicted.png",
                     f"ContourEstimator prediction  (acc {accuracy:.2%})")
    print(f"Saved       → {out}/predicted.png")

    _save_accuracy_curve(steps, accuracies, budget, out / "accuracy_curve.png")
    print(f"Saved       → {out}/accuracy_curve.png")

    print(f"Pixel acc   : {accuracy:.2%}")
    print(f"Elapsed     : {elapsed:.2f} s")
    print(f"Peak memory : {peak_kb / 1024:.1f} MiB")

    return {
        "accuracy": accuracy,
        "elapsed": elapsed,
        "peak_mib": peak_kb / 1024,
        "steps": steps,
        "accuracies": accuracies,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ContourEstimator on synthetic map."
    )
    p.add_argument("--blob_size",  type=float, default=0.35)
    p.add_argument("--n_holes",    type=int,   default=2)
    p.add_argument("--hole_size",  type=float, default=5.0)
    p.add_argument("--n_outliers", type=int,   default=10)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out-dir",    type=str,   default="contour/output",
                   dest="out_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        blob_size=args.blob_size,
        n_holes=args.n_holes,
        hole_size=args.hole_size,
        n_outliers=args.n_outliers,
        seed=args.seed,
        out_dir=args.out_dir,
    )
