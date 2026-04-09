"""
lse/evaluate.py
Run StraddleGPR on a synthetic map and report accuracy.

Usage
-----
    # from repo root
    python lse/evaluate.py
    python lse/evaluate.py --blob_size 0.4 --n_holes 3 --hole_size 6 \
                           --n_outliers 15 --seed 7 --kappa 1.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)   # sklearn ConvergenceWarning

# Allow running from repo root or from inside lse/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raw.synth.synthetic_map import generate_map
from lse.lse import StraddleGPR


def run(
    blob_size: float = 0.35,
    n_holes: int = 2,
    hole_size: float = 5.0,
    n_outliers: int = 10,
    seed: int = 42,
    kappa: float = 1.5,
    refit_interval: int = 50,
    out: str = "lse/result.png",
) -> dict:
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

    # ---- oracle wraps ground truth ----
    def oracle(row: int, col: int) -> int:
        return int(truth[row, col])

    # ---- estimate ----
    est = StraddleGPR(kappa=kappa, refit_interval=refit_interval)
    t0 = time.perf_counter()
    predicted = est.fit(oracle, (H, W), budget=budget)
    elapsed = time.perf_counter() - t0

    accuracy = float((predicted == truth).mean())
    print(f"Pixel acc   : {accuracy:.2%}")
    print(f"Elapsed     : {elapsed:.1f} s")

    # ---- visualise ----
    _save_png(truth, predicted, accuracy, elapsed, out)
    print(f"Saved PNG   → {out}")

    return {"accuracy": accuracy, "elapsed": elapsed}


def _save_png(
    truth: np.ndarray,
    predicted: np.ndarray,
    accuracy: float,
    elapsed: float,
    path: str,
) -> None:
    import matplotlib.pyplot as plt

    error = (predicted != truth).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(16, 3), dpi=110)

    axes[0].imshow(truth, cmap="gray", vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(predicted, cmap="gray", vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")
    axes[1].set_title(f"StraddleGPR  acc={accuracy:.2%}  t={elapsed:.1f}s")
    axes[1].axis("off")

    axes[2].imshow(error, cmap="Reds", vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")
    axes[2].set_title(f"Errors  ({error.sum()} px)")
    axes[2].axis("off")

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate StraddleGPR on synthetic map.")
    p.add_argument("--blob_size",      type=float, default=0.35)
    p.add_argument("--n_holes",        type=int,   default=2)
    p.add_argument("--hole_size",      type=float, default=5.0)
    p.add_argument("--n_outliers",     type=int,   default=10)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--kappa",          type=float, default=1.5)
    p.add_argument("--refit_interval", type=int,   default=50)
    p.add_argument("--out",            type=str,   default="lse/result.png")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        blob_size=args.blob_size,
        n_holes=args.n_holes,
        hole_size=args.hole_size,
        n_outliers=args.n_outliers,
        seed=args.seed,
        kappa=args.kappa,
        refit_interval=args.refit_interval,
        out=args.out,
    )
