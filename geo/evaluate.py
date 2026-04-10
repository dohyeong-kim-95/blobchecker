"""
geo/evaluate.py
Run GeoEstimator on three synthetic maps and report aggregate results.

Outputs (all written to --out-dir, default geo/output/)
-------
  truth_<i>.png          ground truth binary map for blob i
  predicted_<i>.png      predicted binary map for blob i
  accuracy_curve.png     accuracy vs samples for all three blobs
  summary.png            combined summary panel

Printed
-------
  per-blob pixel accuracy / elapsed / peak memory
  aggregate: avg pixel accuracy, max elapsed, max peak memory

Usage
-----
    python geo/evaluate.py
    python geo/evaluate.py --out-dir /tmp/geo_out
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raw.synth.synthetic_map import generate_map
from geo.geo import GeoEstimator


# ---------------------------------------------------------------------------
# Three canonical test blobs
# ---------------------------------------------------------------------------

BLOBS = [
    dict(blob_size=0.35, n_holes=2, hole_size=5.0, n_outliers=10, seed=42),
    dict(blob_size=0.40, n_holes=3, hole_size=6.0, n_outliers=15, seed=7),
    dict(blob_size=0.30, n_holes=1, hole_size=4.0, n_outliers=8,  seed=99),
]

COLORS = ["seagreen", "steelblue", "darkorange"]


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
    all_steps: list[list[int]],
    all_accuracies: list[list[float]],
    budget: int,
    path: Path,
    target: float = 97.5,
) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    for i, (steps, accs) in enumerate(zip(all_steps, all_accuracies)):
        ax.plot(steps, [a * 100 for a in accs],
                marker="o", markersize=3, linewidth=1.5,
                color=COLORS[i], label=f"Blob {i+1} (seed={BLOBS[i]['seed']})")
    ax.axhline(target, color="crimson", linewidth=1, linestyle="--",
               label=f"{target:.0f}% target")
    ax.set_xlabel("Samples queried")
    ax.set_ylabel("Pixel accuracy (%)")
    ax.set_title("Accuracy vs samples  (GeoEstimator, 3 blobs)")
    ax.set_xlim(0, budget)
    ax.set_ylim(0, 101)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_summary(
    results: list[dict],
    all_steps: list[list[int]],
    all_accuracies: list[list[float]],
    budget: int,
    path: Path,
    target: float = 97.5,
) -> None:
    import matplotlib.pyplot as plt

    n = len(results)
    fig = plt.figure(figsize=(13, 3 * n + 4), dpi=110)
    gs = fig.add_gridspec(n + 2, 2,
                          height_ratios=[1] * n + [2, 2],
                          hspace=0.55, wspace=0.25)

    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[i, :])
        ax.imshow(res["predicted"], cmap="gray", vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")
        ax.set_title(
            f"Blob {i+1} (seed={BLOBS[i]['seed']})  acc={res['accuracy']:.2%}",
            fontsize=9)
        ax.axis("off")

    ax_c = fig.add_subplot(gs[n, :])
    for i, (steps, accs) in enumerate(zip(all_steps, all_accuracies)):
        ax_c.plot(steps, [a * 100 for a in accs],
                  marker="o", markersize=3, linewidth=1.5,
                  color=COLORS[i], label=f"Blob {i+1}")
    ax_c.axhline(target, color="crimson", linewidth=1, linestyle="--",
                 label=f"{target:.0f}% target")
    ax_c.set_xlabel("Samples queried")
    ax_c.set_ylabel("Pixel accuracy (%)")
    ax_c.set_title("Accuracy vs samples")
    ax_c.set_xlim(0, budget)
    ax_c.set_ylim(0, 101)
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3)

    avg_acc = np.mean([r["accuracy"] for r in results])
    max_elapsed = max(r["elapsed"] for r in results)
    max_mem = max(r["peak_mib"] for r in results)

    ax_m = fig.add_subplot(gs[n + 1, :])
    ax_m.axis("off")
    lines = [
        f"{'Blob':<6}  {'Accuracy':>9}  {'Elapsed':>9}  {'Peak mem':>10}",
        "-" * 42,
    ]
    for i, res in enumerate(results):
        lines.append(
            f"{'#'+str(i+1):<6}  {res['accuracy']:>9.2%}  "
            f"{res['elapsed']:>8.2f}s  {res['peak_mib']:>8.1f} MiB"
        )
    lines += [
        "-" * 42,
        f"{'avg/max':<6}  {avg_acc:>9.2%}  {max_elapsed:>8.2f}s  {max_mem:>8.1f} MiB",
    ]
    ax_m.text(
        0.05, 0.95, "\n".join(lines),
        va="top", ha="left", fontsize=9, fontfamily="monospace",
        transform=ax_m.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )

    fig.suptitle("GeoEstimator — 3-Blob Summary", fontsize=12, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    checkpoint_interval: int = 50,
    out_dir: str = "geo/output",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    all_steps: list[list[int]] = []
    all_accuracies: list[list[float]] = []
    budget = None  # computed from first blob; same grid size for all

    for i, blob_cfg in enumerate(BLOBS):
        truth = generate_map(**blob_cfg)
        H, W = truth.shape
        b = int(0.15 * H * W)
        if budget is None:
            budget = b

        print(f"\n── Blob {i+1}  (seed={blob_cfg['seed']}) ──")
        print(f"Grid        : {H}×{W}  ({H*W} pixels)")
        print(f"Budget      : {b} samples  (15%)")
        print(f"Truth cover : {truth.mean():.2%}")

        _save_binary_png(truth, out / f"truth_{i+1}.png",
                         f"Ground truth blob {i+1}  (coverage {truth.mean():.2%})")

        steps: list[int] = []
        accuracies: list[float] = []

        def on_checkpoint(n_sampled: int, pred: np.ndarray,
                          _truth: np.ndarray = truth) -> None:
            steps.append(n_sampled)
            accuracies.append(float((_truth == pred).mean()))

        oracle = lambda r, c, _t=truth: int(_t[r, c])

        est = GeoEstimator(checkpoint_interval=checkpoint_interval)
        t0 = time.perf_counter()
        predicted = est.fit(oracle, (H, W), budget=b, checkpoint_fn=on_checkpoint)
        elapsed = time.perf_counter() - t0
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        accuracy = float((predicted == truth).mean())
        peak_mib = peak_kb / 1024

        _save_binary_png(predicted, out / f"predicted_{i+1}.png",
                         f"GeoEstimator blob {i+1}  (acc {accuracy:.2%})")

        print(f"Pixel acc   : {accuracy:.2%}")
        print(f"Elapsed     : {elapsed:.2f} s")
        print(f"Peak memory : {peak_mib:.1f} MiB")

        results.append({
            "accuracy": accuracy,
            "elapsed": elapsed,
            "peak_mib": peak_mib,
            "predicted": predicted,
            "truth": truth,
        })
        all_steps.append(steps)
        all_accuracies.append(accuracies)

    avg_acc = float(np.mean([r["accuracy"] for r in results]))
    max_elapsed = max(r["elapsed"] for r in results)
    max_mem = max(r["peak_mib"] for r in results)

    print(f"\n── Aggregate ──")
    print(f"Avg pixel acc : {avg_acc:.2%}")
    print(f"Max elapsed   : {max_elapsed:.2f} s")
    print(f"Max peak mem  : {max_mem:.1f} MiB")

    _save_accuracy_curve(all_steps, all_accuracies, budget, out / "accuracy_curve.png")
    print(f"\nSaved → {out}/accuracy_curve.png")

    _save_summary(results, all_steps, all_accuracies, budget, out / "summary.png")
    print(f"Saved → {out}/summary.png")

    return {
        "avg_accuracy": avg_acc,
        "max_elapsed": max_elapsed,
        "max_peak_mib": max_mem,
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GeoEstimator on 3 synthetic blobs.")
    p.add_argument("--checkpoint_interval", type=int, default=50)
    p.add_argument("--out-dir", type=str, default="geo/output", dest="out_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(checkpoint_interval=args.checkpoint_interval, out_dir=args.out_dir)
