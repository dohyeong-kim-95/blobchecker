"""
compare.py
Generate a single comparison image showing all estimator results side by side.

Default (fast): reads existing output PNGs from each method's output directory.
With --run-all : re-runs every estimator and generates fresh results.

Layout
------
  Row 0 (maps)  : Ground Truth | LSE | Sparse GP | Geo | Contour
  Row 1 (curves): metrics table| LSE curve | Sparse GP curve | Geo curve | Contour curve

Usage
-----
    python compare.py                    # use cached PNGs
    python compare.py --run-all          # re-run all methods (LSE ~2 min)
    python compare.py --out-dir results  # custom output directory
    python compare.py --skip lse         # skip slow LSE when using --run-all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS = [
    {
        "key":       "lse",
        "label":     "LSE\n(sklearn GP)",
        "truth":     "lse/output/truth.png",
        "predicted": "lse/output/predicted.png",
        "curve":     "lse/output/accuracy_curve.png",
        "color":     "#4878d0",
    },
    {
        "key":       "sparse_gp",
        "label":     "Sparse GP\n(GPyTorch FITC)",
        "truth":     "sparse_gp/output/truth.png",
        "predicted": "sparse_gp/output/predicted.png",
        "curve":     "sparse_gp/output/accuracy_curve.png",
        "color":     "#ee854a",
    },
    {
        "key":       "geo",
        "label":     "Geo\n(Scanline+BFS)",
        "truth":     "geo/output/truth.png",
        "predicted": "geo/output/predicted.png",
        "curve":     "geo/output/accuracy_curve.png",
        "color":     "#9370db",
    },
    {
        "key":       "contour",
        "label":     "Contour\n(Boundary search)",
        "truth":     "contour/output/truth.png",
        "predicted": "contour/output/predicted.png",
        "curve":     "contour/output/accuracy_curve.png",
        "color":     "#2ca02c",
    },
]

# Summary metrics (README values; updated when --run-all is used)
DEFAULT_METRICS = {
    "lse":       {"accuracy": "98.6%", "elapsed": "158 s",  "memory": "~700 MiB"},
    "sparse_gp": {"accuracy": "95.8%", "elapsed": "7.8 s",  "memory": "~695 MiB"},
    "geo":       {"accuracy": "96.7%", "elapsed": "<0.1 s", "memory": "<100 MiB"},
    "contour":   {"accuracy": "97.7%", "elapsed": "0.02 s", "memory": "~93 MiB"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_img(path: str | Path) -> np.ndarray | None:
    p = ROOT / path
    if not p.exists():
        return None
    return mpimg.imread(str(p))


def _run_all(skip: set[str]) -> dict[str, dict]:
    """Re-run each estimator; return dict of result dicts."""
    results: dict[str, dict] = {}

    if "lse" not in skip:
        print("Running LSE (sklearn GP) — may take ~2 minutes …")
        from lse.evaluate import run as lse_run
        results["lse"] = lse_run()

    if "sparse_gp" not in skip:
        print("Running Sparse GP …")
        from sparse_gp.evaluate import run as sgp_run
        results["sparse_gp"] = sgp_run()

    if "geo" not in skip:
        print("Running Geo …")
        from geo.evaluate import run as geo_run
        results["geo"] = geo_run()

    if "contour" not in skip:
        print("Running Contour …")
        from contour.evaluate import run as contour_run
        results["contour"] = contour_run()

    return results


def _metrics_table(metrics: dict[str, dict], skip: set[str]) -> str:
    lines = [
        "Method        Accuracy  Elapsed   Memory",
        "─" * 44,
    ]
    labels = {
        "lse":       "LSE (sklearn)",
        "sparse_gp": "Sparse GP",
        "geo":       "Geo",
        "contour":   "Contour",
    }
    for m in METHODS:
        key = m["key"]
        if key in skip:
            continue
        info = metrics.get(key, {})
        acc  = info.get("accuracy", "—")
        if isinstance(acc, float):
            acc = f"{acc:.2%}"
        ela  = info.get("elapsed", "—")
        if isinstance(ela, float):
            ela = f"{ela:.2f} s"
        mem  = info.get("peak_mib", info.get("memory", "—"))
        if isinstance(mem, float):
            mem = f"{mem:.0f} MiB"
        lines.append(f"{labels[key]:<14}{acc:<10}{ela:<10}{mem}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare(
    run_all: bool = False,
    skip: set[str] | None = None,
    out_dir: str = "output",
) -> None:
    skip = skip or set()
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)

    # ---- optionally re-run methods ----
    fresh_results: dict[str, dict] = {}
    if run_all:
        fresh_results = _run_all(skip)

    # Build per-method metrics (fresh if available, else README defaults)
    metrics: dict[str, dict] = {}
    for m in METHODS:
        key = m["key"]
        if key in fresh_results:
            r = fresh_results[key]
            metrics[key] = {
                "accuracy": r.get("accuracy", "—"),
                "elapsed":  r.get("elapsed",  "—"),
                "peak_mib": r.get("peak_mib", "—"),
            }
        else:
            metrics[key] = DEFAULT_METRICS[key]

    active = [m for m in METHODS if m["key"] not in skip]
    n = len(active)   # number of methods shown

    # ---- figure layout: 2 rows × (1 + n) cols ----
    # col 0: Ground Truth (row 0) + metrics table (row 1)
    # col 1..n: predicted map (row 0) + accuracy curve (row 1)
    fig = plt.figure(figsize=(5 * (n + 1), 9), dpi=110)
    gs  = fig.add_gridspec(2, n + 1, hspace=0.35, wspace=0.12)

    # ---- col 0, row 0: Ground Truth ----
    ax_truth = fig.add_subplot(gs[0, 0])
    truth_img = _load_img("contour/output/truth.png")
    if truth_img is None:
        ax_truth.text(0.5, 0.5, "truth.png\nnot found",
                      ha="center", va="center", transform=ax_truth.transAxes)
    else:
        ax_truth.imshow(truth_img)
    ax_truth.set_title("Ground Truth", fontsize=11, fontweight="bold")
    ax_truth.axis("off")

    # ---- col 0, row 1: metrics table ----
    ax_meta = fig.add_subplot(gs[1, 0])
    ax_meta.axis("off")
    table_txt = _metrics_table(metrics, skip)
    ax_meta.text(
        0.03, 0.97, table_txt,
        va="top", ha="left", fontsize=7.5,
        fontfamily="monospace",
        transform=ax_meta.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )

    # ---- cols 1..n: per-method predicted + curve ----
    for col_idx, m in enumerate(active, start=1):
        key   = m["key"]
        label = m["label"]
        color = m["color"]
        info  = metrics[key]

        acc = info.get("accuracy", "—")
        if isinstance(acc, float):
            acc = f"{acc:.2%}"
        ela = info.get("elapsed", "—")
        if isinstance(ela, float):
            ela = f"{ela:.2f} s"
        mem = info.get("peak_mib", info.get("memory", "—"))
        if isinstance(mem, float):
            mem = f"{mem:.0f} MiB"

        subtitle = f"acc {acc}  |  {ela}  |  {mem}"

        # row 0: predicted map
        ax_map = fig.add_subplot(gs[0, col_idx])
        pred_img = _load_img(m["predicted"])
        if pred_img is None:
            ax_map.text(0.5, 0.5, f"{m['predicted']}\nnot found",
                        ha="center", va="center", transform=ax_map.transAxes,
                        fontsize=9, color="gray")
            ax_map.set_facecolor("#f0f0f0")
        else:
            ax_map.imshow(pred_img)
        ax_map.set_title(f"{label}\n{subtitle}", fontsize=9, fontweight="bold",
                         color=color)
        ax_map.axis("off")

        # row 1: accuracy curve
        ax_curve = fig.add_subplot(gs[1, col_idx])
        curve_img = _load_img(m["curve"])
        if curve_img is None:
            ax_curve.text(0.5, 0.5, f"{m['curve']}\nnot found",
                          ha="center", va="center", transform=ax_curve.transAxes,
                          fontsize=9, color="gray")
            ax_curve.set_facecolor("#f0f0f0")
        else:
            ax_curve.imshow(curve_img)
        ax_curve.axis("off")

    fig.suptitle(
        "Binary Map Estimation — Method Comparison  (50×200 grid, 15% budget)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    out_path = out / "comparison.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate side-by-side comparison of all estimators."
    )
    p.add_argument(
        "--run-all", action="store_true",
        help="Re-run every estimator before compositing (slow for LSE).",
    )
    p.add_argument(
        "--skip", nargs="*", default=[],
        metavar="METHOD",
        help="Methods to skip: lse sparse_gp geo contour",
    )
    p.add_argument(
        "--out-dir", type=str, default="output",
        dest="out_dir",
        help="Directory to save comparison.png (default: output/)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compare(
        run_all=args.run_all,
        skip=set(args.skip),
        out_dir=args.out_dir,
    )
