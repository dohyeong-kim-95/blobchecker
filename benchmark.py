"""
Phase 0 benchmark runner.

Usage
-----
    python benchmark.py                          # all seeds, geometry_first
    python benchmark.py --algo preplanned        # baseline only
    python benchmark.py --algo both              # compare both
    python benchmark.py --seed 0 3 7             # specific seeds
"""

import argparse
import time
import numpy as np

from src.generator import generate_dataset
from src.evaluator import Evaluator, PHASE0_SEEDS
from src.algorithms.preplanned_greedy import PreplannedGreedy
from src.algorithms.geometry_first import GeometryFirstAdaptive

H, W = 50, 200


def run_seed(seed: int, algo_name: str, verbose: bool = True) -> dict:
    if verbose:
        print(f"\n{'='*60}")
        print(f"Seed {seed}  |  algo: {algo_name}")
        print(f"{'='*60}")

    truth_blob, truth_out, truth_full = generate_dataset(H, W, seed)
    evaluator = Evaluator(truth_blob, truth_out, truth_full)

    cap = evaluator.iteration_cap
    if algo_name == "preplanned":
        algorithm = PreplannedGreedy(H, W, cap, radius=1)
    elif algo_name == "geometry_first":
        algorithm = GeometryFirstAdaptive(H, W, cap)
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    result = evaluator.run(algorithm, seed=seed, phase=0)

    if verbose:
        print(result.summary())
        curve = result.accuracy_curve          # (n_iter, 8)
        milestones = [99, 299, 499, 749, 999, 1249, 1499]
        print("\n  Accuracy curve (min over 8 layers):")
        for m in milestones:
            if m < len(curve):
                print(f"    iter {m+1:>4}: {curve[m].min():.4f}")

    return {
        "seed": seed,
        "algo": algo_name,
        "overall_pass": result.overall_pass,
        "per_layer_accuracy": result.per_layer_accuracy.tolist(),
        "min_accuracy": float(result.per_layer_accuracy.min()),
        "accuracy_pass": result.accuracy_pass.tolist(),
        "height_pass": result.height_pass.tolist(),
        "width_pass": result.width_pass.tolist(),
        "elapsed_seconds": result.elapsed_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, nargs="*", default=None,
    )
    parser.add_argument(
        "--algo", choices=["preplanned", "geometry_first", "both"],
        default="geometry_first",
    )
    args = parser.parse_args()

    seeds = args.seed if args.seed is not None else PHASE0_SEEDS
    algos = (
        ["preplanned", "geometry_first"] if args.algo == "both"
        else [args.algo]
    )

    print(f"Phase 0 benchmark  |  grid {H}×{W}  |  "
          f"cap {int(0.15*H*W)}  |  seeds {seeds}  |  algos {algos}")

    all_results: dict[str, list] = {a: [] for a in algos}

    for s in seeds:
        for a in algos:
            all_results[a].append(run_seed(s, a, verbose=True))

    # ── aggregate summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*60}")
    for a in algos:
        rs = all_results[a]
        n = len(rs)
        n_pass = sum(r["overall_pass"] for r in rs)
        mean_acc = np.mean([r["min_accuracy"] for r in rs])
        mean_t   = np.mean([r["elapsed_seconds"] for r in rs])
        print(f"  {a:20s}  pass {n_pass}/{n}  "
              f"mean_min_acc {mean_acc:.4f}  mean_time {mean_t:.1f}s")


if __name__ == "__main__":
    main()
