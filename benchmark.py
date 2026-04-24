"""
Phase 0 benchmark runner.

Usage
-----
    python benchmark.py              # run all 10 seeds
    python benchmark.py --seed 0     # single seed
    python benchmark.py --seed 0 3 7 # specific seeds
"""

import argparse
import json
import sys
import numpy as np

from src.generator import generate_dataset
from src.evaluator import Evaluator, PHASE0_SEEDS
from src.algorithms.preplanned_greedy import PreplannedGreedy

H, W = 50, 200


def run_seed(seed: int, verbose: bool = True) -> dict:
    if verbose:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

    truth_blob, truth_out, truth_full = generate_dataset(H, W, seed)
    evaluator = Evaluator(truth_blob, truth_out, truth_full)

    algorithm = PreplannedGreedy(H, W, evaluator.iteration_cap, radius=1)

    result = evaluator.run(algorithm, seed=seed, phase=0)

    if verbose:
        print(result.summary())
        # Show accuracy curve milestones
        curve = result.accuracy_curve          # (n_iter, 8)
        milestones = [100, 300, 500, 750, 1000, 1250, 1499]
        print("\n  Accuracy curve (min over 8 layers):")
        for m in milestones:
            if m < len(curve):
                print(f"    iter {m+1:>4}: {curve[m].min():.4f}")

    return {
        "seed": seed,
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
        help="Seeds to evaluate (default: all PHASE0_SEEDS)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print machine-readable JSON summary at the end",
    )
    args = parser.parse_args()

    seeds = args.seed if args.seed is not None else PHASE0_SEEDS

    print(f"Phase 0 benchmark  |  grid {H}×{W}  |  "
          f"iteration_cap {int(0.15*H*W)}  |  seeds {seeds}")

    results = []
    for s in seeds:
        results.append(run_seed(s, verbose=True))

    # ---- aggregate summary ----
    n = len(results)
    n_pass = sum(r["overall_pass"] for r in results)
    mean_acc = np.mean([r["min_accuracy"] for r in results])

    print(f"\n{'='*60}")
    print(f"SUMMARY  {n_pass}/{n} seeds passed")
    print(f"Mean min-layer accuracy : {mean_acc:.4f}")
    print(f"{'='*60}")

    if args.json:
        print(json.dumps(results, indent=2))

    sys.exit(0 if n_pass == n else 1)


if __name__ == "__main__":
    main()
