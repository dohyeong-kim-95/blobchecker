"""
Phase 0 benchmark runner.

Usage
-----
    python benchmark.py                          # all seeds, geometry_first
    python benchmark.py --algo preplanned        # baseline only
    python benchmark.py --algo both              # compare both
    python benchmark.py --seed 0 3 7             # specific seeds
    python benchmark.py --suite validation       # validation seeds 100..199
    python benchmark.py --suite both             # public + validation suites
    python benchmark.py --seed-range 100 110     # half-open custom range
"""

import argparse
import numpy as np

from src.evaluator import Evaluator, PHASE0_SEEDS, PHASE0_VALIDATION_SEEDS

H, W = 50, 200


def _select_seed_suites(args) -> dict[str, list[int]]:
    if args.seed is not None:
        return {"custom": args.seed}
    if args.seed_range is not None:
        start, stop = args.seed_range
        if stop <= start:
            raise ValueError("--seed-range requires STOP > START")
        return {"custom": list(range(start, stop))}
    if args.suite == "public":
        return {"public": PHASE0_SEEDS}
    if args.suite == "validation":
        return {"validation": PHASE0_VALIDATION_SEEDS}
    return {
        "public": PHASE0_SEEDS,
        "validation": PHASE0_VALIDATION_SEEDS,
    }


def run_seed(seed: int, algo_name: str, verbose: bool = True) -> dict:
    from src.generator import generate_dataset
    from src.algorithms.preplanned_greedy import PreplannedGreedy
    from src.algorithms.geometry_first import GeometryFirstAdaptive

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
        help="Explicit seed list. Overrides --suite and --seed-range.",
    )
    parser.add_argument(
        "--seed-range", type=int, nargs=2, metavar=("START", "STOP"),
        default=None,
        help="Use a half-open custom seed range [START, STOP).",
    )
    parser.add_argument(
        "--suite", choices=["public", "validation", "both"],
        default="public",
        help="Seed suite to run when --seed/--seed-range is not supplied.",
    )
    parser.add_argument(
        "--algo", choices=["preplanned", "geometry_first", "both"],
        default="geometry_first",
    )
    args = parser.parse_args()

    seed_suites = _select_seed_suites(args)
    algos = (
        ["preplanned", "geometry_first"] if args.algo == "both"
        else [args.algo]
    )

    print(f"Phase 0 benchmark  |  grid {H}×{W}  |  "
          f"cap {int(0.15*H*W)}  |  suites {list(seed_suites)}  |  "
          f"algos {algos}")

    all_results: dict[str, dict[str, list]] = {
        suite_name: {a: [] for a in algos}
        for suite_name in seed_suites
    }

    for suite_name, seeds in seed_suites.items():
        print(f"\nSuite {suite_name}: seeds {seeds}")
        for s in seeds:
            for a in algos:
                all_results[suite_name][a].append(run_seed(s, a, verbose=True))

    # ── aggregate summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*60}")
    for suite_name in seed_suites:
        print(f"\n  Suite: {suite_name}")
        for a in algos:
            rs = all_results[suite_name][a]
            n = len(rs)
            n_pass = sum(r["overall_pass"] for r in rs)
            mean_acc = np.mean([r["min_accuracy"] for r in rs])
            mean_t   = np.mean([r["elapsed_seconds"] for r in rs])
            print(f"    {a:20s}  pass {n_pass}/{n}  "
                  f"mean_min_acc {mean_acc:.4f}  mean_time {mean_t:.1f}s")


if __name__ == "__main__":
    main()
