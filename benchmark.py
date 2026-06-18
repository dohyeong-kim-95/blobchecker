"""
Benchmark runner — skeleton pending Shmoo generator implementation.

Grid: 150 x 200  |  Layers: 16  |  Cap: int(0.15 * H * W) = 4500

Usage (once generator and algorithms are implemented):
    python benchmark.py
    python benchmark.py --seed 0 3 7
    python benchmark.py --suite validation
    python benchmark.py --suite both
    python benchmark.py --seed-range 100 110
"""

import argparse
import sys

import numpy as np

from src.evaluator import Evaluator, PHASE0_SEEDS, PHASE0_VALIDATION_SEEDS

H, W = 150, 200
N_LAYERS = 16


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
    # Generator is implemented; algorithms are not yet designed for this domain.
    from src.shmoo import generate_dataset

    truth_blob, truth_out, truth_full = generate_dataset(
        H=H, W=W, seed=seed, n_layers=N_LAYERS
    )
    evaluator = Evaluator(truth_blob, truth_out, truth_full)

    # TODO: replace with a Shmoo-domain algorithm once designed.
    # from src.algorithms.<new_algo> import NewAlgorithm
    # algorithm = NewAlgorithm(H, W, evaluator.iteration_cap)
    # return evaluator.run(algorithm, seed=seed, phase=1)._as_dict()
    raise NotImplementedError(
        f"Dataset ready (cap={evaluator.iteration_cap}); no Shmoo-domain "
        "algorithm implemented yet. See docs/shmoo_model.md."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Blobchecker benchmark  grid={H}x{W}  layers={N_LAYERS}"
    )
    parser.add_argument(
        "--seed", type=int, nargs="*", default=None,
    )
    parser.add_argument(
        "--seed-range", type=int, nargs=2, metavar=("START", "STOP"), default=None,
    )
    parser.add_argument(
        "--suite", choices=["public", "validation", "both"], default="public",
    )
    parser.add_argument(
        "--algo", default="tbd",
        help="Algorithm name (no implementations yet).",
    )
    args = parser.parse_args()

    seed_suites = _select_seed_suites(args)
    print(
        f"Blobchecker benchmark  |  grid {H}×{W}  |  layers {N_LAYERS}  |  "
        f"cap {int(0.15*H*W)}  |  suites {list(seed_suites)}"
    )
    print("Generator and algorithms not yet implemented — see docs/shmoo_model.md")
    sys.exit(1)


if __name__ == "__main__":
    main()
