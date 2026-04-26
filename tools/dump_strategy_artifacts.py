#!/usr/bin/env python3
"""
Run one strategy and save diagnostic artifacts for visualization.

Examples
--------
    python tools/dump_strategy_artifacts.py --algo geometry_first --seed 0
    python tools/dump_strategy_artifacts.py --algo both --seed 0 --render
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algorithms.geometry_first import GeometryFirstAdaptive
from src.algorithms.preplanned_greedy import PreplannedGreedy
from src.diagnostics import (
    make_error_map,
    phase_id_for_algorithm,
    summarize_final_result,
)
from src.evaluator import _per_layer_accuracy
from src.generator import generate_dataset

H, W = 50, 200


def _build_algorithm(name: str, budget: int):
    if name == "geometry_first":
        return GeometryFirstAdaptive(H, W, budget)
    if name == "preplanned":
        return PreplannedGreedy(H, W, budget, radius=1)
    raise ValueError(f"unknown algorithm: {name}")


def dump_seed(seed: int, algo: str, output_root: Path) -> tuple[Path, Path]:
    truth_blob, truth_out, truth_full = generate_dataset(H, W, seed)
    iteration_cap = int(0.15 * H * W)
    algorithm = _build_algorithm(algo, iteration_cap)

    query_rows = np.zeros(iteration_cap, dtype=np.int16)
    query_cols = np.zeros(iteration_cap, dtype=np.int16)
    query_phases = np.zeros(iteration_cap, dtype=np.uint8)
    query_labels = np.zeros((iteration_cap, truth_full.shape[0]), dtype=np.uint8)
    accuracy_curve = np.zeros((iteration_cap, truth_full.shape[0]), dtype=np.float32)

    for i in range(iteration_cap):
        row, col = algorithm.next_query()
        labels = truth_full[:, row, col].copy()
        algorithm.update(row, col, labels)
        predicted = algorithm.predict()

        query_rows[i] = row
        query_cols[i] = col
        query_phases[i] = phase_id_for_algorithm(algorithm)
        query_labels[i] = labels
        accuracy_curve[i] = _per_layer_accuracy(predicted, truth_blob)

    predicted_final = algorithm.predict()
    summary = summarize_final_result(
        seed=seed,
        algo=algo,
        truth_blob=truth_blob,
        predicted_final=predicted_final,
        iteration_cap=iteration_cap,
        phase=0,
    )

    target_dir = output_root / algo
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seed_{seed:03d}"
    npz_path = target_dir / f"{stem}.npz"
    summary_path = target_dir / f"{stem}_summary.json"

    arrays = {
        "truth_blob": truth_blob,
        "truth_outlier": truth_out,
        "truth_full": truth_full,
        "predicted_final": predicted_final,
        "error_map": make_error_map(predicted_final, truth_blob),
        "query_rows": query_rows,
        "query_cols": query_cols,
        "query_phases": query_phases,
        "query_labels": query_labels,
        "accuracy_curve": accuracy_curve,
    }
    if hasattr(algorithm, "_p"):
        arrays["final_belief"] = algorithm._p.copy()

    np.savez_compressed(npz_path, **arrays)
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return npz_path, summary_path


def _render(npz_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "visualize_strategy.py"), str(npz_path)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--algo",
        choices=["geometry_first", "preplanned", "both"],
        default="geometry_first",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "diagnostics",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render a standalone HTML dashboard after dumping the artifact.",
    )
    args = parser.parse_args()

    algos = ["preplanned", "geometry_first"] if args.algo == "both" else [args.algo]
    for algo in algos:
        npz_path, summary_path = dump_seed(args.seed, algo, args.output_root)
        print(f"wrote {npz_path}")
        print(f"wrote {summary_path}")
        if args.render:
            _render(npz_path)


if __name__ == "__main__":
    main()
