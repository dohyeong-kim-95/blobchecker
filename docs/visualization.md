# Strategy Visualization

The diagnostic workflow separates algorithm execution from rendering:

```bash
python tools/dump_strategy_artifacts.py --algo geometry_first --seed 0 --render
python tools/dump_strategy_artifacts.py --algo both --seed 0
python tools/visualize_strategy.py artifacts/diagnostics/geometry_first/seed_000.npz
```

Generated files live under `artifacts/diagnostics/` and are ignored by git.

## Artifact Contents

Each `.npz` artifact contains:

- `truth_blob`, `truth_outlier`, `truth_full`
- `predicted_final`
- `error_map`
- `query_rows`, `query_cols`, `query_phases`, `query_labels`
- `accuracy_curve`
- `final_belief` when the algorithm exposes a belief tensor

The sibling `*_summary.json` file stores final pass/fail, per-layer accuracy,
and bounding-box metrics.

## Dashboard Reading

Use the dashboard to connect three questions:

1. Did the weakest layer improve over the iteration budget?
2. Did the strategy spend queries on discovery, boundary repair, or entropy
   acquisition?
3. Are final errors mostly false positives, false negatives, boundary misses,
   interior holes, or shape overshoot?

Error map colors:

- black: true positive
- red: false positive
- blue: false negative
- white: true negative

Query phase colors:

- gray: preplanned
- blue: coarse discovery
- orange: boundary refinement
- green: entropy acquisition
