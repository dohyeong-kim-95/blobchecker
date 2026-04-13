# Legacy Wisdom

## What the old project was

This repository originally benchmarked several methods for reconstructing a
single 2D binary blob map from sequential oracle queries on a fixed grid.
The old problem assumed:

- one oracle
- one binary map
- a sample budget of `int(0.15 * H * W)`
- evaluation by pixel accuracy, elapsed time, and peak memory

That architecture is now retired because the new problem is materially
different: one chosen coordinate must be queried across 8 blobs at once, and
the primary cost is iteration count rather than per-blob sample count.

## Useful ideas worth keeping in mind

- Smooth blob boundaries respond well to geometry-first methods.
- Dense GP methods were too expensive relative to their gain.
- Sparse GP improved runtime versus dense GP, but still carried significant
  implementation and memory complexity.
- Row-wise boundary search is a strong primitive when blobs are compact and
  mostly smooth.
- Interior probing is necessary to recover holes; boundary-only methods miss
  internal zeros.
- Background outlier ones are expensive to chase and are not a good primary
  target unless the metric rewards them.
- Evaluation logic should be centralized. The old code duplicated budgets,
  thresholds, plotting, and result formatting across many files.

## Legacy method takeaways

### `geo`

- Best practical baseline from the old codebase.
- Core idea: scanline seed finding, boundary localization, then targeted hole
  discovery.
- Strength: cheap queries, high speed, simple mental model.
- Weakness: strongly shaped by the assumption of one dominant connected blob.

### `contour`

- Strong boundary-oriented baseline.
- Core idea: find left/right boundaries per row, then interpolate labels from
  sampled structure.
- Strength: efficient when the main challenge is boundary recovery.
- Weakness: also assumes one main object and a single shared prediction map.

### `lse`

- Dense GP with straddle acquisition.
- Strength: high reconstruction quality in the old benchmark.
- Weakness: too slow and memory-heavy to be a sensible default.

### `sparse_gp`

- Sparse GP reduced the cost of the dense GP baseline.
- Strength: more scalable than `lse`.
- Weakness: still more complex than justified by the practical gains in the
  old regime.

## Generator wisdom

The synthetic-map generator is a reusable idea even if the implementation is
not retained:

- start from a smooth, perturbed ellipse-like blob
- carve a small number of interior holes
- optionally inject sparse background outliers

For the new task, the right replacement is not a single-map generator but a
stack generator that emits 8 aligned blob maps sharing the same coordinate
system.

## Evaluation lessons

- Keep one canonical task spec and derive all budgets/targets from it.
- Keep one canonical result schema and one evaluator.
- Measure the thing the task actually optimizes.
- For the new task, evaluation should be driven by:
- per-blob accuracy threshold (`>= 98%`)
- iteration count
- minimum blob height/width recovery
- tolerance for missing external positive outliers

## Why the old implementation was removed

The old repository encoded the wrong abstraction boundary:

- `fit(oracle, grid_size, budget)` for one map
- one prediction image per method
- one accuracy curve per method
- method-specific evaluators with duplicated assumptions

The new system should instead start from:

- one shared query planner that chooses a coordinate per iteration
- one stacked oracle returning 8 labels for that coordinate
- per-blob reconstruction state
- stopping logic based on all blobs meeting the required thresholds

## Rebuild guidance

- Centralize the task definition first.
- Treat iteration as the primary budget unit.
- Build the shared-coordinate query interface before any estimator.
- Prefer simple geometry-first baselines before adding surrogate models.
- Only add model complexity if it clearly reduces iterations under the new
  metric.
