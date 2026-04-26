# Blobchecker Decision Brief

This is the single document to read before making project decisions.

## Current Goal

Reconstruct 8 binary blob layers on one shared 2D grid with as few coordinate
queries as possible.

One iteration selects exactly one coordinate `(row, col)`. The oracle returns
the 8 binary labels at that same coordinate, one per layer. Algorithms know only
the grid shape before querying.

## Canonical Contract

The detailed contract lives in [docs/spec.md](docs/spec.md). In short:

| Item | Current value |
|---|---|
| Layers | 8 |
| Phase 0 grid | 50 x 200 |
| Iteration cap | `int(0.15 * H * W)` = 1,500 for Phase 0 |
| Oracle output | `truth_full_mask[:, row, col]` |
| Scoring target | `truth_blob_mask`, not outliers |
| Required output | `predicted_blob_mask.shape == (8, H, W)` |
| Accuracy gate | every layer `>= 98%` |
| Shape gate, Phase 0 | height/width within `max(1, floor(5% * truth_dim))` |
| Shape gate, Phase 1 | exact height/width |
| Overall pass | every layer passes every active gate |

Outliers are observed through the oracle but are not core reconstruction
targets. Predicting an outlier as blob is naturally penalized as a false
positive against `truth_blob_mask`.

## Current Implementation

| Component | File |
|---|---|
| Dataset generator | `src/generator.py` |
| Oracle wrapper | `src/oracle.py` |
| Evaluator and result object | `src/evaluator.py` |
| Abstract algorithm API | `src/algorithms/base.py` |
| Pre-planned baseline | `src/algorithms/preplanned_greedy.py` |
| Current adaptive algorithm | `src/algorithms/geometry_first.py` |
| Benchmark runner | `benchmark.py` |

Run the current benchmark with:

```bash
python benchmark.py --algo both
```

## Latest Known Results

Phase 0, 10 public placeholder seeds `[0..9]`, grid 50 x 200:

| Algorithm | Pass | Mean min accuracy | Mean time |
|---|---:|---:|---:|
| `preplanned` | 0/10 | 0.9279 | 7.0s/seed |
| `geometry_first` before boundary trim | 0/10 | 0.9648 | 1.4s/seed |
| `geometry_first` current journal result | 0/10 | 0.9695 | 1.5s/seed |

The current blocker is accuracy, not the shared-coordinate API. Height recovery
improved after confirmed-zero boundary trimming, but the algorithm remains below
the 98% per-layer accuracy gate.

## Decision Rules

Use these rules when choosing the next change:

1. Preserve the shared-coordinate constraint. Do not introduce per-layer query
   schedules unless they are wrapped by a global one-coordinate-per-iteration
   planner.
2. Keep the evaluator as the source of truth for pass/fail. Algorithms must not
   access `truth_blob_mask` or `truth_outlier_mask`.
3. Prefer geometry-first improvements before GP-style models. Prior work found
   dense and sparse GP complexity hard to justify for this benchmark.
4. Optimize for every-layer pass, not average accuracy. One failing layer fails
   the whole run.
5. Treat outlier recovery as supplemental. Blob accuracy against
   `truth_blob_mask` is the core gate.
6. Keep results tied to seed set, grid shape, algorithm parameters, and command.

## Next Decisions

The most useful next experiments are:

| Priority | Decision | Why |
|---:|---|---|
| 1 | Add left/right boundary targeting and trimming | Current errors are still mostly boundary pixels, especially width/edge uncertainty. |
| 2 | Add targeted interior-hole probing | Up to 3 holes of about 7 x 7 pixels can alone consume most of the 2% error budget. |
| 3 | Improve outlier isolation near the main component | Isolated outliers are handled by connected components; boundary-adjacent outliers can still merge. |
| 4 | Compare sum-entropy vs worst-layer weighting | Averages are not enough; the weakest layer determines pass/fail. |
| 5 | Freeze Phase 0 seeds and benchmark hardware | Timing and official result claims need stable context. |

## Document Map

| Document | Role |
|---|---|
| [README.md](README.md) | Decision brief. Read this first and use it for project choices. |
| [docs/spec.md](docs/spec.md) | Canonical detailed problem and evaluation contract. |
| [journal.md](journal.md) | Time-ordered experiments, failures, lessons, and discarded approaches. |
| [research/08_synthesis_and_recommendations.md](research/08_synthesis_and_recommendations.md) | Research synthesis and method rationale. |
| [docs/archive/legacy_wisdom.md](docs/archive/legacy_wisdom.md) | Archived legacy lessons from the old single-blob project. |

Removed root-level duplicate docs should not be recreated. If a new decision
needs more detail than this brief can hold, update `docs/spec.md` or
`journal.md` and link it from here.
