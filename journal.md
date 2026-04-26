# Blobchecker Journal

This file records experiments, failures, lessons, and discarded approaches.
Use [README.md](README.md) for decisions and [docs/spec.md](docs/spec.md) for
the canonical contract.

## 2026-04-26 — Phase 0 Baselines

### Pre-Planned Greedy Coverage

Approach:

- choose all 1,500 query coordinates before execution
- maximize local spatial coverage greedily
- reconstruct each layer with row-wise scanline filling

Result, 10-seed average:

```text
pass: 0/10
mean_min_accuracy: 0.9279
time: 7.0s/seed
```

Lessons:

- Uniform coverage is simple but leaves boundary uncertainty of several pixels.
- Scanline reconstruction works only when observations are distributed across
  rows and columns in a compatible way.
- Holes are detected only when sampled directly.

### Geometry-First Adaptive, Initial Version

Approach:

- coarse discovery grid
- belief tensor over `(8, H, W)`
- summed uncertainty acquisition:

```text
score(r,c) = sum_k [1 - 4 * (p_k(r,c) - 0.5)^2]
```

- local belief propagation after every query
- prediction by threshold plus largest connected component filtering

Result, 10-seed average:

```text
pass: 0/10
mean_min_accuracy: 0.9648
time: 1.4s/seed
```

Lessons:

- Adaptive querying improves both accuracy and runtime over pre-planned greedy.
- Most remaining error sits on blob boundaries.
- Height was often over-predicted because positive observations propagated
  beyond the real boundary.

## 2026-04-26 — Failed Reconstruction Experiments

### Scanline Prediction for Adaptive Queries

Hypothesis:

Replacing belief-threshold prediction with scanline reconstruction would reduce
boundary overshoot.

Result:

```text
mean_min_accuracy: 0.8869
```

Conclusion:

Rejected. Adaptive queries concentrate near boundaries, so extreme rows often
do not have enough positive samples for scanline filling. Scanline
reconstruction assumes a more uniform query distribution.

### Force Zero on Rows Without Positive Observations

Hypothesis:

If a queried row has no positive label for a layer, force that row to zero.

Result:

```text
mean_min_accuracy: 0.9536
```

Conclusion:

Rejected. Narrow top and bottom blob rows can be missed by coarse sampling and
then wrongly removed. The rule turns over-prediction into under-prediction.

### Positive-Only Propagation

Hypothesis:

Only propagate positive labels so negative boundary observations cannot erode
interior belief.

Result:

```text
mean_min_accuracy: 0.7818
```

Conclusion:

Rejected. Negative propagation is needed to collapse exterior uncertainty and
steer acquisition toward boundaries. Without it, many interior pixels remain at
the prior and exterior pixels keep wasting query budget.

## 2026-04-26 — Boundary Trim Update

Changes:

- `COARSE_STEP_C`: 10 -> 8
- `PROP_RADIUS`: 3 -> 4
- `PROP_DECAY`: 0.65 -> 0.60
- add top/bottom boundary row scans after coarse discovery
- scan layers round-robin instead of exhausting one layer first
- trim predicted top/bottom rows when they have confirmed-zero evidence

Result, 10-seed average:

```text
pass: 0/10
mean_min_accuracy: 0.9695
time: 1.5s/seed
```

Lessons:

- Confirmed-zero observations just outside the predicted boundary are useful
  for correcting belief-propagation overshoot.
- Round-robin boundary scanning matters. Sequential layer scanning spent too
  much of the cap on early layers.
- Scanning around the detected boundary in both directions can be harmful; the
  outside direction is what creates useful confirmed-zero evidence.

Remaining error estimate:

- about 105 pixels per layer still need to be corrected to reach 98%
- likely sources are column-boundary uncertainty, boundary-adjacent outliers,
  and residual shape errors

## 2026-04-26 — Left/Right Boundary Targeting

Hypothesis:

The top/bottom confirmed-zero trim reduced height overshoot, so applying the
same idea to left/right edges should reduce width overshoot without using
truth-specific information.

Changes:

- Phase 2 boundary refinement now scans four sides: top, bottom, left, right.
- Row scans use a coarse column stride; column scans use a coarse row stride so
  each scanned line costs about the same number of queries on the Phase 0 grid.
- Boundary queries are round-robin by layer and side so one layer or one side
  does not consume the whole boundary budget.
- `predict()` now trims confirmed-outside rows and columns. A boundary line is
  removed only when it has observed zero evidence and no observed positive
  evidence for that layer.

Overfitting guard:

- This is a geometric symmetry extension of the existing top/bottom trim, not a
  seed-specific coordinate schedule.
- Accepting it still requires public and validation seed results to move in the
  same direction. Public-only improvement should be treated as overfitting.

Results:

```text
public seeds [0..9]:
  pass: 0/10
  mean_min_accuracy: 0.9693
  mean_time: 1.5s/seed

validation seeds [100..109]:
  pass: 0/10
  mean_min_accuracy: 0.9677
  mean_time: 1.6s/seed
```

Interpretation:

- Width recovery looked stable in these runs: every reported public and
  validation layer had `w_ok=True`.
- Public mean min accuracy was slightly below the previous journal result
  `0.9695`, so this should not be treated as a net accuracy improvement yet.
- Height overshoot remains a visible failure mode on several seeds.

Verification:

- Python syntax compilation passed for the changed modules.
- Boundary-query and trim helper behavior was checked with a local scipy stub.
- A local `.venv` with `requirements.txt` was used for benchmark execution.

Lessons:

- Boundary-targeting changes should be evaluated as a family of four symmetric
  sides. Tuning only the failing visible dimension risks matching the public
  seed suite rather than improving the underlying level-set estimate.
- The validation suite added earlier is now a required gate before treating this
  change as a performance improvement.
- Left/right trim appears to help width stability, but the extra boundary
  budget may be taking useful queries away from entropy acquisition. The next
  iteration should compare boundary budget allocations instead of assuming more
  boundary scans are always better.

## 2026-04-26 — Stateful Boundary Binary Search

Hypothesis:

The fixed four-side boundary scan spends queries on coarse outside lines. A
stateful bracketed binary search should refine bbox edges more efficiently by
querying only midpoints between an observed outside zero and an observed inside
positive.

Changes:

- Phase 2 now builds per-layer boundary tasks from coarse observations.
- A task is created only when there is an inside positive and an outward
  observed zero on the same row or column.
- Each task adaptively updates its bracket after the query result arrives.
- Tasks are round-robin by layer so a single layer does not consume the boundary
  budget.
- Boundary trimming still uses the conservative confirmed-zero/no-positive rule
  for rows and columns.

Results:

```text
public seeds [0..9]:
  pass: 0/10
  mean_min_accuracy: 0.9696
  mean_time: 1.8s/seed

validation seeds [100..109]:
  pass: 0/10
  mean_min_accuracy: 0.9678
  mean_time: 1.8s/seed
```

Interpretation:

- This slightly improved mean min accuracy versus the fixed four-side scan
  result (`0.9693` public, `0.9677` validation subset), but the margin is small.
- Runtime increased from about 1.5-1.6s to about 1.8s because Phase 2 now
  performs stateful task management and more adaptive boundary work.
- The result is not a pass and should not be considered a solved boundary
  method yet.

Verification:

- Python syntax compilation passed for the changed modules.
- Boundary task, bracket update, and trim helper behavior were checked with a
  local scipy stub.
- A local `.venv` with `requirements.txt` was used for benchmark execution.

Lessons:

- Binary search is viable and not obviously overfit: public and validation
  subset moved in the same direction.
- The improvement is small enough that boundary task allocation matters more
  than the high-level idea. The next experiment should tune by principle, not
  public seeds: fewer anchors, fewer steps, or worst-layer-focused boundary
  tasks.
- Height failures remain. Pure left/right width work is not enough for the 98%
  accuracy target.

Disposition:

- Removed from the active implementation after review. The small accuracy gain
  did not justify the extra runtime and task-state complexity.
- Keeping this entry as a caution: binary search may be a valid research idea,
  but in the current implementation it would add noise when evaluating the next
  algorithmic change.

## 2026-04-26 — Strategy Pivot to Budget Ladder

Problem:

- The 15% budget is tight enough that local improvements are hard to interpret.
- Recent boundary experiments changed mean min accuracy by only about 0.0001 to
  0.0003, which is too small to distinguish robust structure from allocation
  noise.
- No current method passes, so optimizing directly under the submission cap
  risks compressing an algorithm that is not structurally capable of passing.

Decision:

- Reframe development as a budget ladder:

```text
Phase A: 50% budget  -> find a pass-capable algorithm structure
Phase B: 30% budget  -> remove redundant queries and rebalance allocations
Phase C: 20% budget  -> compress weakest-layer/boundary/hole priorities
Phase D: 15% budget  -> match the official submission condition
```

Lessons:

- First prove that a structure can pass when query budget is not the binding
  constraint.
- Only after that should query removal, boundary allocation, and runtime/BOPs
  be treated as optimization targets.
- Budget ratio must become an explicit benchmark dimension. Otherwise every
  algorithm change mixes three questions: whether the idea works, whether it
  fits the cap, and whether it overfits the current public seeds.

## Durable Lessons

1. Query strategy and reconstruction are coupled. Do not swap reconstruction
   methods without checking whether the query distribution supports them.
2. Symmetric positive/negative propagation is important for acquisition
   efficiency.
3. The 15% query budget is tight for 8 independent boundaries.
4. Largest connected component filtering removes isolated outliers but not
   outliers merged into the main component.
5. Boundary-directed queries are higher value than uniform interior sampling
   once coarse localization is complete.
6. Every-layer pass/fail means the weakest layer should influence acquisition,
   not just the summed average score.
7. Do not optimize only at 15% until a pass-capable structure exists at a wider
   budget. Use the budget ladder to separate algorithm viability from query
   compression.
