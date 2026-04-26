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
