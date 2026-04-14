# Problem Definition: Shared-Coordinate Multi-Blob Reconstruction

## Summary

Estimate 8 binary blob maps defined on the same 2D coordinate grid.

At each iteration, the algorithm may choose exactly one coordinate `(row, col)`.
That coordinate must then be queried across all 8 blob layers. The result of one
iteration is therefore an 8-value observation vector, one label per blob at the
same spatial location.

The primary goal is to minimize the number of iterations while still recovering
all 8 blob layers to the required accuracy.

---

## Input / Output

### Input

- A shared grid of shape `(H, W)`
- A stacked deterministic oracle over 8 blob layers
- Optional external iteration cap supplied by the evaluator

### Algorithm prior

The algorithm knows **only the grid shape `(H, W)`** before querying begins.
It has no prior knowledge of:

- blob location or size
- blob coverage
- number or size of interior holes
- presence or density of outliers

### Oracle contract

The conceptual oracle is:

```python
query(row: int, col: int) -> tuple[int, int, int, int, int, int, int, int]
```

Equivalent vector form:

```python
query(row: int, col: int) -> np.ndarray  # shape (8,), dtype uint8
```

Where:

- `query(row, col)[k]` is the binary label for blob layer `k` at coordinate `(row, col)`
- each value is in `{0, 1}`
- the oracle is deterministic
- all 8 labels are observed together for the same coordinate
- the oracle returns `truth_full_mask[k, row, col]` which includes both blob and
  outlier pixels — the algorithm cannot directly distinguish the two

### Recommended truth artifact

The dataset generator should emit three aligned tensors:

```python
truth_blob_mask.shape    == (8, H, W)
truth_outlier_mask.shape == (8, H, W)
truth_full_mask.shape    == (8, H, W)
```

With:

```python
truth_full_mask = truth_blob_mask | truth_outlier_mask
```

Interpretation:

- `truth_blob_mask` is the meaningful blob target for core scoring
- `truth_outlier_mask` contains optional external positive outliers
- `truth_full_mask` is the full observed binary layer (what the oracle returns)

### Required prediction output

The estimator must return at least:

```python
predicted_blob_mask.shape == (8, H, W)
```

### Recommended optional prediction output

The estimator may also return:

```python
predicted_outlier_mask.shape == (8, H, W)
```

This separates the core blob reconstruction target from optional outlier
recovery.

---

## Core Constraint: Shared-Coordinate Querying

The defining rule of the task is:

- one iteration selects one coordinate
- that same coordinate must be evaluated on all 8 blobs
- the algorithm may not choose different coordinates for different blobs within
  the same iteration

This makes the optimization problem fundamentally different from solving 8
independent blob-reconstruction problems.

A coordinate selection is the atomic unit of cost.

---

## Query Strategy

The algorithm may use any query strategy:

- **Adaptive (online):** next coordinate is chosen based on prior observations.
  Advantage: can reduce iterations needed to reach required accuracy.
- **Pre-planned (offline):** all query coordinates are fixed before querying
  begins. Advantage: lower computational overhead per iteration.

Both strategies are valid. Trade-offs are reflected in the evaluation ranking
(iteration count and elapsed time are both tracked).

---

## Algorithm Termination

The algorithm always runs until the iteration cap is exhausted. It does not
self-terminate early.

The evaluator (not the algorithm) is responsible for checking accuracy
thresholds. The evaluator has access to ground truth and tracks reconstruction
quality at every iteration via a callback.

---

## Objective

Minimize total iteration count subject to all of the following:

- every blob layer is reconstructed to at least 98% accuracy
- the recovered blob height and width are correct
- isolated positive outliers outside the meaningful blob region are not required

Passing quality comes first. Among passing methods, fewer iterations is better.

---

## Blob Model

### Structural definition

A valid blob for this task is defined structurally, not by a fixed generative
distribution. This ensures benchmarks test general blob-finding capability
rather than adaptation to a specific parametric family.

Required structural properties:

- exactly one meaningful blob per layer
- the blob is 8-connected
- blob boundary is locally coherent (not random pixel-level noise)
- small interior holes may exist inside the blob
- isolated positive outliers may exist outside the blob

### Coverage

Each blob's coverage (fraction of the grid that is positive in
`truth_blob_mask[k]`) is drawn from a distribution with:

- median: 40%
- standard deviation: 5%
- truncated range: [30%, 70%]

Coverage is drawn independently for each of the 8 layers.

### Interior holes

Each blob may contain 0 to 3 interior holes.

Hole properties:

- bounding box size: mean 7×7 pixels, standard deviation 2×2 pixels
- holes are part of the blob reconstruction problem
- incorrectly filling a hole hurts blob accuracy

### External outliers

External positive outliers (pixels outside the meaningful blob that are
nevertheless positive in `truth_full_mask`) are optional:

- a blob may have zero outliers
- a blob may have sparse external outliers
- outlier presence and density are not guaranteed

---

## Acceptance Criteria

A run is considered passing only if all 8 blob layers satisfy all required
checks.

### 1. Per-layer blob accuracy

For each layer `k`:

```python
blob_accuracy_k = mean(predicted_blob_mask[k] == truth_blob_mask[k])
blob_accuracy_k >= 0.98
```

Accuracy is computed over the entire `H × W` grid, not just the blob region.

Note: if the algorithm includes outlier pixels in `predicted_blob_mask`, those
pixels count as false positives against `truth_blob_mask` and hurt accuracy.
This naturally penalizes over-prediction without a separate scoring rule.

### 2. Blob size recovery

For each layer `k`, the estimator must recover the blob's height and width.

- compute the true blob bounding box from `truth_blob_mask[k]`
- compute the predicted blob bounding box from `predicted_blob_mask[k]`
- phase 0 uses a 5% pixelized tolerance
- phase 1 requires exact match

### 3. Outlier tolerance

The evaluator scores blob accuracy on `truth_blob_mask` only. External outliers
are not required reconstruction targets. Outlier recall and precision are
reported as supplemental metrics only.

### 4. Iteration efficiency

Among methods that satisfy all correctness constraints, the better method is the
one that uses fewer coordinate selections.

---

## Cost Model

The cost unit is iteration count, not per-layer label count.

One iteration consists of:

1. choose one coordinate `(row, col)`
2. query that coordinate across all 8 blobs
3. receive 8 labels
4. update the internal state for all 8 reconstructions

Therefore:

- 1 iteration yields up to 8 useful observations
- repeated queries to the same coordinate still count as additional iterations
- methods should strongly prefer globally informative coordinates

---

## Evaluation Architecture

The evaluator is a **separate component** from the algorithm. It:

- holds ground truth (`truth_blob_mask`, `truth_outlier_mask`, `truth_full_mask`)
- wraps the oracle so the algorithm cannot access truth directly
- receives a copy of `predicted_blob_mask` at each iteration (callback contract)
- records per-iteration accuracy for all 8 layers
- produces an accuracy curve as a function of iteration count
- applies acceptance criteria to the final state

The algorithm has no access to ground truth at any point during execution.

---

## Evaluation View

The evaluator reports at least:

- total iterations used
- per-layer blob accuracy for all 8 blobs at final state
- per-layer pass/fail for the 98% threshold
- per-layer pass/fail for blob-height recovery
- per-layer pass/fail for blob-width recovery
- overall pass/fail
- per-iteration accuracy curve for all 8 layers

Recommended aggregate rule:

```text
PASS only if every blob passes every required check.
```

No average-only pass condition should be used.

---

## Baseline Design Implications

This task favors algorithms that exploit shared structure across layers.

Implications:

- query planning must be global, not blob-local
- the next coordinate should be chosen for joint information gain across 8 blobs
- solving each blob independently is a poor fit unless wrapped in a shared
  planner
- geometry-first methods are attractive because they can often reduce
  iterations without heavy model cost
- model complexity is justified only if it clearly reduces shared-coordinate
  query count

Since the algorithm must also distinguish blob pixels from outlier pixels using
only the binary oracle signal, geometric reasoning (e.g. connectivity, spatial
coherence) is a natural tool for blob-outlier separation.

---

## Application Context

This project is intended as a real applied system, not a pure research
benchmark. Implementation complexity and maintainability are first-class
concerns. Algorithmic simplicity and interpretability are valued alongside
raw performance.

---

## Non-Goals

The task does not require:

- recovery of every isolated positive outlier outside the meaningful blob region
- per-blob independent coordinate schedules
- optimizing average accuracy while letting one or more blobs fail
- early stopping logic within the algorithm

---

## Canonical Working Assumptions

Until refined further, this repository should implement against the following
working assumptions:

- exactly 8 blob layers
- one shared `(H, W)` grid
- one coordinate chosen per iteration
- all 8 labels observed for that coordinate
- one meaningful blob per layer
- success requires `>= 98%` blob accuracy on every layer
- success also requires correct blob height and width on every layer
- external positive outliers are optional to recover
- iteration count is the primary efficiency metric
- algorithm always runs to the iteration cap
- evaluator is separate from algorithm and holds ground truth
- algorithm knows only grid shape before querying begins
