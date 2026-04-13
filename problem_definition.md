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
- `truth_full_mask` is the full observed binary layer

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

## Objective

Minimize total iteration count subject to all of the following:

- every blob layer is reconstructed to at least 98% accuracy
- the recovered blob height and width are correct
- isolated positive outliers outside the meaningful blob region are not required

Passing quality comes first. Among passing methods, fewer iterations is better.

---

## Blob Model

For this project, every layer contains exactly one meaningful blob.

Assumed properties of the task:

- each layer has one blob of interest
- blobs may overlap spatially with blobs from other layers
- blob boundaries are expected to be reasonably coherent, not pure noise
- small interior holes may exist inside the blob
- isolated positive outliers may exist outside the blob

Important evaluation rule:

- external isolated positive outliers are not the main target
- recovering them is beneficial, but not required for a passing result

### Phase 0 restriction

Phase 0 uses the same one-blob-per-layer assumption as the long-term task.
There is no multi-blob handling requirement.

---

## Acceptance Criteria

A run is considered passing only if all 8 blob layers satisfy all required
checks.

### 1. Per-layer blob accuracy

For each layer `k`:

```python
blob_accuracy_k >= 0.98
```

The intended interpretation is that the estimator must recover the meaningful
blob shape with high fidelity for every layer, not just on average across
layers.

### 2. Blob size recovery

For each layer `k`, the estimator must recover the blob's:

- height
- width

Working interpretation for this repository:

- compute the true blob bounding box from `truth_blob_mask[k]`
- compute the predicted blob bounding box from `predicted_blob_mask[k]`
- phase 0 may use a bounded tolerance during scaffold development
- the long-term target remains exact height and width recovery

### 3. Outlier tolerance

The evaluator should de-emphasize isolated positive pixels or tiny disconnected
positive components outside the blob target.

Practical meaning:

- the system is optimized for reconstructing the meaningful blob structure
- sparse external positive noise is not a required target
- recovering such outliers is still better than missing them if it can be done
  cheaply

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

## Evaluation View

The evaluator should report at least:

- total iterations used
- per-layer blob accuracy for all 8 blobs
- per-layer pass/fail for the 98% threshold
- per-layer pass/fail for blob-height recovery
- per-layer pass/fail for blob-width recovery
- overall pass/fail

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

---

## Recommendations

These are recommendations, not hard requirements.

- The dataset generator should emit `truth_blob_mask`, `truth_outlier_mask`, and
  `truth_full_mask` separately.
- The predictor should emit `predicted_blob_mask` as the core output and
  `predicted_outlier_mask` as an optional output.
- Supplemental metrics such as blob IoU and positive recall should be reported
  alongside core pass/fail metrics.
- Grid size, overlap regime, hole rate, and outlier rate should be recorded in
  every benchmark artifact.
- Native low-level timing should be preferred over Python-level timing for
  performance comparisons.

---

## Non-Goals

The task does not require:

- recovery of every isolated positive outlier outside the meaningful blob region
- per-blob independent coordinate schedules
- optimizing average accuracy while letting one or more blobs fail

---

## Open Questions

These are still unresolved and should be made explicit in code and evaluation:

1. How should interior holes be weighted relative to overall blob accuracy?
2. Is the grid size fixed for the benchmark or variable across tasks?
3. What dataset family should become the official benchmark distribution?
4. What reference machine and official low-level timing lane should be used?
5. What final hard time and memory budgets should replace the current recommendations?

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
