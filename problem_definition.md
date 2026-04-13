# Problem Definition: Shared-Coordinate Multi-Blob Reconstruction

## Summary

Estimate 8 binary blob maps defined on the same 2D coordinate grid.

At each iteration, the algorithm may choose exactly one coordinate `(row, col)`.
That coordinate must then be queried across all 8 blob layers. The result of one
iteration is therefore an 8-value observation vector, one label per blob at the
same spatial location.

The primary goal is to minimize the number of iterations while still recovering
all 8 blobs to the required accuracy.

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

- `query(row, col)[k]` is the binary label for blob `k` at coordinate `(row, col)`
- each value is in `{0, 1}`
- the oracle is deterministic
- all 8 labels are observed together for the same coordinate

### Output

The estimator must return 8 predicted binary maps:

```python
predicted.shape == (8, H, W)
```

Each predicted layer must be a binary mask for one blob.

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

- every blob is reconstructed to at least 98% accuracy
- the main blob size is recovered correctly in both height and width
- missing isolated positive outliers outside the main blob is allowed

Passing quality comes first. Among passing methods, fewer iterations is better.

---

## Blob Model

Each of the 8 layers contains one primary blob of interest.

Assumed properties of the primary blob:

- the blob is a connected positive region or near-connected region
- the blob may overlap spatially with blobs from other layers
- the blob boundary is expected to be reasonably coherent, not pure noise
- small interior holes may exist
- isolated positive outliers may exist outside the main blob

Important evaluation rule:

- isolated positive outliers outside the main blob do not need to be recovered
- failure to predict such outliers should not prevent a passing result

---

## Acceptance Criteria

A run is considered passing only if all 8 blobs satisfy all required checks.

### 1. Per-blob accuracy

For each blob `k`:

```python
accuracy_k >= 0.98
```

The intended interpretation is that the estimator must recover the main blob
shape with high fidelity for every layer, not just on average across layers.

### 2. Main-blob size recovery

For each blob `k`, the estimator must recover the main blob's:

- height
- width

Working interpretation for this repository:

- compute the true main blob bounding box for layer `k`
- compute the predicted main blob bounding box for layer `k`
- both bounding-box height and bounding-box width must match the true blob
  dimensions

If later a tolerance is desired, it should be added explicitly rather than
assumed.

### 3. Outlier tolerance

The evaluator may ignore isolated positive pixels or small disconnected positive
components outside the main blob.

Practical meaning:

- the system is optimized for reconstructing the main blob
- sparse external positive noise is not a required target

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
- per-blob accuracy for all 8 blobs
- pass/fail for per-blob 98% threshold
- per-blob pass/fail for main-blob height recovery
- per-blob pass/fail for main-blob width recovery
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
- model complexity is justified only if it reduces shared-coordinate query count

---

## Non-Goals

The task does not require:

- recovery of every isolated positive outlier outside the main blob
- per-blob independent coordinate schedules
- optimizing average accuracy while letting one or more blobs fail

---

## Open Questions

These are still unresolved and should be made explicit in code and evaluation:

1. How exactly is the main blob identified if a layer contains multiple positive
   connected components of similar size?
2. Should main-blob height/width require exact equality, or allow a small
   tolerance such as `±1` or `±2` pixels?
3. How are interior holes scored: included fully in accuracy, or treated as a
   softer objective?
4. Is there a fixed maximum iteration budget, or only a best-effort minimize
   objective?
5. Is the grid size fixed for the benchmark or variable across tasks?

---

## Canonical Working Assumptions

Until refined further, this repository should implement against the following
working assumptions:

- exactly 8 blob layers
- one shared `(H, W)` grid
- one coordinate chosen per iteration
- all 8 labels observed for that coordinate
- success requires `>= 98%` accuracy on every blob
- success also requires correct main-blob height and width on every blob
- external positive outliers are optional to recover
- iteration count is the primary efficiency metric
