# Blobchecker Canonical Spec

This document is the detailed source of truth for the problem and evaluation
contract. For decisions, start with [../README.md](../README.md).

## Problem

Estimate 8 binary blob maps on a shared 2D grid.

At each iteration, an algorithm chooses one coordinate `(row, col)`. The same
coordinate is queried across all 8 layers, producing an 8-value observation
vector. A coordinate selection is the atomic cost unit.

The algorithm knows only the grid shape `(H, W)` before querying begins. It has
no prior knowledge of blob position, blob size, coverage, holes, or outlier
presence.

## Data Contract

The dataset generator emits three aligned tensors:

```python
truth_blob_mask.shape    == (8, H, W)
truth_outlier_mask.shape == (8, H, W)
truth_full_mask.shape    == (8, H, W)
truth_full_mask = truth_blob_mask | truth_outlier_mask
```

Meanings:

| Tensor | Meaning |
|---|---|
| `truth_blob_mask` | Meaningful blob target used for core scoring. |
| `truth_outlier_mask` | External positive outliers used for supplemental metrics. |
| `truth_full_mask` | Observed binary field returned by the oracle. |

The oracle contract is:

```python
query(row: int, col: int) -> np.ndarray  # shape (8,), dtype uint8
```

It returns:

```python
truth_full_mask[:, row, col]
```

The algorithm cannot directly distinguish blob positives from outlier positives.

## Blob Model

Each layer has exactly one meaningful blob.

Required structural properties:

| Property | Contract |
|---|---|
| Connectivity | One 8-connected meaningful blob per layer. |
| Coverage | Median 40%, standard deviation 5%, truncated to `[30%, 70%]`. |
| Boundary | Locally coherent, not pixel-level random noise. |
| Holes | 0 to 3 interior holes, about 7 x 7 pixels on average. |
| Outliers | Optional sparse external positive pixels. |

Layers are generated independently.

## Algorithm Contract

An algorithm implements:

```python
next_query() -> tuple[int, int]
update(row: int, col: int, labels: np.ndarray) -> None
predict() -> np.ndarray  # predicted_blob_mask, shape (8, H, W), binary
```

Rules:

- The same coordinate applies to every layer in an iteration.
- Different per-layer coordinates inside one iteration are forbidden.
- Re-querying a coordinate is allowed but still costs one full iteration.
- The algorithm runs to the evaluator's iteration cap.
- The evaluator, not the algorithm, decides pass/fail.

## Evaluator Contract

The evaluator:

- owns `truth_blob_mask`, `truth_outlier_mask`, and `truth_full_mask`
- wraps the oracle so the algorithm sees only `query(row, col)`
- calls `predict()` after every iteration
- records a per-layer accuracy curve
- scores the final prediction

One iteration is:

1. choose one coordinate `(row, col)`
2. query that coordinate once
3. observe 8 labels
4. update estimator state
5. produce an updated `predicted_blob_mask` for evaluator recording

The active iteration cap is:

```python
iteration_cap = int(0.15 * H * W)
```

For Phase 0:

```python
H, W = 50, 200
iteration_cap = 1500
```

## Scoring

Core scoring is against `truth_blob_mask` only.

For each layer `k`:

```python
blob_accuracy_k = mean(predicted_blob_mask[k] == truth_blob_mask[k])
```

Accuracy is computed over the entire `H x W` grid.

A run passes only if every layer passes every active gate. Average-only pass
rules are not allowed.

### Phase 0 Gates

- `iterations_used <= int(0.15 * H * W)`
- every layer has `blob_accuracy_k >= 0.98`
- every layer's predicted height satisfies:

```python
abs(height_pred_k - height_truth_k) <= max(1, floor(0.05 * height_truth_k))
```

- every layer's predicted width satisfies:

```python
abs(width_pred_k - width_truth_k) <= max(1, floor(0.05 * width_truth_k))
```

### Phase 1 Gates

- `iterations_used <= int(0.15 * H * W)`
- every layer has `blob_accuracy_k >= 0.98`
- every layer's predicted height matches exactly
- every layer's predicted width matches exactly

## Ranking

Among candidate methods, rank lexicographically:

1. pass/fail status
2. fewer iterations used
3. stronger supplemental metrics
4. lower elapsed time
5. lower peak memory

Supplemental metrics may include blob IoU, positive recall, positive precision,
boundary error, outlier recall, and outlier precision. They do not replace the
hard gates above.

## Phase 0 Benchmark Context

Current Phase 0 implementation uses:

```python
PHASE0_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

These seeds are still marked as placeholder in code and older docs. Freeze them
before making official benchmark claims.

Open benchmark items:

- reference hardware
- final hard time budget
- final hard memory budget
- Phase 1 grid size range
- Phase 1 seed suite
- whether supplemental metrics become hard gates later
