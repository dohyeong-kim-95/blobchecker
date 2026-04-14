# Evaluation Spec: Shared-Coordinate Multi-Blob Reconstruction

## Status

Draft 0.4

## Purpose

Define how candidate algorithms are evaluated for the shared-coordinate,
8-blob reconstruction task.

This spec turns the problem definition into an executable scoring contract.
It defines:

- what counts as one iteration
- what outputs an estimator must produce
- how correctness is measured
- how phase-0 and phase-1 acceptance differ
- how time and memory are measured
- which implementation languages are preferred for official timing claims
- which parts are recommendations rather than hard pass/fail gates

---

## 1. Evaluation Inputs

The evaluator consumes:

- `truth_blob_mask`: ground-truth meaningful blob tensor of shape `(8, H, W)`
- `truth_outlier_mask`: optional external positive outlier tensor of shape `(8, H, W)`
- `truth_full_mask`: full observed tensor, `truth_blob_mask | truth_outlier_mask`
- `predictor`: an algorithm that selects one coordinate per iteration and
  consumes the 8-label observation vector returned at that coordinate
- optional metadata describing implementation language, build mode, and
  runtime environment

The algorithm receives only the oracle interface:

```python
query(row: int, col: int) -> np.ndarray  # shape (8,), dtype uint8
```

The algorithm does not have access to any truth tensor at any point.

---

## 2. Evaluator Architecture

The evaluator is a **separate component** from the algorithm.

### 2.1 Callback contract

At each iteration, the algorithm must provide its current prediction:

```python
# called by the evaluator at every iteration i
predicted_blob_mask_i = algorithm.predict()  # shape (8, H, W), binary
```

The evaluator records this snapshot alongside the iteration index.

### 2.2 Accuracy curve

The evaluator computes per-layer blob accuracy at each iteration:

```python
blob_accuracy_k_i = mean(predicted_blob_mask_i[k] == truth_blob_mask[k])
```

This produces an accuracy curve of shape `(8, num_iterations)` that shows
how reconstruction quality evolves as queries accumulate.

### 2.3 Truth isolation

The evaluator wraps the oracle so the algorithm can only call:

```python
query(row, col) -> truth_full_mask[:, row, col]
```

The algorithm cannot access `truth_blob_mask` or `truth_outlier_mask` directly.

---

## 3. Required Predictor Output

A predictor run must produce at least:

```python
predicted_blob_mask.shape == (8, H, W)
```

Optional output:

```python
predicted_outlier_mask.shape == (8, H, W)
```

And the evaluation record must include:

- `iterations_used`
- `predicted_blob_mask` at final iteration
- per-iteration accuracy curve for all 8 layers
- optional `predicted_outlier_mask`
- optional per-iteration trace for debugging and analysis
- implementation metadata for the timing lane

All predicted values must be binary.

If `predicted_outlier_mask` is omitted, the evaluator treats it as an all-zero
mask for supplemental outlier metrics.

---

## 4. Cost Model

### 4.1 Iteration accounting

One iteration is defined as:

1. choose one coordinate `(row, col)`
2. query that coordinate once
3. observe 8 labels, one per blob
4. update the estimator state
5. produce an updated `predicted_blob_mask` (for evaluator callback)

Rules:

- the same coordinate is applied to all 8 blobs in that iteration
- different coordinates for different blobs in the same iteration are forbidden
- every coordinate query counts as 1 full iteration
- re-querying a previously used coordinate still counts as a new iteration

### 4.2 Iteration cap

Active cap:

```python
iteration_cap = int(0.15 * H * W)
```

Phase 0 example with 50×200 grid:

```python
iteration_cap = int(0.15 * 50 * 200) = 1500
```

A run that exceeds this cap is an automatic failure.

The algorithm always runs to the cap. Early stopping is not performed by the
algorithm; the evaluator observes the full accuracy curve.

---

## 5. Phase 0 Benchmark

### 5.1 Grid

Fixed grid shape for Phase 0:

```python
H, W = 50, 200
```

### 5.2 Seed suite

Phase 0 uses 10 pre-published fixed seeds.

Seeds are fixed and public so that results are reproducible across
implementations and machines.

```python
PHASE0_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # placeholder — freeze before first official run
```

Each seed produces one independent draw of 8 blob layers, for a total of:

```python
80 blobs = 10 seeds × 8 layers
```

The seed suite will change in subsequent phases. Phase 0 results should be
reported with their seed set explicitly recorded.

### 5.3 Blob generation contract

The generator must produce blobs satisfying the structural constraints in the
problem definition:

- each blob is 8-connected
- coverage drawn from: median 40%, std 5%, truncated to [30%, 70%]
- 0–3 interior holes per blob; hole bounding box drawn from mean 7×7, std 2×2 pixels
- external outliers may or may not be present; their density is variable
- 8 layers are drawn independently (different coverage, position, shape per layer)

The generator must emit:

```python
truth_blob_mask.shape    == (8, H, W)   # meaningful blob only
truth_outlier_mask.shape == (8, H, W)   # external outliers only
truth_full_mask.shape    == (8, H, W)   # blob | outlier (what oracle returns)
```

### 5.4 Phase scope

Phase 0 is the scaffold-building lane.

Goals:

- stabilize the shared-coordinate algorithm skeleton
- validate iteration discipline
- validate high-accuracy reconstruction on the one-blob-per-layer case
- validate the evaluator callback architecture

---

## 6. Phase 1 Benchmark

### 6.1 Grid

Phase 1 target direction uses a variable grid. Exact sizes to be determined.

### 6.2 Seed suite

Phase 1 uses a different seed suite from Phase 0. The Phase 0 seeds are retired
when Phase 1 begins.

### 6.3 Phase scope

Phase 1 is the target lane. Requirements are stricter:

- exact blob height and width recovery (no pixel tolerance)
- continued `>= 98%` accuracy on every layer
- stronger performance on supplemental metrics

---

## 7. Core Scoring Contract

### 7.1 Hard scoring region

Core pass/fail scoring is defined on `truth_blob_mask` only.

- external positive outliers are not required for passing
- recovering outliers is still beneficial through supplemental metrics

### 7.2 Per-layer blob accuracy

For each layer `k`:

```python
blob_accuracy_k = mean(predicted_blob_mask[k] == truth_blob_mask[k])
```

Accuracy is computed over the entire `H × W` grid.

If the algorithm predicts outlier pixels as part of the blob, they count as
false positives against `truth_blob_mask` and reduce accuracy naturally.

### 7.3 Hole treatment

Interior holes inside the blob are part of the blob reconstruction problem.

- holes remain part of `truth_blob_mask` (holes are zeros inside the blob region)
- incorrectly filling a hole hurts blob accuracy
- missing a positive blob region also hurts blob accuracy

### 7.4 Overall correctness rule

A run passes only if every layer passes every active requirement.

No average-only pass rule is allowed.

---

## 8. Bounding-Box Recovery

For each layer `k`, define:

- `bbox_truth_k` from `truth_blob_mask[k]`
- `bbox_pred_k` from `predicted_blob_mask[k]`

The evaluator extracts:

- `height_truth_k`, `width_truth_k`
- `height_pred_k`, `width_pred_k`

### 8.1 Phase 0 scaffold rule

Both dimension errors must satisfy the pixelized tolerance rule:

```python
abs(height_pred_k - height_truth_k) <= max(1, floor(0.05 * height_truth_k))
abs(width_pred_k  - width_truth_k)  <= max(1, floor(0.05 * width_truth_k))
```

### 8.2 Phase 1 target rule

For each layer `k`:

```python
height_pred_k == height_truth_k
width_pred_k  == width_truth_k
```

Exact match required.

---

## 9. Connectivity Recommendation

Connectivity is not part of the hard phase-0 scoring contract.

Recommended convention:

- use 8-neighbor connectivity for diagnostics, visualization, and any
  component-level analysis

Recommended dataset property:

- each `truth_blob_mask[k]` should form one 8-connected blob in phase 0

---

## 10. Outlier Scoring

Outliers are supplemental, not core pass/fail targets.

### 10.1 Hard-rule treatment

- `truth_outlier_mask` is excluded from the hard blob accuracy metric
- failure to recover outliers must not by itself cause failure

### 10.2 Supplemental outlier metrics

If outlier masks are available, the evaluator should report:

- outlier recall
- outlier precision
- outlier count error

---

## 11. Acceptance Summary

### 11.1 Phase 0

A phase-0 run passes if all of the following hold:

- `iterations_used <= int(0.15 * H * W)`
- for every layer, `blob_accuracy_k >= 0.98`
- for every layer, blob height satisfies the phase-0 pixelized tolerance rule
- for every layer, blob width satisfies the phase-0 pixelized tolerance rule
- timing and memory are reported

### 11.2 Phase 1

A phase-1 run passes if all of the following hold:

- `iterations_used <= int(0.15 * H * W)`
- for every layer, `blob_accuracy_k >= 0.98`
- for every layer, blob height matches exactly
- for every layer, blob width matches exactly
- timing and memory are reported

---

## 12. Supplemental Metrics

The following metrics are recommended as secondary diagnostics:

- blob IoU
- positive recall
- positive precision
- false-positive area outside the blob mask
- boundary error statistics
- outlier recall and outlier precision, if outlier channels are used
- per-iteration accuracy curve (primary diagnostic for algorithm efficiency)

---

## 13. Dataset Recommendations

Recommended reporting for every benchmark artifact:

- dataset generator version
- seed set identifier and all seed values
- grid shape
- overlap regime (whether blobs from different layers overlap)
- hole and outlier settings used

---

## 14. Time Budget

### 14.1 Measurement principle

Official elapsed-time claims should measure the algorithm core, excluding
Python orchestration overhead wherever possible.

Measured region should include only:

- coordinate planning
- oracle-state update logic
- reconstruction logic
- stopping logic

### 14.2 Budget recommendation

Because benchmark hardware has not yet been frozen, elapsed-time thresholds are
currently recommendations, not hard gates.

Draft working recommendations:

- phase 0 recommended target: `elapsed_native <= 1.0 s`
- phase 1 recommended target: `elapsed_native <= 250 ms`

---

## 15. Memory Budget

Peak memory should be measured as process peak resident set size.

Draft working recommendations:

- phase 0 recommended target: `peak_memory <= 64 MiB`
- phase 1 recommended target: `peak_memory <= 16 MiB`

Memory must be reported for every benchmarked run.

---

## 16. Ranking Rule

Methods are ranked lexicographically:

1. pass/fail status
2. fewer iterations used
3. higher per-layer robustness on supplemental metrics
4. lower native elapsed time
5. lower peak memory

---

## 17. Required Evaluation Report

Each run should emit a machine-readable report containing at least:

```python
{
    "phase": "phase0" | "phase1",
    "seed": int,
    "grid_shape": [H, W],
    "iteration_cap": int,
    "iterations_used": int,
    "per_layer_blob_accuracy": [float, ... length 8],
    "per_layer_height_truth": [int, ... length 8],
    "per_layer_height_pred": [int, ... length 8],
    "per_layer_width_truth": [int, ... length 8],
    "per_layer_width_pred": [int, ... length 8],
    "per_layer_height_pass": [bool, ... length 8],
    "per_layer_width_pass": [bool, ... length 8],
    "accuracy_pass": [bool, ... length 8],
    "overall_pass": bool,
    "accuracy_curve": [[float x num_iterations] x 8],  # per-layer, per-iteration
    "elapsed_native_seconds": float,
    "peak_memory_mib": float,
    "blob_iou": [float, ... length 8],
    "positive_recall": [float, ... length 8],
    "positive_precision": [float, ... length 8],
    "outlier_recall": [float, ... length 8],
    "outlier_precision": [float, ... length 8],
}
```

---

## 18. Open Items

The following remain intentionally open:

1. Final seed values for `PHASE0_SEEDS` (freeze before first official run)
2. Benchmark reference hardware
3. Final hard time budget
4. Final hard memory budget
5. Variable grid size handling for Phase 1
6. Whether phase-1 hard metrics should eventually incorporate supplemental metrics
