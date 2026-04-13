# Evaluation Spec: Shared-Coordinate Multi-Blob Reconstruction

## Status

Draft 0.3

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
- `truth_full_mask`: full observed tensor, typically `truth_blob_mask | truth_outlier_mask`
- `predictor`: an algorithm that selects one coordinate per iteration and
  consumes the 8-label observation vector returned at that coordinate
- optional metadata describing implementation language, build mode, and
  runtime environment

Conceptual oracle interface:

```python
query(row: int, col: int) -> np.ndarray  # shape (8,), dtype uint8
```

Returned values are binary labels for all 8 blobs at the same coordinate.

---

## 2. Required Predictor Output

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
- `predicted_blob_mask`
- optional `predicted_outlier_mask`
- optional per-iteration trace for debugging and analysis
- implementation metadata for the timing lane

All predicted values must be binary.

If `predicted_outlier_mask` is omitted, the evaluator treats it as an all-zero
mask for supplemental outlier metrics.

---

## 3. Cost Model

### 3.1 Iteration accounting

One iteration is defined as:

1. choose one coordinate `(row, col)`
2. query that coordinate once
3. observe 8 labels, one per blob
4. update the estimator state

Rules:

- the same coordinate is applied to all 8 blobs in that iteration
- different coordinates for different blobs in the same iteration are forbidden
- every coordinate query counts as 1 full iteration
- re-querying a previously used coordinate still counts as a new iteration

### 3.2 Iteration cap

Active cap:

```python
iteration_cap = int(0.15 * H * W)
```

A run that exceeds this cap is an automatic failure.

This cap is active for phase 0 and phase 1 unless superseded later by a stricter
benchmark rule.

---

## 4. Phase Scope

### 4.1 Phase 0

Phase 0 is the scaffold-building lane.

Assumptions:

- each blob layer contains exactly one meaningful blob
- the evaluation dataset follows the one-blob-per-layer contract

Goal:

- stabilize the shared-coordinate algorithm skeleton
- validate iteration discipline
- validate high-accuracy reconstruction on the one-blob case

### 4.2 Phase 1

Phase 1 is the target lane.

Target direction:

- exact blob height and width recovery
- continued `>= 98%` accuracy on every layer
- stronger performance on supplemental metrics

The one-blob-per-layer assumption remains active.

---

## 5. Core Scoring Contract

### 5.1 Hard scoring region

Core pass/fail scoring is defined on `truth_blob_mask` only.

This means:

- the meaningful blob is the required reconstruction target
- external positive outliers are not required for passing
- recovering outliers is still beneficial through supplemental metrics

### 5.2 Per-layer blob accuracy

For each layer `k`:

```python
blob_accuracy_k = mean(predicted_blob_mask[k] == truth_blob_mask[k])
```

This is the primary correctness metric.

### 5.3 Hole treatment

Interior holes inside the blob are part of the blob reconstruction problem.

Therefore:

- holes remain part of `truth_blob_mask`
- incorrectly filling a hole hurts blob accuracy
- missing a positive blob region also hurts blob accuracy

### 5.4 Overall correctness rule

A run passes only if every layer passes every active requirement.

No average-only pass rule is allowed.

---

## 6. Bounding-Box Recovery

For each layer `k`, define:

- `bbox_truth_k` from `truth_blob_mask[k]`
- `bbox_pred_k` from `predicted_blob_mask[k]`

The evaluator extracts:

- `height_truth_k`, `width_truth_k`
- `height_pred_k`, `width_pred_k`

### 6.1 Phase 0 scaffold rule

Phase 0 exists to let the algorithm skeleton stabilize before exact matching is
required.

In phase 0, both dimension errors must satisfy the pixelized tolerance rule:

```python
abs(height_pred_k - height_truth_k) <= max(1, floor(0.05 * height_truth_k))
abs(width_pred_k  - width_truth_k)  <= max(1, floor(0.05 * width_truth_k))
```

Interpretation:

- the allowed error is 5% of the true size, rounded into pixel space
- every blob gets at least a 1-pixel scaffold tolerance in phase 0

### 6.2 Phase 1 target rule

Phase 1 is the intended end-state requirement.

For each layer `k`:

```python
height_pred_k == height_truth_k
width_pred_k  == width_truth_k
```

That is, exact match.

---

## 7. Connectivity Recommendation

Connectivity is not part of the hard phase-0 scoring contract because truth and
prediction are supplied explicitly as blob masks.

Recommended convention:

- use `8-neighbor` connectivity for diagnostics, visualization, and any
  component-level analysis

Recommended dataset property:

- each `truth_blob_mask[k]` should form one `8-connected` blob in phase 0

These are recommendations, not hard gates.

---

## 8. Outlier Scoring

Outliers are supplemental, not core pass/fail targets.

### 8.1 Hard-rule treatment

- `truth_outlier_mask` is excluded from the hard blob accuracy metric
- `predicted_outlier_mask` does not affect the hard blob accuracy metric
- failure to recover outliers must not by itself cause failure

### 8.2 Supplemental outlier metrics

If outlier masks are available, the evaluator should report:

- outlier recall
- outlier precision
- outlier count error

This preserves the intended tradeoff:

- outlier recovery is good
- outlier recovery is not worth sacrificing the core blob objective

---

## 9. Acceptance Phases

### 9.1 Phase 0: skeleton-building lane

A phase-0 run passes if all of the following hold:

- `iterations_used <= int(0.15 * H * W)`
- for every layer, `blob_accuracy_k >= 0.98`
- for every layer, blob height satisfies the phase-0 pixelized tolerance rule
- for every layer, blob width satisfies the phase-0 pixelized tolerance rule
- timing and memory are reported

Phase 0 is intended to validate the reconstruction loop, query planner, and
shared-coordinate architecture.

### 9.2 Phase 1: target lane

A phase-1 run passes if all of the following hold:

- `iterations_used <= int(0.15 * H * W)`
- for every layer, `blob_accuracy_k >= 0.98`
- for every layer, blob height matches exactly
- for every layer, blob width matches exactly
- timing and memory are reported

Exact time and memory thresholds remain recommendations until the benchmark
platform is frozen.

---

## 10. Supplemental Metrics

The following metrics are recommended as secondary diagnostics and should be
reported when practical:

- blob IoU
- positive recall
- positive precision
- false-positive area outside the blob mask
- boundary error statistics
- outlier recall and outlier precision, if outlier channels are used

Rationale:

- earlier work showed that plain pixel accuracy can hide shape defects
- blob IoU provides a stronger shape-overlap signal
- positive recall helps expose missed blob mass even when overall accuracy looks
  strong on sparse maps

These metrics are recommended, not yet hard pass/fail gates.

---

## 11. Dataset Recommendations

The official dataset family is not frozen yet. For now, the following are
recommendations rather than hard rules.

Recommended dataset properties:

- shared grid across all 8 layers
- exactly one blob per layer
- controllable overlap across layers
- occasional interior holes
- very sparse external positive outliers
- fixed seed suites for reproducibility
- explicit storage of `truth_blob_mask`, `truth_outlier_mask`, and `truth_full_mask`

Recommended reporting:

- always record the dataset generator version
- always record the seed set
- always record the overlap regime
- always record hole and outlier settings

---

## 12. Time Budget

### 12.1 Measurement principle

The time budget is not a Python-level budget.

Official elapsed-time claims should measure the algorithm core in a native,
low-level implementation lane, excluding Python orchestration overhead wherever
possible.

Measured region should include only:

- coordinate planning
- oracle-state update logic
- reconstruction logic
- stopping logic

Measured region should exclude:

- plot generation
- debug printing
- file I/O unrelated to required inputs
- report serialization
- Python wrapper overhead, if a wrapper is used only as a harness

### 12.2 Timing lane recommendation

For official timing eligibility, the preferred implementation core is a language
whose first public release predates C in 1972.

Recommended timing-lane languages:

- assembly
- B
- BCPL
- Fortran
- ALGOL
- PL/I

Languages such as Python, C, C++, Rust, Go, Java, and JavaScript may still be
used for correctness experiments, prototyping, and harness work, but their
elapsed times should be treated as non-official.

### 12.3 Budget recommendation

Because benchmark hardware has not yet been frozen, elapsed-time thresholds are
currently recommendations, not hard gates.

Draft working recommendations:

- phase 0 recommended target: `elapsed_native <= 1.0 s`
- phase 1 recommended target: `elapsed_native <= 250 ms`

Until the reference machine is defined:

- elapsed time must always be reported
- elapsed time should inform comparisons
- elapsed time should not yet decide pass/fail by itself

---

## 13. Memory Budget

### 13.1 Measurement principle

Peak memory should be measured as process peak resident set size, or the closest
available low-level equivalent on the reference platform.

Preferred unit:

- MiB

### 13.2 Budget recommendation

As with elapsed time, the memory thresholds are provisional recommendations
until the reference machine and implementation lane are finalized.

Draft working recommendations:

- phase 0 recommended target: `peak_memory <= 64 MiB`
- phase 1 recommended target: `peak_memory <= 16 MiB`

Rules:

- memory usage must be reported for every benchmarked run
- memory should guide comparisons
- memory should not yet decide pass/fail by itself

---

## 14. Ranking Rule

Methods are ranked lexicographically:

1. pass/fail status
2. fewer iterations used
3. higher per-layer robustness on supplemental metrics
4. lower native elapsed time
5. lower peak memory

Interpretation:

- correctness dominates everything else
- iteration efficiency dominates performance micro-optimizations
- supplemental shape metrics help distinguish equally passing methods
- time and memory break ties after correctness and iteration quality

---

## 15. Required Evaluation Report

Each run should emit a machine-readable report containing at least:

```python
{
    "phase": "phase0" | "phase1",
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
    "elapsed_native_seconds": float,
    "peak_memory_mib": float,
    "timing_lane_language": str,
    "timing_lane_official": bool,
    "blob_iou": [float, ... length 8],
    "positive_recall": [float, ... length 8],
    "positive_precision": [float, ... length 8],
    "outlier_recall": [float, ... length 8],
    "outlier_precision": [float, ... length 8],
}
```

Additional debug fields may be added freely.

---

## 16. Open Items

The following remain intentionally open:

1. the official dataset generator and seed suite
2. benchmark reference hardware
3. final hard time budget
4. final hard memory budget
5. whether phase-1 hard metrics should eventually incorporate one or more supplemental metrics

Until these are frozen, the evaluator should treat scores as draft but still
apply the structural rules above.
