# Evaluation Spec: Shared-Coordinate Multi-Blob Reconstruction

## Status

Draft 0.1

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
- which implementation languages are eligible for official timing claims

---

## 1. Evaluation Inputs

The evaluator consumes:

- `truth`: ground-truth blob stack of shape `(8, H, W)`, binary
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
predicted.shape == (8, H, W)
```

And the evaluation record must include:

- `iterations_used`
- `predicted`
- optional per-iteration trace for debugging and analysis
- implementation metadata for the timing lane

All predicted values must be binary.

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

Initial fixed cap:

```python
iteration_cap = int(0.15 * H * W)
```

A run that exceeds this cap is an automatic failure.

This cap is provisional but active for the first benchmark phase.

---

## 4. Main Blob Scoring View

Each layer contains one primary blob of interest plus possible external
positive outliers.

Evaluation is main-blob-centric:

- the main blob must be reconstructed accurately
- isolated positive outliers outside the main blob do not need to be recovered

### 4.1 Main blob selector

The exact rule for identifying the main blob is not yet frozen.

For now, the evaluator must treat main-blob selection as a pluggable policy:

```python
select_main_blob(mask: np.ndarray) -> np.ndarray
```

Requirements for draft evaluation:

- every report must record which selector policy was used
- until the selector is frozen, all published scores are provisional
- the selector must be applied consistently to both truth and prediction when
  computing main-blob-derived metrics

Open item:

- whether the selector should use largest connected component, largest bounding
  box, highest mass near a seed, or another rule remains undecided

---

## 5. Accuracy Metric

### 5.1 Main-blob-focused truth

To avoid penalizing failure to recover external positive outliers, the evaluator
must derive a scored truth mask per blob:

```python
truth_main[k] = select_main_blob(truth[k])
```

This means:

- positives belonging to the ground-truth main blob remain required
- positives outside the selected main blob are not required targets

### 5.2 Per-blob accuracy

For each blob `k`:

```python
accuracy_k = mean(predicted[k] == truth_main[k])
```

This keeps the scoring aligned with the project goal:

- missing external positive outliers is tolerated
- false positives outside the main blob still hurt accuracy

### 5.3 Overall correctness rule

A run passes only if every blob passes every active requirement.

No average-only pass rule is allowed.

---

## 6. Bounding-Box Recovery

For each blob `k`, define:

- `bbox_truth_k` from the selected main blob in truth
- `bbox_pred_k` from the selected main blob in prediction

The evaluator extracts:

- `height_truth_k`, `width_truth_k`
- `height_pred_k`, `width_pred_k`

### 6.1 Phase 0 scaffold rule

Phase 0 exists to let the algorithm skeleton stabilize before exact matching is
required.

In phase 0, both dimension errors must satisfy:

```python
abs(height_pred_k - height_truth_k) / max(height_truth_k, 1) <= 0.05
abs(width_pred_k  - width_truth_k)  / max(width_truth_k, 1)  <= 0.05
```

Interpretation:

- up to 5% relative error is allowed for main-blob height
- up to 5% relative error is allowed for main-blob width

### 6.2 Phase 1 target rule

Phase 1 is the intended end-state requirement.

For each blob `k`:

```python
height_pred_k == height_truth_k
width_pred_k  == width_truth_k
```

That is, exact match.

---

## 7. Acceptance Phases

### 7.1 Phase 0: skeleton-building lane

A phase-0 run passes if all of the following hold:

- `iterations_used <= int(0.15 * H * W)`
- for every blob, `accuracy_k >= 0.98`
- for every blob, main-blob height is within 5%
- for every blob, main-blob width is within 5%
- timing and memory are reported

Phase 0 is intended to validate the reconstruction loop, query planner, and
shared-coordinate architecture.

### 7.2 Phase 1: target lane

A phase-1 run passes if all of the following hold:

- `iterations_used <= int(0.15 * H * W)`
- for every blob, `accuracy_k >= 0.98`
- for every blob, main-blob height matches exactly
- for every blob, main-blob width matches exactly
- timing and memory budgets are satisfied in the official timing lane

---

## 8. Time Budget

### 8.1 Measurement principle

The time budget is not a Python-level budget.

Official elapsed-time claims must measure the algorithm core in a native,
low-level implementation lane, excluding Python orchestration overhead wherever
possible.

Measured region should include only:

- coordinate planning
- oracle-state update logic
- reconstruction logic
- stopping logic

Measured region must exclude:

- plot generation
- debug printing
- file I/O unrelated to required inputs
- report serialization
- Python wrapper overhead, if a wrapper is used only as a harness

### 8.2 Timing lane restriction

For official timing eligibility, the implementation core must be written in a
language whose first public release predates C in 1972.

Examples of eligible timing-lane languages:

- assembly
- B
- BCPL
- Fortran
- ALGOL
- PL/I

Examples that are not eligible for official timing claims in this draft:

- Python
- C
- C++
- Rust
- Go
- Java
- JavaScript

These later languages may still be used for correctness experiments, prototyping,
and harness work, but their elapsed times are considered non-official.

### 8.3 Budget status

Because benchmark hardware has not yet been frozen, elapsed-time thresholds are
currently provisional.

Draft working budgets:

- phase 0 provisional target: `elapsed_native <= 1.0 s`
- phase 1 provisional target: `elapsed_native <= 250 ms`

These numbers must be interpreted only on the designated reference machine once
that machine is defined.

Until then:

- elapsed time must always be reported
- elapsed time is advisory in phase 0
- elapsed time becomes a hard gate only after the reference machine is fixed

---

## 9. Memory Budget

### 9.1 Measurement principle

Peak memory should be measured as process peak resident set size, or the closest
available low-level equivalent on the reference platform.

Preferred unit:

- MiB

### 9.2 Budget status

As with elapsed time, the memory thresholds are provisional until the reference
machine and implementation lane are finalized.

Draft working budgets:

- phase 0 provisional target: `peak_memory <= 64 MiB`
- phase 1 provisional target: `peak_memory <= 16 MiB`

Rules:

- memory usage must be reported for every benchmarked run
- phase-0 memory is advisory
- phase-1 memory becomes a hard gate after the reference environment is fixed

---

## 10. Ranking Rule

Methods are ranked lexicographically:

1. pass/fail status
2. fewer iterations used
3. lower native elapsed time
4. lower peak memory

Interpretation:

- correctness dominates everything else
- iteration efficiency dominates performance micro-optimizations
- time and memory break ties among otherwise equivalent passing methods

---

## 11. Required Evaluation Report

Each run should emit a machine-readable report containing at least:

```python
{
    "phase": "phase0" | "phase1",
    "grid_shape": [H, W],
    "iteration_cap": int,
    "iterations_used": int,
    "per_blob_accuracy": [float, ... length 8],
    "per_blob_height_truth": [int, ... length 8],
    "per_blob_height_pred": [int, ... length 8],
    "per_blob_width_truth": [int, ... length 8],
    "per_blob_width_pred": [int, ... length 8],
    "per_blob_height_pass": [bool, ... length 8],
    "per_blob_width_pass": [bool, ... length 8],
    "accuracy_pass": [bool, ... length 8],
    "overall_pass": bool,
    "elapsed_native_seconds": float,
    "peak_memory_mib": float,
    "timing_lane_language": str,
    "timing_lane_official": bool,
    "main_blob_selector": str,
}
```

Additional debug fields may be added freely.

---

## 12. Open Items

The following remain intentionally open:

1. exact main-blob selector rule
2. benchmark reference hardware
3. final hard time budget
4. final hard memory budget
5. whether phase-1 exact height/width matching should still allow a special
   escape hatch for degenerate tiny blobs

Until these are frozen, the evaluator should treat scores as draft but still
apply the structural rules above.
