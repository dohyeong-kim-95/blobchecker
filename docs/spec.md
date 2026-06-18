# Blobchecker Canonical Spec

This document is the detailed source of truth for the problem and evaluation
contract. For decisions, start with [../README.md](../README.md).

## Problem

Estimate 16 binary blob maps on a shared 2D grid.

At each iteration, an algorithm chooses one coordinate `(row, col)`. The same
coordinate is queried across all 16 layers, producing a 16-value observation
vector. A coordinate selection is the atomic cost unit.

The blob source is a **DRAM Shmoo Plot** physical model: each layer's blob is
the PASS region of a shmoo over `(timing, vref)`. See
[shmoo_model.md](shmoo_model.md) for the generator's canonical design.

The algorithm knows only the grid shape `(H, W) = (150, 200)` before querying
begins. It has no prior knowledge of blob position, blob size, or coverage.

## Data Contract

The dataset generator emits three aligned tensors:

```python
truth_blob_mask.shape    == (16, H, W)   # H=150 (Vref rows), W=200 (timing cols)
truth_outlier_mask.shape == (16, H, W)
truth_full_mask.shape    == (16, H, W)
truth_full_mask = truth_blob_mask | truth_outlier_mask
```

In the current Shmoo domain there are no outliers, so `truth_outlier_mask` is
all zeros and `truth_full_mask == truth_blob_mask`. The three-tensor signature
is kept for backward compatibility.

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

Each layer has exactly one meaningful blob: the PASS region of a DRAM Shmoo
Plot. The full physical model (RC edges, soft saturation, discrete ISI taps,
DCD, RJ, per-layer diversity) is canonical in [shmoo_model.md](shmoo_model.md).

Required structural properties:

| Property | Contract |
|---|---|
| Connectivity | One 8-connected PASS region (eye) per layer. |
| Coverage | Target 50% mean (coverage ladder C1; TODO lower in later stages). |
| Boundary | Physically coherent eye boundary set by ISI/DCD/RJ envelope. |
| Holes | None (eye interior is monotonically PASS). |
| Outliers | None (eye exterior is monotonically FAIL). |
| Diversity | Per-pattern per-layer horizontal skew + per-layer ISI scaling. |

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

For the current grid:

```python
H, W = 150, 200
iteration_cap = 4500
```

### Development Budget Ladder

The official submission cap remains `int(0.15 * H * W)`. During development,
candidate algorithms should also be evaluated at wider query budgets so that
algorithm structure and budget compression are not conflated.

| Development phase | Budget ratio | 150x200 iterations | Purpose |
|---|---:|---:|---|
| Phase A | 50% | 15,000 | Find a structure that can pass before optimizing query count. |
| Phase B | 30% | 9,000 | Remove redundant queries and rebalance phase allocations. |
| Phase C | 20% | 6,000 | Compress weakest-layer and boundary priorities. |
| Phase D | 15% | 4,500 | Match the official submission condition. |

Rules:

- Phase A-C are diagnostic development gates, not relaxed submission gates.
- A method that only passes Phase A is not considered solved.
- When a change improves a lower budget phase, compare it against the same
  method at higher phases to verify that the improvement is structural rather
  than seed-specific noise.
- Report public and validation seed suites separately at every budget ratio.

## Compute Cost Metric

Iteration count is the primary task cost, but it does not capture algorithm
runtime. A low-level implementation lane, such as C, Fortran, or
Nastran-adjacent solver code, must also report an implementation-independent
operation estimate.

The canonical compute metric is **BOPs**: blobchecker normalized operations.
One BOP approximates one scalar primitive operation in a tight native loop. The
metric is not a CPU-cycle claim; it is a stable accounting convention for
comparing algorithms before Python, interpreter, allocator, BLAS, or hardware
effects enter the measurement.

### Counted Region

Count all algorithm-core work between the first call to `next_query()` and the
final `predict()`:

- coordinate planning and acquisition scoring
- oracle-result state update
- belief propagation or reconstruction state updates
- prediction-mask construction
- connected-component, bounding-box, trimming, hole, and outlier logic inside
  the algorithm

Do not count:

- dataset generation
- evaluator scoring
- logging, printing, JSON serialization, plotting, or file I/O
- Python interpreter overhead in prototype runs
- the oracle table lookup itself, except for copying the 8 returned labels into
  algorithm-owned state

### BOP Weight Table

Use this table when deriving an estimated operation count from source code or
native instrumentation:

| Primitive | BOPs |
|---|---:|
| integer add/subtract/compare/min/max/boolean op | 1 |
| floating add/subtract/compare/min/max/abs | 1 |
| integer multiply, bit shift, index arithmetic | 1 |
| floating multiply or fused multiply-add | 1 |
| division, modulo, square root | 8 |
| `log`, `exp`, `pow`, trigonometric function | 32 |
| scalar load or store from contiguous array | 1 |
| scalar load or store from non-contiguous or indirect array | 2 |
| branch with predictable condition | 1 |
| branch with data-dependent unpredictable condition | 3 |
| queue/stack push or pop for BFS/connected components | 4 |
| cache-friendly scan over one scalar array element | load/store plus primitive ops above |
| FFT, dense linear algebra, or solver library call | report separately with the formula below |

For library calls, report both the formula and the realized estimate:

```text
fft2_real(H, W)       = 5 * H * W * log2(H * W) BOPs
dense_matmul(n,m,k)  = 2 * n * m * k BOPs
dense_cholesky(n)    = (1/3) * n^3 BOPs
dense_gp_update(n)   = O(n^3) BOPs unless a tighter implementation count is supplied
```

### Reporting Fields

Every benchmark report that claims native runtime should include:

```python
{
    "estimated_bops_total": int,
    "estimated_bops_per_iteration": float,
    "estimated_bops_breakdown": {
        "planning": int,
        "update": int,
        "prediction": int,
        "reconstruction": int,
        "library_calls": int,
    },
    "native_elapsed_seconds": float,
    "native_bops_per_second": float,
}
```

Where:

```python
native_bops_per_second = estimated_bops_total / native_elapsed_seconds
```

If the implementation is still Python-only, `estimated_bops_total` may be a
source-derived estimate and `native_bops_per_second` should be omitted or marked
as prototype-only.

### Interpretation

BOPs are a proxy for compute time, not a replacement for measured native
elapsed time. They answer different questions:

- iteration count: how many coordinates were consumed
- BOPs: how much algorithmic work was required
- native elapsed time: how fast the implementation actually ran
- memory: whether the method fits the target execution envelope

Research implications for this project:

- summed entropy, boundary scoring, and occupancy updates should remain around
  `O(8 * H * W)` per full-grid scoring pass
- pre-planned greedy coverage is acceptable as a baseline but can be expensive
  if it rescans the full grid for every planned coordinate
- dense GP, MOBO, and solver-heavy CS methods must be justified with BOP and
  memory estimates before implementation

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
4. lower estimated BOPs
5. lower native elapsed time
6. lower peak memory

Supplemental metrics may include blob IoU, positive recall, positive precision,
boundary error, outlier recall, and outlier precision. They do not replace the
hard gates above.

## Phase 0 Benchmark Context

Current Phase 0 implementation uses:

```python
PHASE0_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
PHASE0_VALIDATION_SEEDS = [100, 101, ..., 199]
```

The public seed suite is for development comparison and regression tracking.
The validation seed suite is for overfitting checks before accepting a new
algorithm or parameter change.

Rules:

- Do not tune parameters on validation results repeatedly.
- Report public and validation results separately.
- Treat public-only improvement with validation regression as overfitting until
  disproven.
- Freeze final official seeds before making official benchmark claims.

Open benchmark items:

- reference hardware
- final hard time budget
- final hard BOP budget per phase
- final hard memory budget
- Phase 1 grid size range
- Phase 1 seed suite
- whether supplemental metrics become hard gates later
