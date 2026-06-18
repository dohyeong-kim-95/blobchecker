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

## 2026-06-18 — Domain Pivot to DRAM Shmoo Generator

Problem:

- The gaussian-smoothing generator produced blobs that did not match the
  intended domain. The owner's real domain is DRAM I/O Shmoo characterization.

Decision:

- Replace the data generator with a DRAM Shmoo Plot physical model. The PASS
  region of a shmoo over `(timing, vref)` is the blob. Canonical design lives in
  [docs/shmoo_model.md](docs/shmoo_model.md); the raw requirement prompt is
  preserved in [todo.md](todo.md).

Contract changes:

- Output `(16, 150, 200)`: 16 layers, H=150 rows (Vref), W=200 cols (timing).
- Coverage target 50% (coverage ladder C1; lower in later stages — TODO).
- No holes, no outliers. `truth_outlier=0`, `truth_full==truth_blob`.
- Iteration cap `int(0.15*150*200) = 4500`.

Generator design notes (pre-implementation):

- Layer diversity = per-pattern per-layer horizontal skew (main source) plus
  per-layer ISI tap-magnitude scaling (vertical/height variety). Uniform skew
  across all patterns only translates the eye; it must be per-pattern.
- "All MC pass" collapses to `min/max` over MC per pattern, then a Vref-axis
  broadcast and an 8-pattern AND. Big speedup and a clean boundary.

Implementation traps recorded ahead of coding:

- `np.convolve(..., mode='same')` misaligns the main cursor by one bit for the
  even-length tap kernel. Use causal convolution and verify `received[17]`.
- Sampling needs linear interpolation; nearest-sample aliases RJ because
  `1/64 UI ≈ σ_RJ`.

Status:

- Documentation renewed (README, spec, this journal, shmoo_model). Code is not
  implemented yet; the codebase is being cleaned first.

## 2026-06-18 — Shmoo Generator Implementation

Implemented `src/shmoo/` (signal_gen, channel_model, jitter_model, shmoo_eval,
plotting) plus `tools/visualize_shmoo.py`. Generates `(16, 150, 200)` uint8.

Results (seed 0):

```text
generated (16, 150, 200) in 0.08s
mean coverage 50.0%  [min 43.3%, max 57.2%]
self-checks passed: ISI alignment, MC collapse
```

Verification:

- `received[17]` matches the causal ISI formula (avoids the `np.convolve`
  `mode='same'` one-bit misalignment trap).
- MC-collapse `min/max` threshold is bit-for-bit equal to an explicit
  all-MC/all-pattern AND on a small grid (`_check_mc_collapse`).
- Coverage auto-calibrated to the target via one global guard band, binary
  searched over the per-layer envelopes (which are guard-independent, so the
  search is cheap and stable).

Observations:

- With `sat_k=10` the center voltage saturates to ~0/1, so eye height is nearly
  constant (~568 mV); inter-layer diversity shows up mainly as width
  (0.65-0.78 UI) and coverage. Lower `sat_k` for more vertical variety.
- The single-bit eye opens asymmetrically (~`[-0.2, +0.5] UI`) because the
  leading-edge rise time eats the left of the window. Physically expected.

Dropped `scipy` from requirements (no longer used); added `matplotlib`.

## 2026-06-18 — Coverage Range [20%, 55%]

Changed coverage from a single global 50% target to per-layer targets drawn
uniformly from [20%, 55%], each hit by a per-layer calibrated guard band.

Motivation: the fixed-50% eyes were large and similar, which over-emphasized
the inherent left/right edge asymmetry. Mixing smaller eyes in adds diversity
and tones the asymmetry down.

Result (seed 0):

```text
mean coverage 38.1%  [min 21.0%, max 53.3%]
eye height 270-600 mV, eye width 0.59-0.76 UI
```

Now both vertical (eye height) and horizontal (width) diversity are strong,
unlike the fixed-50% case where height was nearly constant (~568 mV).

API: `generate_dataset(..., coverage_range=(0.20, 0.55))`. A fixed `guard_v`
still overrides calibration. CLI: `--coverage-range LO HI`.

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
8. Generator domain must match the owner's real domain. The blob source is now
   physical (DRAM Shmoo), not synthetic gaussian noise.
9. For the Shmoo generator, two correctness traps dominate: causal ISI
   convolution (not `mode='same'`) and interpolated sampling (not nearest).
