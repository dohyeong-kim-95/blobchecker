# Sparse GP — Level-Set Estimation

Sparse GP implementation of the Straddle sequential sampler, replacing the
dense sklearn GP in `lse/` with a GPyTorch SGPR (FITC approximation) to
meet the time and memory constraints on a 50×200 grid.

## Structure

```
raw/synth/synthetic_map.py    blob generator (ground truth oracle)
sparse_gp/sparse_gp.py       BaseEstimator interface + SparseStraddleGPR
sparse_gp/evaluate.py        end-to-end evaluation script
```

## Quick start

```bash
# from repo root
python sparse_gp/evaluate.py
```

Output:
```
Grid        : 50×200  (10000 pixels)
Budget      : 1500 samples  (15%)
Truth cover : 35.48%
Inducing pts: 200
Pixel acc   : 95.83%
Elapsed     : 7.8 s
Peak memory : ~695 MiB
```

## Algorithm: SparseStraddleGPR

1. **Init** — random space-filling samples (`n_init = refit_interval`)
2. **Fit** — GPyTorch SGPR with `InducingPointKernel` (FITC), `n_init_steps=100` Adam steps
3. **Acquire** — Straddle score: `|μ(x) − 0.5| − κ·σ(x)`, pick argmin over unsampled
4. **Warm refit** — `set_train_data()` + `n_warm_steps=30` Adam steps (keeps hyperparams)
5. **Repeat** — every `refit_interval=50` steps until budget exhausted
6. **Predict** — threshold `μ ≥ 0.5` on full grid

Budget = **15% of total pixels** (1 500 calls for a 50×200 grid).

## Why Sparse GP?

| | sklearn dense GP (`lse/`) | GPyTorch SGPR (`sparse_gp/`) |
|---|---|---|
| Train complexity | O(n³) | O(n·m²), m=200 inducing pts |
| Elapsed (50×200) | ~158 s | **7.8 s** |
| Pixel accuracy | ~98.6% | ~95.8% |

The FITC kernel approximates the full covariance matrix via m inducing
points, reducing the Cholesky bottleneck from O(n³) to O(m³ + n·m²).

## Memory note

Process peak memory (~695 MiB) is dominated by PyTorch's runtime (~600 MiB
base), not by the SGPR data structures (inducing matrix K_mm: 200²×4 B = 160 KB;
kernel cross-matrix K_nm: 1500×200×4 B = 1.2 MB). The 100 MB constraint
requires a non-PyTorch sparse GP implementation.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `blob_size` | 0.35 | Fraction of map covered by blob |
| `n_holes` | 2 | Interior 0-holes inside blob |
| `hole_size` | 5.0 | Hole radius (pixels) |
| `n_outliers` | 10 | Isolated 1-pixels in background |
| `seed` | 42 | RNG seed |
| `kappa` | 1.5 | Exploration weight in Straddle score |
| `refit_interval` | 50 | Steps between GP refits |
| `n_inducing` | 200 | Number of FITC inducing points |
| `n_init_steps` | 100 | Adam steps for initial fit |
| `n_warm_steps` | 30 | Adam steps for warm-start refits |

## Dependencies

```
numpy scipy matplotlib torch gpytorch
```
