# LSE — Level-Set Estimation

Sequential binary map estimation using Gaussian Process Regression.

## Structure

```
raw/synth/synthetic_map.py   blob generator (ground truth oracle)
lse/lse.py                   BaseEstimator interface + StraddleGPR
lse/evaluate.py              end-to-end evaluation script
```

## Quick start

```bash
# from repo root
python lse/evaluate.py
```

Output:
```
Grid        : 50×200  (10000 pixels)
Budget      : 1500 samples  (15%)
Truth cover : 34.67%
Pixel acc   : XX.XX%
Elapsed     : XX.X s
Saved PNG   → lse/result.png
```

## Algorithm: StraddleGPR

1. **Init** — space-filling random samples (`n_init = refit_interval`)
2. **Fit** — sklearn `GaussianProcessRegressor` (RBF + WhiteKernel)
3. **Acquire** — Straddle score: `|μ(x) − 0.5| − κ·σ(x)`, pick argmin
4. **Repeat** — sample `refit_interval` points, refit GP, repeat until budget
5. **Predict** — threshold `μ ≥ 0.5` on full grid

Budget = **15% of total pixels** (1 500 calls for a 50×200 grid).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `blob_size` | 0.35 | Fraction of map covered by the blob |
| `n_holes` | 2 | Interior 0-holes inside blob |
| `hole_size` | 5.0 | Hole radius (pixels) |
| `n_outliers` | 10 | Isolated 1-pixels in background |
| `seed` | 42 | RNG seed |
| `kappa` | 1.5 | Exploration weight in Straddle score |
| `refit_interval` | 50 | GP refit frequency (steps) |

## Dependencies

```
numpy scipy scikit-learn matplotlib
```
