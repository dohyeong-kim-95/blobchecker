# Contour — Level-Set Estimation

Contour-following estimator for binary map reconstruction. No GP, no surrogate
model — exploits the known structure (one smooth blob with rare interior holes)
via scanline binary search and targeted interior probing.

## Structure

```
raw/synth/synthetic_map.py   blob generator (ground truth oracle)
contour/contour.py           BaseEstimator interface + ContourEstimator
contour/evaluate.py          end-to-end evaluation script
```

## Quick start

```bash
python contour/evaluate.py
```

```
Grid        : 50×200  (10000 pixels)
Budget      : 1500 samples  (15%)
Pixel acc   : 97.74%
Elapsed     : 0.02 s
Peak memory : 92.7 MiB
```

## Algorithm: ContourEstimator

### Phase 1 — Log-spaced scanlines  (~5% of budget)
Binary-search each of ~15 rows spaced exponentially outward from the grid
midpoint.  Establishes the blob's vertical extent and seeds Phase 2 with
targeted column probes for narrow segments.

### Phase 2 — Per-row boundary scan  (~80% of budget)
For every row, binary-search for the **left** and **right** blob boundaries:
- Grid-edge check first (`col 0` and `col W-1`): if already 1, that IS the
  boundary — avoids holes terminating the search prematurely.
- Post-boundary probe at offsets +1, +6, +13: if any returns 1 a hole stopped
  the binary search early; re-run from the new seed to find the true boundary.
- `extra_cols` from Phase 1 ensure narrow blob segments are not missed.
- Phase 2b: explicit 0-anchors at `(r, left-1)` and `(r, right+1)` prevent
  interior 1-samples from bleeding across row boundaries in NN prediction.

### Phase 3 — Interior grid sampling  (~15% of budget)
Uniform grid over the blob bounding box.  Detects interior holes (0-regions).

### Prediction
- **Exterior/interior split**: directly from Phase 2 boundary lines — no NN
  bleed across rows.
- **Hole recovery**: nearest-neighbour among interior-sampled points only,
  applied inside the predicted 1-region.

## Performance vs GP alternatives

| | sklearn GP (`lse/`) | Sparse GP (`sparse_gp/`) | Contour (`contour/`) |
|---|---|---|---|
| Pixel accuracy | 98.6% | 95.8% | 97.7% |
| Elapsed | 158 s | 7.8 s | **0.02 s** |
| Peak memory | ~700 MiB | ~695 MiB | **93 MiB** |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `blob_size` | 0.35 | Fraction of map covered by blob |
| `n_holes` | 2 | Interior 0-holes inside blob |
| `hole_size` | 5.0 | Hole radius (pixels) |
| `n_outliers` | 10 | Isolated 1-pixels in background |
| `seed` | 42 | RNG seed |
| `phase1_frac` | 0.05 | Budget fraction for Phase 1 |
| `phase2_frac` | 0.80 | Cumulative budget fraction by end of Phase 2 |

## Dependencies

```
numpy scipy matplotlib
```
