#!/usr/bin/env python3
"""Generate the Shmoo blob dataset, run self-checks, and render PNGs.

Examples
--------
    python tools/visualize_shmoo.py
    python tools/visualize_shmoo.py --seed 3 --selftest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shmoo.channel_model import DEFAULT_TAPS, apply_isi
from src.shmoo.plotting import plot_eye_diagram, plot_shmoo_grid
from src.shmoo.shmoo_eval import (
    PATTERNS,
    ShmooConfig,
    _child_seeds,
    _layer_isi_scale,
    generate_dataset,
    layer_envelopes,
)
from src.shmoo.signal_gen import reconstruct_waveform, soft_saturate


# ---------------------------------------------------------------------------
# Self-checks (the implementation traps from docs/shmoo_model.md)
# ---------------------------------------------------------------------------

def _check_isi_alignment() -> None:
    """received[17] must depend only on the target bit and predecessors."""
    cfg = ShmooConfig()
    p0, p1, p2 = 1, 1, 0
    bits = np.zeros(cfg.n_preamble + 3 + cfg.n_postamble)
    bits[cfg.n_preamble : cfg.n_preamble + 3] = (p0, p1, p2)
    received = apply_isi(bits, DEFAULT_TAPS)
    expected = (
        p1 * DEFAULT_TAPS[0]
        + p0 * DEFAULT_TAPS[1]
        + 0 * DEFAULT_TAPS[2]
        + 0 * DEFAULT_TAPS[3]
    )
    assert abs(received[17] - expected) < 1e-12, (
        f"ISI misalignment: received[17]={received[17]} != {expected}"
    )


def _check_mc_collapse() -> None:
    """min/max-over-MC threshold must equal an explicit all-MC/all-pattern AND."""
    cfg = ShmooConfig(H=40, W=60, n_mc=32)
    guard = 0.05

    # Fast path.
    rng_fast = np.random.default_rng(123)
    mh, ml = layer_envelopes(rng_fast, cfg, isi_scale=1.0)
    vg = cfg.vref_grid[:, None]
    fast = (vg > (ml + guard)[None, :]) & (vg < (mh - guard)[None, :])

    # Slow reference: identical draw order, full samples, explicit ALL reduction.
    rng_slow = np.random.default_rng(123)
    taps = np.asarray(DEFAULT_TAPS) * 1.0
    timing = cfg.timing_grid
    passes_high = np.ones((cfg.H, cfg.W), bool)
    passes_low = np.ones((cfg.H, cfg.W), bool)
    for pattern in PATTERNS:
        bits = np.zeros(cfg.n_preamble + 3 + cfg.n_postamble)
        bits[cfg.n_preamble : cfg.n_preamble + 3] = pattern
        levels = soft_saturate(apply_isi(bits, taps), 0.0, 1.0, cfg.sat_k)
        t_axis, wave = reconstruct_waveform(
            levels, tau_rise=cfg.tau_rise, tau_fall=cfg.tau_fall,
            dcd_offset=cfg.dcd_offset, samples_per_ui=cfg.samples_per_ui,
        )
        skew = float(rng_slow.standard_normal() * cfg.sigma_skew)
        rj = rng_slow.standard_normal((cfg.n_mc, 1)) * cfg.sigma_rj
        t_sample = cfg.target_center + timing[None, :] + rj - skew
        v = np.interp(t_sample.ravel(), t_axis, wave).reshape(cfg.n_mc, cfg.W)
        for m in range(cfg.n_mc):
            if pattern[1] == 1:
                passes_high &= cfg.vref_grid[:, None] < (v[m] - guard)[None, :]
            else:
                passes_low &= cfg.vref_grid[:, None] > (v[m] + guard)[None, :]
    slow = passes_high & passes_low

    assert np.array_equal(fast, slow), "MC-collapse mismatch vs explicit AND"


def run_selftest() -> None:
    _check_isi_alignment()
    _check_mc_collapse()
    print("self-checks passed: ISI alignment, MC collapse")


# ---------------------------------------------------------------------------
# Reproduce one layer's waveforms for the eye diagram
# ---------------------------------------------------------------------------

def _layer_waves(seed: int, layer: int, cfg: ShmooConfig):
    cs = _child_seeds(seed, layer + 1)[layer]
    rng = np.random.default_rng(cs)
    isi_scale = _layer_isi_scale(rng)
    _, _, waves = layer_envelopes(rng, cfg, isi_scale=isi_scale, collect_waves=True)
    return waves


def _eye_metrics(blob_layer: np.ndarray, cfg: ShmooConfig) -> tuple[float, float]:
    """Eye height (mV) and width (UI) of one PASS mask."""
    rows = np.any(blob_layer, axis=1)
    cols = np.any(blob_layer, axis=0)
    if not rows.any():
        return 0.0, 0.0
    vref = cfg.vref_grid
    timing = cfg.timing_grid
    h_mv = (vref[rows].max() - vref[rows].min()) * 1000
    w_ui = timing[cols].max() - timing[cols].min()
    return float(h_mv), float(w_ui)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shmoo blob generator + viz.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--coverage", type=float, default=0.5)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--outdir", type=Path, default=ROOT / "docs" / "images")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()

    cfg = ShmooConfig(H=150, W=200)
    t0 = time.perf_counter()
    blob, out, full = generate_dataset(
        H=cfg.H, W=cfg.W, seed=args.seed, n_layers=args.n_layers,
        target_coverage=args.coverage, cfg=cfg,
    )
    elapsed = time.perf_counter() - t0

    coverages = blob.reshape(args.n_layers, -1).mean(axis=1)
    print(f"\ngenerated {blob.shape} in {elapsed:.2f}s  "
          f"(seed={args.seed}, target cov={args.coverage:.0%})")
    print(f"mean coverage {coverages.mean()*100:.1f}%  "
          f"[min {coverages.min()*100:.1f}%, max {coverages.max()*100:.1f}%]")
    assert np.array_equal(full, blob) and out.sum() == 0, "contract: full==blob, no outliers"

    print(f"\n  {'layer':>5}  {'coverage':>8}  {'eye_h(mV)':>9}  {'eye_w(UI)':>9}")
    for k in range(args.n_layers):
        h_mv, w_ui = _eye_metrics(blob[k], cfg)
        print(f"  {k:>5}  {coverages[k]*100:>7.1f}%  {h_mv:>9.0f}  {w_ui:>9.3f}")

    grid_path = plot_shmoo_grid(blob, cfg, args.outdir / f"shmoo_seed{args.seed:03d}.png")
    waves = _layer_waves(args.seed, 0, cfg)
    eye_path = plot_eye_diagram(waves, cfg, args.outdir / f"eye_seed{args.seed:03d}.png", layer=0)
    print(f"\nwrote {grid_path}")
    print(f"wrote {eye_path}")


if __name__ == "__main__":
    main()
