"""Shmoo pass/fail evaluation and dataset generation.

See docs/shmoo_model.md. The "all Monte Carlo iterations pass" rule collapses to
an extreme value (min/max) over the MC draws per pattern, which is both exact
and far cheaper than a full MC loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel_model import DEFAULT_TAPS, apply_isi
from .jitter_model import sample_rj, sample_skew
from .signal_gen import reconstruct_waveform, soft_saturate

#: All eight 3-bit patterns. The middle bit (index 1) is the sampling target.
PATTERNS: tuple[tuple[int, int, int], ...] = tuple(
    (a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)
)


@dataclass
class ShmooConfig:
    """Physical and grid parameters for a single shmoo layer."""

    H: int = 150                       # vref rows
    W: int = 200                       # timing cols
    n_mc: int = 100
    samples_per_ui: int = 64
    tau_rise: float = 0.15
    tau_fall: float = 0.15
    sigma_rj: float = 0.015
    dcd_offset: float = 0.01
    sat_k: float = 10.0
    sigma_skew: float = 0.06
    n_preamble: int = 16
    n_postamble: int = 16
    vref_lo: float = 0.2               # volts (200 mV)
    vref_hi: float = 0.8               # volts (800 mV)
    timing_lo: float = -0.5            # UI
    timing_hi: float = 0.5             # UI

    @property
    def target_center(self) -> float:
        """Time (UI) of the target bit center: n_preamble + 1 + 0.5."""
        return self.n_preamble + 1.5

    @property
    def timing_grid(self) -> np.ndarray:
        return np.linspace(self.timing_lo, self.timing_hi, self.W)

    @property
    def vref_grid(self) -> np.ndarray:
        return np.linspace(self.vref_lo, self.vref_hi, self.H)


def _pattern_bits(pattern: tuple[int, int, int], cfg: ShmooConfig) -> np.ndarray:
    bits = np.zeros(cfg.n_preamble + 3 + cfg.n_postamble, dtype=float)
    bits[cfg.n_preamble : cfg.n_preamble + 3] = pattern
    return bits


def layer_envelopes(
    rng: np.random.Generator,
    cfg: ShmooConfig,
    *,
    isi_scale: float = 1.0,
    collect_waves: bool = False,
):
    """Compute the eye envelopes for one layer.

    For each pattern the waveform is built once, sampled across the timing grid
    under ``n_mc`` random-jitter draws and a per-pattern skew, and reduced to a
    worst-case effective voltage per timing column:

    - target bit == 1: ``min`` over MC (lowest high level still must beat Vref)
    - target bit == 0: ``max`` over MC (highest low level still must stay below)

    Returns
    -------
    (min_high, max_low) or (min_high, max_low, waves)
        ``min_high[W]`` is the lower envelope of the high patterns; ``max_low[W]``
        the upper envelope of the low patterns. A point passes iff
        ``max_low < vref < min_high``. ``waves`` (optional) is a list of
        ``(pattern, t_axis, wave)`` for plotting.
    """
    taps = np.asarray(DEFAULT_TAPS, dtype=float) * isi_scale
    timing = cfg.timing_grid
    min_high = np.full(cfg.W, np.inf)
    max_low = np.full(cfg.W, -np.inf)
    waves = [] if collect_waves else None

    for pattern in PATTERNS:
        bits = _pattern_bits(pattern, cfg)
        received = apply_isi(bits, taps)
        levels = soft_saturate(received, 0.0, 1.0, cfg.sat_k)
        t_axis, wave = reconstruct_waveform(
            levels,
            tau_rise=cfg.tau_rise,
            tau_fall=cfg.tau_fall,
            dcd_offset=cfg.dcd_offset,
            samples_per_ui=cfg.samples_per_ui,
        )
        # Draw order fixed for reproducibility: skew, then RJ.
        skew = sample_skew(rng, cfg.sigma_skew)
        rj = sample_rj(rng, cfg.n_mc, cfg.sigma_rj)            # (n_mc, 1)
        t_sample = cfg.target_center + timing[None, :] + rj - skew  # (n_mc, W)
        v = np.interp(t_sample.ravel(), t_axis, wave).reshape(cfg.n_mc, cfg.W)

        if pattern[1] == 1:
            min_high = np.minimum(min_high, v.min(axis=0))
        else:
            max_low = np.maximum(max_low, v.max(axis=0))

        if collect_waves:
            waves.append((pattern, t_axis, wave))

    if collect_waves:
        return min_high, max_low, waves
    return min_high, max_low


def _threshold(min_high: np.ndarray, max_low: np.ndarray, cfg: ShmooConfig,
               guard_v: float) -> np.ndarray:
    """Build the PASS mask from envelopes and a guard band."""
    vg = cfg.vref_grid[:, None]                       # (H, 1)
    lo = (max_low + guard_v)[None, :]                 # (1, W)
    hi = (min_high - guard_v)[None, :]
    return ((vg > lo) & (vg < hi)).astype(np.uint8)   # (H, W)


def generate_shmoo(
    rng: np.random.Generator,
    cfg: ShmooConfig | None = None,
    *,
    isi_scale: float = 1.0,
    guard_v: float = 0.0,
) -> np.ndarray:
    """Generate one shmoo PASS mask (one blob layer), shape (H, W) uint8."""
    cfg = cfg or ShmooConfig()
    min_high, max_low = layer_envelopes(rng, cfg, isi_scale=isi_scale)
    return _threshold(min_high, max_low, cfg, guard_v)


def _child_seeds(seed: int, n_layers: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_layers)


def _layer_isi_scale(rng: np.random.Generator) -> float:
    """Per-layer ISI tap-magnitude scaling for vertical/shape diversity."""
    return float(rng.uniform(0.5, 1.5))


def generate_dataset(
    H: int = 150,
    W: int = 200,
    seed: int = 0,
    n_layers: int = 16,
    *,
    coverage_range: tuple[float, float] = (0.20, 0.55),
    guard_v: float | None = None,
    cfg: ShmooConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a Shmoo blob dataset.

    Returns
    -------
    truth_blob_mask    : (n_layers, H, W) uint8 — scoring target (PASS region)
    truth_outlier_mask : (n_layers, H, W) uint8 — all zeros (no outliers)
    truth_full_mask    : (n_layers, H, W) uint8 — equals truth_blob_mask

    Each layer is given an independent coverage target drawn uniformly from
    ``coverage_range``; a per-layer guard band is then calibrated to hit it.
    This spreads coverage across the range (more diversity, smaller eyes mixed
    in). If ``guard_v`` is given, it overrides calibration with a fixed guard
    for every layer.
    """
    cfg = cfg or ShmooConfig(H=H, W=W)
    cfg.H, cfg.W = H, W

    child_seeds = _child_seeds(seed, n_layers)
    cov_targets = np.random.default_rng(seed * 2 + 1).uniform(
        coverage_range[0], coverage_range[1], size=n_layers
    )

    masks = []
    for i, cs in enumerate(child_seeds):
        rng = np.random.default_rng(cs)
        isi_scale = _layer_isi_scale(rng)
        # Envelopes are independent of the guard, so compute once per layer.
        min_high, max_low = layer_envelopes(rng, cfg, isi_scale=isi_scale)
        if guard_v is not None:
            g = guard_v
        else:
            g = _calibrate_guard(
                lambda gg, mh=min_high, ml=max_low: float(_threshold(mh, ml, cfg, gg).mean()),
                cov_targets[i],
            )
        masks.append(_threshold(min_high, max_low, cfg, g))

    blob = np.stack(masks)
    out = np.zeros_like(blob)
    return blob, out, blob.copy()


def _calibrate_guard(coverage_fn, target: float, lo: float = -0.6,
                     hi: float = 0.7, iters: int = 30) -> float:
    """Binary-search the guard band so coverage matches ``target``.

    Coverage decreases monotonically as the guard increases. Negative guards
    dilate the eye beyond the raw envelopes (for targets above raw coverage).
    """
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if coverage_fn(mid) > target:
            lo = mid          # too much PASS area -> widen guard
        else:
            hi = mid
    return (lo + hi) / 2.0
