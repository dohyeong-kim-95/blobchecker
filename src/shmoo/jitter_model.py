"""Jitter sources for the shmoo model.

See docs/shmoo_model.md section 6 and the per-layer diversity section.

Three distinct uses of "jitter":

- **RJ** (random jitter): per-sample Gaussian perturbation of the sampling
  instant. Drives the Monte Carlo pass/fail margin.
- **per-pattern skew**: a fixed horizontal offset drawn once per (pattern,
  layer). The main source of inter-layer blob diversity; must be independent
  per pattern, otherwise the eye merely translates.
- **DCD**: deterministic edge-position shift, handled in ``signal_gen`` during
  waveform reconstruction.
"""

from __future__ import annotations

import numpy as np


def sample_rj(rng: np.random.Generator, n_mc: int, sigma_rj: float) -> np.ndarray:
    """Draw ``n_mc`` random-jitter offsets (in UI), shape (n_mc, 1)."""
    return rng.standard_normal((n_mc, 1)) * sigma_rj


def sample_skew(rng: np.random.Generator, sigma_skew: float) -> float:
    """Draw one per-pattern horizontal skew offset (in UI)."""
    return float(rng.standard_normal() * sigma_skew)
