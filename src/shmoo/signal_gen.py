"""Pulse waveform synthesis: soft saturation and RC-edge reconstruction.

See docs/shmoo_model.md sections 1, 3, 4, 5.
"""

from __future__ import annotations

import numpy as np


def soft_saturate(
    v: np.ndarray | float,
    v_low: float = 0.0,
    v_high: float = 1.0,
    k: float = 10.0,
) -> np.ndarray:
    """Smooth (C-infinity) voltage saturation, replacing hard clipping.

    Models MOSFET output-driver saturation. ``k`` controls sharpness:
    ``k -> inf`` approaches a hard clip, ``k -> 0`` approaches linear.

    Parameters
    ----------
    v : array_like
        Voltage(s) to saturate.
    v_low, v_high : float
        Output rails.
    k : float
        Saturation sharpness.

    Returns
    -------
    np.ndarray
        Saturated voltage(s) in ``[v_low, v_high]``.
    """
    mid = (v_low + v_high) / 2.0
    span = (v_high - v_low) / 2.0
    return mid + span * np.tanh(k * (np.asarray(v, dtype=float) - mid) / span)


def reconstruct_waveform(
    levels: np.ndarray,
    *,
    tau_rise: float = 0.15,
    tau_fall: float = 0.15,
    dcd_offset: float = 0.01,
    samples_per_ui: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a continuous waveform from per-UI effective voltage levels.

    Each UI holds its ISI-affected, saturated level. Between consecutive levels
    an RC exponential transition is applied. DCD shifts the *start* of each
    transition: rising edges move right (+dcd_offset), falling edges move left
    (-dcd_offset). The most recent edge governs each sample; with tau ~ 0.15 UI
    a transition is >99.8% settled within one UI, so the snap-to-previous-level
    start introduces negligible error.

    Parameters
    ----------
    levels : np.ndarray
        Per-bit effective voltage levels, shape (n_bits,).
    tau_rise, tau_fall : float
        RC time constants (in UI) for rising / falling edges.
    dcd_offset : float
        Duty-cycle distortion edge shift (in UI).
    samples_per_ui : int
        Waveform resolution.

    Returns
    -------
    (t_axis, wave) : tuple of np.ndarray
        ``t_axis`` in UI over ``[0, n_bits)``; ``wave`` the sampled voltage.
    """
    levels = np.asarray(levels, dtype=float)
    n_bits = len(levels)
    n = n_bits * samples_per_ui
    t_axis = np.arange(n, dtype=float) / samples_per_ui

    wave = np.full(n, levels[0], dtype=float)
    for i in range(1, n_bits):
        prev, cur = levels[i - 1], levels[i]
        if abs(cur - prev) < 1e-9:
            continue  # flat: previous plateau persists
        rising = cur > prev
        tau = tau_rise if rising else tau_fall
        edge = i + (dcd_offset if rising else -dcd_offset)
        mask = t_axis >= edge
        wave[mask] = prev + (cur - prev) * (1.0 - np.exp(-(t_axis[mask] - edge) / tau))
    return t_axis, wave
