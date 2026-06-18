"""Channel ISI model: discrete causal FIR tap-delay line.

See docs/shmoo_model.md section 2. The main cursor must land on the current
bit, so a *causal* convolution is used rather than ``np.convolve(mode='same')``,
which for an even-length kernel misaligns the main cursor by one bit.
"""

from __future__ import annotations

import numpy as np

#: Main cursor + three post-cursors.
DEFAULT_TAPS: tuple[float, ...] = (1.0, -0.15, -0.08, -0.04)


def apply_isi(bits: np.ndarray, taps: np.ndarray | tuple[float, ...] = DEFAULT_TAPS) -> np.ndarray:
    """Apply a causal post-cursor FIR filter to a bit sequence.

    Implements ``received[i] = sum_k taps[k] * bits[i - k]`` so that the main
    cursor ``taps[0]`` multiplies ``bits[i]`` (the current bit) and the
    post-cursors multiply strictly past bits. With a 16-zero preamble this makes
    ``received[17]`` depend only on the target bit and its predecessors.

    Parameters
    ----------
    bits : np.ndarray
        Transmitted bit sequence (0/1), shape (n_bits,).
    taps : array_like
        FIR tap coefficients; ``taps[0]`` is the main cursor.

    Returns
    -------
    np.ndarray
        ISI-affected voltage sequence, same shape as ``bits``.
    """
    bits = np.asarray(bits, dtype=float)
    taps = np.asarray(taps, dtype=float)
    out = np.zeros_like(bits)
    for k, tap in enumerate(taps):
        if k == 0:
            out += tap * bits
        else:
            out[k:] += tap * bits[:-k]
    return out
