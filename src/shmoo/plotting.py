"""Matplotlib rendering: shmoo PASS/FAIL maps and eye diagram.

Outputs PNGs so they render inline on GitHub.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .shmoo_eval import ShmooConfig


def plot_shmoo_grid(blob: np.ndarray, cfg: ShmooConfig, path: Path,
                    *, title: str = "Synthetic DRAM Shmoo Plot (ISI + DCD + RJ)",
                    ncols: int = 4) -> Path:
    """Render every layer's PASS/FAIL map in a grid. PASS=green, FAIL=dark."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    n = blob.shape[0]
    nrows = int(np.ceil(n / ncols))
    cmap = ListedColormap(["#1f2937", "#22c55e"])  # FAIL, PASS
    extent = [cfg.timing_lo, cfg.timing_hi, cfg.vref_lo * 1000, cfg.vref_hi * 1000]

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 1.9))
    axes = np.atleast_1d(axes).ravel()
    for k in range(len(axes)):
        ax = axes[k]
        if k < n:
            ax.imshow(blob[k], origin="lower", aspect="auto", cmap=cmap,
                      vmin=0, vmax=1, extent=extent)
            ax.set_title(f"layer {k}  cov {blob[k].mean()*100:.0f}%", fontsize=9)
            ax.tick_params(labelsize=7)
            if k % ncols == 0:
                ax.set_ylabel("Vref (mV)", fontsize=8)
            if k >= n - ncols:
                ax.set_xlabel("timing (UI)", fontsize=8)
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_eye_diagram(waves, cfg: ShmooConfig, path: Path,
                     *, layer: int = 0, half_width: float = 1.0) -> Path:
    """Overlay the 8 pattern waveforms over a ±half_width UI window.

    The sampling window [timing_lo, timing_hi] is shown as vertical dashed
    lines. Using half_width=1.0 (default) places both the leading and trailing
    bit transitions fully inside the plot, avoiding the cliff artifact that
    appears when the window is cut exactly at the bit boundaries.

    ``waves`` is a list of ``(pattern, t_axis, wave)`` from
    ``layer_envelopes(..., collect_waves=True)``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.get_cmap("tab10")
    center = cfg.target_center
    for i, (pattern, t_axis, wave) in enumerate(waves):
        rel = t_axis - center
        sel = (rel >= -half_width) & (rel <= half_width)
        label = "".join(str(b) for b in pattern)
        ax.plot(rel[sel], wave[sel] * 1000, color=cmap(i % 10), label=label, lw=1.6)

    # Sampling window boundaries
    ax.axvline(cfg.timing_lo, ls=":", color="#d1d5db", lw=1.2, label="sampling window")
    ax.axvline(cfg.timing_hi, ls=":", color="#d1d5db", lw=1.2)
    ax.axhline(500, ls="--", color="#6b7280", lw=1, label="Vref=500mV")
    ax.axvline(0.0, ls="--", color="#9ca3af", lw=1)
    ax.set_xlabel("timing offset (UI)")
    ax.set_ylabel("voltage (mV)")
    ax.set_title(f"Eye Diagram — layer {layer} (8 patterns, soft-saturated, ±{half_width} UI)")
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-50, 1050)
    ax.legend(ncol=3, fontsize=8, loc="center right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
