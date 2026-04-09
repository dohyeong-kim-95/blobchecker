"""
raw/synth/synthetic_map.py
Generate a 50×200 binary map with a smooth blob, interior holes, and background outliers.

API
---
    from raw.synth.synthetic_map import generate_map
    m = generate_map(blob_size=0.35, n_holes=2, hole_size=5, n_outliers=10, seed=42)

CLI
---
    python raw/synth/synthetic_map.py --blob_size 0.4 --n_holes 3 --hole_size 6 \
                                      --n_outliers 15 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _make_blob(rng: np.random.Generator, H: int, W: int, blob_size: float) -> np.ndarray:
    """Fourier-perturbed ellipse centred near the grid centre."""
    target = int(blob_size * H * W)
    aspect = W / H
    r_x = min(np.sqrt(target / np.pi * aspect), W * 0.45)
    r_y = min(np.sqrt(target / np.pi / aspect), H * 0.45)

    cy = rng.uniform(H * 0.3, H * 0.7)
    cx = rng.uniform(W * 0.3, W * 0.7)

    YY, XX = np.mgrid[0:H, 0:W]
    dy = (YY - cy) / r_y
    dx = (XX - cx) / r_x
    theta = np.arctan2(dy, dx)

    n_modes = 6
    amps   = rng.uniform(0.05, 0.18, n_modes)
    phases = rng.uniform(0.0, 2 * np.pi, n_modes)
    perturb = np.ones_like(theta)
    for a, phi, k in zip(amps, phases, range(2, 2 + n_modes)):
        perturb += a * np.cos(k * theta + phi)

    return (np.sqrt(dy ** 2 + dx ** 2) < perturb).astype(np.uint8)


def _carve_holes(
    rng: np.random.Generator,
    binary: np.ndarray,
    n_holes: int,
    hole_size: float,
) -> np.ndarray:
    """Carve circular 0-regions strictly inside the blob."""
    H, W = binary.shape
    ys_g, xs_g = np.ogrid[:H, :W]
    binary = binary.copy()

    se_r = max(1, int(hole_size))
    struct = np.ones((2 * se_r + 1, 2 * se_r + 1), dtype=bool)
    interior = np.argwhere(binary_erosion(binary.astype(bool), structure=struct))

    placed = 0
    for _ in range(n_holes * 30):
        if placed >= n_holes or len(interior) == 0:
            break
        cy, cx = interior[rng.integers(len(interior))]
        binary[np.sqrt((ys_g - cy) ** 2 + (xs_g - cx) ** 2) <= hole_size] = 0
        placed += 1
        interior = np.argwhere(binary_erosion(binary.astype(bool), structure=struct))

    return binary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_map(
    grid_shape: tuple[int, int] = (50, 200),
    blob_size: float = 0.35,
    n_holes: int = 2,
    hole_size: float = 5.0,
    n_outliers: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Return a (H, W) uint8 binary map.

    Parameters
    ----------
    grid_shape : output grid (H, W)
    blob_size  : target fraction of cells covered by the main blob
    n_holes    : circular 0-regions carved inside the blob
    hole_size  : hole radius in pixels
    n_outliers : isolated 1-pixels placed in the background
    seed       : random seed
    """
    rng = np.random.default_rng(seed)
    H, W = grid_shape

    binary = _make_blob(rng, H, W, blob_size)

    if n_holes > 0 and hole_size > 0:
        binary = _carve_holes(rng, binary, n_holes, hole_size)

    if n_outliers > 0:
        bg = np.argwhere(binary == 0)
        if len(bg):
            for i in rng.choice(len(bg), size=min(n_outliers, len(bg)), replace=False):
                binary[bg[i, 0], bg[i, 1]] = 1

    return binary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic binary map.")
    p.add_argument("--height",     type=int,   default=50)
    p.add_argument("--width",      type=int,   default=200)
    p.add_argument("--blob_size",  type=float, default=0.35)
    p.add_argument("--n_holes",    type=int,   default=2)
    p.add_argument("--hole_size",  type=float, default=5.0)
    p.add_argument("--n_outliers", type=int,   default=10)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out",        type=str,   default="raw/synth/map")
    args = p.parse_args()

    m = generate_map((args.height, args.width),
                     args.blob_size, args.n_holes, args.hole_size,
                     args.n_outliers, args.seed)
    print(f"shape={m.shape}  coverage={m.mean():.2%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out) + ".npy", m)
    print(f"Saved → {out}.npy")


if __name__ == "__main__":
    _cli()
