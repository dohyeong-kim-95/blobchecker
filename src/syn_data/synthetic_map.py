"""
synthetic_map.py
Generates a 50×200 binary map with a smooth blob, optional interior holes,
and optional outlier 1s in the background.

Usage (CLI):
    python synthetic_map.py
    python synthetic_map.py --blob_size 0.4 --n_holes 3 --hole_size 6 \
                            --n_outliers 15 --seed 7 --out src/syn_data/map

Usage (API):
    from src.syn_data.synthetic_map import generate_map, save_map
    m = generate_map(blob_size=0.35, n_holes=2, hole_size=5, n_outliers=10, random_seed=42)
    save_map(m, "src/syn_data/map")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_blob(rng: np.random.Generator, H: int, W: int, blob_size: float) -> np.ndarray:
    """Return a uint8 mask with one smooth connected blob.

    Strategy: Fourier-perturbed ellipse centred near the grid centre.
    The blob is described in polar coordinates from its centre.  A smooth
    random radial perturbation (low-frequency Fourier modes) deforms the
    boundary naturally.
    """
    # Target number of 1-pixels
    target_pixels = int(blob_size * H * W)

    # Base radii of the ellipse, scaled from target area (π r_y r_x ≈ area)
    # We use an aspect ratio close to the grid's own aspect ratio
    aspect = W / H
    r_base_x = np.sqrt(target_pixels / np.pi * aspect)
    r_base_y = np.sqrt(target_pixels / np.pi / aspect)
    # Clamp so the blob fits inside the grid with a margin
    r_base_x = min(r_base_x, W * 0.45)
    r_base_y = min(r_base_y, H * 0.45)

    # Random centre (within the inner 40–60 % of each axis)
    cy = rng.uniform(H * 0.3, H * 0.7)
    cx = rng.uniform(W * 0.3, W * 0.7)

    # Build pixel-level distance in normalised coordinates
    ys = np.arange(H)
    xs = np.arange(W)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")

    dy = (YY - cy) / r_base_y          # normalised vertical distance
    dx = (XX - cx) / r_base_x          # normalised horizontal distance

    # Polar angle for each pixel
    theta = np.arctan2(dy, dx)          # shape (H, W)

    # Smooth radial perturbation as sum of low-freq Fourier modes
    n_modes = 6
    amplitudes = rng.uniform(0.05, 0.18, size=n_modes)
    phases     = rng.uniform(0, 2 * np.pi, size=n_modes)
    freqs      = np.arange(2, 2 + n_modes)

    perturb = np.ones_like(theta)
    for a, phi, k in zip(amplitudes, phases, freqs):
        perturb += a * np.cos(k * theta + phi)

    # Normalised Euclidean distance from centre
    r_norm = np.sqrt(dy ** 2 + dx ** 2)

    # Pixel is inside blob if r_norm < perturb (boundary at r_norm == perturb)
    blob = (r_norm < perturb).astype(np.uint8)
    return blob


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

    placed, attempts = 0, 0
    max_attempts = n_holes * 30

    # Pre-compute erosion-like mask: only candidate centres far enough from edge
    # A hole of radius r must have its centre at least r pixels from the boundary.
    from scipy.ndimage import binary_erosion, generate_binary_structure

    se_r = max(1, int(hole_size))
    struct = np.ones((2 * se_r + 1, 2 * se_r + 1), dtype=bool)
    interior_mask = binary_erosion(binary.astype(bool), structure=struct)

    interior_coords = np.argwhere(interior_mask)
    if len(interior_coords) == 0:
        return binary                  # blob too small to hold any hole

    while placed < n_holes and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(len(interior_coords))
        cy, cx = interior_coords[idx]

        dist = np.sqrt((ys_g - cy) ** 2 + (xs_g - cx) ** 2)
        hole_mask = dist <= hole_size

        binary[hole_mask] = 0
        placed += 1

        # Refresh interior mask so next holes don't overlap prior ones
        interior_mask = binary_erosion(binary.astype(bool), structure=struct)
        interior_coords = np.argwhere(interior_mask)
        if len(interior_coords) == 0:
            break

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
    random_seed: int = 42,
) -> np.ndarray:
    """Return a (H, W) uint8 binary map.

    Parameters
    ----------
    grid_shape  : (H, W) of the output map
    blob_size   : approximate fraction [0, 1] of cells covered by the main blob
    n_holes     : number of circular interior holes (0-regions inside blob)
    hole_size   : radius (pixels) of each hole
    n_outliers  : number of isolated 1-pixels placed in the background
    random_seed : RNG seed for reproducibility
    """
    rng = np.random.default_rng(random_seed)
    H, W = grid_shape

    # ------------------------------------------------------------------
    # 1. Smooth, connected blob (Fourier-perturbed ellipse)
    # ------------------------------------------------------------------
    binary = _make_blob(rng, H, W, blob_size)

    # ------------------------------------------------------------------
    # 2. Interior holes
    # ------------------------------------------------------------------
    if n_holes > 0 and hole_size > 0:
        binary = _carve_holes(rng, binary, n_holes, hole_size)

    # ------------------------------------------------------------------
    # 3. Outlier 1s in background
    # ------------------------------------------------------------------
    if n_outliers > 0:
        bg_coords = np.argwhere(binary == 0)
        if len(bg_coords) > 0:
            chosen_idx = rng.choice(
                len(bg_coords),
                size=min(n_outliers, len(bg_coords)),
                replace=False,
            )
            for i in chosen_idx:
                y, x = bg_coords[i]
                binary[y, x] = 1

    return binary


def save_map(binary: np.ndarray, out_prefix: str | Path) -> None:
    """Save binary map as <prefix>.npy and <prefix>.png."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(out_prefix) + ".npy", binary)
    print(f"Saved .npy → {out_prefix}.npy")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 2.5), dpi=100)
        ax.imshow(binary, cmap="gray", vmin=0, vmax=1, aspect="auto",
                  interpolation="nearest")
        ax.set_title(
            f"Binary map  shape={binary.shape}  "
            f"coverage={binary.mean():.2%}"
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(str(out_prefix) + ".png", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved .png → {out_prefix}.png")
    except ImportError:
        print("matplotlib not installed — PNG skipped.")


def load_map(npy_path: str | Path) -> np.ndarray:
    """Load a previously saved binary map."""
    return np.load(str(npy_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a synthetic 50×200 binary map."
    )
    p.add_argument("--height",     type=int,   default=50,              help="Grid height (default 50)")
    p.add_argument("--width",      type=int,   default=200,             help="Grid width  (default 200)")
    p.add_argument("--blob_size",  type=float, default=0.35,            help="Blob coverage fraction [0,1]")
    p.add_argument("--n_holes",    type=int,   default=2,               help="Interior holes")
    p.add_argument("--hole_size",  type=float, default=5.0,             help="Hole radius (pixels)")
    p.add_argument("--n_outliers", type=int,   default=10,              help="Outlier 1s in background")
    p.add_argument("--seed",       type=int,   default=42,              help="Random seed")
    p.add_argument("--out",        type=str,   default="src/syn_data/map",
                   help="Output prefix without extension")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    binary = generate_map(
        grid_shape=(args.height, args.width),
        blob_size=args.blob_size,
        n_holes=args.n_holes,
        hole_size=args.hole_size,
        n_outliers=args.n_outliers,
        random_seed=args.seed,
    )
    print(f"Map shape : {binary.shape}")
    print(f"1-coverage: {binary.mean():.2%}  ({int(binary.sum())} / {binary.size} cells)")
    save_map(binary, args.out)


if __name__ == "__main__":
    main()
