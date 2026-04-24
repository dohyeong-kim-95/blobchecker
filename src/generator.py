"""
Synthetic blob dataset generator.

Generates 8 independent binary blob layers per seed, satisfying the
structural constraints defined in problem_definition.md.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, label

N_LAYERS = 8


def _generate_blob(H, W, rng, coverage):
    """
    Generate one 8-connected blob with the given coverage fraction.

    Uses a smooth random field with a center-attraction bias so the result
    is one compact connected region with a locally coherent boundary.
    """
    noise = rng.standard_normal((H, W)).astype(np.float32)
    sigma = max(3.0, min(H, W) / 7.0)
    smooth = gaussian_filter(noise, sigma=sigma)

    # Pull blob toward a random interior center to avoid edge-hugging.
    cy = rng.integers(H // 4, max(H // 4 + 1, 3 * H // 4))
    cx = rng.integers(W // 4, max(W // 4 + 1, 3 * W // 4))
    rows, cols = np.ogrid[:H, :W]
    center_pull = -(
        (rows - cy) ** 2 / (H / 2.5) ** 2
        + (cols - cx) ** 2 / (W / 2.5) ** 2
    )
    field = smooth + 0.8 * center_pull

    threshold = np.percentile(field, (1.0 - coverage) * 100.0)
    raw = (field >= threshold).astype(np.uint8)

    # Keep only the largest 8-connected component.
    struct = np.ones((3, 3), dtype=np.int32)
    labeled, n = label(raw, structure=struct)
    if n == 0:
        return np.zeros((H, W), dtype=np.uint8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(np.uint8)


def _carve_holes(blob, rng, H, W):
    """
    Carve 0–3 rectangular holes into the interior of a blob.

    A hole is only committed if the blob remains 8-connected after carving.
    """
    n_holes = int(rng.integers(0, 4))
    if n_holes == 0:
        return blob

    struct = np.ones((3, 3), dtype=np.int32)
    result = blob.copy()
    blob_coords = np.argwhere(blob == 1)

    for _ in range(n_holes):
        if len(blob_coords) < 20:
            break
        idx = rng.integers(len(blob_coords))
        cy, cx = blob_coords[idx]

        hole_h = max(2, int(round(rng.normal(7, 2))))
        hole_w = max(2, int(round(rng.normal(7, 2))))
        r0 = max(0, cy - hole_h // 2)
        r1 = min(H, r0 + hole_h)
        c0 = max(0, cx - hole_w // 2)
        c1 = min(W, c0 + hole_w)

        candidate = result.copy()
        candidate[r0:r1, c0:c1] = 0

        _, n = label(candidate, structure=struct)
        if n == 1:  # still connected
            result = candidate

    return result


def _generate_outliers(blob, rng, H, W):
    """Scatter sparse positive outliers outside the blob region."""
    mask = np.zeros((H, W), dtype=np.uint8)
    if rng.random() < 0.5:
        return mask
    density = rng.uniform(0.001, 0.008)
    coords = np.argwhere(blob == 0)
    if len(coords) == 0:
        return mask
    n = min(int(density * H * W), len(coords))
    chosen = rng.choice(len(coords), size=n, replace=False)
    for idx in chosen:
        r, c = coords[idx]
        mask[r, c] = 1
    return mask


def generate_dataset(H, W, seed):
    """
    Generate 8 independent binary blob layers for the given seed.

    Returns
    -------
    truth_blob_mask    : ndarray shape (8, H, W) uint8  — scoring target
    truth_outlier_mask : ndarray shape (8, H, W) uint8  — supplemental
    truth_full_mask    : ndarray shape (8, H, W) uint8  — oracle source
    """
    rng = np.random.default_rng(seed)

    truth_blob = np.zeros((N_LAYERS, H, W), dtype=np.uint8)
    truth_out = np.zeros((N_LAYERS, H, W), dtype=np.uint8)

    for k in range(N_LAYERS):
        coverage = float(np.clip(rng.normal(0.4, 0.05), 0.3, 0.7))
        blob = _generate_blob(H, W, rng, coverage)
        blob = _carve_holes(blob, rng, H, W)
        truth_blob[k] = blob
        truth_out[k] = _generate_outliers(blob, rng, H, W)

    truth_full = (truth_blob | truth_out).astype(np.uint8)
    return truth_blob, truth_out, truth_full
