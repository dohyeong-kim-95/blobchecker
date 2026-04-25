"""
Shared reconstruction utilities.

Scanline fill + largest-connected-component filter, used by both
PreplannedGreedy and GeometryFirstAdaptive.
"""

import numpy as np
from scipy.ndimage import label as ndlabel

_STRUCT8 = np.ones((3, 3), dtype=np.int32)
HOLE_HALF = 4   # neighbourhood cleared around interior observed zeros


def scanline_reconstruct(
    H: int,
    W: int,
    obs_mask: np.ndarray,      # (H, W) bool
    obs_labels: np.ndarray,    # (8, H, W) uint8
) -> np.ndarray:
    """
    Reconstruct all 8 layers from sparse observations using row-wise scanline fill.

    For each layer and row:
      1. Find leftmost / rightmost observed positive.
      2. Fill the span [left, right] with 1.
      3. Suppress a HOLE_HALF-wide neighbourhood around any interior observed 0.
      4. Rows with no observations inherit from their nearest observed neighbour.
      5. Apply largest-8-connected-component filter to remove isolated outlier
         pixels (which would inflate the bounding-box and hurt accuracy).

    Returns
    -------
    predicted : ndarray (8, H, W) uint8
    """
    predicted = np.zeros((8, H, W), dtype=np.uint8)

    for k in range(8):
        pred_k = _reconstruct_layer(H, W, obs_mask, obs_labels[k])
        predicted[k] = pred_k

    return predicted


def _reconstruct_layer(
    H: int,
    W: int,
    obs_mask: np.ndarray,
    obs_labels_k: np.ndarray,
) -> np.ndarray:
    pred = np.zeros((H, W), dtype=np.uint8)
    observed_rows = []

    for r in range(H):
        row_mask = obs_mask[r]
        if not row_mask.any():
            continue
        observed_rows.append(r)

        queried_cols = np.where(row_mask)[0]
        labels = obs_labels_k[r, queried_cols]
        pos_cols = queried_cols[labels == 1]

        if pos_cols.size == 0:
            continue

        c_left  = int(pos_cols.min())
        c_right = int(pos_cols.max())
        pred[r, c_left : c_right + 1] = 1

        # Suppress neighbourhood around interior observed zeros (holes).
        neg_inside = queried_cols[
            (labels == 0)
            & (queried_cols >= c_left)
            & (queried_cols <= c_right)
        ]
        for c in neg_inside:
            lo = max(c_left, c - HOLE_HALF)
            hi = min(c_right + 1, c + HOLE_HALF + 1)
            pred[r, lo:hi] = 0

    # Rows with no direct observations inherit from nearest observed row.
    if observed_rows:
        obs_arr = np.array(observed_rows)
        for r in range(H):
            if obs_mask[r].any():
                continue
            nearest = obs_arr[np.argmin(np.abs(obs_arr - r))]
            pred[r] = pred[nearest]

    # Largest-CC filter: removes isolated outlier pixels.
    if pred.any():
        labeled, n = ndlabel(pred, structure=_STRUCT8)
        if n > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            pred = (labeled == sizes.argmax()).astype(np.uint8)

    return pred
