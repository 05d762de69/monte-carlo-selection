from __future__ import annotations

import numpy as np


def rasterize_polygon_skimage(xy: np.ndarray, H: int, W: int, *, matlab_1_indexed: bool = True) -> np.ndarray:
    """
    Rasterize a filled polygon into a boolean mask.
    xy is Nx2 with columns [x, y] in pixel coords.

    matlab_1_indexed=True means xy lives in [1..baseGrid] like MATLAB.
    """
    try:
        from skimage.draw import polygon as sk_polygon
    except Exception as e:
        raise RuntimeError("Missing dependency: scikit-image (skimage). Install it.") from e

    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected Nx2 polygon, got {pts.shape}")

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()

    if matlab_1_indexed:
        x -= 1.0
        y -= 1.0

    rr = np.clip(y, 0, H - 1)
    cc = np.clip(x, 0, W - 1)

    r_idx, c_idx = sk_polygon(rr, cc, shape=(H, W))
    m = np.zeros((H, W), dtype=bool)
    m[r_idx, c_idx] = True
    return m
