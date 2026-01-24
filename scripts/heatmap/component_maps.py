# File: scripts/heatmap/component_maps.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class MapBuildConfig:
    base_grid: int
    sigma: float = 1.5
    eps: float = 1e-12
    use_log: bool = False
    clip_min: float = 0.0


def _xy_to_rc(xy: np.ndarray, base_grid: int, matlab_1_indexed: bool) -> np.ndarray:
    """
    Convert XY (x,y) into RC (row,col) index coordinates for image arrays.
    Assumes x is column axis, y is row axis.
    """
    p = np.asarray(xy, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"xy must be Nx2, got {p.shape}")

    x = p[:, 0]
    y = p[:, 1]

    if matlab_1_indexed:
        x = x - 1.0
        y = y - 1.0

    x = np.clip(x, 0.0, base_grid - 1.0)
    y = np.clip(y, 0.0, base_grid - 1.0)

    rc = np.column_stack([y, x])
    return rc


def _rasterize_polyline(rc: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Very fast polyline rasterization. Marks pixels along segment connections.
    """
    from skimage.draw import line

    out = np.zeros((H, W), dtype=np.float32)
    pts = np.asarray(rc, dtype=np.float64)
    if pts.shape[0] < 2:
        return out

    r = pts[:, 0]
    c = pts[:, 1]

    for i in range(len(pts) - 1):
        r0 = int(round(r[i]))
        c0 = int(round(c[i]))
        r1 = int(round(r[i + 1]))
        c1 = int(round(c[i + 1]))
        rr, cc = line(r0, c0, r1, c1)
        rr = np.clip(rr, 0, H - 1)
        cc = np.clip(cc, 0, W - 1)
        out[rr, cc] = 1.0

    return out


def build_component_heatmaps(
    rows_xy: List[np.ndarray],
    resp: np.ndarray,
    occluder_xy: np.ndarray,
    *,
    cfg: MapBuildConfig,
    matlab_1_indexed: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-component spatial heatmaps.

    rows_xy: list length N, each is (Ni,2) XY polygon/polyline for the completion shape.
             We only use the completion polyline region, so you should pass aligned completion curves
             or the full stitched contour. Either works. For occluded-region scoring, you will mask later.
    resp: (N,K) responsibilities
    occluder_xy: (M,2) XY polygon defining occluder in same coordinate space

    Returns:
      maps: (K,H,W) float32 heatmaps, normalized to sum 1 per component over occluder region
      occ_mask: (H,W) bool occluder interior mask
    """
    from scipy.ndimage import gaussian_filter
    from matplotlib.path import Path as MplPath

    N = len(rows_xy)
    if resp.ndim != 2 or resp.shape[0] != N:
        raise ValueError(f"resp must be (N,K) with N=len(rows_xy). Got {resp.shape}, N={N}")

    H = W = int(cfg.base_grid)
    K = int(resp.shape[1])

    # occluder mask in RC image space
    occ_path = MplPath(np.asarray(occluder_xy, dtype=np.float64))
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    if matlab_1_indexed:
        # pts are 0-index. occluder is 1-index. convert pts to 1-index for contains_points
        pts_q = pts + 1.0
    else:
        pts_q = pts

    occ_mask = occ_path.contains_points(pts_q, radius=1e-9).reshape(H, W)

    maps = np.zeros((K, H, W), dtype=np.float32)

    for i in range(N):
        xy = rows_xy[i]
        if xy is None or len(xy) < 2:
            continue

        rc = _xy_to_rc(xy, cfg.base_grid, matlab_1_indexed=matlab_1_indexed)
        line_img = _rasterize_polyline(rc, H, W)

        # smooth to make a density field
        if cfg.sigma and cfg.sigma > 0:
            line_img = gaussian_filter(line_img, sigma=float(cfg.sigma)).astype(np.float32)

        # accumulate into each component with responsibility weight
        w = resp[i].astype(np.float32)
        maps += w[:, None, None] * line_img[None, :, :]

    # mask to occluder region only
    maps *= occ_mask[None, :, :].astype(np.float32)

    # optional log
    if cfg.use_log:
        maps = np.log(maps + cfg.eps)

    # clip and renormalize each component to sum to 1 over occluder
    maps = np.maximum(maps, float(cfg.clip_min)).astype(np.float32)
    sums = maps.reshape(K, -1).sum(axis=1) + float(cfg.eps)
    maps = maps / sums[:, None, None]

    return maps, occ_mask


def save_component_maps_npz(
    path: str | Path,
    maps: np.ndarray,
    occ_mask: np.ndarray,
    *,
    base_grid: int,
    matlab_1_indexed: bool,
    meta: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        maps=maps.astype(np.float32),
        occ_mask=occ_mask.astype(np.uint8),
        base_grid=int(base_grid),
        matlab_1_indexed=bool(matlab_1_indexed),
        meta=np.array([meta], dtype=object),
    )
