# File: scripts/heatmap/scoring.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ScoreWeights:
    alpha: float = 1.0  # weight term
    beta: float = 1.0   # path likelihood term
    gamma: float = 1.0  # coherence term


def _sample_density_along_path(density: np.ndarray, path_xy: np.ndarray, *, base_grid: int, matlab_1_indexed: bool) -> np.ndarray:
    dens = np.asarray(density, dtype=np.float64)
    P = np.asarray(path_xy, dtype=np.float64)
    x = P[:, 0].copy()
    y = P[:, 1].copy()
    if matlab_1_indexed:
        x -= 1.0
        y -= 1.0
    r = np.clip(np.round(y).astype(int), 0, base_grid - 1)
    c = np.clip(np.round(x).astype(int), 0, base_grid - 1)
    return dens[r, c]


def path_loglik_score(density: np.ndarray, path_xy: np.ndarray, *, base_grid: int, matlab_1_indexed: bool, eps: float = 1e-12) -> float:
    vals = _sample_density_along_path(density, path_xy, base_grid=base_grid, matlab_1_indexed=matlab_1_indexed)
    return float(np.mean(np.log(vals + float(eps))))


def _point_to_polyline_dist(points_xy: np.ndarray, curve_xy: np.ndarray) -> np.ndarray:
    """
    Approximate point-to-polyline distance.
    Uses segment projections. O(N*M) but fine for moderate sizes.
    """
    P = np.asarray(points_xy, dtype=np.float64)
    C = np.asarray(curve_xy, dtype=np.float64)
    if C.shape[0] < 2 or P.shape[0] < 1:
        return np.full((P.shape[0],), np.inf, dtype=np.float64)

    # segments
    A = C[:-1]
    B = C[1:]
    AB = B - A
    AB2 = np.sum(AB * AB, axis=1) + 1e-12

    dmin = np.full((P.shape[0],), np.inf, dtype=np.float64)

    for i in range(P.shape[0]):
        p = P[i]
        AP = p[None, :] - A
        t = np.sum(AP * AB, axis=1) / AB2
        t = np.clip(t, 0.0, 1.0)
        proj = A + t[:, None] * AB
        d = np.sqrt(np.sum((proj - p[None, :]) ** 2, axis=1))
        dmin[i] = np.min(d)

    return dmin


def coherence_score(
    member_curves_xy: List[np.ndarray],
    consensus_path_xy: np.ndarray,
    *,
    n_subsample: int = 200,
    stat: str = "median",
) -> float:
    """
    Negative dispersion. Larger is better.
    We sample points from each member curve and measure distance to consensus polyline.
    """
    rng = np.random.default_rng(0)
    dists_all = []

    for xy in member_curves_xy:
        if xy is None or len(xy) < 2:
            continue
        pts = np.asarray(xy, dtype=np.float64)
        if pts.shape[0] > n_subsample:
            idx = rng.choice(pts.shape[0], size=n_subsample, replace=False)
            pts = pts[idx]
        d = _point_to_polyline_dist(pts, consensus_path_xy)
        dists_all.append(d)

    if not dists_all:
        return -np.inf

    dcat = np.concatenate(dists_all, axis=0)
    if stat == "mean":
        disp = float(np.mean(dcat))
    else:
        disp = float(np.median(dcat))
    return -disp


def composite_scores(
    weights: np.ndarray,
    density_maps: np.ndarray,
    paths_xy: List[np.ndarray],
    members_xy: List[List[np.ndarray]],
    *,
    base_grid: int,
    matlab_1_indexed: bool,
    W: ScoreWeights,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      S_total, S_weight, S_path, S_coh all shape (K,)
    """
    K = int(density_maps.shape[0])
    w = np.asarray(weights, dtype=np.float64).reshape(K,)

    S_w = np.log(w + float(eps))

    S_p = np.zeros((K,), dtype=np.float64)
    S_c = np.zeros((K,), dtype=np.float64)

    for k in range(K):
        if paths_xy[k] is None or len(paths_xy[k]) < 2:
            S_p[k] = -np.inf
            S_c[k] = -np.inf
            continue

        S_p[k] = path_loglik_score(
            density_maps[k],
            paths_xy[k],
            base_grid=base_grid,
            matlab_1_indexed=matlab_1_indexed,
            eps=eps,
        )
        S_c[k] = coherence_score(members_xy[k], paths_xy[k], n_subsample=200, stat="median")

    S_total = float(W.alpha) * S_w + float(W.beta) * S_p + float(W.gamma) * S_c
    return S_total, S_w, S_p, S_c
