# File: scripts/heatmap/path_extract.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np


@dataclass
class PathExtractConfig:
    eps: float = 1e-12
    cost_floor: float = 1e-8
    allow_diagonal: bool = True

    # --- ridge-band mode ---
    ridge_q_start: float = 0.90          # start threshold at this quantile inside occluder
    ridge_q_min: float = 0.50            # do not go below this quantile
    ridge_q_step: float = 0.02           # step down until endpoints connect
    ridge_cost_mode: str = "inv_density" # "unit" | "inv_density" | "neg_log"
    ridge_smooth_win: int = 5            # simple 1D smoothing on xy path (odd, >=3)


def _xy_to_rc(xy: np.ndarray, *, matlab_1_indexed: bool) -> Tuple[int, int]:
    x = float(xy[0])
    y = float(xy[1])
    if matlab_1_indexed:
        c = int(round(x)) - 1
        r = int(round(y)) - 1
    else:
        c = int(round(x))
        r = int(round(y))
    return r, c


def _rc_to_xy(rc: Tuple[int, int], *, matlab_1_indexed: bool) -> np.ndarray:
    r, c = int(rc[0]), int(rc[1])
    if matlab_1_indexed:
        return np.array([c + 1, r + 1], dtype=np.float64)
    return np.array([c, r], dtype=np.float64)


def _clip_rc(rc: Tuple[int, int], H: int, W: int) -> Tuple[int, int]:
    r, c = rc
    r = 0 if r < 0 else (H - 1 if r >= H else r)
    c = 0 if c < 0 else (W - 1 if c >= W else c)
    return r, c


def _neighbors(rc: Tuple[int, int], *, allow_diagonal: bool) -> List[Tuple[int, int]]:
    r, c = rc
    n4 = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    if not allow_diagonal:
        return n4
    n8 = n4 + [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
    return n8


def _dijkstra_on_mask(cost: np.ndarray, mask: np.ndarray, start_rc: Tuple[int, int], end_rc: Tuple[int, int],
                      *, allow_diagonal: bool) -> Optional[np.ndarray]:
    """
    Dijkstra on a boolean mask.
    cost is a (H,W) array. It is read only where mask=True.
    Returns path as (L,2) int array of (r,c), or None if unreachable.
    """
    import heapq

    H, W = cost.shape
    sr, sc = start_rc
    tr, tc = end_rc

    if not (0 <= sr < H and 0 <= sc < W and 0 <= tr < H and 0 <= tc < W):
        return None
    if (not mask[sr, sc]) or (not mask[tr, tc]):
        return None

    dist = np.full((H, W), np.inf, dtype=np.float64)
    prev_r = np.full((H, W), -1, dtype=np.int32)
    prev_c = np.full((H, W), -1, dtype=np.int32)

    dist[sr, sc] = 0.0
    pq = [(0.0, sr, sc)]

    while pq:
        d, r, c = heapq.heappop(pq)
        if d != dist[r, c]:
            continue
        if (r, c) == (tr, tc):
            break

        for rr, cc in _neighbors((r, c), allow_diagonal=allow_diagonal):
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                continue
            if not mask[rr, cc]:
                continue
            nd = d + float(cost[rr, cc])
            if nd < dist[rr, cc]:
                dist[rr, cc] = nd
                prev_r[rr, cc] = r
                prev_c[rr, cc] = c
                heapq.heappush(pq, (nd, rr, cc))

    if not np.isfinite(dist[tr, tc]):
        return None

    # reconstruct
    path = []
    r, c = tr, tc
    path.append((r, c))
    while not (r == sr and c == sc):
        pr = int(prev_r[r, c])
        pc = int(prev_c[r, c])
        if pr < 0 or pc < 0:
            return None
        r, c = pr, pc
        path.append((r, c))
    path.reverse()

    return np.asarray(path, dtype=np.int32)


def _smooth_xy(xy: np.ndarray, win: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    win = int(win)
    if win < 3:
        return xy
    if win % 2 == 0:
        win += 1
    pad = win // 2
    out = xy.copy()

    for dim in [0, 1]:
        v = xy[:, dim]
        vp = np.pad(v, (pad, pad), mode="edge")
        k = np.ones((win,), dtype=np.float64) / win
        out[:, dim] = np.convolve(vp, k, mode="valid")

    out[0] = xy[0]
    out[-1] = xy[-1]
    return out


def ridge_band_path_on_density(
    density_map: np.ndarray,
    *,
    occ_mask: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    base_grid: int,
    matlab_1_indexed: bool,
    cfg: PathExtractConfig,
) -> np.ndarray:
    """
    Ridge-band path.
    1) threshold density inside occluder by quantile q.
    2) step q down until start and end are connected.
    3) run Dijkstra restricted to that band.

    Returns path as (L,2) XY points. Raises if no connection found.
    """
    hm = np.asarray(density_map, dtype=np.float64)
    mask_occ = np.asarray(occ_mask, dtype=bool)

    if hm.ndim != 2:
        raise ValueError(f"density_map must be 2D. Got {hm.shape}")
    if mask_occ.shape != hm.shape:
        raise ValueError(f"occ_mask shape {mask_occ.shape} must match density_map {hm.shape}")

    H, W = hm.shape
    if H != int(base_grid) or W != int(base_grid):
        # keep permissive. base_grid is still used for consistency checks downstream
        pass

    s_rc = _clip_rc(_xy_to_rc(np.asarray(start_xy), matlab_1_indexed=matlab_1_indexed), H, W)
    t_rc = _clip_rc(_xy_to_rc(np.asarray(end_xy), matlab_1_indexed=matlab_1_indexed), H, W)

    vals = hm[mask_occ]
    if vals.size == 0:
        raise RuntimeError("occ_mask has no True pixels.")

    # normalize within occluder for stable quantiles
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    hmn = (hm - vmin) / (vmax - vmin + cfg.eps)
    hmn = np.clip(hmn, 0.0, 1.0)

    # choose cost
    if cfg.ridge_cost_mode == "unit":
        cost = np.ones_like(hmn, dtype=np.float64)
    elif cfg.ridge_cost_mode == "neg_log":
        cost = -np.log(hmn + cfg.eps)
        cost = np.maximum(cost, cfg.cost_floor)
    else:  # "inv_density"
        cost = 1.0 / np.maximum(hmn, cfg.cost_floor)

    # step q down until connected
    q = float(cfg.ridge_q_start)
    q_min = float(cfg.ridge_q_min)
    q_step = float(cfg.ridge_q_step)

    best = None
    best_q = None

    while q >= q_min - 1e-12:
        thr = float(np.quantile(hmn[mask_occ], q))
        band = (hmn >= thr) & mask_occ

        # endpoints must be inside. If not, keep lowering q
        if (not band[s_rc]) or (not band[t_rc]):
            q -= q_step
            continue

        path_rc = _dijkstra_on_mask(cost, band, s_rc, t_rc, allow_diagonal=cfg.allow_diagonal)
        if path_rc is not None and path_rc.shape[0] >= 2:
            best = path_rc
            best_q = q
            break

        q -= q_step

    if best is None:
        raise RuntimeError(
            f"Ridge-band failed to connect endpoints. Tried q from {cfg.ridge_q_start} down to {cfg.ridge_q_min}."
        )

    xy = np.stack([_rc_to_xy((int(r), int(c)), matlab_1_indexed=matlab_1_indexed) for r, c in best], axis=0)

    if cfg.ridge_smooth_win and int(cfg.ridge_smooth_win) >= 3:
        xy = _smooth_xy(xy, int(cfg.ridge_smooth_win))

    # stash for debugging if you want it in the notebook prints
    return xy, best_q



def shortest_path_on_density(
    density_map: np.ndarray,
    *,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    base_grid: int,
    matlab_1_indexed: bool,
    cfg: PathExtractConfig,
) -> np.ndarray:
    """
    Your previous default. Kept for backward compatibility.
    This remains a plain shortest path on a global cost surface, no ridge constraint.
    """
    hm = np.asarray(density_map, dtype=np.float64)
    H, W = hm.shape

    s_rc = _clip_rc(_xy_to_rc(np.asarray(start_xy), matlab_1_indexed=matlab_1_indexed), H, W)
    t_rc = _clip_rc(_xy_to_rc(np.asarray(end_xy), matlab_1_indexed=matlab_1_indexed), H, W)

    # normalize to 0..1 for stable costs
    vmin, vmax = float(hm.min()), float(hm.max())
    hmn = (hm - vmin) / (vmax - vmin + cfg.eps)
    hmn = np.clip(hmn, 0.0, 1.0)

    cost = 1.0 / np.maximum(hmn, cfg.cost_floor)
    mask = np.ones_like(hmn, dtype=bool)

    path_rc = _dijkstra_on_mask(cost, mask, s_rc, t_rc, allow_diagonal=cfg.allow_diagonal)
    if path_rc is None:
        raise RuntimeError("shortest_path_on_density failed to connect endpoints.")

    xy = np.stack([_rc_to_xy((int(r), int(c)), matlab_1_indexed=matlab_1_indexed) for r, c in path_rc], axis=0)
    if cfg.ridge_smooth_win and int(cfg.ridge_smooth_win) >= 3:
        xy = _smooth_xy(xy, int(cfg.ridge_smooth_win))
    return xy
