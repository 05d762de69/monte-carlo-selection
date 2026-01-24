from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import json
import numpy as np


@dataclass
class HeatmapResult:
    heat_global: np.ndarray                # H x W
    heat_comp: Dict[int, np.ndarray]       # k -> H x W
    top_components: List[int]
    top_weights: List[float]


def select_top_components(weights: np.ndarray, top_k: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    return np.argsort(w)[::-1][:top_k]


def accumulate_heatmaps(
    rows_xy: Sequence[np.ndarray],
    resp: np.ndarray,
    *,
    H: int,
    W: int,
    top_components: Sequence[int],
    rasterize_fn,
) -> HeatmapResult:
    """
    rows_xy length N. resp shape N x K.
    rasterize_fn(xy, H, W) -> bool mask H x W.
    """
    N = len(rows_xy)
    if resp.shape[0] != N:
        raise ValueError(f"resp rows {resp.shape[0]} must match rows_xy {N}")

    top_components = [int(k) for k in top_components]

    heat_global = np.zeros((H, W), dtype=np.float64)
    heat_comp: Dict[int, np.ndarray] = {k: np.zeros((H, W), dtype=np.float64) for k in top_components}

    for i, xy in enumerate(rows_xy):
        mask = rasterize_fn(xy, H, W)

        heat_global[mask] += 1.0

        r = resp[i]
        for k in top_components:
            wk = float(r[k])
            if wk > 0.0:
                heat_comp[k][mask] += wk

    return HeatmapResult(
        heat_global=heat_global,
        heat_comp=heat_comp,
        top_components=top_components,
        top_weights=[],
    )


def normalize_heatmap(h: np.ndarray) -> np.ndarray:
    m = float(np.max(h))
    if m <= 0:
        return h.astype(np.float64)
    return h / m


def normalize_result(result: HeatmapResult) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    global_norm = normalize_heatmap(result.heat_global)
    comp_norm = {k: normalize_heatmap(v) for k, v in result.heat_comp.items()}
    return global_norm, comp_norm


def save_heatmaps(
    out_dir: str | Path,
    *,
    heat_global: np.ndarray,
    heat_comp: Dict[int, np.ndarray],
    info: dict,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "heat_global.npy", heat_global.astype(np.float32))
    for k, hk in heat_comp.items():
        np.save(out_dir / f"heat_comp_{int(k):03d}.npy", hk.astype(np.float32))

    with (out_dir / "heatmap_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return out_dir
