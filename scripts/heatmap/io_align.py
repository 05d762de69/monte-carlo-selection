from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_infer_paths(paths_txt: str | Path) -> List[str]:
    """
    Load the exact ordering used during inference. One path per line.
    """
    paths_txt = Path(paths_txt)
    with paths_txt.open("r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
    if not paths:
        raise RuntimeError(f"No paths found in: {paths_txt}")
    return paths


def make_path_index(infer_paths: Sequence[str]) -> Dict[str, int]:
    return {p: i for i, p in enumerate(infer_paths)}


def _as_xy_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D XY array, got shape {a.shape}")
    if a.shape[1] == 2:
        return a
    if a.shape[0] == 2:
        return a.T
    raise ValueError(f"Expected Nx2 or 2xN, got shape {a.shape}")


def default_extract_polygon_xy(rec: dict) -> np.ndarray:
    """
    Tries common field names. Update if your JSON schema differs.
    """
    candidate_keys = ["new_shape", "polygon", "xy", "coords", "new_shape_xy", "polygon_xy"]
    for k in candidate_keys:
        if k in rec and rec[k] is not None and rec[k] != "":
            return _as_xy_array(rec[k])

    # nested fallback
    for k in ["shape", "completion"]:
        if k in rec and isinstance(rec[k], dict):
            for kk in ["new_shape", "polygon", "xy", "coords"]:
                if kk in rec[k]:
                    return _as_xy_array(rec[k][kk])

    raise KeyError(
        "Could not find polygon XY in record. "
        "Update default_extract_polygon_xy for your JSONL schema. "
        f"Available keys (sample): {sorted(rec.keys())[:40]}"
    )


def align_polygons_from_jsonl(
    jsonl_path: str | Path,
    infer_paths: Sequence[str],
    *,
    extract_xy: Callable[[dict], np.ndarray] = default_extract_polygon_xy,
    out_file_key: str = "out_file",
    allow_filename_fallback: bool = True,
) -> List[np.ndarray]:
    """
    Returns a list rows_xy of length N aligned to infer_paths order.
    Alignment uses rec[out_file_key] matching full path string from infer_paths.
    If allow_filename_fallback=True, also matches by filename if exact path differs.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing JSONL: {jsonl_path}")

    N = len(infer_paths)
    rows_xy: List[Optional[np.ndarray]] = [None] * N

    path_to_row = make_path_index(infer_paths)

    # Optional filename fallback index
    name_to_row: Dict[str, int] = {}
    if allow_filename_fallback:
        for i, p in enumerate(infer_paths):
            name = Path(p).name
            if name in name_to_row:
                # duplicates. filename fallback would be ambiguous. disable it silently for this case
                name_to_row = {}
                break
            name_to_row[name] = i

    n_read = 0
    n_aligned = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_read += 1

            out_file = rec.get(out_file_key, "")
            if not out_file:
                continue
            out_file = str(out_file)

            row = path_to_row.get(out_file, None)
            if row is None and allow_filename_fallback and name_to_row:
                row = name_to_row.get(Path(out_file).name, None)

            if row is None:
                continue

            if rows_xy[row] is None:
                rows_xy[row] = extract_xy(rec)
                n_aligned += 1

    missing = [i for i, x in enumerate(rows_xy) if x is None]
    if missing:
        raise RuntimeError(
            f"Missing polygons for {len(missing)}/{N} rows. "
            f"Example missing index: {missing[0]}. "
            "This is usually a path mismatch between paths.txt and JSONL out_file."
        )

    return [x for x in rows_xy if x is not None]
