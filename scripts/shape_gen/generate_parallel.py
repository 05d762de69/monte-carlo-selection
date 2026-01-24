# =========================
# PATCH: shape_gen/generate3.py
# =========================
# Apply these edits to your existing file.
# I’m showing the full generate_completions function with only the relevant changes.
# Everything else in your file stays the same.

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import math
import numpy as np

from shapely.geometry import Point, Polygon, LineString, MultiPoint, GeometryCollection
from shapely.prepared import prep
from shapely.ops import unary_union, nearest_points

from shape_gen.segments import extract_random_segment
from shape_gen.render import draw_and_save
from shape_gen.io_mat import unit_to_pixel

from shape_gen.donor_fit import DonorFitParams, donor_occluder_fit

# ... keep the rest of your helpers as-is ...


def generate_completions(
    *,
    silhouette: np.ndarray,
    occluder: np.ndarray,
    start_pt: np.ndarray,
    end_pt: np.ndarray,
    minX: int,
    minY: int,
    wBB: int,
    hBB: int,
    out_w: int,
    out_h: int,
    out_dir: str | Path,
    silhouette_index: int,
    sil_class: str,
    base_grid: int,
    records,
    classes: List[str],
    byClass: Dict[str, np.ndarray],
    n_images: int,
    rng: np.random.Generator,

    # NEW. Start index for completion_index and filename uniqueness in parallel runs.
    start_index: int = 1,

    # donor segment sampling
    fraction: float = 0.40,

    # donor-fit behavior (preserve donor curvature; shrink only if needed)
    shrink_gamma: float = 0.88,
    max_shrink_iters: int = 30,
    smooth_win: int = 5,
    try_mirror: bool = True,

    # behavior
    final_n_samples_mode: str = "match_segment",  # or "matlab_100"
    supersample: int = 4,
    flush_every: int = 500,
    max_attempts_per_image: int = 50,
    require_valid: bool = True,
    save_invalid: bool = False,
    invalid_subdir: str = "_invalid",

    # intersection behavior
    snap_intersections_to_vertices: bool = True,

    # flexible refit spline behavior
    refit_enabled: bool = True,
    refit_n_ctrl: int = 14,
    refit_subdiv: int = 25,
    refit_jitter_sigma: float = 0.015,
    refit_max_attempts: int = 10,
) -> tuple[List["CompletionMeta"], List[str], List[np.ndarray]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_invalid:
        (out_dir / invalid_subdir).mkdir(parents=True, exist_ok=True)

    if n_images <= 0:
        return [], [], []

    sil = np.asarray(silhouette, dtype=np.float64)
    occ_xy = np.asarray(occluder, dtype=np.float64)

    pA_true, pB_true, visible_arc, prepared_occ = _extract_intersections_AB_and_visible_arc(
        sil_xy=sil,
        occ_xy=occ_xy,
        snap_to_silhouette_vertices=bool(snap_intersections_to_vertices),
    )
    occ_poly = Polygon(occ_xy.tolist())

    metas: List[CompletionMeta] = []
    out_files_xy: List[str] = []
    polygons_xy: List[np.ndarray] = []

    produced = 0

    # CHANGED. n_images now means "how many attempt indices to consume in this call".
    target_attempts = int(n_images)
    i = int(start_index)
    end_i = int(start_index) + target_attempts - 1

    # CHANGED. Loop over fixed index range so parallel calls do not collide.
    while i <= end_i:
        attempts = 0
        valid = False
        out_file: Optional[Path] = None
        new_shape_valid: Optional[np.ndarray] = None

        donor_idx: Optional[int] = None
        donor_class: Optional[str] = None
        donor_img_id: Optional[int] = None

        while attempts < max_attempts_per_image and (not valid):
            attempts += 1

            donor_idx, donor_class = _choose_donor_index(rng, records, classes, byClass, sil_class)
            donor = records[donor_idx]
            donor_img_id = getattr(donor, "img_id", None)

            alt_shape = unit_to_pixel(donor.contour_u, base_grid)
            random_segment, _ = extract_random_segment(alt_shape, fraction=fraction, rng=rng)

            n_samples = 100 if final_n_samples_mode == "matlab_100" else int(len(random_segment))
            if n_samples < 20:
                n_samples = 20

            params = DonorFitParams(
                n_samples=int(n_samples),
                enforce_simple=True,
                max_shrink_iters=int(max_shrink_iters),
                shrink_gamma=float(shrink_gamma),
                smooth_win=int(smooth_win),
                try_mirror=bool(try_mirror),
            )

            aligned_segment = donor_occluder_fit(
                donor_segment=random_segment,
                start_pt=start_pt,
                end_pt=end_pt,
                occluder=occ_xy,
                rng=rng,
                params=params,
            )

            if aligned_segment.size == 0:
                continue

            aligned_segment = np.asarray(aligned_segment, dtype=np.float64)

            if not _validate_inside(aligned_segment, prepared_occ):
                continue
            if not _is_simple_polyline(aligned_segment):
                continue

            dA0 = np.sum((aligned_segment[0] - pA_true) ** 2)
            dA1 = np.sum((aligned_segment[-1] - pA_true) ** 2)
            if dA1 < dA0:
                aligned_segment = np.flipud(aligned_segment)

            aligned_segment = aligned_segment.copy()
            aligned_segment[0] = pA_true
            aligned_segment[-1] = pB_true

            if refit_enabled:
                refined = _flexible_refit_spline_inside(
                    aligned_segment,
                    pA=pA_true,
                    pB=pB_true,
                    occ_poly=occ_poly,
                    prepared_occ=prepared_occ,
                    rng=rng,
                    n_ctrl=int(refit_n_ctrl),
                    subdiv=int(refit_subdiv),
                    jitter_sigma=float(refit_jitter_sigma),
                    max_attempts=int(refit_max_attempts),
                )
                if refined.size == 0:
                    continue
                if not _validate_inside(refined, prepared_occ):
                    continue
                if not _is_simple_polyline(refined):
                    continue
                aligned_segment = refined

            new_shape = np.vstack([aligned_segment, visible_arc[1:]])

            if not np.allclose(new_shape[0], new_shape[-1]):
                new_shape = np.vstack([new_shape, new_shape[0]])

            poly = Polygon(new_shape.tolist())
            if (not poly.is_valid) or (poly.area <= 1e-6):
                continue

            # CHANGED. Filename uses the globally unique i.
            out_file = out_dir / f"completion_{silhouette_index:04d}_{i:05d}.png"
            draw_and_save(
                polygons=[new_shape],
                colors=[[0, 0, 0]],
                minX=minX,
                minY=minY,
                wBB=wBB,
                hBB=hBB,
                out_w=out_w,
                out_h=out_h,
                out_file=out_file,
                supersample=supersample,
            )

            valid = True
            new_shape_valid = new_shape.astype(np.float32, copy=False)

        if require_valid and (not valid):
            metas.append(
                CompletionMeta(
                    silhouette_index=int(silhouette_index),
                    completion_index=int(i),
                    sil_class=str(sil_class or ""),
                    donor_class=str(donor_class or (records[donor_idx].category if donor_idx is not None else "") or ""),
                    donor_record_index=int(donor_idx) if donor_idx is not None else -1,
                    donor_img_id=donor_img_id,
                    fraction=float(fraction),
                    shrink_gamma=float(shrink_gamma),
                    max_shrink_iters=int(max_shrink_iters),
                    smooth_win=int(smooth_win),
                    try_mirror=bool(try_mirror),
                    n_samples_mode=str(final_n_samples_mode),
                    refit_enabled=bool(refit_enabled),
                    refit_n_ctrl=int(refit_n_ctrl),
                    refit_subdiv=int(refit_subdiv),
                    refit_jitter_sigma=float(refit_jitter_sigma),
                    refit_max_attempts=int(refit_max_attempts),
                    out_file=str(out_file) if out_file is not None else "",
                    attempts=int(attempts),
                    valid=False,
                )
            )
            if (len(metas) % flush_every) == 0:
                done = (i - int(start_index) + 1)
                print(f"Attempts logged {done}/{target_attempts}. Valid produced {produced}...")
            i += 1
            continue

        if out_file is not None and new_shape_valid is not None:
            produced += 1
            out_files_xy.append(str(out_file))
            polygons_xy.append(new_shape_valid)

        metas.append(
            CompletionMeta(
                silhouette_index=int(silhouette_index),
                completion_index=int(i),
                sil_class=str(sil_class or ""),
                donor_class=str(donor_class or (records[donor_idx].category if donor_idx is not None else "") or ""),
                donor_record_index=int(donor_idx) if donor_idx is not None else -1,
                donor_img_id=donor_img_id,
                fraction=float(fraction),
                shrink_gamma=float(shrink_gamma),
                max_shrink_iters=int(max_shrink_iters),
                smooth_win=int(smooth_win),
                try_mirror=bool(try_mirror),
                n_samples_mode=str(final_n_samples_mode),
                refit_enabled=bool(refit_enabled),
                refit_n_ctrl=int(refit_n_ctrl),
                refit_subdiv=int(refit_subdiv),
                refit_jitter_sigma=float(refit_jitter_sigma),
                refit_max_attempts=int(refit_max_attempts),
                out_file=str(out_file) if out_file is not None else "",
                attempts=int(attempts),
                valid=bool(valid),
            )
        )

        # CHANGED. Progress based on attempts, not produced.
        if ((i - int(start_index) + 1) % flush_every) == 0:
            done = (i - int(start_index) + 1)
            print(f"Attempts {done}/{target_attempts}. Valid produced {produced}. Last attempts={attempts}")

        i += 1

    return metas, out_files_xy, polygons_xy
