"""Paint-local 3D mask: stay near the brush, do NOT take the whole skeleton.

Default: grow only a limited distance through bone from the green paint.
Optional: full connected bone component (old behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class GrowResult:
    mask: np.ndarray  # uint8 0/255 ZYX
    debug: dict


def bone_from_paint(
    ct_zyx: np.ndarray,
    paint_include: np.ndarray,
    *,
    hu_min: float = 200.0,
    hu_max: float = 3000.0,
    dilate_seeds: int = 2,
    paint_exclude: np.ndarray | None = None,
    mode: str = "local",
    grow_radius_mm: float = 8.0,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> GrowResult:
    """Build mask from paint.

    mode:
      - local (default): bone near paint only (radius mm) — what you brushed
      - component: entire connected bone blob touching paint (old, too greedy)
      - paint_only: painted voxels (+ small dilate) only
    """
    ct = np.asarray(ct_zyx, dtype=np.float64)
    paint = np.asarray(paint_include, dtype=bool)
    exclude = np.asarray(paint_exclude, dtype=bool) if paint_exclude is not None else None

    bone = (ct >= hu_min) & (ct <= hu_max)
    if exclude is not None:
        bone = bone & ~exclude

    n_paint = int(paint.sum())
    if n_paint == 0:
        return largest_bone_component(ct, hu_min=hu_min, hu_max=hu_max, exclude=exclude)

    if dilate_seeds > 0:
        paint_d = ndimage.binary_dilation(paint, iterations=int(dilate_seeds))
    else:
        paint_d = paint
    if exclude is not None:
        paint_d = paint_d & ~exclude

    if mode == "component":
        return _full_component(ct, bone, paint_d, hu_min, hu_max, n_paint)
    if mode == "paint_only":
        return _paint_only(ct, bone, paint_d, n_paint)

    # --- local (default): limited grow from paint through bone ---
    return _local_grow(
        ct,
        bone,
        paint_d,
        paint,
        n_paint,
        grow_radius_mm=grow_radius_mm,
        spacing_xyz=spacing_xyz,
        hu_min=hu_min,
        hu_max=hu_max,
    )


def _local_grow(
    ct: np.ndarray,
    bone: np.ndarray,
    paint_d: np.ndarray,
    paint: np.ndarray,
    n_paint: int,
    *,
    grow_radius_mm: float,
    spacing_xyz: tuple[float, float, float],
    hu_min: float,
    hu_max: float,
) -> GrowResult:
    # spacing_xyz is SimpleITK (x,y,z); array is z,y,x
    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    # isotropic voxel step ~ min spacing for dilation iterations
    step = max(1e-6, min(sx, sy, sz))
    radius = max(0.5, float(grow_radius_mm))
    max_iter = int(np.ceil(radius / step))
    max_iter = min(max_iter, 80)  # safety cap

    # Seeds: painted voxels that are bone, OR paint itself (so interior strokes count)
    seeds = paint_d & bone
    if not seeds.any():
        # paint may sit slightly off dense cortex — keep paint and grow outward into bone
        seeds = paint_d.copy()

    # Walkable: bone OR seeds (always keep brushed voxels)
    walk = bone | seeds

    structure = np.ones((3, 3, 3), dtype=bool)
    region = seeds.copy()
    for _ in range(max_iter):
        grown = ndimage.binary_dilation(region, structure=structure) & walk
        if np.array_equal(grown, region):
            break
        region = grown

    # Always include original paint stroke
    mask_bool = region | paint

    debug = {
        "mode": "local_near_paint",
        "hu_min": hu_min,
        "hu_max": hu_max,
        "grow_radius_mm": radius,
        "grow_iterations": max_iter,
        "spacing_xyz": [sx, sy, sz],
        "bone_voxels": int(bone.sum()),
        "paint_voxels": n_paint,
        "seed_voxels": int(seeds.sum()),
        "mask_voxels": int(mask_bool.sum()),
        "ok": int(mask_bool.sum()) > 0,
        "hint": (
            f"Only bone within ~{radius:.1f} mm of your green paint. "
            "Increase grow radius if too small; do NOT use whole-bone mode unless you want hip+femur."
        ),
    }
    if not debug["ok"]:
        debug["error"] = (
            "Local grow produced empty mask. Paint ON the structure, "
            "lower Bone HU min, or increase grow radius."
        )
        debug["hint_ct_at_paint"] = _hu_stats_under_paint(ct, paint_d)

    return GrowResult(mask=(mask_bool.astype(np.uint8)) * 255, debug=debug)


def _paint_only(
    ct: np.ndarray,
    bone: np.ndarray,
    paint_d: np.ndarray,
    n_paint: int,
) -> GrowResult:
    # Prefer bone under paint; fall back to paint voxels
    mask_bool = paint_d & bone
    if not mask_bool.any():
        mask_bool = paint_d
    debug = {
        "mode": "paint_only",
        "paint_voxels": n_paint,
        "mask_voxels": int(mask_bool.sum()),
        "ok": int(mask_bool.sum()) > 0,
        "hint": "Exact paint only (minimal expand).",
    }
    return GrowResult(mask=(mask_bool.astype(np.uint8)) * 255, debug=debug)


def _full_component(
    ct: np.ndarray,
    bone: np.ndarray,
    paint_d: np.ndarray,
    hu_min: float,
    hu_max: float,
    n_paint: int,
) -> GrowResult:
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, n_lab = ndimage.label(bone, structure=structure)
    seed_labels = np.unique(labeled[paint_d & (labeled > 0)])
    seed_labels = seed_labels[seed_labels > 0]
    debug = {
        "mode": "bone_component_from_paint",
        "hu_min": hu_min,
        "hu_max": hu_max,
        "bone_voxels": int(bone.sum()),
        "paint_voxels": n_paint,
        "n_bone_components": int(n_lab),
        "seed_labels": [int(x) for x in seed_labels[:20]],
        "n_seed_labels": int(len(seed_labels)),
        "warning": "FULL bone component — includes entire connected skeleton (hip+femur etc.)",
    }
    if len(seed_labels) == 0:
        debug["error"] = "Paint did not touch bone. Paint ON bright bone or lower HU min."
        debug["hint_ct_at_paint"] = _hu_stats_under_paint(ct, paint_d)
        return GrowResult(mask=np.zeros_like(ct, dtype=np.uint8), debug=debug)
    mask_bool = np.isin(labeled, seed_labels)
    debug["mask_voxels"] = int(mask_bool.sum())
    debug["ok"] = debug["mask_voxels"] > 0
    return GrowResult(mask=(mask_bool.astype(np.uint8)) * 255, debug=debug)


def largest_bone_component(
    ct_zyx: np.ndarray,
    *,
    hu_min: float = 200.0,
    hu_max: float = 3000.0,
    exclude: np.ndarray | None = None,
    min_voxels: int = 500,
) -> GrowResult:
    """Fallback when no paint: largest connected bone blob."""
    ct = np.asarray(ct_zyx, dtype=np.float64)
    bone = (ct >= hu_min) & (ct <= hu_max)
    if exclude is not None:
        bone = bone & ~np.asarray(exclude, dtype=bool)

    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, n_lab = ndimage.label(bone, structure=structure)

    debug = {
        "mode": "largest_bone_component",
        "hu_min": hu_min,
        "hu_max": hu_max,
        "bone_voxels": int(bone.sum()),
        "n_bone_components": int(n_lab),
    }

    if n_lab == 0:
        debug["error"] = f"No bone voxels with HU in [{hu_min}, {hu_max}]. Lower Bone HU min."
        debug["ct_stats"] = {
            "min": float(ct.min()),
            "max": float(ct.max()),
            "mean": float(ct.mean()),
            "p50": float(np.median(ct)),
            "p90": float(np.percentile(ct, 90)),
            "p99": float(np.percentile(ct, 99)),
        }
        return GrowResult(mask=np.zeros_like(ct, dtype=np.uint8), debug=debug)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    best = int(np.argmax(counts))
    sizes = sorted(
        ((int(i), int(c)) for i, c in enumerate(counts) if c >= min_voxels),
        key=lambda t: -t[1],
    )
    debug["top_components"] = sizes[:8]
    debug["chosen_label"] = best
    debug["chosen_voxels"] = int(counts[best])
    mask_bool = labeled == best
    mask = mask_bool.astype(np.uint8) * 255
    debug["mask_voxels"] = int(mask_bool.sum())
    debug["ok"] = debug["mask_voxels"] > 0
    debug["hint"] = "No paint — largest bone. Paint green on the part you want for local ROI."
    return GrowResult(mask=mask, debug=debug)


def _hu_stats_under_paint(ct: np.ndarray, paint: np.ndarray) -> dict:
    vals = ct[paint]
    if vals.size == 0:
        return {}
    return {
        "n": int(vals.size),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
    }
