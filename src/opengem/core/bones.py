"""Auto-split CT into separate bones (connected bright components).

Hip and femur often look connected at low HU due to partial-volume “bridges”.
We use dual-threshold separation:
  1) Core seeds at higher HU (true dense bone)
  2) Grow each seed into the lower-HU bone mask without merging
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from opengem.core import image_io
from opengem.core.surface import SurfaceOptions, mask_to_surface, surface_to_json


BONE_COLORS = [
    0x7EC8FF,
    0x2DD4A8,
    0xF0B429,
    0xF07178,
    0xC792EA,
    0x82AAFF,
    0xFF9E64,
    0x89DDFF,
    0xC3E88D,
    0xFF5370,
    0xBB80B3,
    0x80CBC4,
]


@dataclass
class BoneInfo:
    id: int
    name: str
    label: int
    voxels: int
    volume_mm3: float
    color: str
    color_hex: int
    preview_file: str


@dataclass
class AutoSplitResult:
    n_bones: int
    bones: list[BoneInfo]
    labeled_path: str
    bones_json_path: str
    hu_min: float
    hu_max: float
    min_voxels: int
    debug: dict = field(default_factory=dict)


def _hex_color(i: int) -> tuple[str, int]:
    c = BONE_COLORS[i % len(BONE_COLORS)]
    return f"#{c:06x}", c


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return labeled volume and list of (label, count) sorted by size desc."""
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, n_lab = ndimage.label(mask, structure=structure)
    if n_lab == 0:
        return labeled, []
    counts = np.bincount(labeled.ravel())
    comps = [(int(i), int(c)) for i, c in enumerate(counts) if i > 0]
    comps.sort(key=lambda t: -t[1])
    return labeled, comps


def _open_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    struct = ndimage.generate_binary_structure(3, 1)
    if radius > 1:
        struct = ndimage.iterate_structure(struct, radius)
    return ndimage.binary_opening(mask, structure=struct)


def split_by_hysteresis(
    ct_arr: np.ndarray,
    *,
    hu_low: float,
    hu_high: float,
    hu_max: float,
    min_voxels: int,
    open_core: int = 1,
) -> tuple[np.ndarray, list[tuple[int, int]], dict]:
    """Label dense cores (hu_high), grow into soft bone (hu_low) without merging.

    Returns remapped labels 1..N (only components with final size >= min_voxels).
    """
    low = (ct_arr >= hu_low) & (ct_arr <= hu_max)
    high = (ct_arr >= hu_high) & (ct_arr <= hu_max)
    high = _open_mask(high, open_core)

    core_lab, core_comps = _label_components(high)
    # drop tiny cores early
    core_comps = [(lab, c) for lab, c in core_comps if c >= max(50, min_voxels // 20)]
    debug = {
        "method": "hysteresis",
        "hu_low": hu_low,
        "hu_high": hu_high,
        "n_core_components": len(core_comps),
        "core_top": core_comps[:10],
        "low_voxels": int(low.sum()),
        "high_voxels": int(high.sum()),
    }
    if not core_comps:
        return np.zeros_like(ct_arr, dtype=np.uint16), [], debug

    # Seeds: only kept cores, renumber 1..K temporarily
    seeds = np.zeros_like(core_lab, dtype=np.int32)
    for new_id, (lab, _) in enumerate(core_comps, start=1):
        seeds[core_lab == lab] = new_id
    n_seeds = len(core_comps)

    # Multi-source geodesic grow into `low` (and seeds)
    walk = low | (seeds > 0)
    labels = seeds.copy()
    structure = np.ones((3, 3, 3), dtype=bool)

    # Iterate until stable (or safety cap)
    for it in range(80):
        # Dilate each label into unlabeled walkable voxels
        # Use grey dilation on labels where 0 is background — but max of neighbors
        # would mix labels. Do per-label carefully with binary dilate.
        changed = False
        unlabeled = (labels == 0) & walk
        if not unlabeled.any():
            break
        for lid in range(1, n_seeds + 1):
            region = labels == lid
            if not region.any():
                continue
            grown = ndimage.binary_dilation(region, structure=structure) & unlabeled
            if grown.any():
                labels[grown] = lid
                changed = True
                unlabeled = (labels == 0) & walk
        if not changed:
            break
        debug["grow_iterations"] = it + 1

    # Filter by final size and renumber Bone 1..N
    final_comps = []
    for lid in range(1, n_seeds + 1):
        c = int((labels == lid).sum())
        if c >= min_voxels:
            final_comps.append((lid, c))
    final_comps.sort(key=lambda t: -t[1])

    remapped = np.zeros_like(labels, dtype=np.uint16)
    kept = []
    for new_id, (old_id, c) in enumerate(final_comps, start=1):
        remapped[labels == old_id] = new_id
        kept.append((new_id, c))

    debug["n_bones"] = len(kept)
    debug["bone_sizes"] = kept
    return remapped, kept, debug


def split_by_opening(
    ct_arr: np.ndarray,
    *,
    hu_min: float,
    hu_max: float,
    min_voxels: int,
    open_radius: int,
) -> tuple[np.ndarray, list[tuple[int, int]], dict]:
    bone = (ct_arr >= hu_min) & (ct_arr <= hu_max)
    bone = _open_mask(bone, open_radius)
    labeled, comps = _label_components(bone)
    comps = [(lab, c) for lab, c in comps if c >= min_voxels]
    remapped = np.zeros_like(labeled, dtype=np.uint16)
    kept = []
    for new_id, (lab, c) in enumerate(comps, start=1):
        remapped[labeled == lab] = new_id
        kept.append((new_id, c))
    debug = {
        "method": "opening",
        "hu_min": hu_min,
        "open_radius": open_radius,
        "n_bones": len(kept),
        "bone_sizes": kept,
    }
    return remapped, kept, debug


def auto_split_bones(
    ct: sitk.Image,
    *,
    hu_min: float = 200.0,
    hu_max: float = 3000.0,
    hu_core: float | None = None,
    open_radius: int = 2,
    min_voxels: int = 1000,
    max_preview_bones: int = 12,
    out_dir: Path,
    on_progress: Callable[[str, int, str], None] | None = None,
) -> AutoSplitResult:
    """Detect separate bone components and write masks + surface previews."""

    def prog(step: str, pct: int, msg: str) -> None:
        if on_progress:
            on_progress(step, pct, msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bones_dir = out_dir / "bones"
    bones_dir.mkdir(exist_ok=True)

    prog("threshold", 5, f"Loading CT, low HU≥{hu_min}…")
    ct_arr, spacing, origin = image_io.to_numpy(ct)
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    vox_mm3 = abs(sx * sy * sz)

    # Core threshold: dense bone that separates hip vs femur
    if hu_core is None:
        hu_core = max(hu_min + 150.0, 400.0)

    prog("label", 15, f"Dual-threshold split (core HU≥{hu_core:.0f}, fill HU≥{hu_min:.0f})…")
    remapped, kept, dbg = split_by_hysteresis(
        ct_arr,
        hu_low=hu_min,
        hu_high=hu_core,
        hu_max=hu_max,
        min_voxels=min_voxels,
        open_core=1,
    )

    # Fallback if hysteresis found < 2: stronger morphological open at low HU
    if len(kept) < 2:
        prog("label", 30, f"Hysteresis found {len(kept)} bone(s) — trying open r={open_radius}…")
        remapped2, kept2, dbg2 = split_by_opening(
            ct_arr,
            hu_min=hu_min,
            hu_max=hu_max,
            min_voxels=min_voxels,
            open_radius=max(open_radius, 2),
        )
        if len(kept2) > len(kept):
            remapped, kept, dbg = remapped2, kept2, dbg2
        elif len(kept) == 0 and kept2:
            remapped, kept, dbg = remapped2, kept2, dbg2

    # Last resort: single threshold open=2
    if len(kept) == 0:
        prog("label", 35, "Last resort: HU threshold + open…")
        remapped, kept, dbg = split_by_opening(
            ct_arr, hu_min=hu_min, hu_max=hu_max, min_voxels=max(200, min_voxels // 2), open_radius=2
        )

    prog("label", 45, f"Kept {len(kept)} separate bones")
    if not kept:
        return AutoSplitResult(
            n_bones=0,
            bones=[],
            labeled_path="",
            bones_json_path="",
            hu_min=hu_min,
            hu_max=hu_max,
            min_voxels=min_voxels,
            debug={**dbg, "error": "No separate bones found — try lower HU min"},
        )

    bone_infos: list[BoneInfo] = []
    for bone_id, c in kept:
        css, hx = _hex_color(bone_id - 1)
        bone_infos.append(
            BoneInfo(
                id=bone_id,
                name=f"Bone {bone_id}",
                label=bone_id,
                voxels=c,
                volume_mm3=float(c * vox_mm3),
                color=css,
                color_hex=hx,
                preview_file=f"outputs/bones/bone_{bone_id}_preview.json",
            )
        )

    labeled_img = image_io.from_numpy(remapped.astype(np.uint16), spacing=spacing, origin=origin)
    labeled_img.SetDirection(ct.GetDirection())
    image_io.write_image(out_dir / "bones_labeled.nrrd", labeled_img)

    opts = SurfaceOptions(
        use_gaussian=True,
        use_polygon_smooth=True,
        pad_slices=1,
        smooth_iterations=10,
        gaussian_std=1.2,
    )
    n_mesh = min(len(bone_infos), max_preview_bones)
    for bi in range(n_mesh):
        info = bone_infos[bi]
        pct = 50 + int(45 * (bi + 1) / max(n_mesh, 1))
        prog("mesh", pct, f"Meshing {info.name} ({info.voxels} vx)…")
        mask_u8 = ((remapped == info.id).astype(np.uint8)) * 255
        mask_img = image_io.from_numpy(mask_u8, spacing=spacing, origin=origin)
        mask_img.SetDirection(ct.GetDirection())
        try:
            poly = mask_to_surface(mask_img, opts)
            preview = surface_to_json(poly, max_points=35_000)
        except Exception as e:
            preview = {"points": [], "faces": [], "n_points": 0, "n_faces": 0, "error": str(e)}
        (bones_dir / f"bone_{info.id}_preview.json").write_text(json.dumps(preview))
        image_io.write_image(bones_dir / f"bone_{info.id}_mask.nrrd", mask_img)

    for bi in range(n_mesh, len(bone_infos)):
        bone_infos[bi].preview_file = ""

    bones_meta = {
        "n_bones": len(bone_infos),
        "hu_min": hu_min,
        "hu_max": hu_max,
        "hu_core": hu_core,
        "min_voxels": min_voxels,
        "open_radius": open_radius,
        "spacing": [sx, sy, sz],
        "bones": [asdict(b) for b in bone_infos],
        "labeled_file": "outputs/bones_labeled.nrrd",
        "debug": dbg,
    }
    (out_dir / "bones.json").write_text(json.dumps(bones_meta, indent=2))
    prog("done", 100, f"Done — {len(bone_infos)} bones")

    return AutoSplitResult(
        n_bones=len(bone_infos),
        bones=bone_infos,
        labeled_path="outputs/bones_labeled.nrrd",
        bones_json_path="outputs/bones.json",
        hu_min=hu_min,
        hu_max=hu_max,
        min_voxels=min_voxels,
        debug=dbg,
    )


def union_bone_masks(
    out_dir: Path,
    bone_ids: list[int],
    *,
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    direction: Any = None,
) -> sitk.Image | None:
    out_dir = Path(out_dir)
    bones_dir = out_dir / "bones"
    if not bone_ids:
        return None
    acc = None
    for bid in bone_ids:
        p = bones_dir / f"bone_{bid}_mask.nrrd"
        if not p.exists():
            continue
        img = sitk.ReadImage(str(p))
        arr = sitk.GetArrayFromImage(img) > 0
        acc = arr if acc is None else (acc | arr)
    if acc is None:
        return None
    mask = image_io.from_numpy((acc.astype(np.uint8) * 255), spacing=spacing, origin=origin)
    if direction is not None:
        mask.SetDirection(direction)
    return mask
