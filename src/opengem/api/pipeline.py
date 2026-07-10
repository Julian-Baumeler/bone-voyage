"""Run pipeline steps for a project."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from opengem.api.workspace import Workspace
from opengem.core import image_io
from opengem.core.export import (
    export_abaqus,
    export_ansys,
    export_ascii_ugrid,
    export_lsdyna,
    export_vtu,
)
from opengem.core.material import map_modulus
from opengem.core.matmap_xml import MatMapConfig, config_from_dict, save_matmap
from opengem.core.segment import estimate_graphcut_memory_bytes, graph_cut_3d
from opengem.core.surface import SurfaceOptions, mask_to_surface, surface_to_json, write_stl, write_vtp
from opengem.core.volume import VolumeOptions, tetrahedralize, ug_stats, write_vtu

logger = logging.getLogger(__name__)


def run_step(ws: Workspace, project_id: str, step: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    ws.set_error(project_id, None)
    ws.set_status(project_id, current_step=step, state="running")
    try:
        if step == "segment":
            result = _step_segment(ws, project_id, params)
        elif step == "surface":
            result = _step_surface(ws, project_id, params)
        elif step == "volume":
            result = _step_volume(ws, project_id, params)
        elif step == "material":
            result = _step_material(ws, project_id, params)
        elif step == "export":
            result = _step_export(ws, project_id, params)
        else:
            raise ValueError(f"Unknown step: {step}")
        ws.set_status(project_id, current_step=step, state="done", last_result=result)
        return result
    except Exception as e:
        logger.exception("Step %s failed", step)
        ws.set_error(project_id, str(e))
        ws.set_status(project_id, current_step=step, state="error")
        raise


def _step_segment(ws: Workspace, project_id: str, params: dict) -> dict:
    ct = image_io.read_image(ws.resolve(project_id, "ct"))
    fg = image_io.read_image(ws.resolve(project_id, "fg_seeds"))
    bg = image_io.read_image(ws.resolve(project_id, "bg_seeds"))
    sigma = float(params.get("sigma", 50.0))
    direction = params.get("boundary_direction", "bidirectional")
    mem = estimate_graphcut_memory_bytes(image_io.to_numpy(ct)[0].shape)
    out = graph_cut_3d(ct, fg, bg, sigma=sigma, boundary_direction=direction)
    rel = "outputs/segmentation.nrrd"
    dest = ws.path(project_id) / rel
    image_io.write_image(dest, out)
    ws.set_file(project_id, "mask", rel)
    return {"mask": rel, "estimated_memory_bytes": mem}


def _get_or_make_mask(ws: Workspace, project_id: str, params: dict):
    """Return a binary mask image: uploaded/segmented mask, or bone threshold from CT."""
    import SimpleITK as sitk

    files = ws.get(project_id).files
    if "mask" in files:
        mask = image_io.read_image(ws.resolve(project_id, "mask"))
        return mask, "mask"

    if "ct" not in files:
        raise FileNotFoundError(
            "Need a CT or a segmentation mask. Upload a CT first, then either "
            "upload a mask or run Surface (auto bone-threshold from CT)."
        )

    ct = image_io.read_image(ws.resolve(project_id, "ct"))
    # Bone HU threshold (default 200 — works for femur tutorial CT)
    hu_min = float(params.get("bone_hu_min", 200.0))
    hu_max = float(params.get("bone_hu_max", 3000.0))
    binary = sitk.BinaryThreshold(
        ct,
        lowerThreshold=hu_min,
        upperThreshold=hu_max,
        insideValue=255,
        outsideValue=0,
    )
    # Light cleanup so marching cubes gets a cleaner surface
    binary = sitk.BinaryMorphologicalOpening(binary, [1, 1, 1])
    binary = sitk.BinaryMorphologicalClosing(binary, [1, 1, 1])
    rel = "outputs/mask_from_bone_threshold.nrrd"
    image_io.write_image(ws.path(project_id) / rel, binary)
    ws.set_file(project_id, "mask", rel)
    return binary, f"bone_threshold_HU[{hu_min},{hu_max}]"


def _step_surface(ws: Workspace, project_id: str, params: dict) -> dict:
    mask, mask_source = _get_or_make_mask(ws, project_id, params)

    opts = SurfaceOptions(
        use_median=bool(params.get("use_median", False)),
        use_gaussian=bool(params.get("use_gaussian", True)),
        gaussian_std=float(params.get("gaussian_std", 2.2)),
        use_polygon_smooth=bool(params.get("use_polygon_smooth", True)),
        pad_slices=int(params.get("pad_slices", 1)),
        threshold=float(params.get("threshold", 127.5)),
    )
    poly = mask_to_surface(mask, opts)
    if poly.GetNumberOfPoints() < 4:
        raise RuntimeError(
            "Surface is empty — no bone voxels found. "
            "Try a lower bone HU min (e.g. 100) or upload a segmentation mask."
        )
    rel_stl = "outputs/surface.stl"
    rel_vtp = "outputs/surface.vtp"
    write_stl(ws.path(project_id) / rel_stl, poly)
    write_vtp(ws.path(project_id) / rel_vtp, poly)
    ws.set_file(project_id, "surface", rel_stl)
    preview = surface_to_json(poly)
    preview_path = ws.path(project_id) / "outputs" / "surface_preview.json"
    preview_path.write_text(json.dumps(preview))
    ws.set_file(project_id, "surface_preview", "outputs/surface_preview.json")
    return {
        "surface": rel_stl,
        "mask_source": mask_source,
        "n_points": preview["n_points"],
        "n_faces": preview["n_faces"],
    }


def _step_volume(ws: Workspace, project_id: str, params: dict) -> dict:
    from opengem.core.surface import read_surface

    files = ws.get(project_id).files
    # Auto-run surface if missing (CT-only projects)
    if "surface" not in files:
        if "ct" in files or "mask" in files:
            logger.info("No surface yet — generating surface before volume mesh")
            _step_surface(ws, project_id, params)
        else:
            raise FileNotFoundError(
                "No surface mesh yet. Upload a CT, go to step 3 (Surface) and click "
                "“Generate surface”, then try volume mesh again."
            )

    surf = read_surface(ws.resolve(project_id, "surface"))
    opts = VolumeOptions(
        cell_size=float(params.get("cell_size", 3.0)),
        quadratic=bool(params.get("quadratic", True)),
        backend=str(params.get("backend", "auto")),
    )
    ug = tetrahedralize(surf, opts)
    if ug.GetNumberOfCells() == 0:
        raise RuntimeError("Volume mesher produced 0 tetrahedra. Try a larger cell size or check the surface.")
    rel = "outputs/volume.vtu"
    write_vtu(ws.path(project_id) / rel, ug)
    ws.set_file(project_id, "volume", rel)
    return {"volume": rel, **ug_stats(ug)}


def _step_material(ws: Workspace, project_id: str, params: dict) -> dict:
    from opengem.core.volume import read_vtu

    ct = image_io.read_image(ws.resolve(project_id, "ct"))
    mesh = read_vtu(ws.resolve(project_id, "volume"))
    if "matmap" in params:
        cfg = config_from_dict(params["matmap"])
    else:
        cfg = MatMapConfig()
        # sensible defaults for demo without phantom
        cfg.auto_fit_rho_ct = False
        cfg.density.slope = 0.001
        cfg.density.offset = 0.0
        cfg.density.ash_enabled = False
        cfg.density.app_enabled = False

    result = map_modulus(mesh, ct, cfg)
    rel = "outputs/material_mapped.vtu"
    write_vtu(ws.path(project_id) / rel, result.mesh)
    ws.set_file(project_id, "mapped", rel)
    matmap_rel = "outputs/config.matmap"
    save_matmap(ws.path(project_id) / matmap_rel, cfg)
    ws.set_file(project_id, "matmap", matmap_rel)
    return {"mapped": rel, "stats": result.stats}


def _step_export(ws: Workspace, project_id: str, params: dict) -> dict:
    from opengem.core.volume import read_vtu

    key = "mapped" if "mapped" in ws.get(project_id).files else "volume"
    mesh = read_vtu(ws.resolve(project_id, key))
    fmt = params.get("format", "vtu")
    out_dir = ws.outputs(project_id)
    files: dict[str, str] = {}

    if fmt == "vtu":
        p = export_vtu(out_dir / "export.vtu", mesh)
        files["vtu"] = str(p.relative_to(ws.path(project_id)))
    elif fmt == "ascii":
        p = export_ascii_ugrid(out_dir / "export.txt", mesh)
        files["ascii"] = str(p.relative_to(ws.path(project_id)))
    elif fmt == "abaqus":
        p = export_abaqus(out_dir / "export.inp", mesh)
        files["abaqus"] = str(p.relative_to(ws.path(project_id)))
    elif fmt == "ansys":
        p = export_ansys(out_dir / "export.mac", mesh)
        files["ansys"] = str(p.relative_to(ws.path(project_id)))
    elif fmt == "lsdyna":
        m, mats = export_lsdyna(out_dir / "mesh.k", out_dir / "mats.k", mesh)
        files["lsdyna_mesh"] = str(m.relative_to(ws.path(project_id)))
        files["lsdyna_mats"] = str(mats.relative_to(ws.path(project_id)))
    elif fmt == "all":
        for f, name in [
            (export_vtu(out_dir / "export.vtu", mesh), "vtu"),
            (export_ascii_ugrid(out_dir / "export.txt", mesh), "ascii"),
            (export_abaqus(out_dir / "export.inp", mesh), "abaqus"),
            (export_ansys(out_dir / "export.mac", mesh), "ansys"),
        ]:
            files[name] = str(Path(f).relative_to(ws.path(project_id)))
        m, mats = export_lsdyna(out_dir / "mesh.k", out_dir / "mats.k", mesh)
        files["lsdyna_mesh"] = str(m.relative_to(ws.path(project_id)))
        files["lsdyna_mats"] = str(mats.relative_to(ws.path(project_id)))
    else:
        raise ValueError(f"Unknown format: {fmt}")

    for k, rel in files.items():
        ws.set_file(project_id, f"export_{k}", rel)
    return {"files": files}
