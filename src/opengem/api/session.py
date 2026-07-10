"""Single auto-session (no project management for users)."""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from opengem.api.workspace import Workspace
from opengem.core import image_io
from opengem.core.export import export_ascii_ugrid, export_vtu
from opengem.core.material import map_modulus
from opengem.core.matmap_xml import MatMapConfig
from opengem.core.bones import union_bone_masks
from opengem.core.region_grow import bone_from_paint
from opengem.core.surface import SurfaceOptions, mask_to_surface, surface_to_json, write_stl, write_vtp
from opengem.core.volume import VolumeOptions, tetrahedralize, ug_stats, write_vtu

logger = logging.getLogger(__name__)

SESSION_NAME = "OpenGEM session"


class Session:
    def __init__(self, ws: Workspace | None = None) -> None:
        self.ws = ws or Workspace()
        self._id: str | None = None

    @property
    def id(self) -> str | None:
        return self._id

    def ensure(self) -> str:
        if self._id and (self.ws.root / self._id).exists():
            return self._id
        # Reuse most recent session that has a CT (survives server restart)
        restored = self._find_latest_with_ct()
        if restored:
            self._id = restored
            return self._id
        meta = self.ws.create(SESSION_NAME)
        self._id = meta.id
        return self._id

    def _find_latest_with_ct(self) -> str | None:
        try:
            dirs = sorted(
                [p for p in self.ws.root.iterdir() if p.is_dir() and (p / "meta.json").exists()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for p in dirs:
                try:
                    meta = self.ws.get(p.name)
                    if "ct" in meta.files:
                        return p.name
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def reset(self) -> str:
        # Keep old sessions on disk for debug; just open a fresh one
        meta = self.ws.create(SESSION_NAME)
        self._id = meta.id
        return self._id

    def path(self) -> Path:
        return self.ws.path(self.ensure())

    def meta(self) -> dict:
        return {
            "id": self.ensure(),
            "files": self.ws.get(self.ensure()).files,
            "status": self.ws.get(self.ensure()).status,
            "last_error": self.ws.get(self.ensure()).last_error,
        }


def run_auto_pipeline(
    session: Session,
    *,
    paint: np.ndarray | None = None,
    hu_min: float = 200.0,
    hu_max: float = 3000.0,
    cell_size: float = 4.0,
    quadratic: bool = True,
    do_material: bool = True,
    grow_mode: str = "local",
    grow_radius_mm: float = 8.0,
    selected_bone_ids: list[int] | None = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    """One-shot: mask (from paint or largest bone) → surface → volume → material.

    on_progress(step_id, percent, message) is called for UI progress.
    """
    log: list[str] = []
    debug: dict[str, Any] = {"steps": {}, "log": log}
    pid = session.ensure()
    ws = session.ws

    def progress(step: str, pct: int, msg: str) -> None:
        log.append(msg)
        if on_progress:
            try:
                on_progress(step, pct, msg)
            except Exception:
                pass

    def fail(msg: str, **extra: Any) -> dict:
        debug["ok"] = False
        debug["error"] = msg
        debug.update(extra)
        log.append(f"ERROR: {msg}")
        if on_progress:
            try:
                on_progress("error", 100, msg)
            except Exception:
                pass
        ws.set_error(pid, msg)
        ws.set_status(pid, state="error", current_step="auto")
        return debug

    try:
        if "ct" not in ws.get(pid).files:
            return fail("No CT loaded. Drop a CT file first.")

        progress("load", 5, "Loading CT…")
        ct = image_io.read_image(ws.resolve(pid, "ct"))
        ct_arr, spacing, origin = image_io.to_numpy(ct)
        progress(
            "load",
            12,
            f"CT loaded {ct_arr.shape} HU=[{ct_arr.min():.0f},{ct_arr.max():.0f}]",
        )
        debug["ct"] = {
            "shape": list(ct_arr.shape),
            "spacing": list(spacing),
            "hu_min": float(ct_arr.min()),
            "hu_max": float(ct_arr.max()),
            "hu_p90": float(np.percentile(ct_arr, 90)),
            "hu_p99": float(np.percentile(ct_arr, 99)),
        }

        # --- mask ---
        # Prefer selected auto-detected bones (checkbox selection)
        mask_img = None
        if selected_bone_ids:
            progress("mask", 18, f"Using selected bones: {selected_bone_ids}")
            mask_img = union_bone_masks(
                ws.outputs(pid),
                selected_bone_ids,
                spacing=spacing,
                origin=origin,
                direction=ct.GetDirection(),
            )
            if mask_img is not None:
                n_fg = int((image_io.to_numpy(mask_img)[0] > 0).sum())
                debug["steps"]["mask"] = {
                    "mode": "selected_bones",
                    "bone_ids": selected_bone_ids,
                    "mask_voxels": n_fg,
                }
                progress("mask", 30, f"Mask from bones {selected_bone_ids}: {n_fg} voxels")
                if n_fg == 0:
                    mask_img = None

        if mask_img is None:
            progress(
                "mask",
                18,
                f"Building mask (mode={grow_mode}, radius={grow_radius_mm}mm)…",
            )
            include = None
            exclude = None
            if paint is not None:
                if paint.shape != ct_arr.shape:
                    return fail(
                        f"Paint shape {paint.shape} ≠ CT {ct_arr.shape}. Reload CT and paint again."
                    )
                include = paint == 1
                exclude = paint == 2
                progress(
                    "mask",
                    20,
                    f"Paint: include={int(include.sum())} exclude={int(exclude.sum())}",
                )
            else:
                progress("mask", 20, "No paint / bones — largest bone component")

            grow = bone_from_paint(
                ct_arr,
                include if include is not None else np.zeros(ct_arr.shape, dtype=bool),
                hu_min=hu_min,
                hu_max=hu_max,
                dilate_seeds=2,
                paint_exclude=exclude,
                mode=grow_mode if include is not None and include.any() else "local",
                grow_radius_mm=grow_radius_mm,
                spacing_xyz=spacing,
            )
            debug["steps"]["mask"] = grow.debug
            progress(
                "mask",
                30,
                f"Mask ready: {grow.debug.get('mask_voxels', 0)} voxels ({grow.debug.get('mode')})",
            )

            if grow.debug.get("error") or int(grow.debug.get("mask_voxels", 0)) == 0:
                return fail(
                    grow.debug.get("error")
                    or "Empty mask — nothing to build a 3D model from.",
                    mask_debug=grow.debug,
                )

            mask_img = image_io.from_numpy(grow.mask, spacing=spacing, origin=origin)
            mask_img.SetDirection(ct.GetDirection())

        mask_rel = "outputs/mask_auto.nrrd"
        image_io.write_image(ws.path(pid) / mask_rel, mask_img)
        ws.set_file(pid, "mask", mask_rel)

        # --- surface ---
        progress("surface", 35, "Generating surface mesh (marching cubes)…")
        opts = SurfaceOptions(
            use_gaussian=True,
            use_polygon_smooth=True,
            pad_slices=2,
            smooth_iterations=20,
        )
        try:
            poly = mask_to_surface(mask_img, opts)
        except Exception as e:
            return fail(f"Surface generation crashed: {e}", traceback=traceback.format_exc())

        n_pts = int(poly.GetNumberOfPoints())
        n_cells = int(poly.GetNumberOfCells())
        debug["steps"]["surface"] = {"n_points": n_pts, "n_cells": n_cells}
        progress("surface", 50, f"Surface: {n_pts} points, {n_cells} cells")

        if n_pts < 4 or n_cells < 1:
            return fail(
                "Surface is empty/too small after marching cubes. "
                "Mask may be too thin or disconnected. Try painting more of the bone "
                "or lower Bone HU min.",
                mask_voxels=grow.debug.get("mask_voxels"),
            )

        progress("surface", 55, "Saving surface + 3D preview…")
        write_stl(ws.path(pid) / "outputs/surface.stl", poly)
        write_vtp(ws.path(pid) / "outputs/surface.vtp", poly)
        ws.set_file(pid, "surface", "outputs/surface.stl")
        # Keep preview small for browser (large JSON breaks job polling)
        preview = surface_to_json(poly, max_points=40_000)
        (ws.path(pid) / "outputs/surface_preview.json").write_text(json.dumps(preview))
        ws.set_file(pid, "surface_preview", "outputs/surface_preview.json")
        debug["steps"]["surface"]["preview_points"] = preview["n_points"]
        debug["steps"]["surface"]["preview_faces"] = preview["n_faces"]
        debug["preview_url"] = "outputs/surface_preview.json"

        # --- volume ---
        progress(
            "volume",
            60,
            f"Volume meshing (tet fill, cell≈{cell_size}mm) — often the slow step…",
        )
        try:
            ug = tetrahedralize(
                poly,
                VolumeOptions(cell_size=cell_size, quadratic=quadratic, backend="delaunay"),
            )
        except Exception as e:
            debug["ok"] = True
            debug["partial"] = True
            debug["error"] = f"Volume mesh failed (surface OK): {e}"
            debug["has_surface"] = True
            debug["surface_preview"] = True
            progress("volume", 90, f"Volume failed (surface OK): {e}")
            ws.set_status(pid, state="partial", current_step="auto")
            return debug

        stats = ug_stats(ug)
        debug["steps"]["volume"] = stats
        progress("volume", 78, f"Volume: {stats['n_cells']} cells, {stats['n_points']} points")

        if stats["n_cells"] == 0:
            debug["ok"] = True
            debug["partial"] = True
            debug["has_surface"] = True
            debug["error"] = (
                "Volume mesher produced 0 tetrahedra (surface is still shown). "
                "Try larger cell size, or the surface may not be watertight."
            )
            progress("volume", 90, debug["error"])
            ws.set_status(pid, state="partial", current_step="auto")
            return debug

        write_vtu(ws.path(pid) / "outputs/volume.vtu", ug)
        ws.set_file(pid, "volume", "outputs/volume.vtu")

        # --- material ---
        if do_material:
            progress("material", 82, "Material mapping (CT density → Young's modulus)…")
            cfg = MatMapConfig()
            cfg.auto_fit_rho_ct = False
            cfg.density.slope = 0.001
            cfg.density.offset = 0.0
            cfg.density.ash_enabled = False
            cfg.density.app_enabled = False
            cfg.do_peel = True
            cfg.number_of_extends = 2
            try:
                result = map_modulus(ug, ct, cfg)
                write_vtu(ws.path(pid) / "outputs/material_mapped.vtu", result.mesh)
                ws.set_file(pid, "mapped", "outputs/material_mapped.vtu")
                export_ascii_ugrid(ws.path(pid) / "outputs/export.txt", result.mesh)
                export_vtu(ws.path(pid) / "outputs/export.vtu", result.mesh)
                ws.set_file(pid, "export_ascii", "outputs/export.txt")
                ws.set_file(pid, "export_vtu", "outputs/export.vtu")
                debug["steps"]["material"] = result.stats
                progress("material", 95, f"Materials mapped: E mean={result.stats.get('E_node_mean', '?')}")
            except Exception as e:
                progress("material", 95, f"Material mapping failed (mesh still OK): {e}")
                debug["steps"]["material"] = {"error": str(e)}

        debug["ok"] = True
        debug["has_surface"] = True
        debug["has_volume"] = stats["n_cells"] > 0
        debug["downloads"] = {
            k: v
            for k, v in ws.get(pid).files.items()
            if k.startswith("export") or k in ("mapped", "volume", "surface", "mask")
        }
        ws.set_error(pid, None)
        ws.set_status(pid, state="done", current_step="auto")
        progress("done", 100, "Pipeline complete")
        return debug

    except Exception as e:
        logger.exception("auto pipeline")
        return fail(str(e), traceback=traceback.format_exc())
