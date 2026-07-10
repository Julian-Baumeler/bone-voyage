"""Material mapping: CT → nodal/element Young's modulus (MITK-GEM parity)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from opengem.core.density import BoneDensityParams, PowerLawSet, apply_density, apply_power_law
from opengem.core.matmap_xml import MatMapConfig

logger = logging.getLogger(__name__)


@dataclass
class MappingResult:
    mesh: vtk.vtkUnstructuredGrid
    node_e_name: str = "E"
    cell_e_name: str = "E"
    stats: dict | None = None


def map_modulus(
    mesh: vtk.vtkUnstructuredGrid,
    ct_image: sitk.Image,
    config: MatMapConfig | None = None,
    point_array_name: str = "E",
    cell_array_name: str = "E",
) -> MappingResult:
    """Assign Young's modulus from CT via density + power laws.

    Simplified but faithful pipeline vs MaterialMappingFilter:
    1. Sample HU at each mesh node (linear interp in physical space)
    2. Convert HU → ρ_app → E
    3. Optional peel/extend on a VOI image for surface nodes (new method)
    4. Element E = inverse-distance-to-centroid weighted nodal average
    """
    cfg = config or MatMapConfig()
    dens = cfg.density
    if cfg.auto_fit_rho_ct and len(cfg.calibration_points) >= 2:
        from opengem.core.density import fit_calibration

        hu = [p[0] for p in cfg.calibration_points]
        rho = [p[1] for p in cfg.calibration_points]
        slope, offset = fit_calibration(hu, rho, cfg.calibration_unit)
        dens.slope = slope
        dens.offset = offset
        logger.info("Calibration fit: slope=%s offset=%s", slope, offset)

    ct_arr = sitk.GetArrayFromImage(ct_image).astype(np.float64)  # z,y,x
    spacing = np.array(ct_image.GetSpacing(), dtype=float)  # x,y,z
    origin = np.array(ct_image.GetOrigin(), dtype=float)

    # Optionally build extended E field image for better surface sampling
    e_field = None
    if cfg.do_peel or cfg.number_of_extends > 0:
        e_field = _build_extended_e_field(ct_arr, spacing, origin, mesh, dens, cfg)

    n_pts = mesh.GetNumberOfPoints()
    node_e = np.zeros(n_pts, dtype=float)
    for i in range(n_pts):
        p = np.array(mesh.GetPoint(i))
        if e_field is not None:
            val = _sample_field(e_field, p, spacing, origin)
        else:
            hu = _sample_ct(ct_arr, p, spacing, origin)
            rho = float(dens.evaluate(hu))
            val = float(cfg.power_laws.evaluate(max(rho, 0.0)))
        node_e[i] = max(val, cfg.min_value)

    out = vtk.vtkUnstructuredGrid()
    out.DeepCopy(mesh)

    arr = numpy_to_vtk(node_e, deep=True)
    arr.SetName(point_array_name)
    out.GetPointData().AddArray(arr)
    out.GetPointData().SetActiveScalars(point_array_name)

    cell_e = _nodes_to_elements(out, node_e)
    carr = numpy_to_vtk(cell_e, deep=True)
    carr.SetName(cell_array_name)
    out.GetCellData().AddArray(carr)

    stats = {
        "E_node_min": float(node_e.min()) if len(node_e) else 0,
        "E_node_max": float(node_e.max()) if len(node_e) else 0,
        "E_node_mean": float(node_e.mean()) if len(node_e) else 0,
        "E_cell_mean": float(cell_e.mean()) if len(cell_e) else 0,
        "n_points": int(n_pts),
        "n_cells": int(out.GetNumberOfCells()),
        "density_slope": dens.slope,
        "density_offset": dens.offset,
    }
    return MappingResult(mesh=out, node_e_name=point_array_name, cell_e_name=cell_array_name, stats=stats)


def _sample_ct(
    arr_zyx: np.ndarray,
    point_xyz: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
) -> float:
    """Trilinear sample of CT at physical point."""
    # continuous index
    ix = (point_xyz[0] - origin[0]) / spacing[0]
    iy = (point_xyz[1] - origin[1]) / spacing[1]
    iz = (point_xyz[2] - origin[2]) / spacing[2]
    z, y, x = arr_zyx.shape
    return float(_trilinear(arr_zyx, iz, iy, ix, z, y, x))


def _sample_field(
    field_zyx: np.ndarray,
    point_xyz: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
) -> float:
    return _sample_ct(field_zyx, point_xyz, spacing, origin)


def _trilinear(arr, z, y, x, nz, ny, nx) -> float:
    if z < 0 or y < 0 or x < 0 or z > nz - 1 or y > ny - 1 or x > nx - 1:
        # clamp
        z = min(max(z, 0), nz - 1)
        y = min(max(y, 0), ny - 1)
        x = min(max(x, 0), nx - 1)
    z0 = int(np.floor(z))
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    z1 = min(z0 + 1, nz - 1)
    y1 = min(y0 + 1, ny - 1)
    x1 = min(x0 + 1, nx - 1)
    zd = z - z0
    yd = y - y0
    xd = x - x0
    c00 = arr[z0, y0, x0] * (1 - xd) + arr[z0, y0, x1] * xd
    c01 = arr[z0, y1, x0] * (1 - xd) + arr[z0, y1, x1] * xd
    c10 = arr[z1, y0, x0] * (1 - xd) + arr[z1, y0, x1] * xd
    c11 = arr[z1, y1, x0] * (1 - xd) + arr[z1, y1, x1] * xd
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd
    return float(c0 * (1 - zd) + c1 * zd)


def _nodes_to_elements(mesh: vtk.vtkUnstructuredGrid, node_e: np.ndarray) -> np.ndarray:
    """Inverse-distance-to-centroid weighted average (MaterialMappingFilter::nodesToElements)."""
    n_cells = mesh.GetNumberOfCells()
    out = np.zeros(n_cells, dtype=float)
    for i in range(n_cells):
        cell = mesh.GetCell(i)
        n = cell.GetNumberOfPoints()
        if n == 0:
            continue
        centroid = np.zeros(3)
        pts = []
        ids = []
        for j in range(n):
            pid = cell.GetPointId(j)
            p = np.array(mesh.GetPoint(pid))
            pts.append(p)
            ids.append(pid)
            centroid = (centroid * j + p) / (j + 1)
        dists = []
        for p in pts:
            d = float(np.linalg.norm(p - centroid))
            dists.append(d if d > 0 else 1.0)
        min_d = min(dists)
        weights = [min_d / d for d in dists]
        denom = sum(weights)
        val = sum(w * node_e[pid] for w, pid in zip(weights, ids)) / denom
        out[i] = val
    return out


def _build_extended_e_field(
    ct_zyx: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    mesh: vtk.vtkUnstructuredGrid,
    dens: BoneDensityParams,
    cfg: MatMapConfig,
) -> np.ndarray:
    """Convert CT VOI to E, stencil by mesh surface, peel + extend (new method)."""
    # Full field E from CT
    e = apply_power_law(apply_density(ct_zyx, dens), cfg.power_laws).astype(np.float64)

    # Build binary stencil from mesh surface using SimpleITK/vtk is expensive;
    # approximate: mark voxels near any node as inside via distance ball,
    # better: use vtk PolyDataToImageStencil on extracted surface.
    stencil = _mesh_stencil(ct_zyx.shape, spacing, origin, mesh)
    mask = stencil.copy()

    if cfg.do_peel:
        mask = _peel_mask(mask)

    for _ in range(max(0, cfg.number_of_extends)):
        e, mask = _extend_image(e, mask)

    return e


def _mesh_stencil(
    shape_zyx: tuple[int, int, int],
    spacing: np.ndarray,
    origin: np.ndarray,
    mesh: vtk.vtkUnstructuredGrid,
) -> np.ndarray:
    """Binary mask of voxels inside mesh surface."""
    # Extract surface
    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(mesh)
    geom.Update()
    surf = geom.GetOutput()

    z, y, x = shape_zyx
    # Build vtk image of zeros
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(x, y, z)
    vtk_img.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
    vtk_img.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    vtk_img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    vtk_img.GetPointData().GetScalars().Fill(0)

    pd_to_st = vtk.vtkPolyDataToImageStencil()
    pd_to_st.SetInputData(surf)
    pd_to_st.SetOutputSpacing(vtk_img.GetSpacing())
    pd_to_st.SetOutputOrigin(vtk_img.GetOrigin())
    pd_to_st.SetOutputWholeExtent(vtk_img.GetExtent())
    pd_to_st.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(vtk_img)
    stencil.SetStencilConnection(pd_to_st.GetOutputPort())
    stencil.ReverseStencilOn()
    stencil.SetBackgroundValue(1)
    stencil.Update()
    out = stencil.GetOutput()
    arr = vtk_to_numpy(out.GetPointData().GetScalars()).reshape((z, y, x))
    return (arr > 0).astype(np.uint8)


def _peel_mask(mask: np.ndarray) -> np.ndarray:
    """Erode 3x3x3 then take eroded core (new method peel)."""
    from scipy import ndimage

    eroded = ndimage.binary_erosion(mask.astype(bool), structure=np.ones((3, 3, 3))).astype(np.uint8)
    return eroded


def _extend_image(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance-weighted 3x3x3 extend (MaterialMappingFilter::inplaceExtendImage)."""
    kernel = np.array(
        [
            [
                [1 / np.sqrt(3), 1 / np.sqrt(2), 1 / np.sqrt(3)],
                [1 / np.sqrt(2), 1.0, 1 / np.sqrt(2)],
                [1 / np.sqrt(3), 1 / np.sqrt(2), 1 / np.sqrt(3)],
            ],
            [
                [1 / np.sqrt(2), 1.0, 1 / np.sqrt(2)],
                [1.0, 0.0, 1.0],
                [1 / np.sqrt(2), 1.0, 1 / np.sqrt(2)],
            ],
            [
                [1 / np.sqrt(3), 1 / np.sqrt(2), 1 / np.sqrt(3)],
                [1 / np.sqrt(2), 1.0, 1 / np.sqrt(2)],
                [1 / np.sqrt(3), 1 / np.sqrt(2), 1 / np.sqrt(3)],
            ],
        ],
        dtype=float,
    )
    from scipy import ndimage

    m = mask.astype(float)
    masked = img * m
    conv_img = ndimage.convolve(masked, kernel, mode="constant", cval=0.0)
    conv_mask = ndimage.convolve(m, kernel, mode="constant", cval=0.0)

    out_img = img.copy()
    out_mask = mask.copy()
    grow = (conv_mask > 0) & (mask == 0)
    vals = np.zeros_like(img)
    np.divide(conv_img, conv_mask, out=vals, where=conv_mask > 0)
    # maxval behaviour
    better = grow & (out_img < vals)
    out_img[better] = vals[better]
    also = grow & ~better
    # still mark grown voxels that weren't better? MITK sets value when !mask and conv_mask
    # and if maxval only updates if larger; mask becomes 1 either way
    out_img[grow & ~better] = np.maximum(out_img[grow & ~better], vals[grow & ~better])
    # Actually re-read: if maxval: if imagePoints[i] < val: imagePoints[i]=val; always maskPoints[i]=1
    out_img[grow] = np.where(out_img[grow] < vals[grow], vals[grow], out_img[grow])
    out_mask[grow] = 1
    return out_img, out_mask
