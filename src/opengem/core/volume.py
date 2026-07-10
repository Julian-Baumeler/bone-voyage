"""Surface → tetrahedral volume mesh + linear→quadratic conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


@dataclass
class VolumeOptions:
    """Meshing options. cell_size ~ characteristic edge length in mm."""

    cell_size: float = 3.0
    radius_edge_ratio: float = 2.0
    quadratic: bool = True
    backend: str = "auto"  # auto | gmsh | delaunay


def tetrahedralize(
    surface: vtk.vtkPolyData,
    options: VolumeOptions | None = None,
) -> vtk.vtkUnstructuredGrid:
    opts = options or VolumeOptions()
    backend = opts.backend
    if backend == "auto":
        backend = "gmsh" if _gmsh_available() else "delaunay"

    if backend == "gmsh":
        try:
            ug = _mesh_gmsh(surface, opts)
        except Exception:
            ug = _mesh_delaunay(surface, opts)
    else:
        ug = _mesh_delaunay(surface, opts)

    if opts.quadratic:
        ug = tetra_to_quad(ug)
    return ug


def tetra_to_quad(tetra: vtk.vtkUnstructuredGrid) -> vtk.vtkUnstructuredGrid:
    """Port of MITK-GEM MeshHelpers::tetraToQuad (4-node → 10-node tets)."""
    pairs = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
    nodenum = tetra.GetNumberOfPoints()
    edges: dict[tuple[int, int], int] = {}
    quad = vtk.vtkUnstructuredGrid()
    quad.Allocate(tetra.GetNumberOfCells())

    for i in range(tetra.GetNumberOfCells()):
        cell = tetra.GetCell(i)
        tetpts = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        if len(tetpts) < 4:
            continue
        quadpts = [0] * 10
        for j in range(4):
            quadpts[j] = tetpts[j]
        for j, (a, b) in enumerate(pairs):
            pair = (tetpts[a], tetpts[b])
            if pair[0] > pair[1]:
                pair = (pair[1], pair[0])
            if pair not in edges:
                edges[pair] = nodenum
                mid = nodenum
                nodenum += 1
            else:
                mid = edges[pair]
            quadpts[j + 4] = mid
        ids = vtk.vtkIdList()
        for p in quadpts:
            ids.InsertNextId(p)
        quad.InsertNextCell(vtk.VTK_QUADRATIC_TETRA, ids)

    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(tetra.GetNumberOfPoints() + len(edges))
    for i in range(tetra.GetNumberOfPoints()):
        pts.SetPoint(i, tetra.GetPoint(i))
    for (a, b), mid_id in edges.items():
        p1 = np.array(tetra.GetPoint(a))
        p2 = np.array(tetra.GetPoint(b))
        m = 0.5 * (p1 + p2)
        pts.SetPoint(mid_id, float(m[0]), float(m[1]), float(m[2]))
    quad.SetPoints(pts)
    return quad


def write_vtu(path: str | Path, ug: vtk.vtkUnstructuredGrid) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(str(path))
    w.SetInputData(ug)
    w.Write()


def read_vtu(path: str | Path) -> vtk.vtkUnstructuredGrid:
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    return r.GetOutput()


def ug_stats(ug: vtk.vtkUnstructuredGrid) -> dict:
    return {
        "n_points": int(ug.GetNumberOfPoints()),
        "n_cells": int(ug.GetNumberOfCells()),
        "cell_type": _primary_cell_type(ug),
    }


def _primary_cell_type(ug: vtk.vtkUnstructuredGrid) -> str:
    if ug.GetNumberOfCells() == 0:
        return "empty"
    t = ug.GetCellType(0)
    return {
        vtk.VTK_TETRA: "VTK_TETRA",
        vtk.VTK_QUADRATIC_TETRA: "VTK_QUADRATIC_TETRA",
    }.get(t, str(t))


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401

        return True
    except ImportError:
        return False


def _mesh_gmsh(surface: vtk.vtkPolyData, opts: VolumeOptions) -> vtk.vtkUnstructuredGrid:
    import gmsh
    import tempfile

    # Write temporary STL
    with tempfile.TemporaryDirectory() as td:
        stl = str(Path(td) / "surf.stl")
        w = vtk.vtkSTLWriter()
        w.SetFileName(stl)
        w.SetInputData(surface)
        w.Write()

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("opengem")
            gmsh.merge(stl)
            gmsh.model.mesh.classifySurfaces(40 * np.pi / 180, True, True, 180 * np.pi / 180)
            gmsh.model.mesh.createGeometry()
            # volume from surfaces
            s = gmsh.model.getEntities(2)
            sl = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
            gmsh.model.geo.addVolume([sl])
            gmsh.model.geo.synchronize()
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", opts.cell_size * 0.5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", opts.cell_size)
            gmsh.model.mesh.generate(3)
            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            coords = np.array(coords).reshape(-1, 3)
            # map gmsh tags to 0-based
            tag_to_i = {int(t): i for i, t in enumerate(node_tags)}
            etypes, etags, enodes = gmsh.model.mesh.getElements(3)
            ug = vtk.vtkUnstructuredGrid()
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(len(coords))
            for i, c in enumerate(coords):
                pts.SetPoint(i, float(c[0]), float(c[1]), float(c[2]))
            ug.SetPoints(pts)
            ug.Allocate(sum(len(t) for t in etags))
            for etype, nodes in zip(etypes, enodes):
                # 4 = 4-node tetra
                if etype != 4:
                    continue
                n = np.array(nodes, dtype=int).reshape(-1, 4)
                for tet in n:
                    ids = vtk.vtkIdList()
                    for t in tet:
                        ids.InsertNextId(tag_to_i[int(t)])
                    ug.InsertNextCell(vtk.VTK_TETRA, ids)
            if ug.GetNumberOfCells() == 0:
                raise RuntimeError("gmsh produced no tetrahedra")
            return ug
        finally:
            gmsh.finalize()


def _mesh_delaunay(surface: vtk.vtkPolyData, opts: VolumeOptions) -> vtk.vtkUnstructuredGrid:
    """Constrained-ish volume fill: tetrahedralize points of surface + interior samples.

    Not as robust as tetgen/gmsh for complex femur geometry, but works offline
    without extra binaries. Prefer gmsh when available.
    """
    # Refine surface slightly
    tris = vtk.vtkTriangleFilter()
    tris.SetInputData(surface)
    tris.Update()
    surf = tris.GetOutput()

    # Collect surface points
    n = surf.GetNumberOfPoints()
    if n < 4:
        raise ValueError("Surface has too few points for volume meshing")

    pts_np = vtk_to_numpy(surf.GetPoints().GetData())
    bounds = surf.GetBounds()  # xmin,xmax,ymin,ymax,zmin,zmax

    # Interior grid samples inside bounding box, keep those inside closed surface
    select = vtk.vtkSelectEnclosedPoints()
    select.SetSurfaceData(surf)
    select.SetTolerance(1e-6)

    step = max(opts.cell_size, 1e-3)
    xs = np.arange(bounds[0] + step, bounds[1], step)
    ys = np.arange(bounds[2] + step, bounds[3], step)
    zs = np.arange(bounds[4] + step, bounds[5], step)
    interior = []
    # Cap samples for performance
    max_samples = 40_000
    grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    if len(grid) > max_samples:
        rng = np.random.default_rng(0)
        grid = grid[rng.choice(len(grid), size=max_samples, replace=False)]

    probe_pts = vtk.vtkPoints()
    for p in grid:
        probe_pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    probe_pd = vtk.vtkPolyData()
    probe_pd.SetPoints(probe_pts)
    select.SetInputData(probe_pd)
    select.Update()
    inside = select.GetOutput().GetPointData().GetArray("SelectedPoints")
    for i in range(probe_pts.GetNumberOfPoints()):
        if inside.GetTuple1(i) > 0.5:
            interior.append(probe_pts.GetPoint(i))

    all_pts = vtk.vtkPoints()
    for i in range(n):
        all_pts.InsertNextPoint(surf.GetPoint(i))
    for p in interior:
        all_pts.InsertNextPoint(p)

    cloud = vtk.vtkPolyData()
    cloud.SetPoints(all_pts)

    del3 = vtk.vtkDelaunay3D()
    del3.SetInputData(cloud)
    del3.SetTolerance(0.001)
    del3.SetAlpha(0)  # full Delaunay
    del3.BoundingTriangulationOff()
    del3.Update()
    ug = del3.GetOutput()

    # Keep only tets whose centroid is inside the surface
    return _filter_tets_inside(ug, surf)


def _filter_tets_inside(
    ug: vtk.vtkUnstructuredGrid, surface: vtk.vtkPolyData
) -> vtk.vtkUnstructuredGrid:
    if ug.GetNumberOfCells() == 0:
        return ug

    centroids = vtk.vtkPoints()
    for i in range(ug.GetNumberOfCells()):
        cell = ug.GetCell(i)
        c = [0.0, 0.0, 0.0]
        for j in range(cell.GetNumberOfPoints()):
            p = ug.GetPoint(cell.GetPointId(j))
            c[0] += p[0]
            c[1] += p[1]
            c[2] += p[2]
        n = cell.GetNumberOfPoints() or 1
        centroids.InsertNextPoint(c[0] / n, c[1] / n, c[2] / n)

    pd = vtk.vtkPolyData()
    pd.SetPoints(centroids)
    select = vtk.vtkSelectEnclosedPoints()
    select.SetInputData(pd)
    select.SetSurfaceData(surface)
    select.SetTolerance(1e-4)
    select.Update()
    inside = select.GetOutput().GetPointData().GetArray("SelectedPoints")

    out = vtk.vtkUnstructuredGrid()
    out.SetPoints(ug.GetPoints())
    out.Allocate(ug.GetNumberOfCells())
    for i in range(ug.GetNumberOfCells()):
        if inside.GetTuple1(i) > 0.5:
            cell = ug.GetCell(i)
            ids = vtk.vtkIdList()
            for j in range(cell.GetNumberOfPoints()):
                ids.InsertNextId(cell.GetPointId(j))
            out.InsertNextCell(cell.GetCellType(), ids)
    # Remove unused points
    cleaner = vtk.vtkStaticCleanUnstructuredGrid() if hasattr(vtk, "vtkStaticCleanUnstructuredGrid") else None
    if cleaner is not None:
        cleaner.SetInputData(out)
        cleaner.Update()
        return cleaner.GetOutput()
    return out
