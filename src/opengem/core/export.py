"""Export meshes to FE solver formats (VTU, ASCII ugrid, Ansys, Abaqus, LS-DYNA)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from opengem.core.volume import write_vtu


def export_vtu(path: str | Path, mesh: vtk.vtkUnstructuredGrid) -> Path:
    path = Path(path)
    write_vtu(path, mesh)
    return path


def export_ascii_ugrid(
    path: str | Path,
    mesh: vtk.vtkUnstructuredGrid,
    point_e_name: str = "E",
    cell_e_name: str = "E",
) -> Path:
    """MITK-GEM AsciiUgridFileWriterService layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_pts = mesh.GetNumberOfPoints()
    n_cells = mesh.GetNumberOfCells()
    point_e = _get_array(mesh.GetPointData(), point_e_name, n_pts)
    cell_e = _get_array(mesh.GetCellData(), cell_e_name, n_cells)

    # detect nodes per cell
    ppc = 4
    if n_cells:
        t = mesh.GetCellType(0)
        if t == vtk.VTK_QUADRATIC_TETRA:
            ppc = 10
        elif t == vtk.VTK_TETRA:
            ppc = 4

    with path.open("w") as f:
        f.write("#BEGIN NODES\n")
        f.write("#COMMENT Structure: node_number, x, y, z, TC\n")
        f.write("#COMMENT TC is the Young´s moduli at the nodes for method C.\n")
        for i in range(n_pts):
            x, y, z = mesh.GetPoint(i)
            f.write(f"{i + 1}, {x:12.4f}, {y:12.4f}, {z:12.4f}, {point_e[i]:12.4f}\n")
        f.write("#END NODES\n")

        f.write(f"#BEGIN ELEMENTS {ppc}\n")
        f.write(f"#COMMENT Structure: elem_nr, n1, ... , n{ppc}, EA, EB\n")
        f.write("#COMMENT EA, EB are the Young´s moduli at the elements for method A and B respectively.\n")
        for i in range(n_cells):
            cell = mesh.GetCell(i)
            ids = [cell.GetPointId(j) + 1 for j in range(cell.GetNumberOfPoints())]
            # pad if needed
            while len(ids) < ppc:
                ids.append(ids[-1] if ids else 1)
            e = cell_e[i]
            f.write(f"{i + 1}, " + ", ".join(str(x) for x in ids[:ppc]) + f", {e:12.4f}, {e:12.4f}\n")
        f.write(f"#END ELEMENTS {ppc}\n")

        # surface section (optional simplified)
        f.write("#BEGIN SURFACE\n")
        f.write("#COMMENT Structure: element_number, n1, n2, n3, n4, n5, n6\n")
        f.write("#END SURFACE\n")
    return path


def export_abaqus(
    path: str | Path,
    mesh: vtk.vtkUnstructuredGrid,
    cell_e_name: str = "E",
    poisson: float = 0.3,
    density_scale: float = 1.0,
) -> Path:
    """Write a simple Abaqus .inp with solid tets and *ELASTIC materials binned by E."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_pts = mesh.GetNumberOfPoints()
    n_cells = mesh.GetNumberOfCells()
    cell_e = _get_array(mesh.GetCellData(), cell_e_name, n_cells)

    # bin materials
    n_bins = min(50, max(1, n_cells))
    if n_cells == 0:
        bins = np.array([1.0])
        digitized = np.array([], dtype=int)
    else:
        emax = max(float(cell_e.max()), 1.0)
        bins = np.linspace(1.0, emax, n_bins)
        digitized = np.digitize(cell_e, bins, right=True)
        digitized = np.clip(digitized, 0, len(bins) - 1)

    eltype = "C3D10" if (n_cells and mesh.GetCellType(0) == vtk.VTK_QUADRATIC_TETRA) else "C3D4"

    with path.open("w") as f:
        f.write("*HEADING\nOpenGEM export\n")
        f.write("*NODE\n")
        for i in range(n_pts):
            x, y, z = mesh.GetPoint(i)
            f.write(f"{i + 1}, {x:.6f}, {y:.6f}, {z:.6f}\n")
        f.write(f"*ELEMENT, TYPE={eltype}, ELSET=BONE\n")
        for i in range(n_cells):
            cell = mesh.GetCell(i)
            ids = [str(cell.GetPointId(j) + 1) for j in range(cell.GetNumberOfPoints())]
            f.write(f"{i + 1}, " + ", ".join(ids) + "\n")
        # materials
        used = sorted(set(int(d) for d in digitized)) if n_cells else []
        for bi in used:
            mid = bi + 1
            e_val = float(bins[bi])
            f.write(f"*MATERIAL, NAME=MAT{mid}\n")
            f.write("*ELASTIC\n")
            f.write(f"{e_val:.6e}, {poisson}\n")
            f.write(f"*SOLID SECTION, ELSET=BONE, MATERIAL=MAT{mid}\n")
    return path


def export_ansys(
    path: str | Path,
    mesh: vtk.vtkUnstructuredGrid,
    cell_e_name: str = "E",
    poisson: float = 0.3,
) -> Path:
    """Minimal Ansys APDL snippet (nodes + et + e + mp)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_pts = mesh.GetNumberOfPoints()
    n_cells = mesh.GetNumberOfCells()
    cell_e = _get_array(mesh.GetCellData(), cell_e_name, n_cells)
    quadratic = n_cells and mesh.GetCellType(0) == vtk.VTK_QUADRATIC_TETRA

    with path.open("w") as f:
        f.write("/PREP7\n")
        f.write(f"ET,1,{'187' if quadratic else '285'}\n")  # SOLID187 / SOLID285
        for i in range(n_pts):
            x, y, z = mesh.GetPoint(i)
            f.write(f"N,{i + 1},{x:.6f},{y:.6f},{z:.6f}\n")
        for i in range(n_cells):
            cell = mesh.GetCell(i)
            ids = [cell.GetPointId(j) + 1 for j in range(cell.GetNumberOfPoints())]
            e_val = float(cell_e[i]) if n_cells else 1000.0
            mat_id = i + 1
            f.write(f"MP,EX,{mat_id},{e_val:.6e}\n")
            f.write(f"MP,PRXY,{mat_id},{poisson}\n")
            f.write(f"MAT,{mat_id}\n")
            if quadratic and len(ids) >= 10:
                f.write(
                    "EN,"
                    + str(i + 1)
                    + ","
                    + ",".join(str(x) for x in ids[:10])
                    + "\n"
                )
            else:
                f.write(
                    "EN,"
                    + str(i + 1)
                    + ","
                    + ",".join(str(x) for x in ids[:4])
                    + "\n"
                )
        f.write("FINISH\n")
    return path


def export_lsdyna(
    path_mesh: str | Path,
    path_mats: str | Path,
    mesh: vtk.vtkUnstructuredGrid,
    cell_e_name: str = "E",
    offset: int = 0,
    poisson: float = 0.3,
    n_bins: int = 500,
) -> tuple[Path, Path]:
    """LS-DYNA keyword mesh + materials (from VtuToKfile_example.py)."""
    path_mesh = Path(path_mesh)
    path_mats = Path(path_mats)
    path_mesh.parent.mkdir(parents=True, exist_ok=True)

    n_pts = mesh.GetNumberOfPoints()
    n_cells = mesh.GetNumberOfCells()
    cell_e = _get_array(mesh.GetCellData(), cell_e_name, n_cells)
    if n_cells:
        bins = np.linspace(1.0, max(float(cell_e.max()), 1.0), n_bins)
        digitized = np.digitize(cell_e, bins)
        digitized = np.clip(digitized, 1, n_bins)
    else:
        bins = np.array([1.0])
        digitized = np.array([], dtype=int)

    with path_mesh.open("w") as out_mesh:
        out_mesh.write("*KEYWORD\n*NODE\n")
        for k in range(n_pts):
            p = mesh.GetPoint(k)
            nnum = k + 1 + offset
            out_mesh.write(f"{nnum:8d}{p[0]:16.6f}{p[1]:16.6f}{p[2]:16.6f}{0:8d}{0:8d}\n")
        out_mesh.write("*ELEMENT_SOLID\n")
        for k in range(n_cells):
            c = mesh.GetCell(k)
            enum = k + 1 + offset
            pid = int(digitized[k]) + offset
            out_mesh.write(f"{enum:8d}{pid:8d}\n")
            ids = [c.GetPointId(j) + 1 + offset for j in range(c.GetNumberOfPoints())]
            # pad to 8 for solid format simplicity
            while len(ids) < 8:
                ids.append(ids[-1])
            line = "".join(f"{i:8d}" for i in ids[:8])
            out_mesh.write(line + "\n")
        out_mesh.write("*END\n")

    used = sorted(set(int(d) for d in digitized)) if n_cells else []
    with path_mats.open("w") as out_mats:
        for k in used:
            ID = k
            # mean stiffness in bin
            e_vals = cell_e[digitized == k]
            stiff = float(e_vals.mean()) if len(e_vals) else float(bins[min(k - 1, len(bins) - 1)])
            # reverse power-law-ish density estimate from original script
            dens = (stiff / 6850.0) ** (1 / 1.49) if stiff > 0 else 0.0
            out_mats.write("*PART\n")
            out_mats.write(f"{ID:10d}{ID:10d}{ID:10d}{' ':10}{' ':10}{0:10d}{0:10d}\n")
            out_mats.write("*SECTION_SOLID\n")
            out_mats.write(f"{ID:10d}{16:10d}\n")
            out_mats.write("*MAT_ELASTIC\n")
            out_mats.write(f"{ID:10d}{dens:10.4E}{stiff:10.4E}{poisson:10.4f}{0:10d}{0:10d}{0:10d}\n")
        out_mats.write("*END\n")
    return path_mesh, path_mats


def _get_array(data, name: str, n: int) -> np.ndarray:
    if n == 0:
        return np.zeros(0)
    a = data.GetArray(name)
    if a is None:
        # try first array
        if data.GetNumberOfArrays() > 0:
            a = data.GetArray(0)
        else:
            return np.zeros(n)
    return vtk_to_numpy(a).astype(float).reshape(-1)[:n]
