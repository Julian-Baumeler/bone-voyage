"""Binary mask → surface mesh (MITK-GEM voxel2mesh parity)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


@dataclass
class SurfaceOptions:
    use_median: bool = False
    median_kernel: int = 3
    use_gaussian: bool = True
    gaussian_std: float = 2.2
    gaussian_radius: float = 0.49
    threshold: float = 127.5
    use_polygon_smooth: bool = True
    smooth_iterations: int = 15
    smooth_relaxation: float = 0.1
    pad_slices: int = 1


def mask_to_surface(
    mask: sitk.Image,
    options: SurfaceOptions | None = None,
) -> vtk.vtkPolyData:
    opts = options or SurfaceOptions()

    img = mask
    if opts.pad_slices > 0:
        from opengem.core.image_io import pad_image

        img = pad_image(img, opts.pad_slices)

    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # z,y,x
    # Ensure binary-ish 0/255 for threshold path
    if arr.max() <= 1.0:
        arr = arr * 255.0

    vtk_img = _numpy_to_vtk_image(arr, img)

    if opts.use_median:
        med = vtk.vtkImageMedian3D()
        med.SetInputData(vtk_img)
        k = opts.median_kernel
        med.SetKernelSize(k, k, k)
        med.Update()
        vtk_img = med.GetOutput()

    if opts.use_gaussian:
        g = vtk.vtkImageGaussianSmooth()
        g.SetInputData(vtk_img)
        g.SetDimensionality(3)
        g.SetStandardDeviations(opts.gaussian_std, opts.gaussian_std, opts.gaussian_std)
        g.SetRadiusFactors(opts.gaussian_radius, opts.gaussian_radius, opts.gaussian_radius)
        g.Update()
        vtk_img = g.GetOutput()

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetValue(0, opts.threshold)
    mc.ComputeNormalsOn()
    mc.Update()
    poly = mc.GetOutput()

    if opts.use_polygon_smooth and poly.GetNumberOfPoints() > 0:
        sm = vtk.vtkSmoothPolyDataFilter()
        sm.SetInputData(poly)
        sm.SetNumberOfIterations(opts.smooth_iterations)
        sm.SetRelaxationFactor(opts.smooth_relaxation)
        sm.FeatureEdgeSmoothingOff()
        sm.BoundarySmoothingOn()
        sm.Update()
        poly = sm.GetOutput()

    # Clean
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.Update()
    return clean.GetOutput()


def write_stl(path: str | Path, poly: vtk.vtkPolyData) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    w = vtk.vtkSTLWriter()
    w.SetFileName(str(path))
    w.SetInputData(poly)
    w.Write()


def write_vtp(path: str | Path, poly: vtk.vtkPolyData) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(str(path))
    w.SetInputData(poly)
    w.Write()


def read_surface(path: str | Path) -> vtk.vtkPolyData:
    path = Path(path)
    if path.suffix.lower() == ".stl":
        r = vtk.vtkSTLReader()
    elif path.suffix.lower() in (".vtp", ".vtk"):
        r = vtk.vtkXMLPolyDataReader() if path.suffix.lower() == ".vtp" else vtk.vtkPolyDataReader()
    else:
        r = vtk.vtkSTLReader()
    r.SetFileName(str(path))
    r.Update()
    return r.GetOutput()


def _numpy_to_vtk_image(arr_zyx: np.ndarray, ref: sitk.Image) -> vtk.vtkImageData:
    z, y, x = arr_zyx.shape
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(x, y, z)
    sp = ref.GetSpacing()  # x,y,z
    org = ref.GetOrigin()
    vtk_img.SetSpacing(sp[0], sp[1], sp[2])
    vtk_img.SetOrigin(org[0], org[1], org[2])
    flat = np.ascontiguousarray(arr_zyx.ravel(order="C"))
    # VTK expects x-fastest; SimpleITK array is z,y,x so ravel C is x-fastest within y within z — correct
    vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_arr.SetName("scalars")
    vtk_img.GetPointData().SetScalars(vtk_arr)
    return vtk_img


def surface_to_json(poly: vtk.vtkPolyData, max_points: int = 200_000) -> dict:
    """Lightweight mesh JSON for Three.js."""
    pts = vtk_to_numpy(poly.GetPoints().GetData()) if poly.GetNumberOfPoints() else np.zeros((0, 3))
    polys = poly.GetPolys()
    faces: list[int] = []
    if polys is not None:
        ids = vtk_to_numpy(polys.GetData())
        i = 0
        while i < len(ids):
            n = int(ids[i])
            if n == 3:
                faces.extend([int(ids[i + 1]), int(ids[i + 2]), int(ids[i + 3])])
            i += n + 1
    if len(pts) > max_points:
        # simple decimation for preview
        dec = vtk.vtkQuadricDecimation()
        dec.SetInputData(poly)
        target = max_points / max(len(pts), 1)
        dec.SetTargetReduction(max(0.0, min(0.95, 1.0 - target)))
        dec.Update()
        return surface_to_json(dec.GetOutput(), max_points=max_points * 2)
    return {
        "points": pts.astype(float).tolist(),
        "faces": faces,
        "n_points": int(len(pts)),
        "n_faces": int(len(faces) // 3),
    }
