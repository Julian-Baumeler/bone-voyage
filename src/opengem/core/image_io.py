"""Image load/save via SimpleITK."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_image(path: str | Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def write_image(path: str | Path, image: sitk.Image) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def to_numpy(image: sitk.Image) -> tuple[np.ndarray, tuple[float, ...], tuple[float, ...]]:
    """Return (array zyx or whatever sitk uses, spacing xyz, origin xyz)."""
    arr = sitk.GetArrayFromImage(image)  # z, y, x
    spacing = tuple(float(s) for s in image.GetSpacing())  # x, y, z
    origin = tuple(float(o) for o in image.GetOrigin())
    return arr, spacing, origin


def from_numpy(
    arr: np.ndarray,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    origin: tuple[float, ...] = (0.0, 0.0, 0.0),
    direction: tuple[float, ...] | None = None,
) -> sitk.Image:
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    if direction is not None:
        img.SetDirection(direction)
    return img


def resample(
    image: sitk.Image,
    scale: float = 0.5,
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    """Down/upsample by isotropic scale factor on spacing (scale < 1 = fewer voxels)."""
    old_size = np.array(image.GetSize(), dtype=float)
    old_spacing = np.array(image.GetSpacing(), dtype=float)
    new_spacing = old_spacing / scale
    new_size = [int(round(s * scale)) for s in old_size]
    new_size = [max(1, s) for s in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(s) for s in new_spacing))
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def pad_image(image: sitk.Image, slices: int = 1) -> sitk.Image:
    """Add empty border slices so marching cubes can close surfaces."""
    arr = sitk.GetArrayFromImage(image)
    pad = int(slices)
    padded = np.pad(arr, pad_width=pad, mode="constant", constant_values=0)
    out = sitk.GetImageFromArray(padded)
    spacing = image.GetSpacing()
    origin = list(image.GetOrigin())
    # shift origin so physical coordinates stay consistent
    origin[0] -= pad * spacing[0]
    origin[1] -= pad * spacing[1]
    origin[2] -= pad * spacing[2]
    out.SetSpacing(spacing)
    out.SetOrigin(tuple(origin))
    out.SetDirection(image.GetDirection())
    return out


def crop_to_bounds(
    image: sitk.Image,
    min_index: tuple[int, int, int],
    max_index: tuple[int, int, int],
) -> sitk.Image:
    size = [max_index[i] - min_index[i] + 1 for i in range(3)]
    return sitk.RegionOfInterest(image, size, list(min_index))
