"""Synthetic CT sphere → mask → surface → volume → material → export."""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from opengem.core import image_io
from opengem.core.export import export_ascii_ugrid, export_vtu
from opengem.core.material import map_modulus
from opengem.core.matmap_xml import MatMapConfig
from opengem.core.surface import SurfaceOptions, mask_to_surface
from opengem.core.volume import VolumeOptions, tetrahedralize, write_vtu


@pytest.fixture
def synthetic_ct_mask(tmp_path):
    # small volume for speed
    shape = (32, 32, 32)
    zz, yy, xx = np.mgrid[0:32, 0:32, 0:32]
    center = 16
    r = 10
    dist2 = (zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2
    mask = (dist2 <= r**2).astype(np.uint8) * 255
    ct = np.where(mask > 0, 400.0 + (r**2 - dist2) * 2.0, -1000.0).astype(np.float32)
    ct_img = image_io.from_numpy(ct, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0))
    mask_img = image_io.from_numpy(mask, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0))
    ct_path = tmp_path / "ct.nrrd"
    mask_path = tmp_path / "mask.nrrd"
    image_io.write_image(ct_path, ct_img)
    image_io.write_image(mask_path, mask_img)
    return ct_img, mask_img, tmp_path


def test_full_synthetic_pipeline(synthetic_ct_mask):
    ct_img, mask_img, tmp = synthetic_ct_mask
    poly = mask_to_surface(
        mask_img,
        SurfaceOptions(use_gaussian=True, pad_slices=1, use_polygon_smooth=True),
    )
    assert poly.GetNumberOfPoints() > 50

    ug = tetrahedralize(
        poly,
        VolumeOptions(cell_size=3.0, quadratic=True, backend="delaunay"),
    )
    assert ug.GetNumberOfCells() > 0
    assert ug.GetNumberOfPoints() > 4

    cfg = MatMapConfig()
    cfg.auto_fit_rho_ct = False
    cfg.density.slope = 0.001
    cfg.density.offset = 0.0
    cfg.density.ash_enabled = False
    cfg.density.app_enabled = False
    cfg.do_peel = False
    cfg.number_of_extends = 0

    result = map_modulus(ug, ct_img, cfg)
    assert result.stats["n_points"] > 0
    assert result.stats["E_node_max"] >= result.stats["E_node_min"]

    vtu = tmp / "mapped.vtu"
    export_vtu(vtu, result.mesh)
    assert vtu.exists() and vtu.stat().st_size > 100

    txt = tmp / "mapped.txt"
    export_ascii_ugrid(txt, result.mesh)
    text = txt.read_text()
    assert "#BEGIN NODES" in text
    assert "#BEGIN ELEMENTS" in text
