"""Auto bone split: two disconnected bright blobs → Bone 1 and Bone 2."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from opengem.core.bones import auto_split_bones
from opengem.core import image_io


def test_two_spheres_two_bones(tmp_path: Path):
    z, y, x = 48, 48, 48
    ct = np.full((z, y, x), -1000.0, dtype=np.float32)
    # sphere A
    zz, yy, xx = np.ogrid[:z, :y, :x]
    ct[(zz - 12) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2 <= 8**2] = 900
    # sphere B (larger, disconnected)
    ct[(zz - 36) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2 <= 10**2] = 900

    img = image_io.from_numpy(ct, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0))
    out = tmp_path / "out"
    result = auto_split_bones(
        img,
        hu_min=200,
        hu_core=400,
        min_voxels=100,
        open_radius=2,
        max_preview_bones=4,
        out_dir=out,
    )
    assert result.n_bones >= 2, result.debug
    assert result.bones[0].name == "Bone 1"
    assert result.bones[1].name == "Bone 2"
    assert result.bones[0].voxels >= result.bones[1].voxels
    assert (out / "bones" / "bone_1_preview.json").exists()
    assert (out / "bones.json").exists()


def test_hysteresis_separates_bridged_bones(tmp_path: Path):
    """Two dense cores linked by medium-HU bridge → still 2 bones."""
    z, y, x = 40, 40, 40
    ct = np.full((z, y, x), -1000.0, dtype=np.float32)
    zz, yy, xx = np.ogrid[:z, :y, :x]
    # two dense cores
    ct[(zz - 10) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2 <= 6**2] = 800
    ct[(zz - 30) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2 <= 6**2] = 800
    # soft bridge that would merge at HU>=200
    ct[10:30, 18:22, 18:22] = 250

    img = image_io.from_numpy(ct, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0))
    result = auto_split_bones(
        img, hu_min=200, hu_core=500, min_voxels=50, open_radius=1, out_dir=tmp_path / "o"
    )
    assert result.n_bones >= 2, result.debug
