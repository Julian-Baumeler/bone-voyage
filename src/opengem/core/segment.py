"""3D GraphCut segmentation (Boykov–Kolmogorov), MITK-GEM GraphCut3D parity."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

BoundaryDirection = Literal["bidirectional", "bright_to_dark", "dark_to_bright"]


def estimate_graphcut_memory_bytes(shape: tuple[int, ...]) -> int:
    """Rough upper bound: nodes + 6-neighbour edges."""
    n = int(np.prod(shape))
    # ~64 bytes/node + edges
    return n * 80 + n * 6 * 32


def graph_cut_3d(
    image: sitk.Image,
    foreground: sitk.Image,
    background: sitk.Image,
    sigma: float = 50.0,
    boundary_direction: BoundaryDirection = "bidirectional",
    fg_value: int = 255,
    bg_value: int = 0,
) -> sitk.Image:
    """Segment CT volume from FG/BG seed masks (nonzero = seed).

    Edge weight: exp(-(I_i - I_j)^2 / (2 σ^2)), 6-connected neighbourhood.
    Terminals: source for FG seeds, sink for BG seeds (hard constraints).
    """
    try:
        import maxflow  # PyMaxflow
    except ImportError as e:
        raise ImportError(
            "PyMaxflow is required for GraphCut. Install with: pip install PyMaxflow"
        ) from e

    img = sitk.GetArrayFromImage(image).astype(np.float64)  # z,y,x
    fg = sitk.GetArrayFromImage(foreground)
    bg = sitk.GetArrayFromImage(background)

    if img.shape != fg.shape or img.shape != bg.shape:
        raise ValueError(
            f"Shape mismatch: image {img.shape}, fg {fg.shape}, bg {bg.shape}"
        )

    z, y, x = img.shape
    n = z * y * x

    def idx(zi: int, yi: int, xi: int) -> int:
        return zi * (y * x) + yi * x + xi

    g = maxflow.Graph[float]()
    nodes = g.add_nodes(n)

    # 6-connected: +x, +y, +z only (bidirectional edges)
    sigma2 = 2.0 * sigma * sigma
    hard = 1e12

    def n_weight(a: float, b: float) -> float:
        return float(np.exp(-((a - b) ** 2) / sigma2))

    for zi in range(z):
        for yi in range(y):
            for xi in range(x):
                i = idx(zi, yi, xi)
                center = img[zi, yi, xi]
                # +x
                if xi + 1 < x:
                    j = idx(zi, yi, xi + 1)
                    w = n_weight(center, img[zi, yi, xi + 1])
                    _add_edge(g, i, j, w, center, img[zi, yi, xi + 1], boundary_direction)
                # +y
                if yi + 1 < y:
                    j = idx(zi, yi + 1, xi)
                    w = n_weight(center, img[zi, yi + 1, xi])
                    _add_edge(g, i, j, w, center, img[zi, yi + 1, xi], boundary_direction)
                # +z
                if zi + 1 < z:
                    j = idx(zi + 1, yi, xi)
                    w = n_weight(center, img[zi + 1, yi, xi])
                    _add_edge(g, i, j, w, center, img[zi + 1, yi, xi], boundary_direction)

                if fg[zi, yi, xi] > 0:
                    g.add_tedge(nodes[i], hard, 0.0)
                if bg[zi, yi, xi] > 0:
                    g.add_tedge(nodes[i], 0.0, hard)

    flow = g.maxflow()
    logger.info("GraphCut maxflow = %s", flow)

    out = np.zeros_like(img, dtype=np.uint8)
    for zi in range(z):
        for yi in range(y):
            for xi in range(x):
                i = idx(zi, yi, xi)
                if g.get_segment(nodes[i]) == 0:
                    out[zi, yi, xi] = fg_value
                else:
                    out[zi, yi, xi] = bg_value

    result = sitk.GetImageFromArray(out)
    result.CopyInformation(image)
    return result


def _add_edge(
    g,
    i: int,
    j: int,
    w: float,
    center: float,
    neighbor: float,
    direction: BoundaryDirection,
) -> None:
    if direction == "bright_to_dark":
        if center > neighbor:
            g.add_edge(i, j, w, 1.0)
        else:
            g.add_edge(i, j, 1.0, w)
    elif direction == "dark_to_bright":
        if center > neighbor:
            g.add_edge(i, j, 1.0, w)
        else:
            g.add_edge(i, j, w, 1.0)
    else:
        g.add_edge(i, j, w, w)
