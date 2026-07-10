# Feature parity: MITK-GEM → OpenGEM

| Feature | MITK-GEM | OpenGEM | Notes |
|---------|----------|---------|-------|
| Platform | Qt/MITK desktop | **Web (localhost)** | User pivot |
| GraphCut 3D | maxflow / optional GridCut | PyMaxflow | 6-connected, σ edge weights |
| Surface | median/gauss/MC/smooth | same (VTK) | Padding supported |
| Volume mesh | tetgen + CGAL | gmsh optional + Delaunay fallback | Not bit-identical connectivity |
| Quadratic tets | tetraToQuad | ported | 10-node |
| Material map | peel/extend, power laws | ported (new extend method) | `.matmap` XML load/save |
| ASCII ugrid | yes | yes | Node/element layout |
| Ansys / Abaqus / LS-DYNA | Matlab/Python scripts | built-in export | Simplified but usable |
| Paint seeds in GUI | MITK tools | upload seed masks (v1) | Interactive brush = later |
| DICOM browser | MITK | NIfTI/NRRD first | DICOM zip later |

## Intentional deltas

- Default volume mesher may be Delaunay interior sampling if gmsh is not installed.
- Interactive brush painting is not in v0.1 (upload masks/seeds instead).
- UG visualization is Three.js surface preview, not full VTU scalar coloring yet.
