import vtk

from opengem.core.volume import tetra_to_quad


def test_tetra_to_quad_counts():
    # single tet
    ug = vtk.vtkUnstructuredGrid()
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(0, 0, 0)
    pts.InsertNextPoint(1, 0, 0)
    pts.InsertNextPoint(0, 1, 0)
    pts.InsertNextPoint(0, 0, 1)
    ug.SetPoints(pts)
    ids = vtk.vtkIdList()
    for i in range(4):
        ids.InsertNextId(i)
    ug.InsertNextCell(vtk.VTK_TETRA, ids)

    q = tetra_to_quad(ug)
    assert q.GetNumberOfCells() == 1
    assert q.GetCellType(0) == vtk.VTK_QUADRATIC_TETRA
    # 4 corners + 6 mids = 10
    assert q.GetNumberOfPoints() == 10
