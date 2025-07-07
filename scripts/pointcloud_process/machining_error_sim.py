"""
Description : 模拟加工表面的粗糙度
"""

import vtk
import numpy as np
import random

def ply_to_numpy(polydata):
    points = polydata.GetPoints()
    normals = polydata.GetPointData().GetNormals()
    
    pts_np = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
    nrm_np = np.array([normals.GetTuple(i) for i in range(normals.GetNumberOfTuples())])
    return pts_np, nrm_np

def orthogonal_basis(normal):
    n = normal / np.linalg.norm(normal)
    # 选择一个与 n 不平行的向量作为参考
    if abs(n[0]) < 0.9:
        tmp = np.array([1, 0, 0])
    elif abs(n[1]) < 0.9:
        tmp = np.array([0, 1, 0])
    else:
        tmp = np.array([0, 0, 1])
    
    u = np.cross(n, tmp)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-6:
        # 应急措施，如果仍然为零（极端情况下），返回单位基底
        u = np.array([1, 0, 0])
    else:
        u /= u_norm
    
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v

def apply_surface_wave(pts, nrms, scale=0.05, freq=10):
    new_pts = np.zeros_like(pts)
    magnitudes = np.zeros(len(pts))

    for i, (p, n) in enumerate(zip(pts, nrms)):
        u, v = orthogonal_basis(n)
        # 将局部坐标投影成面上的x, y
        xu = np.dot(p, u)
        yv = np.dot(p, v)
        wave = scale * np.sin(freq * xu) * np.cos(freq * yv)
        offset = u * wave + v * wave  # 双方向扰动
        new_p = p + offset
        new_pts[i] = new_p
        magnitudes[i] = np.linalg.norm(offset)

    return new_pts, magnitudes

def numpy_to_polydata(points, normals, scalars):
    polydata = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_normals = vtk.vtkFloatArray()
    vtk_normals.SetNumberOfComponents(3)
    vtk_normals.SetName("Normals")
    
    vtk_scalars = vtk.vtkFloatArray()
    vtk_scalars.SetName("OffsetMagnitude")

    for i in range(len(points)):
        vtk_points.InsertNextPoint(points[i])
        vtk_normals.InsertNextTuple(normals[i])
        vtk_scalars.InsertNextValue(scalars[i])

    polydata.SetPoints(vtk_points)
    polydata.GetPointData().SetNormals(vtk_normals)
    polydata.GetPointData().SetScalars(vtk_scalars)

    glyphFilter = vtk.vtkVertexGlyphFilter()
    glyphFilter.SetInputData(polydata)
    glyphFilter.Update()
    
    return glyphFilter.GetOutput()

def visualize_modes(original_polydata, deformed_polydata, mode):
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    def create_actor(data, color=None, use_scalar=False):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        if use_scalar:
            mapper.SetScalarModeToUsePointData()
            mapper.ScalarVisibilityOn()
            mapper.SetScalarRange(data.GetPointData().GetScalars().GetRange())
        else:
            mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if color:
            actor.GetProperty().SetColor(color)
        actor.GetProperty().SetPointSize(5)
        return actor

    if mode == "both":
        actor1 = create_actor(original_polydata, (0, 0, 1))
        actor2 = create_actor(deformed_polydata, (1, 0, 0))
        renderer.AddActor(actor1)
        renderer.AddActor(actor2)
    elif mode == "heatmap":
        actor = create_actor(deformed_polydata, use_scalar=True)
        renderer.AddActor(actor)

    renderer.SetBackground(1, 1, 1)
    renderWindow.Render()
    interactor.Start()

if __name__ == "__main__":
    filename = "testcase\cube.ply"
    mode = "heatmap"  # or "both"
    freq = 0.1
    scale = 0.05

    reader = vtk.vtkPLYReader()
    reader.SetFileName(filename)
    reader.Update()
    original = reader.GetOutput()

    pts, nrms = ply_to_numpy(original)

    new_pts, magnitudes = apply_surface_wave(pts, nrms, scale=scale, freq=freq)
    deformed_poly = numpy_to_polydata(new_pts, nrms, magnitudes)

    visualize_modes(original, deformed_poly, mode)
