"""
Description : 对点云文件进行刚性变换，并计算出变换回去的逆变换矩阵
"""

import vtk
import numpy as np

def apply_transform(polydata, translation, rotation_degrees, rotation_axis):
    transform = vtk.vtkTransform()
    transform.RotateWXYZ(rotation_degrees, *rotation_axis)
    transform.Translate(*translation)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    return transform_filter.GetOutput(), transform

def get_inverse_matrix(transform):
    inverse_transform = vtk.vtkTransform()
    inverse_transform.DeepCopy(transform)
    inverse_transform.Inverse()

    matrix_vtk = inverse_transform.GetMatrix()
    inverse_matrix = np.array([[matrix_vtk.GetElement(i, j) for j in range(4)] for i in range(4)])
    return inverse_matrix

def create_actor_from_polydata(polydata, point_size=2, color=(1, 0, 0)):
    vertex_filter = vtk.vtkVertexGlyphFilter()
    vertex_filter.SetInputData(polydata)
    vertex_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(vertex_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(point_size)
    actor.GetProperty().SetColor(color)  # RGB
    return actor

def visualize_side_by_side(actors):
    renderer = vtk.vtkRenderer()
    for actor in actors:
        renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    render_window.Render()
    interactor.Start()

def export_ply(polydata, output_path):
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToASCII()
    writer.Write()

if __name__ == "__main__":
    ply_path = "output/MachinedPartModel_0630.ply"
    output_ply_path = "output/MachinedPartModel_0630_Transformed.ply"

    # 读取点云
    reader = vtk.vtkPLYReader()
    reader.SetFileName(ply_path)
    reader.Update()
    original_polydata = reader.GetOutput()

    # 定义变换
    translation = [6, 1, 4]  # 平移
    rotation_degrees = 1     # 旋转角度
    rotation_axis = [0.4, 0.1, 0.6]  # 绕Z轴

    # 应用变换
    transformed_polydata, transform = apply_transform(original_polydata, translation, rotation_degrees, rotation_axis)

    # 输出逆矩阵
    inverse_matrix = get_inverse_matrix(transform)
    print("逆变换矩阵 (4x4):")
    print(inverse_matrix)

    # 导出PLY
    export_ply(transformed_polydata, output_ply_path)
    print(f"变换后的点云已导出到: {output_ply_path}")

    # 可视化对比
    original_actor = create_actor_from_polydata(original_polydata, point_size=2, color=(0, 0, 1))  # 蓝色
    transformed_actor = create_actor_from_polydata(transformed_polydata, point_size=2, color=(1, 0, 0))  # 红色

    visualize_side_by_side([original_actor, transformed_actor])
