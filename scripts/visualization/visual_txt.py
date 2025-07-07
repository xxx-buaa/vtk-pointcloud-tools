"""
Description : txt格式点云文件的可视化
"""

import vtk
import numpy as np

def visualize_xyz_point_cloud(filepath):
    """
    可视化只包含XYZ坐标的TXT点云文件。

    参数:
    filepath (str): TXT点云文件的路径。
    """
    # 1. 读取点云数据
    try:
        points_data = np.loadtxt(filepath)
    except Exception as e:
        print(f"错误: 无法读取文件 '{filepath}' 或文件格式不正确。请确保每行包含XYZ坐标并以空格或逗号分隔。")
        print(f"详细错误: {e}")
        return

    if points_data.shape[1] != 3:
        print(f"错误: 文件 '{filepath}' 中的数据列数不正确。预期3列 (XYZ)，实际 {points_data.shape[1]} 列。")
        return

    # **添加调试信息：打印点云数量**
    num_points = points_data.shape[0]
    print(f"调试信息: 已从文件 '{filepath}' 读取 {num_points} 个点。")

    # 2. 创建VTK点集
    points = vtk.vtkPoints()
    # 3. 创建VTK多边形数据对象
    polydata = vtk.vtkPolyData()

    # 将NumPy数组中的点添加到VTK点集中
    for i in range(points_data.shape[0]):
        points.InsertNextPoint(points_data[i, 0], points_data[i, 1], points_data[i, 2])

    # 将点集设置到多边形数据对象中
    polydata.SetPoints(points)

    # 4. 创建VTK的顶点（Vertex）单元格，每个点一个顶点
    vertices = vtk.vtkCellArray()
    for i in range(points_data.shape[0]):
        vertices.InsertNextCell(1)  # 每个单元格包含1个点
        vertices.InsertCellPoint(i) # 添加点的索引
    polydata.SetVerts(vertices)

    # 5. 创建映射器（Mapper）
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    # 6. 创建演员（Actor）
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # 设置点的颜色（可选）
    actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 红色
    # 设置点的大小（可选）
    actor.GetProperty().SetPointSize(2)

    # 7. 创建渲染器（Renderer）
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)  # 设置背景颜色（深蓝色）

    # 8. 创建渲染窗口（Render Window）
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName("XYZ Point Cloud Visualization")
    render_window.SetSize(800, 600) # 设置窗口大小

    # 9. 创建交互器（Render Window Interactor）
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    # 10. 初始化并启动交互器
    render_window.Render()
    render_window_interactor.Initialize()
    render_window_interactor.Start()

if __name__ == "__main__":
    file_path = "D:\\PythonProjects\\PointCloud\\src\\GX2.txt"

    # 调用函数可视化点云
    visualize_xyz_point_cloud(file_path)