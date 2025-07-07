"""
Description : 仅可视化stl文件
"""

import vtk
from stl import mesh

# 读取STL文件
def read_stl(file_path):
    # 创建STL读取器
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

# 创建可视化渲染器
def visualize_stl(stl_data):
    # 创建映射器
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(stl_data)

    # 创建演员（显示模型）
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # 创建边缘渲染器（渲染三角形的边）
    edge_filter = vtk.vtkExtractEdges()
    edge_filter.SetInputData(stl_data)
    edge_mapper = vtk.vtkPolyDataMapper()
    edge_mapper.SetInputConnection(edge_filter.GetOutputPort())
    
    edge_actor = vtk.vtkActor()
    edge_actor.SetMapper(edge_mapper)
    edge_actor.GetProperty().SetColor(0, 0, 0)  # 黑色边缘

    # 创建一个渲染器，设置背景为黑色
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.7, 0.7, 0.7)  # 黑色背景
    
    # 创建渲染窗口
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    
    # 创建窗口交互器
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    # 将演员添加到渲染器
    renderer.AddActor(actor)
    renderer.AddActor(edge_actor)
    
    # 设置蓝色面
    actor.GetProperty().SetColor(0, 0, 1)  # 蓝色面
    
    # 启动渲染和交互
    render_window.Render()
    render_window_interactor.Start()

# 主程序
if __name__ == "__main__":
    # file_path = 'src\\GX.stl'  # 替换为您的STL文件路径
    file_path = 'mdl/CylinderSegmentReconstruction_0630_Transformed.stl'
    stl_data = read_stl(file_path)
    stl_mesh = mesh.Mesh.from_file(file_path)
    print(f"STL文件中一共有 {len(stl_mesh.vectors)} 个三角面片。")
    visualize_stl(stl_data)
