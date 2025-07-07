"""
Description : 同时可视化两个ply文件
"""

import vtk

# 1. 读取PLY文件函数
def read_ply_file(file_path):
    ply_reader = vtk.vtkPLYReader()
    ply_reader.SetFileName(file_path)
    ply_reader.Update()
    return ply_reader.GetOutput()

# 2. 可视化点云函数
def visualize_point_cloud(poly_data, color=(1, 1, 1), scale_factor=1.0):
    points = poly_data.GetPoints()
    point_cloud = vtk.vtkPolyData()
    point_cloud.SetPoints(points)
    
    # 设置点的大小
    point_mapper = vtk.vtkPolyDataMapper()
    point_mapper.SetInputData(point_cloud)
    point_actor = vtk.vtkActor()
    point_actor.SetMapper(point_mapper)
    point_actor.GetProperty().SetColor(color)  # 设置颜色
    
    return point_actor

# 3. 可视化法向量函数
def visualize_normals(poly_data, scale_factor=1.0, color=(1, 1, 1)):
    # 创建箭头符号来表示法向量
    arrow_source = vtk.vtkArrowSource()

    # 创建Glyph3D对象来表示每个点的法向量
    glyph3D = vtk.vtkGlyph3D()
    glyph3D.SetInputData(poly_data)
    glyph3D.SetSourceConnection(arrow_source.GetOutputPort())
    glyph3D.SetVectorModeToUseNormal()  # 使用法向量
    glyph3D.SetScaleModeToDataScalingOff()  # 设置固定的法线长度
    glyph3D.SetScaleFactor(scale_factor)  # 设置箭头大小

    # 创建箭头的mapper和actor
    glyph_mapper = vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyph3D.GetOutputPort())
    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.GetProperty().SetColor(color)  # 设置法向量颜色
    
    return glyph_actor

# 4. 读取target和source PLY文件
target_poly_data = read_ply_file('testcase/test0630/aquarius_cleaned_source.ply')
source_poly_data = read_ply_file('testcase/test0630/aquarius_cleaned_target.ply')

# 5. 创建渲染器
renderer = vtk.vtkRenderer()

# 6. 可视化target点云和法向量
target_point_actor = visualize_point_cloud(target_poly_data, color=(0, 1, 0))  # 绿色
# target_glyph_actor = visualize_normals(target_poly_data, scale_factor=1.0, color=(0.5, 0.8, 1))  # 淡蓝色

# 7. 可视化source点云和法向量
source_point_actor = visualize_point_cloud(source_poly_data, color=(1, 0, 0))  # 红色
# source_glyph_actor = visualize_normals(source_poly_data, scale_factor=1.0, color=(1, 1, 0))  # 黄色

# 8. 将所有的actor添加到渲染器
renderer.AddActor(target_point_actor)
# renderer.AddActor(target_glyph_actor)
renderer.AddActor(source_point_actor)
# renderer.AddActor(source_glyph_actor)

# 9. 设置背景颜色
renderer.SetBackground(0.1, 0.1, 0.1)  # 黑色背景

# 10. 创建渲染窗口
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# 11. 创建交互器
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)
render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

# 12. 启动可视化
render_window.Render()
render_window_interactor.Start()
