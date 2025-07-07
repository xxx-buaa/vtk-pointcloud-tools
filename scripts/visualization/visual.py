"""
Description : ply格式点云的可视化，同时显示每个点的法向量
Bug : 不能显示点，只能显示法向量
"""

import vtk

# 1. 读取PLY文件
ply_reader = vtk.vtkPLYReader()
# ply_reader.SetFileName('src\RoughTheoryModel.ply')
ply_reader.SetFileName('testcase/test0702/RebuiltModels/bird.ply')
ply_reader.Update()

# 2. 获取点云数据
poly_data = ply_reader.GetOutput()

# 3. 提取法向量
normals = poly_data.GetPointData().GetNormals()

# 4. 可视化点云
points = poly_data.GetPoints()
point_cloud = vtk.vtkPolyData()
point_cloud.SetPoints(points)

# 设置点的大小
point_mapper = vtk.vtkPolyDataMapper()
point_mapper.SetInputData(point_cloud)

point_actor = vtk.vtkActor()
point_actor.SetMapper(point_mapper)

# 5. 可视化法向量
# 创建一个箭头符号来表示法向量
arrow_source = vtk.vtkArrowSource()

# 创建Glyph3D对象来表示每个点的法向量
glyph3D = vtk.vtkGlyph3D()
glyph3D.SetInputData(poly_data)
glyph3D.SetSourceConnection(arrow_source.GetOutputPort())
glyph3D.SetVectorModeToUseNormal()  # 使用法向量
glyph3D.SetScaleModeToDataScalingOff()  # 设置固定的法线长度
glyph3D.SetScaleFactor(1)  # 设置箭头大小

# 创建箭头的mapper和actor
glyph_mapper = vtk.vtkPolyDataMapper()
glyph_mapper.SetInputConnection(glyph3D.GetOutputPort())

glyph_actor = vtk.vtkActor()
glyph_actor.SetMapper(glyph_mapper)

# 6. 设置渲染器和可视化器
renderer = vtk.vtkRenderer()
renderer.AddActor(point_actor)
renderer.AddActor(glyph_actor)
renderer.SetBackground(0.1, 0.1, 0.1)  # 设置背景颜色为黑色

# 7. 创建渲染窗口和交互式渲染器
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)
render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

# 8. 启动可视化
render_window.Render()
render_window_interactor.Start()
