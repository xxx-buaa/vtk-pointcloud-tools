import vtk

# 1. 读取 PLY 文件
reader = vtk.vtkPLYReader()
reader.SetFileName("testcase/test0703/RebuiltModels/aquarius.ply")
reader.Update()
polydata = reader.GetOutput()

points = polydata.GetPoints()
num_points = points.GetNumberOfPoints()

# 2. 获取点颜色（RGBA）
colors = polydata.GetPointData().GetScalars()

if colors is None:
    raise ValueError("该点云文件没有包含颜色数据")

# 3. 复制颜色数据并偏移向黄色
new_colors = vtk.vtkUnsignedCharArray()
new_colors.SetNumberOfComponents(4)
new_colors.SetNumberOfTuples(colors.GetNumberOfTuples())
new_colors.SetName("Colors")

# 颜色偏移量（你可以调节）
yellow_weight = 0.5

for i in range(colors.GetNumberOfTuples()):
    r, g, b, a = colors.GetTuple4(i)

    new_r = int((1 - yellow_weight) * r + yellow_weight * 250)
    new_g = int((1 - yellow_weight) * g + yellow_weight * 250)
    new_b = int((1 - yellow_weight) * b + yellow_weight * 0)

    new_colors.SetTuple4(i, new_r, new_g, new_b, a)

# 4. 替换点云中的颜色
# polydata.GetPointData().SetScalars(new_colors)

# 5. 可视化
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(polydata)
# mapper.SetColorModeToDirectScalars()  # 关键：直接使用 RGBA 上色
# mapper.ScalarVisibilityOn()           # 启用标量可见性

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(1, 1, 1)  # 白背景

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

render_window.Render()
interactor.Start()

