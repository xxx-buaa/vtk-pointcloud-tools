import vtk

# 打开PLY文件并打印前10行
ply_file_path = 'testcase/test0703/RebuiltModels/aquarius.ply'

with open(ply_file_path, 'r', encoding='latin1') as f:
    for i in range(20):
        line = f.readline()
        print(line.strip())

# 1. 创建读取器
reader = vtk.vtkPLYReader()
reader.SetFileName(ply_file_path)
reader.Update()

# 2. 获取点数据
polydata = reader.GetOutput()
points = polydata.GetPoints()
num_points = points.GetNumberOfPoints()

# 3. 打印前10个点（或最多num_points个）
for i in range(min(10, num_points)):
    x, y, z = points.GetPoint(i)
    print(f"Point {i}: ({x}, {y}, {z})")

    # 获取点颜色
    colors = polydata.GetPointData().GetScalars()
    if colors:
        r, g, b, a = colors.GetTuple4(0)  # 第一个点的颜色
        print(f"Color of point 0: ({r}, {g}, {b}, {a})")

    # 获取法向量
    normals = polydata.GetPointData().GetNormals()
    if normals:
        nx, ny, nz = normals.GetTuple3(0)
        print(f"Normal of point 0: ({nx}, {ny}, {nz})")
