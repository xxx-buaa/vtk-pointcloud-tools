"""
Description : 将stl转换至点云（.ply或.txt），通过均匀插入点的方式，可自定义点云数量
"""

import vtk
import numpy as np
from stl import mesh

def read_stl(file_path):
    """读取STL文件并返回vtkPolyData对象"""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def render_polydata(polydata):
    """渲染vtkPolyData对象"""
    # 创建Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    # 创建Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # 创建Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)  # 设置背景颜色
    
    # 创建Render Window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    
    # 创建Render Window Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    # 开始渲染
    render_window.Render()
    interactor.Start()

def extract_points_and_normals(polydata):
    """从vtkPolyData中提取点坐标并返回NumPy数组"""
    points_vtk = polydata.GetPoints()
    num_points = points_vtk.GetNumberOfPoints()
    points = np.array([points_vtk.GetPoint(i) for i in range(num_points)])
    
    # 尝试从PolyData获取法向量
    normals_vtk = polydata.GetPointData().GetNormals()
    
    # 如果法向量为空，计算法向量
    if normals_vtk is None:
        normals_vtk = vtk.vtkFloatArray()
        normals_vtk.SetNumberOfComponents(3)
        normals_vtk.SetName("Normals")
        
        # 为每个点初始化法向量为零
        normals = np.zeros((num_points, 3), dtype=np.float64)
        
        # 计算每个三角形的法向量并赋给点
        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                pt_ids = cell.GetPointIds()
                p0 = points_vtk.GetPoint(pt_ids.GetId(0))
                p1 = points_vtk.GetPoint(pt_ids.GetId(1))
                p2 = points_vtk.GetPoint(pt_ids.GetId(2))
                
                # 计算法向量
                v1 = np.array(p1) - np.array(p0)
                v2 = np.array(p2) - np.array(p0)
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)  # 单位化法向量
                
                # 将法向量添加到每个顶点
                for j in range(3):
                    point_id = pt_ids.GetId(j)
                    normals[point_id] += normal
        
        # 单位化每个点的法向量
        for i in range(num_points):
            if np.linalg.norm(normals[i]) > 1e-6:  # 避免除以零
                normals[i] /= np.linalg.norm(normals[i])

        points_with_normals = np.hstack((points, normals))
        # 将法向量添加到vtkPolyData中
        for i in range(num_points):
            normals_vtk.InsertNextTuple(normals[i])

        polydata.GetPointData().SetNormals(normals_vtk)
    
    return points_with_normals

def sample_uniformly(polydata, num_points):
    """
    对vtkPolyData进行均匀采样，返回采样后的点的NumPy数组
    """
    # 获取所有三角形
    triangles = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:
            pt_ids = cell.GetPointIds()
            triangle = [polydata.GetPoint(pt_ids.GetId(j)) for j in range(3)]
            triangles.append(triangle)
    
    triangles = np.array(triangles)  # Shape: (num_triangles, 3, 3)
    
    # 计算每个三角形的面积
    def triangle_area(tri):
        a = tri[1] - tri[0]
        b = tri[2] - tri[0]
        cross = np.cross(a, b)
        return 0.5 * np.linalg.norm(cross)
    
    # 计算所有三角形的面积
    areas = np.array([triangle_area(tri) for tri in triangles])

    # 计算每个三角形的概率
    total_area = np.sum(areas)
    probabilities = areas / total_area
    
    # 选择要采样的三角形索引
    sampled_tri_indices = np.random.choice(len(triangles), size=num_points, p=probabilities)
    
    # 在选定的三角形上均匀采样点
    sampled_points_with_normals = []
    for idx in sampled_tri_indices:
        tri = triangles[idx]
        # 生成两个随机数
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        # 保证采样点均匀分布在三角形上
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        point = tri[0] + r1 * (tri[1] - tri[0]) + r2 * (tri[2] - tri[0])
        
        # 计算该点的法向量（假设所有法向量相同）
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        normal = normal / np.linalg.norm(normal)  # 单位化法向量

        sampled_points_with_normals.append(np.concatenate([point, normal]))
    
    sampled_points_with_normals = np.array(sampled_points_with_normals)
    return sampled_points_with_normals

def random_translate_and_rotate(points_with_normals):
    """对点云数据进行随机平移和旋转操作"""
    # 随机平移
    translation_vector = np.random.uniform(-10, 10, size=3)
    points_with_normals[:, :3] += translation_vector  # 平移点坐标

    # 随机旋转
    angle = np.random.uniform(0, 0.1 * np.pi)  # 随机旋转角度
    axis = np.random.randn(3)  # 随机旋转轴
    axis /= np.linalg.norm(axis)  # 归一化旋转轴
    
    # 计算旋转矩阵 (Rodrigues' rotation formula)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    ux, uy, uz = axis
    rotation_matrix = np.array([
        [cos_angle + ux**2 * (1 - cos_angle), ux * uy * (1 - cos_angle) - uz * sin_angle, ux * uz * (1 - cos_angle) + uy * sin_angle],
        [uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy**2 * (1 - cos_angle), uy * uz * (1 - cos_angle) - ux * sin_angle],
        [uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle, cos_angle + uz**2 * (1 - cos_angle)]
    ])
    
    # 对点和法向量应用旋转
    points_with_normals[:, :3] = points_with_normals[:, :3].dot(rotation_matrix.T)  # 旋转点
    points_with_normals[:, 3:] = points_with_normals[:, 3:].dot(rotation_matrix.T)  # 旋转法向量
    
    return points_with_normals

def save_ply_with_normals(points, file_path):
    """将点云数据保存为TXT文件"""
    with open(file_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]} {point[3]} {point[4]} {point[5]}\n")

def save_xyz_to_txt(points, file_path):
    """
    将点云数据（只包含XYZ坐标）保存为TXT文件。
    points 期望是一个 NumPy 数组，其中每行包含点的 [x, y, z, nx, ny, nz] 信息，
    但此函数只会提取前三个坐标 (x, y, z)。
    """
    with open(file_path, 'w') as f:
        for point in points:
            # 只写入前三个元素 (x, y, z)
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

if __name__ == "__main__":
    # 指定STL文件路径
    # stl_file = "src\\GX1.stl"  # 替换为你的STL文件路径
    stl_file = "D:/PythonProjects/PointCloud/mdl/CylinderSegmentReconstruction_0630_Transformed.stl"
    
    # 读取STL文件
    polydata = read_stl(stl_file)
    
    # 渲染STL模型
    render_polydata(polydata)
    
    # 提取原始点数据和法向量
    original_points_with_normals = extract_points_and_normals(polydata)
    print(f"原始点数量: {original_points_with_normals.shape[0]}")
    
    # 均匀化点云
    desired_num_points = 400000  # 设置所需点数
    uniform_points_with_normals = sample_uniformly(polydata, desired_num_points)
    print(f"均匀化后的点数量: {uniform_points_with_normals.shape[0]}")
    
    # 对点云数据进行随机平移和旋转
    # transformed_points = random_translate_and_rotate(uniform_points_with_normals)
    
    # 保存点云和法向量到PLY文件
    output_ply = "output/CylinderSegmentReconstruction_0630_Transformed.ply"
    save_ply_with_normals(uniform_points_with_normals, output_ply)
    # save_ply_with_normals(transformed_points, output_ply)
    print(f"点云和法向量已保存到 {output_ply}")

    # 保存XYZ坐标到TXT文件
    # output_txt = "src\\GX.txt"
    # save_xyz_to_txt(uniform_points_with_normals, output_txt)
    # print(f"XYZ坐标已保存到 {output_txt}")