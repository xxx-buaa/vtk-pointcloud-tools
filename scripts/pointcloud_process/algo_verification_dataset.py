"""
Description : 算法验证案例的数据集构建

步骤：
1. 读取文件xxx.ply
2. 随机选取80%的点，随机选取两次
3. 一个作为target输出，文件名记为"xxx_target.ply"
4. 一个进行随机刚性变换，输出为source，记为"xxx_source.ply"
5. 记录下该刚性变换的逆变换矩阵，作为ground_truth.txt
"""

import vtk
import numpy as np
import os
import random

def read_ply_with_normals(file_path):
    """读取 PLY 点云，返回 N×6 的数组（xyz + normals）"""
    reader = vtk.vtkPLYReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    
    points = polydata.GetPoints()
    normals = polydata.GetPointData().GetNormals()

    num_points = points.GetNumberOfPoints()
    data = []
    for i in range(num_points):
        p = points.GetPoint(i)
        if normals:
            n = normals.GetTuple(i)
        else:
            n = (0.0, 0.0, 0.0)
        data.append(p + n)
    return np.array(data)

def write_ply_with_normals(points_with_normals, file_path):
    """保存带法向量的PLY点云"""
    points = vtk.vtkPoints()
    normals = vtk.vtkFloatArray()
    normals.SetNumberOfComponents(3)
    normals.SetName("Normals")
    
    for row in points_with_normals:
        points.InsertNextPoint(row[:3])
        normals.InsertNextTuple(row[3:])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetNormals(normals)

    writer = vtk.vtkPLYWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToASCII()
    writer.Write()

def random_sample(points_with_normals, ratio=0.8):
    """随机采样80%的点"""
    N = points_with_normals.shape[0]
    num_sample = int(N * ratio)
    indices = np.random.choice(N, num_sample, replace=False)
    return points_with_normals[indices]

def split_point_cloud(data, source_ratio):
    """将点云按比例划分为 source 和 target，source 随机取，target 为剩下部分"""
    num_points = data.shape[0]
    indices = np.arange(num_points)
    np.random.shuffle(indices)

    split_idx = int(source_ratio * num_points)
    source_idx = indices[:split_idx]
    target_idx = indices[split_idx:]

    source = data[source_idx]
    target = data[target_idx]

    return source, target

def generate_random_rigid_transform(
    angle_range_degrees=(0.5, 5),  # 控制旋转角度范围（单位：度）
    translation_range=0.5         # 控制平移范围（对称范围 [-t, t]）
    ):
    """生成随机旋转 + 平移的刚性变换矩阵 T（4x4）"""
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    # 随机角度（单位：弧度）
    angle_deg = np.random.uniform(*angle_range_degrees)
    angle_rad = np.deg2rad(angle_deg)

    ux, uy, uz = axis
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)

    R = np.array([
        [cos + ux*ux*(1 - cos),    ux*uy*(1 - cos) - uz*sin, ux*uz*(1 - cos) + uy*sin],
        [uy*ux*(1 - cos) + uz*sin, cos + uy*uy*(1 - cos),    uy*uz*(1 - cos) - ux*sin],
        [uz*ux*(1 - cos) - uy*sin, uz*uy*(1 - cos) + ux*sin, cos + uz*uz*(1 - cos)]
    ])

    t = np.random.uniform(-translation_range, translation_range, (3, 1))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T

def apply_manual_rigid_transform(t_path, translation, rotation_degrees, rotation_axis):
    with open(t_path, "w") as f:
        f.write(f"translation: {translation}\n")
        f.write(f"rotation_degrees: {rotation_degrees}\n")
        f.write(f"rotation_axis: {rotation_axis}\n")
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    angle_rad = np.deg2rad(rotation_degrees)
    ux, uy, uz = rotation_axis
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)

    R = np.array([
        [cos + ux**2 * (1 - cos), ux*uy*(1 - cos) - uz*sin, ux*uz*(1 - cos) + uy*sin],
        [uy*ux*(1 - cos) + uz*sin, cos + uy**2*(1 - cos), uy*uz*(1 - cos) - ux*sin],
        [uz*ux*(1 - cos) - uy*sin, uz*uy*(1 - cos) + ux*sin, cos + uz**2*(1 - cos)],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    return T

def apply_rigid_transform(points_with_normals, T):
    """对点和法向量进行刚性变换（点乘 R + t，法向量只乘 R）"""
    R = T[:3, :3]
    t = T[:3, 3]

    transformed = np.empty_like(points_with_normals)
    transformed[:, :3] = points_with_normals[:, :3] @ R.T + t
    transformed[:, 3:] = points_with_normals[:, 3:] @ R.T
    return transformed

def save_matrix_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt="%.6f")
    """
    with open(file_path, "w") as f:
        for row in matrix:
            line = ",".join(f"{val:6f}," for val in row)
            f.write(line + "\n")    
    """

def visualize_two_pointclouds(source, target):
    def numpy_to_vtk_polydata(points_with_normals, color):
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        for i, row in enumerate(points_with_normals):
            pid = points.InsertNextPoint(row[:3])
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(pid)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetPointSize(0.01)
        return actor

    # 配色：source 蓝紫色 #a9b5e5，target 橙黄色 #f1ad2e
    actor_source = numpy_to_vtk_polydata(source, (169/255, 181/255, 229/255))
    actor_target = numpy_to_vtk_polydata(target, (241/255, 173/255, 46/255))

    renderer = vtk.vtkRenderer()

    renderer.AddActor(actor_source)
    renderer.AddActor(actor_target)
    renderer.SetBackground(1.0, 1.0, 1.0)  # 白色背景

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    render_window.Render()    # 添加 XYZ 坐标轴
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(50, 50, 50)  # 坐标轴长度，可以根据点云尺度调整
    axes.AxisLabelsOn()
    axes.SetShaftTypeToCylinder()

    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(interactor)
    widget.SetViewport(0.0, 0.0, 0.2, 0.2)  # 坐标轴在左下角显示
    widget.SetEnabled(1)
    widget.InteractiveOn()

    interactor.Start()

if __name__ == "__main__":
    input_path = "testcase/test0702/RebuiltModels/nefertiti.ply"
    name_prefix = os.path.splitext(os.path.basename(input_path))[0]
    
    # 创建以文件名为名的子目录
    output_path = "testcase/test0702"
    output_dir = os.path.join(output_path, name_prefix)
    os.makedirs(output_dir, exist_ok=True)

    # 刚性变换的参数
    use_manual_tranform = True
    write_file = False
    translation = [7, 0, -7]  # 平移
    rotation_degrees = 75     # 旋转角度
    rotation_axis = [-1, 2, -1.6]  # 绕Z轴

    # 1. 读取点云
    data = read_ply_with_normals(input_path)

    # 2. 随机选取 80% 的点，选两次
    # target = random_sample(data, ratio=0.4)
    # source = random_sample(data, ratio=0.6)
    source, pre_target = split_point_cloud(data, source_ratio=0.4916)
    target = random_sample(pre_target, ratio=0.5328)
    
    # 3. 保存 target 点云
    target_path = os.path.join(output_dir, f"{name_prefix}_target.ply")

    # 4. 对 source 进行刚性变换并保存
    if use_manual_tranform:
        t_path = os.path.join(output_dir, f"{name_prefix}_Tlog.txt")
        T = apply_manual_rigid_transform(t_path, translation, rotation_degrees, rotation_axis)
    else:
        T = generate_random_rigid_transform()

    transformed_source = apply_rigid_transform(source, T)
    source_path = os.path.join(output_dir, f"{name_prefix}_source.ply")

    # 5. 保存逆矩阵
    T_inv = np.linalg.inv(T)
    gt_path = os.path.join(output_dir, f"{name_prefix}_ground_truth.txt")
    
    if write_file:
        write_ply_with_normals(target, target_path)
        write_ply_with_normals(transformed_source, source_path)
        save_matrix_txt(T_inv, gt_path)

    # 7. 输出信息和变换矩阵
    print(f"[✓] Target: {target_path}")
    print(f"[✓] Source (Transformed): {source_path}")
    print(f"[✓] Ground truth inverse matrix: {gt_path}")
    print(T)
    print("\nInverse Rigid Transform Matrix (4x4):")
    print(T_inv)
    print(T @ T_inv)

    # 6. 同屏显示source和target配准效果
    print("\nLaunching visualization window...")
    visualize_two_pointclouds(transformed_source, target)
