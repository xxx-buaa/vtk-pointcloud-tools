"""
Description : 25/07/03 最新修改
增加了让模型颜色进行偏移的操作
针对完整包含x, y, z, nx, ny, nz, r, g, b, a的模型来说
"""

import vtk
import numpy as np
import os
import struct
import random
import heapq
from sklearn.neighbors import NearestNeighbors
     


def read_ply_with_all_data(file_path):
    """读取 PLY 点云，返回 N×6 的数组（xyz + normals）"""
    reader = vtk.vtkPLYReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    
    points = polydata.GetPoints()
    normals = polydata.GetPointData().GetNormals()
    colors = polydata.GetPointData().GetScalars() # Scalars usually hold color data

    num_points = points.GetNumberOfPoints()
    data = []

    for i in range(num_points):
        # XYZ坐标
        p = points.GetPoint(i)
        if normals:
            n = normals.GetTuple(i)
        else:
            n = (0.0, 0.0, 0.0)
        if colors:
            c = colors.GetTuple(i)
        else:
            c = (255.0, 255.0, 255.0, 255.0)
        data.append(p + n + c)
    return np.array(data)

def write_ply_with_all_data(points_data, file_path, shift_color_to=None):
    """
    Saves PLY point cloud data including xyz, normals, and colors.
    points_data should be an N x 10 numpy array (x, y, z, nx, ny, nz, r, g, b, a).
    Optionally applies a color shift.
    """    
    current_colors = np.copy(points_data[:, 6:10]).astype(np.float32) # Ensure float for calculations

    yellow_weight = 0.9 # This weight is for the target color shift, adjust as needed

    if shift_color_to:
        for i in range(current_colors.shape[0]):
            r, g, b, a = current_colors[i]

            # 应用黄色偏移，并裁剪到 [0, 255]
            new_r = np.clip((1 - yellow_weight) * r + yellow_weight * shift_color_to[0], 0, 255)
            new_g = np.clip((1 - yellow_weight) * g + yellow_weight * shift_color_to[1], 0, 255)
            new_b = np.clip((1 - yellow_weight) * b + yellow_weight * shift_color_to[2], 0, 255)
            
            current_colors[i] = (new_r, new_g, new_b, a) # Alpha remains unchanged

    points_data[:, 6:10] = current_colors.astype(np.uint8)

    with open(file_path, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(b"comment VCGLIB generated\n")
        f.write(f"element vertex {len(points_data)}\n".encode("utf-8"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(b"property uchar alpha\n")
        f.write(b"element face 0\n")
        f.write(b"property list uchar int vertex_indices\n")
        f.write(b"end_header\n")

        for row in points_data:
            # 写入 float: x, y, z, nx, ny, nz
            f.write(struct.pack('<6f', *row[0:6]))  # 小端 float
            # 写入 uchar: r, g, b, a
            f.write(struct.pack('<4B', int(row[6]), int(row[7]), int(row[8]), int(row[9])))

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
    # np.random.shuffle(indices)

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

def apply_rigid_transform(points_data, T):
    """对点和法向量进行刚性变换（点乘 R + t，法向量只乘 R）"""
    R = T[:3, :3]
    t = T[:3, 3]

    transformed = np.empty_like(points_data)
    transformed[:, :3] = points_data[:, :3] @ R.T + t
    transformed[:, 3:6] = points_data[:, 3:6] @ R.T
    transformed[:, 6:10] = points_data[:, 6:10] 
    return transformed

def save_matrix_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt="%.6f")
    """
    with open(file_path, "w") as f:
        for row in matrix:
            line = ",".join(f"{val:6f}," for val in row)
            f.write(line + "\n")    
    """

def visualize_two_pointclouds_old(source, target):
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

def visualize_two_pointclouds_new(source_data, target_data):
    """
    Visualizes two point clouds using their embedded colors.
    source_data and target_data should be N x 10 numpy arrays.
    """
    def numpy_to_vtk_polydata(points_data):
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(4) # RGBA
        colors.SetName("Colors")

        for i, row in enumerate(points_data):
            pid = points.InsertNextPoint(row[:3]) # XYZ
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(pid)
            
            # Add RGBA color
            r, g, b, a = row[6:10].astype(int)
            colors.InsertNextTuple4(r, g, b, a)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
        polydata.GetPointData().SetScalars(colors) # Set colors as scalars

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # Point size for visualization
        actor.GetProperty().SetPointSize(1.0) # Adjust point size as needed for better visibility
        return actor

    # Create actors for source and target using their embedded colors
    actor_source = numpy_to_vtk_polydata(source_data)
    actor_target = numpy_to_vtk_polydata(target_data)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor_source)
    renderer.AddActor(actor_target)
    renderer.SetBackground(1.0, 1.0, 1.0)  # White background

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    # Add XYZ coordinate axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(50, 50, 50)  # Axis length, adjust based on point cloud scale
    axes.AxisLabelsOn()
    axes.SetShaftTypeToCylinder()

    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(interactor)
    widget.SetViewport(0.0, 0.0, 0.2, 0.2)  # Axes displayed in the bottom-left corner
    widget.SetEnabled(1)
    widget.InteractiveOn()

    render_window.Render()
    interactor.Start()

def visualize_with_toggle_from_files(target_data, source_data, transformed_source_data):
    """
    通过读取文件可视化点云，通过按键'1'和'2'切换显示：
    - '1': 显示 target 和 source
    - '2': 显示 target 和 transformed_source
    """  
    def numpy_to_vtk_polydata(points_data):
        """与visualize_two_pointclouds_new中的实现完全一致"""
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(4) # RGBA
        colors.SetName("Colors")

        for i in range(points_data.shape[0]):
            row = points_data[i]
            pid = points.InsertNextPoint(row[:3]) # XYZ
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(pid)
            
            # 使用点云自带的RGBA颜色
            r, g, b, a = row[6:10].astype(int)
            colors.InsertNextTuple4(r, g, b, a)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
        polydata.GetPointData().SetScalars(colors) # Set colors as scalars

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # 与原始函数相同的点大小设置
        actor.GetProperty().SetPointSize(1.0)
        return actor

    # 创建渲染器和窗口
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)  # 白色背景
    
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # 创建点云actor - 使用与原始函数相同的创建方法
    target_actor = numpy_to_vtk_polydata(target_data)
    source_actor = numpy_to_vtk_polydata(source_data)
    transformed_source_actor = numpy_to_vtk_polydata(transformed_source_data)
    
    # 初始状态：显示target和source
    renderer.AddActor(source_actor)
    renderer.AddActor(target_actor)
    transformed_source_actor.SetVisibility(False)  # 初始隐藏
    
    # 添加坐标轴 - 与原始函数相同
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(50, 50, 50)
    axes.AxisLabelsOn()
    axes.SetShaftTypeToCylinder()

    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(interactor)
    widget.SetViewport(0.0, 0.0, 0.2, 0.2)
    widget.SetEnabled(1)
    widget.InteractiveOn()
    
    # 键盘回调函数
    def keypress_callback(obj, event):
        key = interactor.GetKeySym()
        if key == '1':
            source_actor.SetVisibility(True)
            transformed_source_actor.SetVisibility(False)
            print("Showing: Target and Source")
        elif key == '2':
            source_actor.SetVisibility(False)
            transformed_source_actor.SetVisibility(True)
            print("Showing: Target and Transformed Source")
        render_window.Render()
    
    # 设置交互器样式 - 与原始函数相同
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    interactor.AddObserver("KeyPressEvent", keypress_callback)
    
    # 添加变换后的源点云到渲染器（但初始不可见）
    renderer.AddActor(transformed_source_actor)
    
    print("\n=== Visualization Controls ===")
    print("Press '1': Show Target and Source")
    print("Press '2': Show Target and Transformed Source")
    
    # 开始交互
    render_window.Render()
    interactor.Start()

def apply_local_deformation(points_data, deformation_ratio=0.2, displacement=0.1, seed_index=None, smooth_radius=0.5):
    """
    对点云中的连续区域应用法向位移，位移量在边界处平滑过渡
    :param points_data: 点云数据 (N x 10)
    :param deformation_ratio: 需要变形的点比例
    :param displacement: 最大位移大小
    :param seed_index: 种子点的索引 (如果为None则随机选择)
    :param smooth_radius: 平滑过渡区域的半径比例 (0.0-1.0)
    :return: 变形后的点云
    """
    n_points = points_data.shape[0]
    n_deform = int(n_points * deformation_ratio)
    
    if n_deform == 0:
        return points_data.copy()
    
    # 提取坐标和法向量
    coords = points_data[:, :3]
    normals = points_data[:, 3:6]
    
    # 使用k-d树加速近邻搜索
    nn = NearestNeighbors(n_neighbors=50, algorithm='kd_tree').fit(coords)
    
    # 选择种子点
    if seed_index is None:
        seed_index = np.random.randint(0, n_points)
    
    # 区域生长算法选择连续区域
    visited = np.zeros(n_points, dtype=bool)
    region_indices = []
    distances = np.full(n_points, np.inf)  # 存储到种子点的距离
    queue = []
    
    # 初始化种子点
    heapq.heappush(queue, (0.0, seed_index))
    visited[seed_index] = True
    distances[seed_index] = 0.0
    
    while queue and len(region_indices) < n_deform:
        dist, idx = heapq.heappop(queue)
        region_indices.append(idx)
        
        # 如果已达到目标点数，退出
        if len(region_indices) >= n_deform:
            break
        
        # 查找当前点的近邻
        current_point = coords[idx]
        _, neighbor_indices = nn.kneighbors(current_point.reshape(1, -1))
        neighbor_indices = neighbor_indices[0]
        
        # 添加未访问的近邻点
        for neighbor_idx in neighbor_indices:
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                # 计算到种子点的距离（使用欧氏距离）
                neighbor_dist = np.linalg.norm(coords[neighbor_idx] - coords[seed_index])
                distances[neighbor_idx] = neighbor_dist
                heapq.heappush(queue, (neighbor_dist, neighbor_idx))
    
    # 计算区域内的最大距离
    max_dist = np.max(distances[region_indices])
    min_dist = np.min(distances[region_indices])
    
    # 创建点云副本
    deformed_data = points_data.copy()
    
    # 对选定区域应用位移（带距离衰减）
    for idx in region_indices:
        normal = normals[idx]
        # 确保法向量是单位向量
        norm_normal = normal / np.linalg.norm(normal)
        
        # 计算点到种子点的归一化距离 (0-1)
        norm_dist = (distances[idx] - min_dist) / (max_dist - min_dist + 1e-8)
        
        # 计算距离衰减权重（使用高斯函数）
        # 当 norm_dist 接近 0（中心）时权重接近 1，接近 1（边界）时权重接近 0
        weight = np.exp(-(norm_dist)**2 / (2 * smooth_radius**2))
        
        # 应用带权重的位移
        deformed_data[idx, :3] += weight * displacement * norm_normal
    
    print(f"Applied deformation to {len(region_indices)} points (seed index: {seed_index})")
    return deformed_data

def apply_region_drop(points_data, drop_ratio=0.2, seed_index=None, visualize=False):
    """
    在点云中的连续区域丢弃点
    :param points_data: 点云数据 (N x 10)
    :param drop_ratio: 需要丢弃的点比例
    :param seed_index: 种子点的索引 (如果为None则随机选择)
    :param visualize: 是否可视化丢弃区域 (用于调试)
    :return: 丢弃后的点云
    """
    n_points = points_data.shape[0]
    n_drop = int(n_points * drop_ratio)
    
    if n_drop == 0:
        return points_data.copy()
    
    # 提取坐标
    coords = points_data[:, :3]
    
    # 使用k-d树加速近邻搜索
    nn = NearestNeighbors(n_neighbors=50, algorithm='kd_tree').fit(coords)
    
    # 选择种子点
    if seed_index is None:
        seed_index = np.random.randint(0, n_points)
    
    # 区域生长算法选择连续区域
    visited = np.zeros(n_points, dtype=bool)
    drop_indices = []
    queue = []
    
    # 初始化种子点
    heapq.heappush(queue, (0.0, seed_index))
    visited[seed_index] = True
    
    while queue and len(drop_indices) < n_drop:
        _, idx = heapq.heappop(queue)
        drop_indices.append(idx)
        
        # 查找当前点的近邻
        current_point = coords[idx]
        _, neighbor_indices = nn.kneighbors(current_point.reshape(1, -1))
        neighbor_indices = neighbor_indices[0]
        
        # 添加未访问的近邻点
        for neighbor_idx in neighbor_indices:
            if not visited[neighbor_idx] and len(drop_indices) < n_drop:
                visited[neighbor_idx] = True
                # 计算到种子点的距离
                dist = np.linalg.norm(coords[neighbor_idx] - coords[seed_index])
                heapq.heappush(queue, (dist, neighbor_idx))
    
    # 创建保留点的掩码
    keep_mask = np.ones(n_points, dtype=bool)
    keep_mask[drop_indices] = False
    
    # 可视化丢弃区域 (调试用)
    if visualize:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制保留的点
        kept_coords = coords[keep_mask]
        ax.scatter(kept_coords[:, 0], kept_coords[:, 1], kept_coords[:, 2], 
                  s=1, c='blue', alpha=0.5, label='Kept points')
        
        # 绘制丢弃的点
        dropped_coords = coords[drop_indices]
        ax.scatter(dropped_coords[:, 0], dropped_coords[:, 1], dropped_coords[:, 2], 
                  s=2, c='red', alpha=0.8, label='Dropped points')
        
        # 标记种子点
        seed_point = coords[seed_index]
        ax.scatter(seed_point[0], seed_point[1], seed_point[2], 
                  s=100, c='green', marker='*', label='Seed point')
        
        ax.set_title(f'Region Drop Visualization (Dropped {len(drop_indices)} points)')
        ax.legend()
        plt.show()
    
    print(f"Dropped {len(drop_indices)} points in a continuous region (seed index: {seed_index})")
    return points_data[keep_mask]


if __name__ == "__main__":

    input_path = "testcase/test0723/origin/aquarius.ply"
    name_prefix = os.path.splitext(os.path.basename(input_path))[0]
    
    # 创建以文件名为名的子目录
    output_path = "testcase/test0723/output"
    output_dir = os.path.join(output_path, name_prefix)
    os.makedirs(output_dir, exist_ok=True)

    # 刚性变换的参数
    use_manual_tranform = True
    write_file = True
    translation = [0.035, 0, 0.5]  # 平移
    rotation_degrees = 25.84     # 旋转角度
    rotation_axis = [-1, -0.1, 0]  # 绕Z轴

    # 颜色偏移的目标值 (RGB)
    source_color_shift = (255, 180, 0) # Yellowish-orange for target
    target_color_shift = (90, 100, 220) # Bluish-purple for source

    # 添加局部变形参数
    apply_deformation = True  # 是否应用局部变形
    deformation_ratio = 0.1      # 变形点比例
    displacement = 0.05       # 最大位移大小（根据点云尺度调整）
    smooth_radius = 0.3       # 平滑过渡区域半径（0.0-1.0，值越大过渡越平缓）
    seed_index = 1000         # 种子点索引（可选，设为None则随机选择）

    # 添加区域丢失参数
    apply_region_drop_flag = True  # 是否应用区域丢失
    drop_ratio = 0.01              # 丢弃点比例
    drop_seed_index = 500           # 丢弃区域种子点索引
    visualize_drop = False           # 是否可视化丢弃区域

    # 1. 读取点云
    data = read_ply_with_all_data(input_path)

    # 2. 随机选取 80% 的点，选两次
    target = random_sample(data, ratio=0.4)
    source = random_sample(data, ratio=0.6)
    #source, target = split_point_cloud(data, source_ratio=0.6) # 0.4916
    # target = random_sample(pre_target, ratio=0.5328)
    
    if apply_region_drop_flag:
        source = apply_region_drop(
            source, 
            drop_ratio=drop_ratio, 
            seed_index=drop_seed_index,
            visualize=visualize_drop
        )

    if apply_deformation:
        source = apply_local_deformation(
            source, 
            deformation_ratio=deformation_ratio,
            displacement=displacement,
            seed_index=None,
            smooth_radius=smooth_radius
        )



    
    # 3. 定义输出路径
    target_path = os.path.join(output_dir, f"{name_prefix}_target.ply")
    source_path = os.path.join(output_dir, f"{name_prefix}_source.ply")
    transformed_source_path = os.path.join(output_dir, f"{name_prefix}_transformed_source.ply")    
    t_path = os.path.join(output_dir, f"{name_prefix}_Tlog.txt")
    gt_path = os.path.join(output_dir, f"{name_prefix}_ground_truth.txt")


    # 4. 对 source 进行刚性变换
    #"""
    if use_manual_tranform:
        T = apply_manual_rigid_transform(t_path, translation, rotation_degrees, rotation_axis)
    else:
        T = generate_random_rigid_transform()
    #"""
    #T = np.eye(4)
    
    transformed_source = apply_rigid_transform(source, T)

    # 5. 计算逆矩阵
    T_inv = np.linalg.inv(T)
    
    if write_file:
        write_ply_with_all_data(target, target_path, shift_color_to=target_color_shift)
        write_ply_with_all_data(source, source_path, shift_color_to=source_color_shift)        
        write_ply_with_all_data(transformed_source, transformed_source_path, shift_color_to=source_color_shift)
        save_matrix_txt(T_inv, gt_path)

    # 6. 输出信息和变换矩阵
    print(f"[✓] Target: {target_path}")
    print(f"[✓] Source (Transformed): {source_path}")
    print(f"[✓] Ground truth inverse matrix: {gt_path}")
    print(T)
    print("\nInverse Rigid Transform Matrix (4x4):")
    print(T_inv)
    print(T @ T_inv)

    # 7. 同屏显示source和target配准效果
    print("\nLaunching visualization window...")
    if write_file:
        visualize_source = read_ply_with_all_data(source_path)
        visualize_target = read_ply_with_all_data(target_path)
        visualize_transformed_source = read_ply_with_all_data(transformed_source_path)
        #visualize_two_pointclouds_new(visualize_source, visualize_target)
        #print("\nLaunching visualization window with toggle control...")
        visualize_with_toggle_from_files(visualize_target, visualize_source, visualize_transformed_source)
    else:
        # If not writing files, visualize the in-memory transformed data (without the saved color shifts)
        print("Note: 'write_file' is False, so visualization will use original colors or default if no colors were in input PLY.")
        visualize_two_pointclouds_old(transformed_source, target)