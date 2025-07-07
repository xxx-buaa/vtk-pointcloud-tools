"""
Description : 探究点云模型颜色效果
"""
import vtk
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy
import math

# 1. 读取PLY文件
def read_ply_colors(filename):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(filename)
    reader.Update()

    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()

    # 提取颜色（RGBA 或 RGB）
    colors_vtk = polydata.GetPointData().GetScalars()
    colors_np = vtk_to_numpy(colors_vtk)

    # 只保留 RGB（三通道）
    if colors_np.shape[1] == 4:
        colors_np = colors_np[:, :3]

    return colors_np, num_points

# 2. 构造接近3:2比例的图像大小
def calculate_image_shape(n_points):
    width = int(np.sqrt((3/2) * n_points))
    height = int(width * 2 / 3)
    if width * height < n_points:
        height += 1
    return height, width

# 3. 可视化颜色平铺图像
def show_color_image(colors_np, height, width):
    # 补足不足的像素点
    total_pixels = height * width
    if colors_np.shape[0] < total_pixels:
        pad = np.zeros((total_pixels - colors_np.shape[0], 3), dtype=np.uint8)
        colors_np = np.vstack((colors_np, pad))

    # reshape成图像
    image = colors_np.reshape((height, width, 3))
    plt.imshow(image)
    plt.title(f'Point Cloud Colors ({width}×{height})')
    plt.axis('off')
    plt.show()

# 4. 主流程
filename = 'testcase/models/aquarius.ply'  # 替换为你的PLY文件路径
colors_np, n_points = read_ply_colors(filename)
height, width = calculate_image_shape(n_points)
show_color_image(colors_np, height, width)
