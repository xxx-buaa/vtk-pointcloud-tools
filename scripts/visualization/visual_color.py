"""
Description : 探究点云模型颜色效果
"""
from PIL import Image

import vtk
import numpy as np
import matplotlib.pyplot as plt
from vtkmodules.util.numpy_support import vtk_to_numpy
import math

# 1. 读取PLY文件
def read_ply_colors(filename):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(filename)
    reader.Update()

    polydata = reader.GetOutput()
    num_points = polydata.GetNumberOfPoints()

    # 提取颜色（RGBA 或 RGB）
    point_data = polydata.GetPointData()

    # 打印所有属性名，调试用
    print("Available point data arrays:")
    for i in range(point_data.GetNumberOfArrays()):
        print(f"- {point_data.GetArrayName(i)}")
    
    colors_vtk = point_data.GetArray("RGBA")

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
    # 保存为PNG图片
    img = Image.fromarray(image)
    img.save('testcase/aquarius.png')

    colors_int = colors_np.astype(np.int32)  # 或 np.int64

# 按 RGB 字典序排序：R优先，其次G，再B
    weights = colors_int[:, 0] * 256 * 256 + colors_int[:, 1] * 256 + colors_int[:, 2]

    sorted_idx = np.argsort(weights)
    sorted_colors = colors_np[sorted_idx]

    image = sorted_colors.reshape((height, width, 3))

    img = Image.fromarray(image)
    img.save('testcase/reshape_aquarius.png')

    plt.imshow(image)
    plt.title(f'Point Cloud Colors ({width}×{height})')
    plt.axis('off')
    plt.show()
    
    print(f"完成")

# 4. 主流程
filename = 'testcase/aquarius.ply'  # 替换为你的PLY文件路径
colors_np, n_points = read_ply_colors(filename)

np.save('testcase/colors.npy', colors_np)

height, width = calculate_image_shape(n_points)
show_color_image(colors_np, height, width)