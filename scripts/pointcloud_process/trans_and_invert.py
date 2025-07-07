"""
Description : 计算筒段位姿变换的逆变换
"""

import numpy as np
import vtk

# 四个平台边缘点（顺时针/逆时针顺序）
p1 = np.array([-1362.2883, 543.5320, -460.7065])
p2 = np.array([-1495.3304, 338.0093, -743.0293])
p3 = np.array([-1378.8234, 29.5273, -571.2750])
p4 = np.array([-1247.7611, 232.6520, -296.6372])
platform_pts = np.array([p1, p2, p3, p4])

def construct_alignment_matrix(pts):
    center = np.mean(pts, axis=0)
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    
    z_axis = np.cross(v1, v2)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # 选择平台的一条边，投影到平台面上作为 X 方向
    x_temp = pts[1] - pts[0]
    x_axis = x_temp - np.dot(x_temp, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # R_local: 模型局部坐标系
    R_local = np.column_stack([x_axis, y_axis, z_axis])
    R_world = np.eye(3)

    # R_align = R_world @ R_local.T
    R_align = R_world @ R_local.T
    T = -R_align @ center

    # 构造 4x4 变换矩阵
    M = np.eye(4)
    M[:3, :3] = R_align
    M[:3, 3] = T
    return M

def apply_transform_to_stl(input_path, output_path, transform_matrix):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(input_path)
    reader.Update()
    polydata = reader.GetOutput()

    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, transform_matrix[i, j])

    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)

    filter = vtk.vtkTransformPolyDataFilter()
    filter.SetInputData(polydata)
    filter.SetTransform(transform)
    filter.Update()

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(filter.GetOutput())
    writer.Write()
    print(f"变换完成，已保存为: {output_path}")

def construct_inverse_alignment_matrix(pts):
    center = np.mean(pts, axis=0)
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    
    z_axis = np.cross(v1, v2)
    z_axis = z_axis / np.linalg.norm(z_axis)

    x_temp = pts[1] - pts[0]
    x_axis = x_temp - np.dot(x_temp, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    R_local = np.column_stack([x_axis, y_axis, z_axis])

    M_inv = np.eye(4)
    M_inv[:3, :3] = R_local
    M_inv[:3, 3] = center
    return M_inv

if __name__ == "__main__":
    # matrix = construct_alignment_matrix(platform_pts)
    # apply_transform_to_stl("src\\GX.stl", "CylinderSegmentOutput.stl", matrix)

    # 逆变换：恢复模型原始位置
    inverse_matrix = construct_inverse_alignment_matrix(platform_pts)
    apply_transform_to_stl("mdl/CylinderSegmentReconstruction_0630.stl", "mdl/CylinderSegmentReconstruction_0630_Transformed.stl", inverse_matrix)
