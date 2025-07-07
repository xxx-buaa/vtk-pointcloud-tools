"""
Description : 25/07/04 逻辑仍然是选择文件夹，读取文件及输出的log，然后对文件进行变换与可视化
"""

import os
import re
from tkinter import Tk, filedialog
from sys import exit

import numpy as np
import vtk
from vtkmodules.vtkIOPLY import vtkPLYReader
from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkPolyDataMapper, vtkActor
from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkRenderWindowInteractor
from vtkmodules.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor

def read_matrix_from_txt(txt_path):
    """读取4x4矩阵（16个float）"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        values = [float(val) for line in f for val in line.strip().split()]
    if len(values) != 16:
        raise ValueError("Matrix txt 不包含 16 个数字")
    return np.array(values).reshape((4, 4))

def extract_log_matrix_and_meta(log_path):
    """从log中提取矩阵和method/u_value"""
    with open(log_path, 'r', encoding='gbk') as f:
        content = f.read()

    # 提取变换矩阵
    pattern = r"res_trans\s*((?:\s*-?\d+\.?\d*e?-?\d*\s*){16})"
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"{log_path} 中未找到 res_trans")

    matrix_values = list(map(float, match.group(1).split()))
    matrix = np.array(matrix_values).reshape((4, 4))

    # 提取method（basename后直到_log_前）
    filename = os.path.basename(log_path)
    method_match = re.match(rf"{base_name}_(.+)_log_([\d\.eE\-]+)\.txt", filename)
    if not method_match:
        raise ValueError(f"无法解析方法名和u值：{filename}")
    method = method_match.group(1)
    u_value = method_match.group(2)
    return matrix, method, u_value

def vtk_read_ply(filename):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def apply_transformation(polydata, matrix):
    transform = vtk.vtkTransform()
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, matrix[i, j])
    transform.SetMatrix(vtk_matrix)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    return transform_filter.GetOutput()

def create_actor(polydata):
    vertex_filter = vtkVertexGlyphFilter()
    vertex_filter.SetInputData(polydata)
    vertex_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(vertex_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(1.7)
    return actor

def visualize(source, target, title, screenshot_path=None):
    source_actor = create_actor(source)
    target_actor = create_actor(target)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(source_actor)
    renderer.AddActor(target_actor)
    renderer.SetBackground(1.0, 1.0, 1.0)  # White background

    camera = vtk.vtkCamera()
    """aquarius
    """  
    camera.SetPosition(-9.0, -1.4, 1.8)         # 相机在 Z=+2 位置上方
    camera.SetFocalPoint(0, -1.4, 1.8)         # 看向原点
    camera.SetViewUp(0, 0, -1)             # Z 轴向下为 up 方向  
    """monkeys
    camera.SetPosition(-12.0, 2, 1.1)
    camera.SetFocalPoint(0, 3, 0.3)
    camera.SetViewUp(0, -1, 0)
    camera.Elevation(5.0)   # ✅ 绕 X 轴旋转 5 度 
    camera.Roll(0.5) # 绕 Z 轴旋转 0.5 度
    """
    renderer.SetActiveCamera(camera)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 800)
    render_window.SetWindowName(title)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.SetRenderWindow(render_window)

    """
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
    """

    render_window.Render()

    # 加了一个自动截图功能
    if screenshot_path:
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(render_window)
        w2i.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(screenshot_path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        print(f"[截图已保存] {screenshot_path}")

    # 显示窗体
    interactor.Start()  # 会阻塞，直到窗口关闭

if __name__ == "__main__":
    global base_name

    # 文件夹选择
    Tk().withdraw()
    folder = filedialog.askdirectory(title="选择包含PLY和LOG的文件夹")
    if not folder:
        print("取消了文件夹选择，程序退出")
        exit(0)
    base_name = os.path.basename(folder)

    # 文件路径
    source_path = os.path.join(folder, f"{base_name}_source.ply")
    target_path = os.path.join(folder, f"{base_name}_target.ply")
    gt_path = os.path.join(folder, f"{base_name}_ground_truth.txt")

    # 点云读取
    source = vtk_read_ply(source_path)
    target = vtk_read_ply(target_path)

    # 1. 可视化原始 input
    visualize(source, target, title="input",
                screenshot_path=os.path.join(folder, f"{base_name}_input.png"))

    # 2. ground truth 可视化
    if os.path.exists(gt_path):
        gt_matrix = read_matrix_from_txt(gt_path)
        transformed = apply_transformation(source, gt_matrix)
        visualize(transformed, target, title=f"{base_name} ground truth",
                    screenshot_path=os.path.join(folder, f"{base_name}_ground_truth.png"))
    else:
        print("未找到 ground truth 文件，跳过该步骤。")

    # 3. 遍历所有log
    method_order = ["ICP", "AA_ICP", "FICP", "RICP", "PPL", "RPPL", "SparseICP", "SICPPPL", "EXPICP"]
    log_map = {}  # method: path

    for f in os.listdir(folder):
        if f.startswith(base_name) and f.endswith(".txt") and "_log_" in f:
            match = re.match(rf"{base_name}_(.+)_log_([\d\.eE\-]+)\.txt", f)
            if match:
                method = match.group(1)
                if method in method_order:
                    log_map[method] = os.path.join(folder, f)

    for method in method_order:
        if method not in log_map:
            print(f"跳过方法 {method}：未找到对应文件")
            continue
        log_path = log_map[method]
        try:
            matrix, _, u_value = extract_log_matrix_and_meta(log_path)
            transformed = apply_transformation(source, matrix)
            visualize(transformed, target, title=f"{base_name} {method} {u_value}", 
                        screenshot_path=os.path.join(folder, f"{base_name}_{method}_{u_value}.png"))
        except Exception as e:
            print(f"跳过方法 {method}，原因：{e}")

    """
    log_files = sorted([f for f in os.listdir(folder)
                        if f.startswith(base_name) and f.endswith(".txt") and "_log_" in f])
    for log_file in log_files:
        log_path = os.path.join(folder, log_file)
        try:
            matrix, method, u_value = extract_log_matrix_and_meta(log_path)
            transformed = apply_transformation(source, matrix)
            visualize(transformed, target, title=f"{base_name} {method} {u_value}")
        except Exception as e:
            print(f"跳过文件 {log_file}，原因：{e}")
    """