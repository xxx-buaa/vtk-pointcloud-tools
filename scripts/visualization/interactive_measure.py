"""
Description : 通过交互点击测量筒段两个点的距离
"""

import vtk
from stl import mesh
import numpy as np

# 读取STL文件
def read_stl(file_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

# 自定义交互器实现测量两个点距离
class PointPickerInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, renderer):
        self.AddObserver("LeftButtonPressEvent", self.on_left_button_press_event)
        self.renderer = renderer
        self.points = []
        self.point_picker = vtk.vtkPointPicker()

    def on_left_button_press_event(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        self.point_picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        picked_position = self.point_picker.GetPickPosition()
        if picked_position != (0.0, 0.0, 0.0):
            self.points.append(picked_position)
            print(f"点 {len(self.points)}: {picked_position}")
            if len(self.points) == 2:
                dist = np.linalg.norm(np.array(self.points[0]) - np.array(self.points[1]))
                print(f"两点距离为: {dist:.3f}")
                self.points.clear()
        self.OnLeftButtonDown()

# 可视化函数
def visualize_stl(stl_data):
    # 用平面沿 Y=0 剖分
    clip_plane = vtk.vtkPlane()
    clip_plane.SetOrigin(0, 0, 0)
    clip_plane.SetNormal(0, 1, 0)  # 沿 Y 方向剖切

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(stl_data)
    clipper.SetClipFunction(clip_plane)
    clipper.Update()

    # 剖分后数据映射
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(clipper.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # 蓝色面

    # 渲染器
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.7, 0.7, 0.7)
    renderer.AddActor(actor)

    # 窗口和交互器
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # 设置自定义交互样式
    style = PointPickerInteractorStyle(renderer)
    interactor.SetInteractorStyle(style)

    # 开始渲染
    render_window.Render()
    interactor.Start()

# 主程序
if __name__ == "__main__":
    file_path = 'src\\GX.stl'  # 替换为实际路径
    stl_data = read_stl(file_path)
    stl_mesh = mesh.Mesh.from_file(file_path)
    print(f"STL文件中一共有 {len(stl_mesh.vectors)} 个三角面片。")
    visualize_stl(stl_data)
