"""
Description : 写了个窗体，选择文件进行可视化，同时输出该点云文件的点云数
"""

import vtk
import tkinter as tk
from tkinter import filedialog, Menu
from vtkmodules.vtkIOPLY import vtkPLYReader
from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkPolyDataMapper, vtkActor
from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkRenderWindowInteractor
from vtkmodules.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor

class PlyViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PLY 可视化工具")
        self.root.geometry("400x100")

        # 菜单栏
        self.menu_bar = Menu(self.root)
        file_menu = Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="打开 PLY 文件...", command=self.load_and_visualize)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        self.menu_bar.add_cascade(label="文件", menu=file_menu)
        self.root.config(menu=self.menu_bar)

        # 状态栏
        self.status_label = tk.Label(self.root, text="尚未加载任何模型", anchor='w')
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # 初始化渲染组件
        self.render_window = None
        self.interactor = None

    def load_and_visualize(self):
        ply_path = filedialog.askopenfilename(
            title="选择 PLY 文件",
            filetypes=[("PLY 文件", "*.ply")]
        )
        if not ply_path:
            return

        # 读取 PLY
        reader = vtkPLYReader()
        reader.SetFileName(ply_path)
        reader.Update()
        polydata = reader.GetOutput()

        # 点数统计
        num_points = polydata.GetNumberOfPoints()
        self.status_label.config(text=f"当前模型共有 {num_points} 个点")

        # 创建 actor
        actor = self.create_actor_from_polydata(polydata)
        # actor = self.create_actor_from_polydata_color_drifted(polydata)

        # 如果已有窗口，则关闭
        if self.render_window:
            self.render_window.Finalize()
            self.interactor.TerminateApp()

        # 新建 VTK 渲染窗口
        self.render_window = vtkRenderWindow()
        self.render_window.SetSize(800, 600)
        renderer = vtkRenderer()
        renderer.SetBackground(1, 1, 1)
        renderer.AddActor(actor)
        renderer.ResetCamera()

        self.render_window.AddRenderer(renderer)

        self.interactor = vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.render_window.Render()
        self.interactor.Start()

    def create_actor_from_polydata(self, polydata, point_size=0.001, color=(0.8, 0.2, 0.2)):
        vertex_filter = vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(vertex_filter.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)
        actor.GetProperty().SetColor(color)
        return actor

    def create_actor_from_polydata_color_drifted(self, polydata, point_size=0.001):

        points = polydata.GetPoints()
        # 2. 获取点颜色（RGBA）
        colors = polydata.GetPointData().GetScalars()

        # 3. 复制颜色数据并偏移向黄色
        new_colors = vtk.vtkUnsignedCharArray()
        new_colors.SetNumberOfComponents(4)
        new_colors.SetNumberOfTuples(colors.GetNumberOfTuples())
        new_colors.SetName("Colors")

        # 颜色偏移量（你可以调节）
        yellow_weight = 0.9

        for i in range(colors.GetNumberOfTuples()):
            r, g, b, a = colors.GetTuple4(i)

            new_r = int((1 - yellow_weight) * r + yellow_weight * 255)
            new_g = int((1 - yellow_weight) * g + yellow_weight * 180)
            new_b = int((1 - yellow_weight) * b + yellow_weight * 0)

            new_colors.SetTuple4(i, new_r, new_g, new_b, a)

        # 4. 替换点云中的颜色
        polydata.GetPointData().SetScalars(new_colors)

        vertex_filter = vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(vertex_filter.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)
        mapper.SetColorModeToDirectScalars()  # 关键：直接使用 RGBA 上色
        mapper.ScalarVisibilityOn()           # 启用标量可见性
        # actor.GetProperty().SetColor(colors)
        return actor

if __name__ == "__main__":
    root = tk.Tk()
    app = PlyViewerApp(root)
    root.mainloop()
