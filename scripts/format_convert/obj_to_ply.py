"""
Description : 将OBJ文件转换为点云（.ply或.txt），通过三角面片均匀采样方式，可自定义点云数量
"""

import vtk
import numpy as np

def read_obj(file_path):
    """读取OBJ文件并返回vtkPolyData对象"""
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def render_polydata(polydata):
    """渲染vtkPolyData对象"""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)
    
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    render_window.Render()
    interactor.Start()

def sample_uniformly(polydata, num_points):
    """从三角面片中均匀采样点并返回点+法向量"""
    triangles = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:
            pt_ids = cell.GetPointIds()
            triangle = [polydata.GetPoint(pt_ids.GetId(j)) for j in range(3)]
            triangles.append(triangle)
    
    triangles = np.array(triangles)

    def triangle_area(tri):
        a = tri[1] - tri[0]
        b = tri[2] - tri[0]
        return 0.5 * np.linalg.norm(np.cross(a, b))
    
    areas = np.array([triangle_area(tri) for tri in triangles])
    total_area = np.sum(areas)
    probabilities = areas / total_area
    
    sampled_indices = np.random.choice(len(triangles), size=num_points, p=probabilities)
    sampled_points_with_normals = []

    for idx in sampled_indices:
        tri = triangles[idx]
        r1, r2 = np.random.rand(), np.random.rand()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        point = tri[0] + r1 * (tri[1] - tri[0]) + r2 * (tri[2] - tri[0])
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        normal /= np.linalg.norm(normal)
        sampled_points_with_normals.append(np.concatenate([point, normal]))

    return np.array(sampled_points_with_normals)

def save_ply_with_normals(points, file_path):
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
    with open(file_path, 'w') as f:
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

if __name__ == "__main__":
    obj_file = "testcase/models/nefertiti.obj"  # 替换为实际路径
    polydata = read_obj(obj_file)
    render_polydata(polydata)

    desired_num_points = 1680000  # 设置采样点数
    sampled_points = sample_uniformly(polydata, desired_num_points)
    print(f"采样点数: {sampled_points.shape[0]}")

    output_ply = "testcase/test0702/RebuiltModels/nefertiti.ply"
    save_ply_with_normals(sampled_points, output_ply)
    # save_xyz_to_txt(sampled_points, output_txt)
    print(f"已保存PLY文件至: {output_ply}")
