"""
Description : 数stl文件共有多少个三角面片
"""

from stl import mesh

def count_triangles_in_stl(file_path):
    # 读取STL文件
    stl_mesh = mesh.Mesh.from_file(file_path)
    
    # 获取三角形的数量
    num_triangles = len(stl_mesh.vectors)
    
    return num_triangles

# 使用示例
file_path = 'src/fa8_007_068.stl'  # 替换为你自己的STL文件路径
triangle_count = count_triangles_in_stl(file_path)

print(f"STL文件中一共有 {triangle_count} 个三角面片。")
