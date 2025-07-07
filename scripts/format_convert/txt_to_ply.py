"""
Description : 将txt格式转换至ply格式
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse
import os

def read_txt_xyz(file_path):
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 3:
                points.append(vals)
    return np.array(points)

def estimate_normals(points, k=10):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    _, indices = nbrs.kneighbors(points)
    normals = []
    for i, neighbors in enumerate(indices):
        neighbors = points[neighbors[1:]]  # exclude itself
        centroid = np.mean(neighbors, axis=0)
        cov = np.cov((neighbors - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        normal /= np.linalg.norm(normal)
        normals.append(normal)
    return np.array(normals)

def save_ply_xyz(points, file_path):
    with open(file_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def save_ply_with_normals(points, normals, file_path):
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
        for p, n in zip(points, normals):
            f.write(f"{p[0]} {p[1]} {p[2]} {n[0]} {n[1]} {n[2]}\n")

if __name__ == "__main__":

    txt_file = "src/GX2.txt"

    points = read_txt_xyz(txt_file)

    output_ply = "src/GX2.ply"

    save_ply_xyz(points, output_ply)

    print(f"点云已保存到 {output_ply}")
