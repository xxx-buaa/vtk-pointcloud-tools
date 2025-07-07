"""
Description : 对舱段模型在不同u值条件下配准后的数据进行处理与可视化
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False

def extract_posratio(filepath):
    """从日志文件中提取每次迭代的 PosRatio 值"""
    posratios = []
    pattern = re.compile(r"Iter:\s*\d+\s*\|.*?PosRatio:\s*([\d\.]+)")
    with open(filepath, 'r', encoding='gbk', errors='replace') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                posratios.append(float(match.group(1)))
    return np.array(posratios)

def extract_param_from_filename(filename):
    """从文件名中提取 log_ 后面的参数值"""
    match = re.search(r"log_(\d*\.?\d+)", filename)
    return match.group(1) if match else "unknown"

# 设置日志文件夹路径
log_folder = 'testcase/PosRatioProcessing'
file_pattern = os.path.join(log_folder, 'Cylinder_EXPICP_log_*.txt')
log_files = sorted(glob(file_pattern))  # 可确保顺序一致

# 绘图
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(log_files)))  # 自动生成颜色

for idx, file_path in enumerate(log_files):
    param = extract_param_from_filename(os.path.basename(file_path))
    posratio = extract_posratio(file_path)
    plt.plot(range(1, len(posratio)+1), posratio, label=f"v = {param}", color=colors[idx])

plt.title("PosRatio 迭代趋势对比")
plt.xlabel("Iteration")
plt.ylabel("PosRatio")
plt.legend(title="参数值", loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
