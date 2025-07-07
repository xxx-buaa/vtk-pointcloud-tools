"""
Description : 对舱段模型在不同u值条件下配准后的数据进行处理与可视化，增强了显示效果
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'  # 支持中文显示
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

def extract_posratio_and_u_changes(filepath):
    """
    提取 PosRatio 数值，以及 u 值发生变化的迭代编号
    返回：
        posratios: List[float]
        u_change_iters: List[int]
    """
    posratios = []
    u_change_iters = []
    last_u = None

    pattern = re.compile(r"Iter:\s*(\d+)\s*\| v: .*? \| u: ([\deE\.\-]+) mm \| PosRatio:\s*([\d\.]+)")

    with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iter_num = int(match.group(1))
                u_value = float(match.group(2))
                pos = float(match.group(3))

                posratios.append(pos)

                if last_u is None or not np.isclose(u_value, last_u, rtol=1e-6, atol=1e-9):
                    u_change_iters.append(iter_num)
                    last_u = u_value

    print(f"✅ {os.path.basename(filepath)} - 提取 {len(posratios)} 个 PosRatio，u变化 {len(u_change_iters)} 次")
    return np.array(posratios), np.array(u_change_iters)

def extract_param_from_filename(filename):
    """从文件名中提取 log_ 后面的参数值"""
    match = re.search(r"log_(\d*\.?\d+)", filename)
    return match.group(1) if match else "unknown"

# 设置日志文件夹路径
log_folder = 'testcase/test0703/PosRatioProcessing'
file_pattern = os.path.join(log_folder, 'Cylinder_EXPICP_log_*.txt')
log_files = sorted(glob(file_pattern))  # 可确保顺序一致

# 初始化全局最大值追踪
global_max_ratio = -np.inf
global_max_info = {"file": None, "param": None, "iter": None, "u_value": None, "value": None}

# 绘图
plt.figure(figsize=(10, 6))
colors = plt.get_cmap("tab20").colors  # 自动生成颜色

for idx, file_path in enumerate(log_files):
    """    
    param = extract_param_from_filename(os.path.basename(file_path))
    posratio = extract_posratio(file_path)
    plt.plot(range(1, len(posratio)+1), posratio, label=f"v = {param}", color=colors[idx])
    """
    
    param = extract_param_from_filename(os.path.basename(file_path))
    posratios, u_change_iters = extract_posratio_and_u_changes(file_path)
    iters = np.arange(1, len(posratios)+1)

    # 画 PosRatio 曲线
    plt.plot(iters, posratios, label=f"log_{param}", color=colors[idx])

    # 画 × 标记
    for i in u_change_iters:
        if i <= len(posratios):  # 防止越界
            plt.scatter(i, posratios[i - 1], marker='x', color=colors[idx], s=60)
    
        # ✅ 查找当前文件中最大值及其索引
    local_max_idx = np.argmax(posratios)
    local_max_val = posratios[local_max_idx]

    # ✅ 从文件中提取该迭代对应的 u 值
    pattern = re.compile(r"Iter:\s*(\d+)\s*\| v: .*? \| u: ([\deE\.\-]+) mm \| PosRatio:\s*([\d\.]+)")
    with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iter_num = int(match.group(1))
                u_val = float(match.group(2))
                pos = float(match.group(3))
                if np.isclose(pos, local_max_val, rtol=1e-6):
                    # ✅ 找到匹配的行
                    if local_max_val > global_max_ratio:
                        global_max_ratio = local_max_val
                        global_max_info = {
                            "file": os.path.basename(file_path),
                            "param": param,
                            "iter": iter_num,
                            "u_value": u_val,
                            "value": local_max_val
                        }
                    break


print("\n📈 全部日志文件中 PosRatio 最大值统计：")
print(f"最大值: {global_max_info['value']:.5f}")
print(f"文件名: {global_max_info['file']}")
print(f"对应参数: log_{global_max_info['param']}")
print(f"迭代编号: {global_max_info['iter']}")
print(f"对应 u 值: {global_max_info['u_value']} mm")

plt.title("PosRatio 迭代趋势对比")
plt.xlabel("Iteration")
plt.ylabel("PosRatio")
plt.legend(title="参数值", loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
