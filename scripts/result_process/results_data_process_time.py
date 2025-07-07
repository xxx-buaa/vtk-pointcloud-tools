"""
Description : 对点云配准后的输出数据进行处理和图表绘制
但是横坐标变为了时间
"""

import os
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from tkinter import Tk, filedialog

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20

def select_folder():
    """弹出文件夹选择窗口"""
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="选择包含log文件的文件夹")
    return folder_path

def is_valid_log_filename(filename):
    """判断文件名是否符合格式：model_algorithm_log_value.txt"""
    return re.match(r"^[^\\/_]+_.+_log_\d*\.?\d+\.txt$", filename) is not None

def extract_rmse_and_time(filepath):
    """从文件中提取 RMSE 序列和总时间"""
    rmse_list = []
    time_total = None
    rmse_pattern = re.compile(r"gt_mse\s*[:=]\s*([0-9eE\.\-]+)")
    time_pattern = re.compile(r"time total:([0-9eE\.\-]+)")

    with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            rmse_match = rmse_pattern.search(line)
            if rmse_match:
                rmse_list.append(float(rmse_match.group(1)))
            if "time total" in line:
                time_match = time_pattern.search(line)
                if time_match:
                    time_total = float(time_match.group(1))
    return np.array(rmse_list), time_total

def extract_algorithm_name(filename):
    """从文件名中提取算法名（中间可能有多个下划线）"""
    # 例如：aquarius_AA_ICP_log_1.txt
    # 提取 AA_ICP
    match = re.match(r"^[^_]+_(.+)_log_\d*\.?\d+\.txt$", filename)
    return match.group(1) if match else "Unknown"

# === 主流程 ===

# 指定算法顺序
algorithm_order = ["ICP", "AA-ICP", "FICP", "RICP", "SparseICP", "PPL", "RPPL", "SICPPPL", "ARPPL"]

folder = select_folder()
if not folder:
    print("❌ 未选择任何文件夹")
    exit()

log_files = [f for f in os.listdir(folder) if f.endswith('.txt') and is_valid_log_filename(f)]
if not log_files:
    print("⚠️ 没有找到符合命名规则的日志文件")
    exit()

plt.figure(figsize=(25, 10))
# plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))  # 每个主刻度间隔为 0.1
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.1))       # 每 0.1 作为主刻度
ax.yaxis.set_minor_locator(AutoMinorLocator(5))         # 每主刻度再分5个次刻度
# colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
# colors = plt.cm.viridis(np.linspace(0, 1, len(log_files)))  # 自动生成颜色
# 构建算法对应颜色的字典
custom_colors = {
    "ICP": "#1f77b4",       # 蓝色，常用默认
    "AA-ICP": "#d62728",    # 红色，强对比
    "FICP": "#2ca02c",      # 绿色，强对比
    "RICP": "#7f7f7f",      # 灰色，专门标识
    "SparseICP": "#9467bd", # 紫色
    "PPL": "#bcbd22",     # 黄绿色
    "RPPL": "#17becf",      # 青色
    "SICPPPL": "#e377c2",    # 粉紫
    "ARPPL": "#ff7f0e",       # 橙色
}
highlight_algorithms = ["RICP", "ICP", "FICP", "AA-ICP"]

alg_to_file = {}

for filename in log_files:
    alg_name = extract_algorithm_name(filename)
    if alg_name not in alg_to_file:
        alg_to_file[alg_name] = []
    alg_to_file[alg_name].append(filename)

# 遍历指定顺序进行绘图
colors = plt.cm.tab10(np.linspace(0, 1, len(algorithm_order)))
for idx, alg in enumerate(algorithm_order):
    if alg not in alg_to_file:
        print(f"⚠️ 未找到算法 {alg} 对应的文件")
        continue
    for filename in alg_to_file[alg]:
        filepath = os.path.join(folder, filename)
        rmse, time_total = extract_rmse_and_time(filepath)
        if rmse.size == 0 or time_total is None:
            print(f"⚠️ 文件 {filename} 中 RMSE 或时间无效")
            continue

        time_axis = np.linspace(0, time_total, len(rmse))

        # iterations = np.arange(1, len(rmse) + 1)
        # plt.plot(iterations, rmse, label=f"{alg} ({time_total:.2f}s)", color=colors[idx])
        # plt.plot(iterations, rmse, label=f"{alg} ({time_total:.2f}s)", color=algorithm_color_map[alg])
        plt.plot(time_axis, rmse, label=alg, color=custom_colors.get(alg, "black"), linewidth = 2.5)
        
        # 在末端加三角形标记
        # if alg in highlight_algorithms:
        #     plt.scatter(time_axis[-1], rmse[-1],
        #                 marker="^", s=100, color=custom_colors.get(alg, "black"), zorder=5)
        
        # 如果运行时间超过 xlim 上限，则在 20s 位置附近添加一个注释
        # x_limit = 19
        x_limit = 14
        if time_total > x_limit:
            # 找到最接近 20s 的有效索引
            idx_20s = np.searchsorted(time_axis, x_limit)
            if idx_20s >= len(rmse):
                # 如果超出 rmse 长度，改为 len(rmse) - 1（就是最接近20s之前的那个）
                idx_20s = len(rmse) - 1
            elif idx_20s > 0 and abs(time_axis[idx_20s] - x_limit) > abs(time_axis[idx_20s - 1] - x_limit):
                # 如果前一个点更接近 20s，则用前一个点
                idx_20s -= 1

            # 取该点 RMSE 作为标注位置
            rmse_value = rmse[idx_20s]
            plt.text(
                x_limit * 0.995, rmse_value * 1.5,
                f"{alg} ({time_total:.1f}s)",
                fontsize=18,
                color=custom_colors.get(alg, "black"),
                verticalalignment='bottom',
                horizontalalignment='right'
            )

# for idx, filename in enumerate(sorted(log_files)):
#     filepath = os.path.join(folder, filename)
#     algorithm_name = extract_algorithm_name(filename)
#     rmse, time_total = extract_rmse_and_time(filepath)
#     if rmse.size == 0:
#         print(f"⚠️ 文件 {filename} 中未提取到 RMSE")
#         continue

#     iterations = np.arange(1, len(rmse)+1)
#     plt.plot(iterations, rmse, label=f"{algorithm_name} ({time_total:.2f}s)", color=colors[idx])

#     # 标注运行时间
#     if time_total is not None:
#         plt.text(iterations[-1], rmse[-1],
#                  f"{algorithm_name} ({time_total:.2f}s)",
#                  fontsize=16, color=colors[idx])

# plt.title("RMSE 随迭代次数变化")
plt.xlabel("Time(sec)")
plt.ylabel("RMSE")
plt.yscale('log')
plt.xlim(-0.5, 14)
# plt.legend(loc="lower left", bbox_to_anchor=(1.02, 0), borderaxespad=0, frameon=False)
plt.legend(loc="lower right", frameon=True)
plt.grid(False)
plt.tight_layout()

#保存图片
#bbox_inches='tight'表示指定将图表多余的空白区域裁减掉
plt.savefig('testcase/test0704/monkeys.png', bbox_inches='tight')

#显示图片
plt.show()
