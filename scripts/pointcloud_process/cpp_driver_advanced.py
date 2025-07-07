"""
Description : 正式版，对zcl点云配准程序的驱动脚本
使用python驱动cpp编译好的exe文件，可对程序进行输入，并将输出记录到log.txt文件
"""

import tkinter as tk
from tkinter import filedialog
import numpy as np
import subprocess
import os

def choose_directory():
    """打开文件选择对话框"""
    root = tk.Tk()
    root.withdraw()  # 不显示主窗口
    folder_selected = filedialog.askdirectory(title="选择一个包含点云和gt的文件夹")
    root.destroy
    return folder_selected

def read_ground_truth(file_path):
    """读取一个4×4的矩阵"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            matrix_values = []
            for line in lines:
                matrix_values.extend(map(float, line.strip().split()))
            if len(matrix_values) != 16:
                raise ValueError(f"Expected 16 values for 4x4 matrix, got {len(matrix_values)}")
            return np.array(matrix_values).reshape((4, 4))
    except Exception as e:
        print(f"Error reading ground truth from {file_path}: {e}")
        return None

def main():
    # 1. 弹出文件管理器选择一个文件夹
    selected_folder = choose_directory()
    if not selected_folder:
        print("未选择任何文件夹。")
        return

    out_path = selected_folder
    print(f"选择的文件夹路径为：{out_path}")

    # 2. 拼接文件路径
    basename = os.path.basename(out_path)
    file_source = os.path.join(out_path, f"{basename}_source.ply")
    file_target = os.path.join(out_path, f"{basename}_target.ply")
    ground_truth_file = os.path.join(out_path, f"{basename}_ground_truth.txt")

    if not os.path.exists(file_source):
        print(f"Error: Source file not found at {file_source}")
        return
    if not os.path.exists(file_target):
        print(f"Error: Target file not found at {file_target}")
        return
    if not os.path.exists(ground_truth_file):
        print(f"Error: Ground truth file not found at {ground_truth_file}")
        return

    # 3. 读取真值矩阵
    gt_trans_matrix = read_ground_truth(ground_truth_file)
    if gt_trans_matrix is None:
        return

    gt_trans_flat = gt_trans_matrix.flatten().tolist()
    gt_trans_args = [str(x) for x in gt_trans_flat]

    # 4. 可选算法列表
    algorithms = ["ICP", "AA_ICP", "FICP", "RICP", "PPL", "RPPL", "SparseICP", "SICPPPL", "EXPICP"]
    
    # 可执行文件路径（你 C++ 编译后生成的 .exe 文件）
    exe_path = r"D:/C++_Projects/PCL_Deploy/x64/Release/PCL_Deploy.exe"  # 请修改成你自己的路径

    # 检查 exe 是否存在
    if not os.path.isfile(exe_path):
        print(f"可执行文件不存在: {exe_path}")
        exit()

    # 调用 C++ 程序并监听输出
    # 5. 循环调用每个算法
    for algo in algorithms:
        print(f"\n\n--- Running {algo} ---")
        log_path = os.path.join(out_path, f"{basename}_{algo}_log.txt")

        args = [exe_path, file_source, file_target, out_path + os.sep, algo]
        args.extend(gt_trans_args)

        with open(log_path, "w") as logfile:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end="")       # 实时输出到终端
                logfile.write(line)       # 保存日志到文件
            proc.wait()

        input(f"\n[{algo}] 可视化窗口已结束，按回车键继续执行下一个算法...")

if __name__ == "__main__":
    main()