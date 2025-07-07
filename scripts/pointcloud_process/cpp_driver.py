"""
Description : 使用python驱动cpp编译好的exe文件，可对程序进行输入，并将输出记录到log.txt文件
"""

import tkinter as tk
from tkinter import filedialog, TK
import subprocess
import os

# 打开文件选择对话框
root = tk.Tk()
root.withdraw()  # 不显示主窗口
ply_path = filedialog.askopenfilename(
    title="选择一个 PLY 文件",
    filetypes=[("PLY files", "*.ply")]
)
name_prefix = os.path.splitext(os.path.basename(ply_path))[0]
output_dir = 'testcase/test0630'
os.makedirs(output_dir, exist_ok=True)

# 判断是否选择了文件
if not ply_path:
    print("未选择文件，退出。")
    exit()

# 可执行文件路径（你 C++ 编译后生成的 .exe 文件）
exe_path = r"D:/C++_Projects/PCL_Deploy/x64/Release/PCL_Deploy.exe"  # 请修改成你自己的路径

# 检查 exe 是否存在
if not os.path.isfile(exe_path):
    print(f"可执行文件不存在: {exe_path}")
    exit()

# 日志文件路径
log_file = os.path.join(output_dir, f"{name_prefix}_log.txt")
# 调用 C++ 程序并监听输出
with open(log_file, "w", encoding="utf-8") as log:
    print(f"启动程序: {exe_path} {ply_path}")
    process = subprocess.Popen(
        [exe_path, ply_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # 自动解码为字符串
    )

    # 实时读取输出
    while True:
        out_line = process.stdout.readline()
        err_line = process.stderr.readline()

        if out_line:
            print(out_line.strip())
            log.write(out_line)
        if err_line:
            print("错误:", err_line.strip())

        if not out_line and not err_line and process.poll() is not None:
            break

    # 获取程序返回值
    return_code = process.wait()

print(f"\n运行结束，退出码: {return_code}")
print(f"日志已保存到: {os.path.abspath(log_file)}")
