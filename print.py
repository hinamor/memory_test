import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def read_cpu_fragmentation_from_excel(file_path):
    """
    读取Excel文件中的平均CPU碎片率列数据
    :param file_path: Excel文件路径
    :return: 清洗后的CPU碎片率列表
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return None
    
    try:
        # 读取Excel文件（默认读取第一个工作表）
        df = pd.read_excel(file_path)
        
        # 查找目标列（支持列名的模糊匹配，防止列名微小差异）
        target_col = None
        for col in df.columns:
            if "平均CPU碎片率" in str(col) and "%" in str(col):
                target_col = col
                break
        
        if target_col is None:
            print(f"错误：文件 {file_path} 中未找到'平均CPU碎片率(%)'列！")
            print(f"文件中的列名：{list(df.columns)}")
            return None
        
        # 提取数据并清洗（去除空值、非数值型数据）
        cpu_frag_data = df[target_col].dropna()  # 去除空值
        cpu_frag_data = pd.to_numeric(cpu_frag_data, errors='coerce').dropna()  # 转为数值型，去除非数值
        
        if len(cpu_frag_data) == 0:
            print(f"警告：文件 {file_path} 中'平均CPU碎片率(%)'列无有效数据！")
            return None
        
        return cpu_frag_data.tolist()
    
    except Exception as e:
        print(f"读取文件 {file_path} 失败：{str(e)}")
        return None

def plot_cpu_fragmentation_comparison(file1_path, file2_path, file1_label="文件1", file2_label="文件2"):
    """
    绘制两个Excel文件的CPU碎片率对比折线图
    :param file1_path: 第一个Excel文件路径
    :param file2_path: 第二个Excel文件路径
    :param file1_label: 第一个文件在图表中的标签
    :param file2_label: 第二个文件在图表中的标签
    """
    # 读取两个文件的数据
    data1 = read_cpu_fragmentation_from_excel(file1_path)
    data2 = read_cpu_fragmentation_from_excel(file2_path)
    
    # 检查数据是否有效
    if data1 is None and data2 is None:
        print("两个文件都无有效数据，无法绘制图表！")
        return
    if data1 is None:
        print(f"仅 {file2_label} 有有效数据，仅绘制该文件的折线图")
    if data2 is None:
        print(f"仅 {file1_label} 有有效数据，仅绘制该文件的折线图")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制折线
    if data1 is not None:
        ax.plot(range(1, len(data1)+1), data1, marker='o', linewidth=2, label=file1_label, color='#1f77b4')
    if data2 is not None:
        ax.plot(range(1, len(data2)+1), data2, marker='s', linewidth=2, label=file2_label, color='#ff7f0e')
    
    # 设置图表样式
    ax.set_title('平均CPU碎片率对比折线图', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('调度轮次', fontsize=12)
    ax.set_ylabel('平均CPU碎片率 (%)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)  # 添加网格
    ax.legend(fontsize=11, loc='upper right')  # 添加图例
    
    # 设置坐标轴刻度
    ax.tick_params(axis='both', labelsize=10)
    
    # 调整布局，防止标签被截断
    plt.tight_layout()
    
    # 保存图表（可选）
    save_path = "cpu_fragmentation_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至：{save_path}")
    
    # 显示图表
    plt.show()

def get_latest_excel_file(folder_path):
    """
    获取指定文件夹下最新修改的Excel文件（.xlsx）
    :param folder_path: 文件夹路径
    :return: 最新Excel文件的完整路径，若不存在则返回None
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在！")
        return None
    
    # 获取文件夹下所有Excel文件
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    if not excel_files:
        print(f"警告：文件夹 {folder_path} 中未找到任何Excel文件（.xlsx）！")
        return None
    
    # 按文件修改时间排序，取最新的文件
    latest_file = max(excel_files, key=os.path.getmtime)
    file_mtime = os.path.getmtime(latest_file)
    file_mtime_str = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"找到文件夹 {folder_path} 下最新的Excel文件：")
    print(f"  文件名：{os.path.basename(latest_file)}")
    print(f"  修改时间：{file_mtime_str}")
    
    return latest_file

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 配置默认的两个文件夹路径（你可以根据实际情况修改）
    DEFAULT_FOLDER1 = "result/增加碎片权重"
    DEFAULT_FOLDER2 = "result/不增加碎片权重"
    
    # # 让用户选择操作模式
    # print("请选择操作模式：")
    # print("1 - 自动获取指定文件夹下的最新Excel文件")
    # print("2 - 手动输入Excel文件路径")
    
    # while True:
    #     choice = input("\n请输入选择（1/2）：").strip()
    #     if choice in ["1", "2"]:
    #         break
    #     print("输入无效，请输入 1 或 2！")
    
    # if choice == "1":
        # 模式1：自动获取最新文件
        # 可自定义文件夹路径，回车使用默认值
    # folder1 = input(f"\n请输入第一个文件夹路径（默认：{DEFAULT_FOLDER1}）：").strip() or DEFAULT_FOLDER1
    # folder2 = input(f"请输入第二个文件夹路径（默认：{DEFAULT_FOLDER2}）：").strip() or DEFAULT_FOLDER2
    
    # 获取最新文件
    EXCEL_FILE1 = get_latest_excel_file(DEFAULT_FOLDER1)
    EXCEL_FILE2 = get_latest_excel_file(DEFAULT_FOLDER2)
    
    # 检查文件是否获取成功
    if EXCEL_FILE1 is None or EXCEL_FILE2 is None:
        print("无法获取有效的Excel文件，程序退出！")
        exit(1)
    
    # else:
    #     # 模式2：手动输入文件路径
    #     EXCEL_FILE1 = input("\n请输入第一个Excel文件路径：").strip()
    #     EXCEL_FILE2 = input("请输入第二个Excel文件路径：").strip()
        
    #     # 检查文件是否存在
    #     if not os.path.exists(EXCEL_FILE1):
    #         print(f"错误：文件 {EXCEL_FILE1} 不存在！")
    #         exit(1)
    #     if not os.path.exists(EXCEL_FILE2):
    #         print(f"错误：文件 {EXCEL_FILE2} 不存在！")
    #         exit(1)
    
    # 自定义图表标签（回车使用默认值）
    label1 = DEFAULT_FOLDER1
    label2 = DEFAULT_FOLDER2
    
    # 绘制对比折线图
    print("\n开始绘制图表...")
    plot_cpu_fragmentation_comparison(
        file1_path=EXCEL_FILE1,
        file2_path=EXCEL_FILE2,
        file1_label=label1,
        file2_label=label2
    )