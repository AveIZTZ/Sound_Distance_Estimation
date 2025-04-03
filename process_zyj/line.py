import os
import pandas as pd

def count_csv_rows(folder_path):
    max_rows = 0
    max_file = None
    
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    row_count = sum(1 for _ in f) - 1  # 减去表头行
                
                if row_count > max_rows:
                    max_rows = row_count
                    max_file = file
            except Exception as e:
                print(f"文件 {file} 处理时出错: {e}")
    
    if max_file:
        print(f"最大行数的CSV文件: {max_file}, 行数: {max_rows}")
    else:
        print("文件夹中没有CSV文件或无法读取文件。")

# 设置要检查的文件夹路径
folder_path = "data/input/metadata_dev/test_zyj"  # 替换为你的CSV文件夹路径
count_csv_rows(folder_path)