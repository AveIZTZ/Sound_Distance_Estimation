import os
import numpy as np
import pandas as pd

fold_name = "fold91830"

def trim_csv_by_npy(npy_folder, csv_folder):
    for npy_file in os.listdir(npy_folder):
        if npy_file.startswith(fold_name) and npy_file.endswith(".npy"):
            npy_path = os.path.join(npy_folder, npy_file)
            csv_path = os.path.join(csv_folder, npy_file.replace(".npy", ".csv"))
            
            if not os.path.exists(csv_path):
                print(f"对应的 CSV 文件未找到: {csv_path}")
                continue
            
            # 读取 .npy 文件，获取行数
            data = np.load(npy_path)
            num_rows = data.shape[0]  # 获取 .npy 的行数
            
            # 读取 .csv 文件
            df = pd.read_csv(csv_path, header=None)
            
            if len(df) > num_rows:
                df = df.iloc[:num_rows]  # 截取前 num_rows 行
                df.to_csv(csv_path, index=False, header=None)  # 保存修改后的 CSV 文件
                print(f"已修改: {csv_path}，保留前 {num_rows} 行")
            else:
                print(f"{csv_path} 无需修改（行数匹配）")

if __name__ == "__main__":
    npy_folder = "data/processed/mic_dev_label"  # 替换为 .npy 文件所在目录
    csv_folder = "data/input/metadata_dev/test"  # 替换为 .csv 文件所在目录
    trim_csv_by_npy(npy_folder, csv_folder)