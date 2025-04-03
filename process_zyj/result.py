import os
import pandas as pd
import numpy as np

# 设置文件夹路径
folder1 = 'data/input/metadata_dev/test'  # 替换为 folder1 的路径
folder2 = 'results/test_WER_90_200ms_0.2'  # 替换为 folder2 的路径
folder3 = "/home/yujiezhu/data/test_WER_90/test_WER_90_200ms/test/0.2/distance/"

# 如果 folder3 不存在，创建该文件夹
if not os.path.exists(folder3):
    os.makedirs(folder3)
# 获取 folder1 中所有的 .csv 文件
fold_name = "fold91202_"
csv_files = [f for f in os.listdir(folder1) if f.endswith('.csv') and f.startswith(fold_name)]
csv_files.sort()  # 按文件名排序
difference_list = []
RDE_list = []

# 遍历每个 csv 文件
for csv_file in csv_files:
    # 构建 folder1 和 folder2 中的文件路径
    file1_path = os.path.join(folder1, csv_file)
    file2_path = os.path.join(folder2, csv_file)
    
    # 读取第一个 csv 文件
    df1 = pd.read_csv(file1_path, header=None)
    
    # 读取第二个 csv 文件
    df2 = pd.read_csv(file2_path, header=None)
    
    # 根据第一个 csv 文件的行数截取第二个 csv 文件的数据
    num_rows = len(df1)
    df2_subset = df2.head(num_rows)
    
    # 计算第二列的平均值
    mean_value = df2_subset.iloc[:, 1].mean() # 假设第二列是从 0 开始的索引
    tv = mean_value**2 - (0.63)**2    
    if tv < 0:
        mean_value = mean_value - 0.63
    else:
        mean_value = np.sqrt(tv)
    distance = df1.iloc[0, 5]
    difference = np.abs(distance - mean_value)
    difference_list.append(difference)
    RDE = difference / distance
    RDE_list.append(RDE)

    # 将 distance 保存为 .npy 文件
    distance_filename = f"{csv_file.replace(fold_name, '').replace('.csv', '.npy')}"
    distance_path = os.path.join(folder3, distance_filename)
    #np.save(distance_path, mean_value)
    
    print(f'{csv_file} 的真实值：{distance}，估计值: {mean_value}，相差: {difference}, 相差百分比: {RDE}')

difference_mean = np.mean(difference_list)
print(f'所有文件的平均相差: {difference_mean}')
RDE_mean = np.mean(RDE_list)
print(f'所有文件的平均相差百分比: {RDE_mean}')