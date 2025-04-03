import os
import shutil

# 定义源路径和目标路径
source_input_dir = 'data_temp/input'
source_processed_dir = 'data_temp/processed'
target_input_dir = 'data/input'
target_processed_dir = 'data/processed'

# 确保目标目录存在，不存在则创建
os.makedirs(target_input_dir, exist_ok=True)
os.makedirs(target_processed_dir, exist_ok=True)

def copy_all_files_with_subfolders(source_dir, target_dir, file_extension=None):
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"源目录 {source_dir} 不存在！")
        return
    
    # 遍历源目录中的所有文件和子目录
    for root, _, files in os.walk(source_dir):
        # 计算相对路径，保留子文件夹结构
        relative_path = os.path.relpath(root, source_dir)
        target_subfolder = os.path.join(target_dir, relative_path)
        files.sort()

        # 如果目标子文件夹不存在，则创建
        if not os.path.exists(target_subfolder):
            os.makedirs(target_subfolder)
        
        for file_name in files:
            source_path = os.path.join(root, file_name)
            
            # 如果指定了文件扩展名，则检查是否匹配
            if file_extension and not file_name.endswith(file_extension):
                continue
            
            # 构造目标路径
            target_path = os.path.join(target_subfolder, file_name)
            
            # 直接覆盖目标文件（如果存在）
            print(f"复制并覆盖：{source_path} -> {target_path}")
            shutil.copy(source_path, target_path)
    
# 执行文件复制
copy_all_files_with_subfolders(source_input_dir, target_input_dir)
copy_all_files_with_subfolders(source_processed_dir, target_processed_dir)

print("复制完成！")