import os
import shutil
import time
import scipy.io
import pandas as pd

# 设置路径（请自行修改）
audio_src_dir = "/home/yujiezhu/data/data_for_SED/train/noisy/"  # 原始音频文件夹
audio_dst_dir = "/home/yujiezhu/code/sound_distance_estimation/data_new/input/mic_dev/train"  # 目标音频文件夹
mat_src_dir   = "/home/yujiezhu/data/data_for_SED/train/data/"  # .mat 文件存放路径
csv_dst_dir   = "/home/yujiezhu/code/sound_distance_estimation/data_new/input/metadata_dev/train"  # .csv 文件存放路径

# 确保目标文件夹存在
os.makedirs(audio_dst_dir, exist_ok=True)
os.makedirs(csv_dst_dir, exist_ok=True)

fold_name = "fold11"

# 复制音频文件并重命名
def copy_and_rename_audio(src_1, dst_1, src_2, dst_2):
    filenames = sorted(os.listdir(src_1))
    filenames.sort()
    idx = 0

    for filename in filenames:
        if filename.lower().endswith((".wav", ".flac", ".mp3")):
            src_path = os.path.join(src_1, filename)
            new_name = f"{fold_name}_{filename}"
            dst_path = os.path.join(dst_1, new_name)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {src_path} -> {dst_path}")

            mat_path = os.path.join(src_2, filename.replace(".wav", ".mat"))
            mat_data = scipy.io.loadmat(mat_path)
            rmic        = mat_data['rmic']
            rsrc        = mat_data['rsrc']
            distance    = rsrc[0,0] - rmic[0,0]
            #distance    = mat_data['dis']
            distance    = distance.reshape(-1, 1)

            df = pd.DataFrame(distance, columns=["Distance"])
            df.insert(0, "Index", range(1, len(df) + 1))
            for i in range(1, 5):
                df.insert(i + 1, f"Col{i}", 0)
            df.insert(5, "Distance", df.pop("Distance"))
            df_repeated = pd.DataFrame()
            for i in range(100):
                df_copy = df.copy()
                df_copy["Index"] = df_copy["Index"] + (i * len(df))
                df_repeated = pd.concat([df_repeated, df_copy], ignore_index=True)
            csv_path = os.path.join(dst_2, new_name.replace(".wav", ".csv"))
            df_repeated.to_csv(csv_path, index=False, header=False)
            print(f"Converted: {mat_path} -> {csv_path}")
            idx = idx + 1

# 执行任务
copy_and_rename_audio(audio_src_dir, audio_dst_dir, mat_src_dir, csv_dst_dir)