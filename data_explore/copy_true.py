import pandas as pd
import shutil
import os

# 读取CSV文件
csv_file = 'K:/dataset/Disic2024challenge/train-metadata.csv'
df = pd.read_csv(csv_file)

# 过滤出target等于1的行
positive_cases = df[df['target'] == 1]

# 获取所有isic_id
positive_isic_ids = positive_cases['isic_id'].values

# 定义源文件夹和目标文件夹
source_folder = 'K:/dataset/Disic2024challenge/train-image/image'
destination_folder = 'K:/dataset/Disic2024challenge/true_imgs'

# 创建目标文件夹（如果不存在）
os.makedirs(destination_folder, exist_ok=True)

# 复制图像文件
for isic_id in positive_isic_ids:
    file_name = f"{isic_id}.jpg"
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder, file_name)

    if os.path.exists(source_path):
        shutil.copy2(source_path, destination_path)
        print(f"复制 {file_name} 到 {destination_folder}")
    else:
        print(f"文件 {file_name} 不存在于 {source_folder}")

print("复制完成")
