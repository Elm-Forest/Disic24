import pandas as pd

# 读取CSV文件
file_path = 'K:/dataset/Disic2024challenge/train-metadata.csv'
df = pd.read_csv(file_path)

# 预览前20行
print("CSV文件的前20行内容:")
print(df.head(20))
