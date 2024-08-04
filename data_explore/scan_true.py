import pandas as pd

# 读取CSV文件
csv_file = 'K:/dataset/Disic2024challenge/train-metadata.csv'
df = pd.read_csv(csv_file)

# 过滤出target等于1的行
positive_cases = df[df['target'] == 1]

# 打印isic_id
print("target等于1的isic_id:")
print(positive_cases['isic_id'].values)
