import pandas as pd
from sklearn.utils import shuffle

# 加载CSV文件
df = pd.read_csv('K:/dataset/Disic2024challenge/train-metadata.csv')

# 获取target值为1的行
target_1 = df[df['target'] == 1]

# 计算target为1的行数
num_target_1 = len(target_1)

# 从target值为0的行中随机选择同等数量的行
target_0 = df[df['target'] == 0].sample(n=num_target_1)

# 合并两个DataFrame
result_df = pd.concat([target_1, target_0])

# 打乱数据行的顺序
result_df = shuffle(result_df)

# 将结果写入新的CSV文件
result_df.to_csv('balanced_dataset.csv', index=False)
