import pandas as pd

# 假设你已经加载了数据到df
csv_file = 'K:/dataset/Disic2024challenge/train-metadata.csv'
df = pd.read_csv(csv_file, low_memory=False)

# 列出要统计的分类列
categorical_columns = [
    'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location',
    'tbp_lv_location_simple', 'attribution'
]

# 统计每个分类属性的类别数量
category_counts = {col: df[col].value_counts() for col in categorical_columns}

# 打印统计结果
for col, counts in category_counts.items():
    print(f"Category counts for {col}:\n{counts}\n")
