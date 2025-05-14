import pandas as pd
import numpy as np

# 读取Excel文件
file_path = '世界各国总人口.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 选择第3列到第27列（索引从2到26，因为Python索引从0开始）
columns_to_use = df.iloc[:, 2:65]

# 检查数据类型，确保所有列都是数值类型
if not all(columns_to_use.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
    print("警告：某些列的数据类型不是数值类型，可能会导致计算错误。")
    # 将非数值列转换为数值类型（如果适用）
    columns_to_use = columns_to_use.apply(pd.to_numeric, errors='coerce')

# 检查是否有缺失值
if columns_to_use.isnull().any().any():
    print("警告：数据中存在缺失值（NaN），计算平均值时将被忽略。")

# 计算每行的非空值的平均数
row_means = columns_to_use.mean(axis=1)

# 将结果保存到新的列中
df['Row_Mean'] = row_means

# 输出结果
print(df[['Row_Mean']])

# 如果需要保存到新的Excel文件
output_file_path = '各国总人口-平均数.xlsx'  # 替换为你想保存的文件路径
df.to_excel(output_file_path, index=False)