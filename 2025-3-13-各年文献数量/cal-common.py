import pandas as pd

# 读取Excel文件
df = pd.read_excel('总-年份.xlsx')

# 计算每个年份的重复数量
year_counts = df['Publication Year'].value_counts().reset_index()
year_counts.columns = ['Publication Year', 'Count']

# 去重，保留每个年份的第一行
df_unique = df.drop_duplicates(subset=['Publication Year'])

# 合并去重后的数据和重复数量
df_result = pd.merge(df_unique, year_counts, on='Publication Year', how='left')

# 保存结果到新的Excel文件
df_result.to_excel('总-年份-数量.xlsx', index=False)

print("去重后的数据及重复数量已写入Excel文件 'your_file_unique_with_counts.xlsx'")