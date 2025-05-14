import pandas as pd

# 读取Excel文件
file_path = '总-期刊名称.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 对"Source Title"列进行去重
unique_journals = df['Source Title'].unique()

# 如果你想保留其他列的数据，只删除完全重复的行，可以使用：
df_deduplicated = df.drop_duplicates(subset=['Source Title'])

# 保存结果到新文件
output_path = 'deduplicated_journals.xlsx'
df_deduplicated.to_excel(output_path, index=False)

print(f"去重完成，结果已保存到 {output_path}")
print(f"原始记录数: {len(df)}，去重后记录数: {len(df_deduplicated)}")