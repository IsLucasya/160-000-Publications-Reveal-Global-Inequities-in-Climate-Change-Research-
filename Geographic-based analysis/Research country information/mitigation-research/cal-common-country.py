import pandas as pd

# 读取Excel文件
df = pd.read_excel('processed_file.xlsx')  # 请根据你的文件路径修改

# 计算每个国家出现的次数
country_counts = df['research country'].value_counts()

# 创建一个新的DataFrame，包含唯一的国家及其重复次数
unique_countries = pd.DataFrame({
    'research country': country_counts.index,
    'count': country_counts.values
})

# 将结果保存为新的Excel文件
unique_countries.to_excel('output_file.xlsx', index=False)

print("去重和计数完成，结果已保存为output_file.xlsx")
