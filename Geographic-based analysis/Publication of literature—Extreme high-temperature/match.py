import pandas as pd

# 读取文件1
file1 = pd.read_excel('总-国家-数量-经纬度.xlsx')

# 读取文件2
file2 = pd.read_excel('国家-1979-2016增量.xlsx')

# 获取文件2中的国家名称列表
countries_in_file2 = file2['CTR_MN_NM'].unique()

# 过滤文件1，只保留文件2中存在的国家
filtered_file1 = file1[file1['research country'].isin(countries_in_file2)]

# 保存过滤后的文件1
filtered_file1.to_excel('filtered_file1.xlsx', index=False)

# 找出文件2中存在但文件1中没有的国家
countries_not_in_file1 = set(countries_in_file2) - set(file1['research country'].unique())

# 将这些国家从文件2中提取出来
missing_countries_data = file2[file2['CTR_MN_NM'].isin(countries_not_in_file1)]

# 保存文件2中存在但文件1中没有的国家数据
missing_countries_data.to_excel('missing_countries.xlsx', index=False)

print("过滤完成，结果已保存到 'filtered_file1.xlsx'")
print("文件2中存在但文件1中没有的国家数据已保存到 'missing_countries.xlsx'")