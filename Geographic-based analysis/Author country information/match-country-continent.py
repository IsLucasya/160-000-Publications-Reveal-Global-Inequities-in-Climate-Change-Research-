import pandas as pd

# 读取文件1和文件2
file1 = pd.read_excel('国家-数量-经纬度.xlsx')  # 替换为实际文件名
file2 = pd.read_excel('各大洲国家-分类.xlsx')  # 替换为实际文件名

# 创建一个字典，存储国家与大洲的映射关系
continent_mapping = {}

# 遍历文件2的每一列（每个大洲）
for continent in file2.columns:
    # 获取该大洲下的国家列表
    countries = file2[continent].dropna().tolist()
    # 将国家与大洲的映射关系存入字典
    for country in countries:
        continent_mapping[country] = continent

# 在文件1中新增一列“Continent”，存储匹配到的大洲
file1['Continent'] = file1['Country'].map(continent_mapping)

# 保存结果到新的Excel文件
file1.to_excel('匹配结果.xlsx', index=False)

print("匹配完成，结果已保存到 '匹配结果.xlsx'")