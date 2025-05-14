import pandas as pd

# 读取文件1 - 大洲与国家对应关系
# 假设文件1中每个大洲作为列名，下面是对应的国家
df_continents = pd.read_excel('各大洲国家-分类.xlsx')

# 转换文件1的数据格式，创建国家到大洲的映射字典
continent_mapping = {}
for continent in df_continents.columns:
    countries = df_continents[continent].dropna().tolist()  # 去除空值并转换为列表
    for country in countries:
        continent_mapping[country] = continent

# 读取文件2 - 包含国家信息的数据
df_countries = pd.read_excel('split_continents_output.xlsx')

# 添加新列'Continent'，根据'Country'列映射大洲信息
df_countries['Continent'] = df_countries['Country'].map(continent_mapping)

# 保存结果到新文件或直接输出
df_countries.to_excel('文件2_带大洲信息.xlsx', index=False)

# 如果需要查看结果
print(df_countries.head())