import pandas as pd

# 读取文件1
df1 = pd.read_excel('各大洲国家-分类.xlsx')

# 读取文件2
df2 = pd.read_excel('总-作者国家-引用次数.xlsx')

# 创建一个字典，将国家映射到大洲
continent_mapping = {}
for continent in df1.columns:
    # 去除NaN值并转换为列表，同时清洗国家名称
    countries = df1[continent].dropna().apply(lambda x: x.strip().lower()).tolist()
    for country in countries:
        continent_mapping[country] = continent

# 清洗文件2中的国家名称
df2['author Country'] = df2['author Country'].apply(lambda x: x.strip().lower())

# 在文件2中创建一个新列“Continent”，并将国家映射到大洲
df2['Continent'] = df2['author Country'].map(continent_mapping)

# 检查是否有未匹配的国家
unmatched_countries = df2[df2['Continent'].isna()]['author Country'].unique()
if len(unmatched_countries) > 0:
    print("以下国家未能匹配到大洲：", unmatched_countries)
else:
    print("所有国家均已成功匹配到大洲。")

# 保存结果到新的Excel文件
df2.to_excel('总-作者国家-引用次数-大洲.xlsx', index=False)

print("处理完成，结果已保存到 '总-作者国家-引用次数-大洲.xlsx'")