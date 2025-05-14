import pandas as pd

# 读取Excel文件
df = pd.read_excel('总-作者国家-近5年IF-大洲.xlsx')

# 按“Continent”列分组，计算各大洲的平均影响因子
# 1. 计算各大洲的文献数量
continent_counts = df['Continent'].value_counts()

# 2. 计算各大洲的影响因子总和
continent_if_sum = df.groupby('Continent')['5年IF'].sum()

# 3. 计算各大洲的平均影响因子
average_if_by_continent = continent_if_sum / continent_counts

# 确保索引匹配并处理NaN
average_if_by_continent = average_if_by_continent.reindex(continent_counts.index, fill_value=0)

# 输出结果
print("各大洲的文献数量：")
print(continent_counts)
print("\n各大洲的引用数量总和：")
print(continent_if_sum)
print("\n各大洲的平均引用数量：")
print(average_if_by_continent)

# 如果需要保存结果到新的Excel文件
result = pd.DataFrame({
    'Continent': continent_counts.index,
    '文献数量': continent_counts.values,
    '引用数量总和': continent_if_sum.reindex(continent_counts.index, fill_value=0).values,
    '平均引用数量': average_if_by_continent.values
})

result.to_excel('average_if_by_continent.xlsx', index=False)