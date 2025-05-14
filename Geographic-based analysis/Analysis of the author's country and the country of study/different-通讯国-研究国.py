import pandas as pd

# 读取Excel文件（假设文件名为'data.xlsx'，请根据实际文件名修改）
df = pd.read_excel('作者大洲-研究大洲.xlsx')

# 去重：基于“author Continent”和“research Continent”两列，保留第一次出现的行
# 如果有“数量”列，则对数量求和
if '数量' in df.columns:
    df = df.groupby(['author Continent', 'research Continent'])['数量'].sum().reset_index()
else:
    df = df.drop_duplicates(subset=['author Continent', 'research Continent'], keep='first')

# 1. 统计“author Continent”和“research Continent”相同的情况
same_continent = df[df['author Continent'] == df['research Continent']]
same_continent_counts = same_continent.groupby('author Continent')['数量'].sum() if '数量' in df.columns else same_continent['author Continent'].value_counts()

# 2. 统计“author Continent”和“research Continent”不同的情况
different_continent = df[df['author Continent'] != df['research Continent']]
different_count = different_continent['数量'].sum() if '数量' in df.columns else len(different_continent)

# 创建不同大洲的组合统计
different_combinations = different_continent.groupby(['author Continent', 'research Continent'])['数量'].sum() if '数量' in df.columns else different_continent.groupby(['author Continent', 'research Continent']).size()

# 创建结果DataFrame
results = {
    '相同大洲统计': pd.DataFrame({
        '大洲': same_continent_counts.index,
        '数量': same_continent_counts.values
    }),
    '不同大洲总数': pd.DataFrame({
        '描述': ['不同大洲的总数'],
        '数量': [different_count]
    }),
    '不同大洲组合': pd.DataFrame({
        '作者所在大洲': different_combinations.index.get_level_values(0),
        '研究大洲': different_combinations.index.get_level_values(1),
        '数量': different_combinations.values
    })
}

# 将结果导出到Excel文件，每个统计结果放在不同的工作表中
with pd.ExcelWriter('continent_analysis_results.xlsx') as writer:
    results['相同大洲统计'].to_excel(writer, sheet_name='相同大洲统计', index=False)
    results['不同大洲总数'].to_excel(writer, sheet_name='不同大洲总数', index=False)
    results['不同大洲组合'].to_excel(writer, sheet_name='不同大洲组合', index=False)

print("分析完成，结果已保存到 'continent_analysis_results.xlsx' 文件中")