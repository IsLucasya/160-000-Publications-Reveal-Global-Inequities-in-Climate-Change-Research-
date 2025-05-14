import pandas as pd

# 读取Excel文件
# 假设文件名为 'research_data.xlsx'，请根据实际情况修改文件名
df = pd.read_excel('作者大洲-研究大洲-加补充大洲.xlsx')

# 创建两个空字典来存储统计结果
same_continent_counts = {}  # 相同大洲的统计
diff_continent_counts = {}  # 不同大洲的统计

# 遍历数据进行统计
for index, row in df.iterrows():
    author_cont = row['author Continent']
    research_cont = row['research Continent']

    # 如果大洲相同
    if author_cont == research_cont:
        if author_cont in same_continent_counts:
            same_continent_counts[author_cont] += 1
        else:
            same_continent_counts[author_cont] = 1
    # 如果大洲不同
    else:
        # 创建不同大洲的组合键
        diff_key = f"{author_cont} - {research_cont}"
        if diff_key in diff_continent_counts:
            diff_continent_counts[diff_key] += 1
        else:
            diff_continent_counts[diff_key] = 1

# 将统计结果转换为DataFrame
same_df = pd.DataFrame(list(same_continent_counts.items()),
                       columns=['Continent', 'Count'])
same_df['Type'] = 'Same Continent'

diff_df = pd.DataFrame(list(diff_continent_counts.items()),
                       columns=['Continent Combination', 'Count'])
diff_df['Type'] = 'Different Continent'

# 添加标题行
same_df.columns = ['相同大洲', '数量', '类型']
diff_df.columns = ['大洲组合', '数量', '类型']

# 创建Excel writer对象
with pd.ExcelWriter('continent_analysis_results.xlsx') as writer:
    # 写入相同大洲的统计
    same_df.to_excel(writer, sheet_name='Analysis', startrow=0, index=False)
    # 写入不同大洲的统计（在相同大洲统计下方）
    diff_df.to_excel(writer, sheet_name='Analysis',
                     startrow=len(same_df) + 2, index=False)

print("分析完成，结果已导出到 'continent_analysis_results.xlsx' 文件中")