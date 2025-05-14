import pandas as pd

# 读取文件1和文件2
file1 = pd.read_excel('研究国家-数量-极端温度暴露人数增量.xlsx')
file2 = pd.read_excel('研究国家-数量-2023年GDP.xlsx')

# 合并两个文件，基于国家名称
merged_data = pd.merge(file1, file2, left_on='research country', right_on='research country', how='left')

# 提取匹配到的数据
matched_data = merged_data[['research country', '文献数量','极端温度暴露人数增量', 'X']]

# 提取未匹配到的国家
unmatched_data = merged_data[merged_data['X'].isna()][['research country', 'X']]

# 导出匹配到的数据到新文件
matched_data.to_excel('匹配到的国家GDP.xlsx', index=False)

# 导出未匹配到的国家到新文件
unmatched_data.to_excel('未匹配到的国家.xlsx', index=False)

print("处理完成，结果已导出到相应文件中。")