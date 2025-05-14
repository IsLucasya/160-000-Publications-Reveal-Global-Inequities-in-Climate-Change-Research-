import pandas as pd

# 读取文件1和文件2
file1 = pd.read_excel('总-被研究国家-数量-经纬度.xlsx')
file2 = pd.read_excel('国家-1979-2016增量.xlsx')

# 合并两个文件，基于国家名称
merged_data = pd.merge(file1, file2, left_on='research country', right_on='CTR_MN_NM', how='left')

# 选择需要的列
result = merged_data[['research country', '增量占比（极端高温人口）']]

# 将结果保存到新的Excel文件
result.to_excel('结果文件.xlsx', index=False)

print("处理完成，结果已保存到 '结果文件.xlsx'")