import pandas as pd

# 读取文件1和文件2
file1 = pd.read_excel('研究国家-数量-2023年GDP.xlsx')
file2 = pd.read_excel('国家划分（按照收入水平）.xlsx')

# 合并两个文件，基于国家名称
merged_data = pd.merge(file1, file2, left_on='research country', right_on='TableName', how='left')

# 只保留需要的列
result = merged_data[['research country', 'IncomeGroup']]

# 将结果保存到新的Excel文件
result.to_excel('结果文件.xlsx', index=False)

print("处理完成，结果已保存到 '结果文件.xlsx'")