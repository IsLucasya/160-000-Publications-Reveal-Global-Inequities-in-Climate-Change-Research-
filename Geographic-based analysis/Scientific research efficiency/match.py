import pandas as pd

# 读取文件1和文件2
file1 = pd.read_excel('国家-科研人数.xlsx')  # 替换为你的文件1路径
file2 = pd.read_excel('总-国家-数量-经纬度.xlsx')  # 替换为你的文件2路径

# 合并两个文件，基于“Country Name”和“author Country”列
# 注意：如果列名不完全一致，可以使用left_on和right_on参数
merged_data = pd.merge(
    file1,
    file2,
    left_on='Country Name',  # 文件1中的国家名称列
    right_on='author Country',  # 文件2中的国家名称列
    how='left'  # 以文件1为基础，保留文件1中的所有行
)

# 保存结果到新的Excel文件
merged_data.to_excel('合并后的文件.xlsx', index=False)

print("匹配完成，结果已保存到 '合并后的文件.xlsx'")