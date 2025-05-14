# 获取对应国家的经纬度
import pandas as pd

# 读取两个Excel文件
file1 = pd.read_excel('世界各国经纬度坐标.xlsx')  # 文件1，包含“Location”, “Longitude”, “Latitude”
file2 = pd.read_excel('国家-数量.xlsx')  # 文件2，包含“research country” 和 “count”

# 合并数据：通过文件1中的“Location”列与文件2中的“research country”列进行匹配
merged_df = pd.merge(file2, file1, left_on='research country', right_on='Country', how='left')

# 保存结果为一个新的Excel文件
merged_df.to_excel('国家-数量-经纬度.xlsx', index=False)

print("匹配完成，结果已保存为国家-数量-经纬度.xlsx")

