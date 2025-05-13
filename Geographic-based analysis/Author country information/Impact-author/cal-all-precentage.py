import pandas as pd

# 读取Excel文件
file = pd.read_excel('国家-数量-经纬度-大洲.xlsx')  # 替换为实际文件名

# 按“Continent”列分组，并计算“文献占比”的总和
continent_total_percentage = file.groupby('Continent')['文献占比'].sum().reset_index()

# 重命名列以便更清晰
continent_total_percentage.columns = ['Continent', '总文献占比']

# 保存结果到新的Excel文件
continent_total_percentage.to_excel('影响-各大洲总占比.xlsx', index=False)

print("计算完成，结果已保存到 '各大洲总占比.xlsx'")