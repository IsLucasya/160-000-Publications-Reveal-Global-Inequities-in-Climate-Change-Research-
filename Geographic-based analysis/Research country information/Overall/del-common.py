import pandas as pd

# 读取 Excel 文件
file_path = '总-分-国家-数量-经纬度.xlsx'  # 修改为你的文件路径
df = pd.read_excel(file_path)

# 按照国家名称合并相同国家的数量，并保留第一条经纬度信息
df_grouped = df.groupby('research country', as_index=False).agg({
    'count': 'sum',  # 汇总数量
    'longitude': 'first',  # 保留第一个国家的经度
    'latitude': 'first'    # 保留第一个国家的纬度
})

# 输出处理后的结果，可以保存为新的 Excel 文件
df_grouped.to_excel('总-国家-数量-经纬度.xlsx', index=False)

# 查看处理后的数据
print(df_grouped)
