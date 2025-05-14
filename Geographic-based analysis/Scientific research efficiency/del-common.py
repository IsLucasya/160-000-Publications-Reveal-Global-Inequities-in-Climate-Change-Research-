import pandas as pd

# 读取 Excel 文件
file_path = '总-国家-数量-人口密度-经纬度.xlsx'  # 修改为你的文件路径
df = pd.read_excel(file_path)

# 按照国家名称合并相同国家的数量，并保留第一条经纬度信息
df_grouped = df.groupby('Country', as_index=False).agg({
    'count': 'sum',
    'average_density': 'first',# 汇总数量
    'longitude': 'first',  # 保留第一个国家的经度
    'latitude': 'first'
    # 保留第一个国家的纬度
})

# 输出处理后的结果，可以保存为新的 Excel 文件
df_grouped.to_excel('processed_file.xlsx', index=False)

# 查看处理后的数据
print(df_grouped)
