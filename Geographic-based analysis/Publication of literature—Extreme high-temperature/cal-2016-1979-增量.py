import pandas as pd

# 读取Excel文件
df = pd.read_excel('国家-年份-暴露人口.xlsx')

# 筛选出1979年和2016年的数据
df_1979 = df[df['year'] == 1979][['CTR_MN_NM', 'Exposure']]
df_2016 = df[df['year'] == 2016][['CTR_MN_NM', 'Exposure']]

# 将1979年和2016年的数据合并到一个DataFrame中
merged_df = pd.merge(df_1979, df_2016, on='CTR_MN_NM', suffixes=('_1979', '_2016'))

# 计算增加数
merged_df['Increase'] = merged_df['Exposure_2016'] - merged_df['Exposure_1979']

# 保存结果到新的Excel文件
merged_df.to_excel('exposure_increase.xlsx', index=False)

print("处理完成，结果已保存到 'exposure_increase.xlsx'")