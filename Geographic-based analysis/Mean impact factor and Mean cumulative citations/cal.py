#计算各国平均影响因子
import pandas as pd

# 读取Excel文件
df = pd.read_excel('总-作者国家-引用次数-大洲.xlsx')  # 替换为你的文件路径

# 检查列名是否正确，如果不匹配请调整
# print(df.columns)  # 可以取消注释查看列名

# 计算每个国家的平均影响因子
average_if_by_country = df.groupby('author Country')['Times Cited'].mean().reset_index()

# 重命名列使其更清晰
average_if_by_country = average_if_by_country.rename(columns={'Times Cited': '平均引用次数'})

# 打印结果
print(average_if_by_country)

# 可选：将结果保存到新的Excel文件
average_if_by_country.to_excel('average_impact_factor_by_country.xlsx', index=False)