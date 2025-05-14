import pandas as pd
import numpy as np

# 1. 读取原始数据
df = pd.read_excel("总-作者国家-引用次数-大洲.xlsx")  # 替换为你的文件路径

# 2. 按大洲（Continent）分组计算统计量
continent_stats = df.groupby("Continent")["Times Cited"].agg([
    ("Mean_Times_Cited", "mean"),          # 计算均值
    ("Std_Dev", "std"),           # 计算标准差（默认ddof=1，样本标准差）
    ("Count", "count")         # 计算样本量
])

# 3. 计算标准误差（SE）
continent_stats["SE"] = continent_stats["Std_Dev"] / np.sqrt(continent_stats["Count"])

# 4. 重置索引（将Continent从索引变为列）
continent_stats = continent_stats.reset_index()

# 5. 保存结果
output_path = "continent_IF_stats.xlsx"
continent_stats.to_excel(output_path, index=False)

print(f"计算完成！结果已保存至: {output_path}")
print(continent_stats)  # 预览结果