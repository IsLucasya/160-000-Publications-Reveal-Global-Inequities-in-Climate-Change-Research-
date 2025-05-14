import pandas as pd
import numpy as np

# 1. 读取原始数据
df = pd.read_excel("总-作者国家-引用次数-大洲-南北地区.xlsx")  # 替换为你的文件路径

# 2. 按地区（region）分组计算统计量
region_stats = df.groupby("region")["Times Cited"].agg([
    ("Mean_IF", "mean"),          # 计算均值
    ("Std_Dev", "std"),           # 计算标准差（样本标准差，ddof=1）
    ("Count", "count")            # 计算样本量
])

# 3. 计算标准误差（SE = SD / √n）
region_stats["SE"] = region_stats["Std_Dev"] / np.sqrt(region_stats["Count"])

# 4. 重置索引（将region从索引变为列）
region_stats = region_stats.reset_index()

# 5. 保存结果
output_path = "region_IF_stats.xlsx"
region_stats.to_excel(output_path, index=False)

print(f"计算完成！结果已保存至: {output_path}")
print(region_stats)  # 预览结果