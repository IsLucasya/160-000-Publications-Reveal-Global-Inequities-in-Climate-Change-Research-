import pandas as pd
import numpy as np

# 1. 读取原始数据
df = pd.read_excel("总-作者国家-引用次数-大洲.xlsx")  # 替换为你的文件路径

# 2. 按国家分组计算统计量
country_stats = df.groupby("author Country")["Times Cited"].agg([
    ("Mean_Times Cited", "mean"),          # 国家平均影响因子
    ("Std_Dev", "std"),           # 国家内标准差（样本标准差）
    ("Count", "count")            # 国家文献数量
]).reset_index()                  # 将国家从索引变为列

# 3. 计算标准误差（SE = SD / √n）
country_stats["SE"] = country_stats["Std_Dev"] / np.sqrt(country_stats["Count"])

# 4. 按平均IF排序（可选）
country_stats = country_stats.sort_values("Mean_Times Cited", ascending=False)

# 5. 保存结果
output_path = "country_Mean_Times Cited-stats.xlsx"
country_stats.to_excel(output_path, index=False)

print(f"计算完成！结果已保存至: {output_path}")
print(country_stats.head())  # 预览前5个国家