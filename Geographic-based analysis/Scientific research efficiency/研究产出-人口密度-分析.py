import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm

# 1. 读取数据
# 假设你的 Excel 文件名为 "data.xlsx"，且数据在第一个 sheet 中
df = pd.read_excel("总体-国家-数量-人口密度-经纬度.xlsx")

# 2. 数据清洗
# 检查缺失值
print("缺失值情况：")
print(df.isnull().sum())

# 删除缺失值（如果有）
df = df.dropna()

# 检查异常值（例如文献总数或人口密度为负数）
df = df[(df["count"] > 0) & (df["average_density"] > 0)]

# 对文献总数和人口密度取对数以减少数据偏差
df["log_count"] = np.log(df["count"])
df["log_density"] = np.log(df["average_density"])

# 3. 相关性分析
# 计算皮尔逊相关系数
corr, p_value = pearsonr(df["log_count"], df["log_density"])
print(f"皮尔逊相关系数: {corr:.3f}, p值: {p_value:.3f}")

# 4. 绘制散点图
plt.figure(figsize=(8, 6))
sns.scatterplot(x="log_density", y="log_count", data=df, alpha=0.6)
plt.title("科研产出与人口密度的关系（对数尺度）")
plt.xlabel("人口密度（对数）")
plt.ylabel("文献总数（对数）")
plt.grid(True)
plt.show()

# 5. 回归分析
# 添加常数项（截距）
X = sm.add_constant(df["log_density"])  # 自变量：人口密度（对数）
y = df["log_count"]  # 因变量：文献总数（对数）

# 拟合线性回归模型
model = sm.OLS(y, X).fit()

# 输出回归结果
print(model.summary())

# 6. 结果解释
# 根据回归系数和显著性水平，判断人口密度对科研产出的影响
if model.params["log_density"] > 0 and model.pvalues["log_density"] < 0.05:
    print("人口密度对科研产出有显著正向影响。")
else:
    print("人口密度对科研产出无显著影响。")