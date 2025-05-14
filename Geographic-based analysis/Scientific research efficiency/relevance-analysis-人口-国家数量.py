import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 读取Excel数据
df = pd.read_excel("总-国家-数量-人口密度-经纬度.xlsx")

# 数据清洗
df.dropna(subset=['count', 'average_density'], inplace=True)

# 计算相关系数
corr_coef, p_value = stats.pearsonr(df['count'], df['average_density'])

# 线性回归分析
slope, intercept, r_value, p_val, std_err = stats.linregress(
    df['average_density'], df['count']
)
r_squared = r_value**2

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['average_density'], df['count'], alpha=0.6, edgecolors='w')

# 添加回归线（修正拼写错误）
x_values = np.linspace(df['average_density'].min(), df['average_density'].max(), 100)
y_values = slope * x_values + intercept  # 确保变量名与回归结果一致
plt.plot(x_values, y_values, color='r', linestyle='--', linewidth=2)

# 设置对数坐标轴（根据数据分布可选）
plt.xscale('log')  # 如果人口密度范围较大则启用
plt.yscale('log')  # 如果论文数量范围较大则启用

# 添加标注
plt.text(0.05, 0.95,
         f'R² = {r_squared:.2f}',
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top')

# 图表装饰
plt.xlabel('Population Density (people/km²)', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.title('Correlation between Published paper Count and Population Density in a country(all)', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()