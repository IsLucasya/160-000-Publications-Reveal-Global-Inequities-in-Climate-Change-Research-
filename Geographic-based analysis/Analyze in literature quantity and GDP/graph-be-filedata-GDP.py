# 国家被研究文献数量和 GDP 的关系
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from adjustText import adjust_text
from scipy import stats
import matplotlib as mpl  # 新增全局字体设置库

# 设置全局字体为Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14  # 默认字体大小

# 读取 Excel 文件
data = pd.read_excel('研究国家-数量-2023年GDP.xlsx')

# 提取变量
X = data['X'].values.reshape(-1, 1)
Y = data['Y'].values
countries = data['research country']

# 选择需要标注的国家
annotate_gdp = data.nlargest(4, 'X')
annotate_papers = data.nlargest(4, 'Y')
remaining_data = data.drop(index=annotate_gdp.index.tolist() + annotate_papers.index.tolist())
annotate_sparse = remaining_data.sort_values(by=['X', 'Y']).iloc[::len(remaining_data)//8][:8]
annotate_indices = pd.concat([annotate_gdp, annotate_papers, annotate_sparse]).drop_duplicates().index

# 对数据取对数
data['log_X'] = np.log1p(data['X'])
data['log_Y'] = np.log1p(data['Y'])

# 创建图形
plt.figure(figsize=(12, 8), dpi=300)  # 提高分辨率

# 绘制散点图（修改点透明度为0.7更美观）
plt.scatter(data['log_X'], data['log_Y'], alpha=0.7, color='blue', label='Data points')
plt.scatter(data.loc[annotate_indices, 'log_X'],
            data.loc[annotate_indices, 'log_Y'],
            color='red', alpha=1, label='Annotated countries')

# 国家标签配置（统一使用Times New Roman）
text_positions = {
    'United States': (0.4, -0.1),
    'China': (-0.2, 0.1),
    'India': (-0.2, 0.1),
    'Japan': (0.1, -0.1),
    'Germany': (-0.4, 0.2),
    'United Kingdom': (-0.7, 0.1),
    'Australia': (-0.4, 0.1),
    'Bangladesh': (-0.4, 0.3),
    'Serbia': (-0.4, 0.5),
    'Estonia': (-0.8, 0.5),
    'Benin': (-0.4, 0.5),
    'Qatar': (0.5, -0.1),
    'Somalia': (-1.5, 0.1),
    'Tuvalu': (0.1, -0.1),
    'Austria': (-0.5, 0.9),
    'Tunisia': (-0.3, 0.1),
    'Costa Rica': (-1.2, 0.4),
    'North Macedonia': (-1.0, 0.2),
    'Armenia': (0.6, -0.3),
    'Kazakhstan': (0.7, -0.2),
    'Swaziland': (-0.3, -0.5),
    'Palau': (0.1, -0.1),
}

texts = []
for i in annotate_indices:
    country = countries[i]
    x_offset, y_offset = text_positions.get(country, (0, 0))
    texts.append(plt.text(
        data['log_X'][i] + x_offset,
        data['log_Y'][i] + y_offset,
        country,
        fontsize=14,  # 统一标签字体大小
        fontname='Times New Roman',  # 显式指定字体
        ha='center',
        va='center'
    ))

# 调整标签位置（优化连接线为虚线）
adjust_text(texts,
            arrowprops=dict(
                arrowstyle='-',
                color='gray',
                linestyle='--',  # 改为虚线
                lw=0.000001,
                shrinkA=0,
                shrinkB=0
            ),
            expand_points=(1.5, 1.5),
            force_text=(0.8, 0.8))

# 回归分析
log_X = data['log_X'].values.reshape(-1, 1)
log_Y = data['log_Y'].values
model = LinearRegression().fit(log_X, log_Y)
x_range = np.linspace(data['log_X'].min(), data['log_X'].max(), 100)
plt.plot(x_range, model.predict(x_range.reshape(-1, 1)),
         color='red',
         linewidth=2,
         label='Regression line (95% CI)')

# 置信区间（改用更透明的填充色）
n = len(log_X)
s_err = np.sqrt(np.sum((log_Y - model.predict(log_X))**2)/(n-2))
confs = stats.t.ppf(0.975, n-2) * s_err * np.sqrt(
    1/n + (x_range - np.mean(log_X))**2/np.sum((log_X - np.mean(log_X))**2))
plt.fill_between(x_range,
                 model.predict(x_range.reshape(-1, 1)).ravel() - confs,
                 model.predict(x_range.reshape(-1, 1)).ravel() + confs,
                 color='red', alpha=0.15)  # 降低透明度

# 添加R²标签（使用数学字体）
r2_text = plt.text(
    0.98, 0.02,
    r'$R^2 = {:.2f}$'.format(model.score(log_X, log_Y)),
    transform=plt.gca().transAxes,
    fontsize=14,
    fontname='Times New Roman',
    ha='right'
)

# 图例设置（统一字体）
legend = plt.legend(
    frameon=True,
    fontsize=14,
    prop={'family': 'Times New Roman'}
)
legend.get_frame().set_edgecolor('gray')

plt.tight_layout()
plt.savefig('GDP_vs_Papers.png', dpi=300)  # 保存高清图
plt.show()