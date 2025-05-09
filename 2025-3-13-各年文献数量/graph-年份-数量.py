import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = '总-年份-数量.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 提取年份和文章数量
years = data['Publication Year']
documents = data['Count']

# 假设预测区间为±10%的文章数量
lower_bounds = documents * 0.9  # 下限
upper_bounds = documents * 1.1  # 上限

# 创建柱状图
plt.figure(figsize=(15, 7))
plt.bar(years, documents, color='skyblue', label='Number of Documents')

# 添加预测区间（误差线）
plt.errorbar(years, documents, yerr=[documents - lower_bounds, upper_bounds - documents],
             fmt='none', ecolor='red', capsize=5, label='Prediction Interval')

# 设置横轴和纵轴 - 可自由调整的参数
x_tick_interval = 5  # 横轴刻度间隔（年）
y_tick_interval = 2500  # 纵轴刻度间隔
y_max_limit = 25000  # 纵轴最大值

plt.xticks(range(min(years), max(years) + 1, x_tick_interval), fontsize=15)  # 可调整横轴刻度间隔
plt.yticks(range(0, y_max_limit + 1, y_tick_interval), fontsize=15)  # 可调整纵轴刻度
plt.ylim(0, y_max_limit)  # 纵轴范围

# 添加标题和标
plt.title('', fontsize=22)
plt.xlabel('Publication Year', fontsize=22)
plt.ylabel('Number of Documents', fontsize=22)

# 添加网格
plt.grid(True, linestyle='', alpha=0.4)

# 找到最大值及其对应的年份
max_doc = documents.max()
max_year = years[documents.idxmax()]

# 调整最大值标注的垂直位置的参数
max_annotation_offset = 1000  # 可以调整这个值来上下移动标注位置

# 按照要求格式标注最大值
plt.annotate(f'Max: {int(max_doc)}',
             xy=(max_year, max_doc),
             xytext=(max_year - 0.5, max_doc + max_annotation_offset),  # 添加垂直偏移量
             ha='right',  # 左对齐
             fontsize=12,
             color='green',  # 绿色
             fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green'))  # 添加箭头

# 显示图例
plt.legend(fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()