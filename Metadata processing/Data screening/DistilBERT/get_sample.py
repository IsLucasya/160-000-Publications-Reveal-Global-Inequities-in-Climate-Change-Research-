# 分层抽样代码
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

# 文件路径
file_path = 'ssss.xlsx'

# 读取 Excel 文件
try:
    data = pd.read_excel(file_path, sheet_name='Sheet1', engine='openpyxl')
    print("数据加载成功！")
except ImportError as e:
    raise ImportError("请确保安装了 'openpyxl' 库以支持读取 .xlsx 文件。使用以下命令安装：\n pip install openpyxl")

# 检查元数据列是否存在
required_columns = ['Publication Type',  'Publication Year', 'Document Type', 'Author Keywords']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"缺少以下列：'{', '.join(missing_columns)}'，请检查数据集是否包含这些列")

# 打印 'Publication Year' 列的信息
# print("'Publication Year' 列的前几行:")
# print(data['Publication Year'].head())
# print("\n'Publication Year' 列的数据类型:", data['Publication Year'].dtype)
# print("'Publication Year' 列中的空值数量:", data['Publication Year'].isnull().sum())

# 只处理 'Publication Year' 列，不删除其他列的空值
data['Publication Year'] = pd.to_numeric(data['Publication Year'], errors='coerce')

# 对 'Publication Year' 进行分箱处理，将 NaN 值分到一个单独的类别
data['Year_Group'] = pd.cut(data['Publication Year'],
                            bins=[-np.inf, 2000, 2010, 2015, 2020, np.inf],
                            labels=['Very Old', 'Old', 'Medium', 'Recent', 'Very Recent'])
data['Year_Group'] = data['Year_Group'].cat.add_categories('Unknown').fillna('Unknown')

# 使用更少的特征进行组合分层
data['strata'] = (
    data['Publication Type'].astype(str) + '_' +
    data['Document Type'].astype(str) + '_' +
    data['Year_Group'].astype(str)
)

# 处理稀有类别
strata_counts = data['strata'].value_counts()
rare_strata = strata_counts[strata_counts < 2].index
data.loc[data['strata'].isin(rare_strata), 'strata'] = 'Other'

# 分层抽样
try:
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.01, random_state=42)
    for train_index, sample_index in split.split(data, data['strata']):
        sample_data = data.iloc[sample_index]
except ValueError:
    print("分层抽样失败，将使用简单随机抽样。")
    sample_data = data.sample(frac=0.1, random_state=42)

# 删除分层依据的临时列
sample_data = sample_data.drop(columns=['strata', 'Year_Group'])

# 保存分层抽样后的数据到 CSV 文件
output_file_path = 'Climate mitigation_sample_data.csv'
sample_data.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"分层抽样后的样本大小为: {len(sample_data)} 行")
print(f"抽样后的数据已保存至：{output_file_path}")