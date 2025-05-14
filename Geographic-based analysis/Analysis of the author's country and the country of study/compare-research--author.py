#对比研究对象国家和作者国家
import os
import pandas as pd

# 读取Excel文件
file_path = '作者国家-研究国家-每列一个国家.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 提取需要分析的列
research_countries = df['research country']
author_countries = df['author country']


# 定义一个函数来判断文献是否研究的是本国情况
def is_local_research(research_country, author_country):
    # 确保research_country是有效的字符串
    if pd.isna(research_country) or not isinstance(research_country, str) or research_country.strip() == "":
        return False  # 如果研究国家是NaN或空字符串，认为没有研究本国情况

    # 将研究对象的国家和作者的国家转为小写以进行不区分大小写的比较
    research_country_list = [country.strip().lower() for country in research_country.split(',')]
    author_country = author_country.strip().lower()

    # 判断作者所在国家是否在研究对象的国家列表中
    return author_country in research_country_list


# 应用函数，判断每行文献是否研究的是本国情况
df['Is_Local_Research'] = [
    is_local_research(research_countries[i], author_countries[i]) for i in range(len(df))
]

# 计算“研究本国情况”和“研究其他国家情况”的数量和占比
local_research_count = df['Is_Local_Research'].sum()  # 本国情况文献数量
other_research_count = len(df) - local_research_count  # 其他国家情况文献数量
total_count = len(df)  # 总文献数量

local_research_percentage = (local_research_count / total_count) * 100  # 本国情况占比
other_research_percentage = (other_research_count / total_count) * 100  # 其他国家情况占比

# 输出结果
print(f"研究本国情况的文献数量: {local_research_count}")
print(f"研究其他国家情况的文献数量: {other_research_count}")
print(f"研究本国情况的占比: {local_research_percentage:.2f}%")
print(f"研究其他国家情况的占比: {other_research_percentage:.2f}%")

# 创建文件夹（如果不存在）
local_research_folder = '研究本国情况的文献'
other_research_folder = '研究其他国家情况的文献'

os.makedirs(local_research_folder, exist_ok=True)  # 创建文件夹（如果不存在）
os.makedirs(other_research_folder, exist_ok=True)  # 创建文件夹（如果不存在）

# 分别筛选出研究本国情况和研究其他国家情况的文献
df_local_research = df[df['Is_Local_Research'] == True]
df_other_research = df[df['Is_Local_Research'] == False]

# 保存为Excel文件
local_research_path = os.path.join(local_research_folder, 'local_research.xlsx')
other_research_path = os.path.join(other_research_folder, 'other_research.xlsx')

df_local_research.to_excel(local_research_path, index=False)
df_other_research.to_excel(other_research_path, index=False)

# 输出文件路径
print(f"研究本国情况的文献已保存到 {local_research_path}")
print(f"研究其他国家情况的文献已保存到 {other_research_path}")

# 可选择保存分析结果到新的Excel文件
output_file_path = 'research_country_analysis.xlsx'
df.to_excel(output_file_path, index=False)
print(f"处理完成，结果已保存到 {output_file_path}")
