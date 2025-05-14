import pandas as pd

# 读取Excel文件
df = pd.read_excel('国家名称.xlsx')


# 定义一个函数来处理每个单元格，并返回去重后的国家列表以及去重数量
def split_and_deduplicate(cell):
    # 如果单元格为空，直接返回空列表和0
    if pd.isna(cell):
        return [], 0

    # 按逗号分隔并去除重复项
    countries = [country.strip() for country in cell.split(',')]
    unique_countries = list(dict.fromkeys(countries))  # 使用dict.fromkeys去重并保持顺序

    # 返回去重后的国家列表和去重数量
    return unique_countries, len(countries) - len(unique_countries)


# 应用函数到“research country”列，并创建新列存储去重数量
df[['research country', 'duplicates_removed']] = df['research country'].apply(
    lambda x: pd.Series(split_and_deduplicate(x))
)

# 将拆分后的国家列表扩展到多个列
df = df.explode('research country')

# 保存处理后的数据到新的Excel文件
df.to_excel('processed_file.xlsx', index=False)

# 打印去重的总数
total_duplicates_removed = df['duplicates_removed'].sum()
print(f"Total duplicates removed: {total_duplicates_removed}")