import pandas as pd

# 读取两个Excel文件
file1 = pd.read_excel('总-作者国家-近5年IF-大洲.xlsx')  # 替换为你的文件1路径
file2 = pd.read_excel('全球南方.xlsx')  # 替换为你的文件2路径

# 创建不区分大小写的国家-地区映射字典
# 将国家名称统一转为小写作为键，同时保留原始的地区信息
country_region_map = {
    str(country).lower(): region
    for country, region in zip(file2['全球南方国家'], file2['region'])
}

# 定义一个函数来获取地区，不区分大小写
def get_region(country):
    # 将查询的国家名称也转为小写进行比较
    return country_region_map.get(str(country).lower(), pd.NA)

# 在文件1中创建新列'region'，存储对应的地区信息
file1['region'] = file1['author Country'].apply(get_region)

# 保存结果到新文件或覆盖原文件
file1.to_excel('结果文件.xlsx', index=False)  # 替换为你想要的输出路径

print("处理完成，结果已保存！")