import pandas as pd

# 读取原始的 CSV 文件，不指定编码
input_file_path = '国家-数量-经纬度.xlsx'  # 输入文件的路径
output_file_path = '国家-数量-经纬度.csv'  # 输出文件的路径

try:
    # 读取原始的 CSV 文件
    df = pd.read_excel(input_file_path, engine='openpyxl')

    # 将 DataFrame 保存为 UTF-8 格式的 CSV 文件
    df.to_csv(output_file_path, index=False, encoding='utf-8')

    print(f"文件已成功转换为 UTF-8 格式并保存至: {output_file_path}")
except Exception as e:
    print(f"转换过程中发生错误: {e}")
