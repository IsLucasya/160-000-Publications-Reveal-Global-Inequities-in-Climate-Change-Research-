#使用豆包API
import pandas as pd
from openai import OpenAI
from openai import APIError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests

# 设置豆包 API 的相关信息
API_KEY = "99de9618-800b4a0c1"  # 替换为您的豆包 API Key
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"  # 豆包 API 的基础 URL
NEW_MODEL = "ep-20250218141541-f6ch7"  # 使用的模型

# 城市、大学、研究机构到国家的映射
city_institution_to_country = {
    "Groningen": "Netherlands",
    "Berlin-Brandenburg": "Germany",
    "Harvard University": "United States",
    "Oxford University": "United Kingdom",
    "Stanford University": "United States",
    "MIT": "United States",
    "Max Planck Institute": "Germany",
    "ETH Zurich": "Switzerland",
    # 可以在此处继续添加更多的城市和机构映射
}

# 读取 Excel 文件
def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)  # 读取 Excel 文件
        return df
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return None

# 清理文本中的特殊字符
def clean_text(text):
    # 移除特殊字符
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

# 调用豆包 API 提取国家名称，并确保返回英文国家名称
def extract_countries_with_doubag(text):
    try:
        # 如果文本为空或无效，直接返回 "No mentioned"
        if not text or text.strip() == "":
            print("Text is empty or None")
            return "No mentioned"

        # 构建请求体
        user_prompt = f"""你是一个助手，负责从文本中提取与国家相关的信息。如果文本中提到的国家，返回国家英文名称，如果没有提到国家，返回'No mentioned'。以下是文本：
{text}"""

        # 向豆包API发送请求
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": NEW_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个专业的助手，负责从文本中提取相关国家名称。"},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )

        # 检查响应是否成功
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            return "No mentioned"

        # 解析响应内容
        response_data = response.json()
        countries = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # 如果返回结果为 "No mentioned"，清理掉任何解释内容，直接返回 "No mentioned"
        if "No mentioned" in countries:
            return "No mentioned"

        # 确保返回的是英文名称，如果是中文名称，则翻译
        country_translation = {
            "美国": "United States",
            "中国": "China",
            "法国": "France",
            "德国": "Germany",
            "日本": "Japan",
            "俄罗斯": "Russia",
            "印度": "India",
            # 可以在此处添加更多的翻译
        }

        # 如果 API 返回的是中文名称，我们做相应的英文翻译
        countries = country_translation.get(countries.strip(), countries)

        return countries
    except Exception as e:
        print(f"Error during API request: {e}")
        return "No mentioned"

# 处理单行数据
def process_row(row):
    try:
        title = row["title"]
        abstract = row["abstract"]

        # 如果 title 和 abstract 都为空，跳过处理
        if pd.isna(title) and pd.isna(abstract):
            print(f"Skipping row due to empty 'title' and 'abstract': {row.name}")
            return "Skipped"

        # 如果 title 或 abstract 为空，仅使用非空列
        if pd.isna(title):
            text = str(abstract)
        elif pd.isna(abstract):
            text = str(title)
        else:
            text = f"{title} {abstract}"

        # 清理文本
        text = clean_text(text)

        # 检查文本是否包含已知的城市、大学或研究机构
        for key, country in city_institution_to_country.items():
            if key.lower() in text.lower():  # 不区分大小写匹配
                return country

        # 调用豆包 API 提取国家名称
        countries = extract_countries_with_doubag(text)

        # 如果未提取到国家名称，设置为 'No mentioned'
        if not countries or countries.lower() == "no mentioned":
            countries = "No mentioned"

        return countries
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的异常堆栈
        return "Error"

# 处理 Excel 文件中的每一行
def process_excel(file_path, output_file, num_threads=4):
    # 读取 Excel 文件
    df = read_excel(file_path)
    if df is None:
        return

    # 检查是否包含所需的列
    if "title" not in df.columns or "abstract" not in df.columns:
        print("The Excel file is missing 'title' or 'abstract' columns.")
        return

    # 初始化一个列表来存储提取的国家名称
    countries_list = [None] * len(df)

    # 使用多线程处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务
        futures = {executor.submit(process_row, row): idx for idx, row in df.iterrows()}

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx = futures[future]
            try:
                countries_list[idx] = future.result()
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                countries_list[idx] = "Error"

    # 将提取的国家名称添加到 DataFrame 中
    df["countries"] = countries_list

    # 保存结果到新的 Excel 文件
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

# 主函数
if __name__ == "__main__":
    # 输入文件路径和输出文件路径
    input_file = "test.xlsx"  # 替换为您的输入文件路径
    output_file = "test-作者国家-研究国家.xlsx"  # 替换为您的输出文件路径

    # 设置线程数
    num_threads = 4  # 可以根据需要调整线程数

    # 处理 Excel 文件
    process_excel(input_file, output_file, num_threads)
