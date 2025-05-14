# 调用DEEPSEEK API
import pandas as pd
from openai import AsyncOpenAI
from openai import APIError
from tqdm import tqdm
import asyncio

# 设置 DeepSeek API Key 和 Base URL
API_KEY = "sk-a"  # 替换为您的 DeepSeek API Key
BASE_URL = "https://api.deepseek.com"

# 读取 Excel 文件
def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)  # 读取 Excel 文件
        return df
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return None


# 异步调用 DeepSeek API 提取国家名称
async def extract_countries_with_deepseek_async(text, retries=3):
    for i in range(retries):
        try:
            # 初始化 OpenAI 异步客户端
            client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

            # 构造请求消息
            response = await client.chat.completions.create(
                model="deepseek-chat",  # 使用 DeepSeek 的模型
                messages=[
                    {"role": "system", "content": "You are an assistant that extracts country names from text. If the text is related to a specific country, return only the English name of the country. If the text is not related to any country, return only 'No mentioned'."},
                    {"role": "user", "content": text}
                ],
                stream=False
            )

            # 解析返回的结果
            result = response.choices[0].message.content.strip()

            # 返回处理后的结果
            return result if result and result.lower() != "no mentioned" else "No mentioned"

        except APIError as e:
            print(f"API call failed (attempt {i + 1}): {e}")
            await asyncio.sleep(2)  # 等待 2 秒后重试
    return "No mentioned"


# 异步处理单行数据
async def process_row_async(row):
    title = row["title"]
    abstract = row["abstract"]

    # 结合标题和摘要
    text = f"{title} {abstract}"

    # 调用 DeepSeek API 提取国家名称
    countries = await extract_countries_with_deepseek_async(text)

    # 返回国家名称
    return countries


# 异步处理 Excel 文件中的每一行
async def process_excel_async(file_path, output_file):
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

    # 创建任务列表
    tasks = [process_row_async(row) for _, row in df.iterrows()]

    # 使用 tqdm 显示进度条
    for idx, task in tqdm(enumerate(asyncio.as_completed(tasks)), total=len(tasks), desc="Processing"):
        try:
            countries_list[idx] = await task
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            countries_list[idx] = "No mentioned"

    # 将提取的国家名称添加到 DataFrame 中
    df["countries"] = countries_list

    # 保存结果到新的 Excel 文件
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")


# 主函数
if __name__ == "__main__":
    # 输入文件路径和输出文件路径
    input_file = "test.xlsx"  # 替换为您的输入文件路径
    output_file = "影响-deepseek-研究国家-优化.xlsx"  # 替换为您的输出文件路径

    # 运行异步任务
    asyncio.run(process_excel_async(input_file, output_file))