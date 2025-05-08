import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests

# API settings
API_KEY = "99de9618-8008-4dc9-8d87-4085f8b4a0c1"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
NEW_MODEL = "ep-20250218141541-f6ch7"


def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return None


def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()


def extract_countries_with_doubag(text):
    try:
        if not text or text.strip() == "":
            return "No mentioned"

        user_prompt = f"""Analyze the following text and identify the country names. If a city, institution, or address is mentioned, return the country where it is located. Only return the country name in English.

Rules:
1. Return only the country name in English, nothing else
2. For cities or addresses, identify and return their corresponding country
3. If multiple countries are found, return them separated by semicolons
4. If no country or location is found, return exactly 'No mentioned'
5. Do not include any explanations or additional text

Example outputs:
- United States
- France
- Germany; United States
- No mentioned

Text to analyze:
{text}"""

        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": NEW_MODEL,
                "messages": [
                    {"role": "system",
                     "content": "You are a location analysis expert. Return only country names in English."},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )

        if response.status_code != 200:
            return "No mentioned"

        response_data = response.json()
        countries = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # 清理响应，确保只包含国家名称
        if not countries or "No mentioned" in countries.lower():
            return "No mentioned"

        # 移除任何额外的解释文本，只保留国家名称
        countries = re.sub(r'[^a-zA-Z; ]', '', countries)
        countries = re.sub(r'\s+', ' ', countries).strip()

        return countries

    except Exception as e:
        return "No mentioned"


def process_row(row):
    try:
        title = row["title"]
        abstract = row["abstract"]

        if pd.isna(title) and pd.isna(abstract):
            return "No mentioned"

        text = ""
        if not pd.isna(title):
            text += str(title) + " "
        if not pd.isna(abstract):
            text += str(abstract)

        text = clean_text(text)
        countries = extract_countries_with_doubag(text)

        return countries
    except Exception as e:
        return "No mentioned"


def process_excel(file_path, output_file, num_threads=4):
    df = read_excel(file_path)
    if df is None:
        return

    if "title" not in df.columns or "abstract" not in df.columns:
        print("The Excel file is missing 'title' or 'abstract' columns.")
        return

    countries_list = [None] * len(df)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_row, row): idx for idx, row in df.iterrows()}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx = futures[future]
            try:
                countries_list[idx] = future.result()
            except Exception as e:
                countries_list[idx] = "No mentioned"

    df["countries"] = countries_list
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    input_file = "test.xlsx"
    output_file = "test.xlsx-研究国家.xlsx"
    num_threads = 4
    process_excel(input_file, output_file, num_threads)