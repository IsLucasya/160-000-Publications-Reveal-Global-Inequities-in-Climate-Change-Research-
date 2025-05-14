# 去除大洲中的逗号
import pandas as pd

# 读取Excel文件
df = pd.read_excel('补充-作者国家-研究大洲.xlsx')  # 替换为你的文件路径

# 拆分continents列，并展开为多行
df = (df.set_index(['Country'])['continents']
      .str.split(', ', expand=True)
      .stack()
      .reset_index(level=1, drop=True)
      .reset_index(name='continents'))

# 保存结果到新文件
df.to_excel('split_continents_output.xlsx', index=False)

print("处理完成，结果已保存到 split_continents_output.xlsx")