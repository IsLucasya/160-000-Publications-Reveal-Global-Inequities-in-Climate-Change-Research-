# 载入所需的库
# pandas 是一个数据处理的工具包，这里用于加载我们要分析的数据。
# torch 是 PyTorch 的核心库，提供了深度学习的功能。
# torch.utils.data 包含 Dataset 和 DataLoader，用于加载数据。
# transformers 包含一些预训练的深度学习模型，这里我们使用 DistilBERT。
# tqdm 是一个用来展示进度条的库，可以让运行时看到任务完成的进度。

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm

# 自定义数据集类
# ClimateDataset 类是我们自定义的数据集类，它帮助我们处理数据，使其适合模型的输入格式。
# __init__ 函数用于初始化。我们将数据、tokenizer（用于将文本转换成数字形式的工具）、以及最大长度 max_len 传给它。
# __len__ 函数返回数据的长度。
# __getitem__ 函数定义如何取出每个数据项。它会从数据中取出 review（评论），使用 tokenizer 处理文本，将其转换为模型可以理解的格式。返回一个字典，包含：
# input_ids：代表文本的数字编码。
# attention_mask：告诉模型哪些部分是重要的，哪些部分可以忽略。
class ClimateDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        review = str(self.data.iloc[index]['Abstract'])
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

# 预测函数
# predict 函数用于将训练好的模型用于预测新数据。
# model.eval() 告诉模型现在进入预测模式。
# torch.no_grad()：关闭梯度计算以节省资源，因为我们只需要预测，不需要训练。
# 循环 for batch in tqdm(data_loader, desc="Predicting") 逐步取出每个数据批次。
# 模型输出 outputs，然后通过 torch.max 找到每个样本的预测类别，将它们加到 predictions 列表中。
# 最后返回所有预测。
def predict(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())

    return predictions

# 主函数
def main():
    # 加载要预测的数据
    df = pd.read_csv('新-气候适应-所有文献相关性预测前.csv')

    # 初始化 tokenizer 和模型
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 加载训练好的模型参数
    model.load_state_dict(torch.load('best_model_state.bin', map_location=torch.device('cpu')))

    # 准备数据集和 DataLoader
    dataset = ClimateDataset(df, tokenizer, max_len=512)
    data_loader = DataLoader(dataset, batch_size=16)

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 进行预测
    predictions = predict(model, data_loader, device)

    # 将预测结果添加到数据集并保存到文件中
    df['label'] = predictions
    df.to_csv('新-气候适应-所有文献相关性预测后.csv', index=False)
    print("预测结果已保存到 新-气候适应-所有文献相关性预测后.csv 文件中")

if __name__ == '__main__':
    main()
