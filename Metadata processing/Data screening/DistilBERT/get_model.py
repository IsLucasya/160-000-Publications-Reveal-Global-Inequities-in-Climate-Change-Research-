# 利用distilbert模型进行气候变化领域的二分类任务
# 思路：
# 导入相应的库：数据处理，深度学习，transformer，机器学习，训练进度库
# 自定义数据集类
# 训练函数
# 评估函数
# 主函数

# 数据处理库
import numpy as np
import pandas as pd
# torch 用于深度学习任务
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
# transformers 库中的 DistilBERT 模型用于文本的序列分类任务
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# 机器学习评估，评估模型性能的指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# tqdm 用于显示训练进度
from tqdm import tqdm

# 设置随机种子以确保可重复性，确保每次运行时结果的一致性。
torch.manual_seed(42)
np.random.seed(42)


# 数据集类
class ClimateDataset(Dataset):
    # 初始化数据集
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    # 返回数据集的长度
    def __len__(self):
        return len(self.data)

    # 处理每个文本，将其转化为模型可以理解的输入格式，根据索引 index 获取样本
    def __getitem__(self, index):
        review = str(self.data.iloc[index]['Abstract'])
        label = self.data.iloc[index]['relevance']

        # 确保标签为整数
        label = int(label)

        # 使用tokenizer.encode_plus将文本转化为input_ids,attention_mask，这些是模型输入所需的格式
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
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 训练函数
# 模型训练模式：调用 model.train()
# 遍历数据集：从 data_loader 中获取批次数据。
# 前向传播：计算模型的输出和损失
# 反向传播：计算梯度并更新模型参数。
# 计算准确率和平均损失：最后返回当前训练周期的准确率和平均损失。
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0
    num_examples = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 检查标签是否正确
        if labels.min() < 0 or labels.max() > 1:
            raise ValueError(f"Invalid label values in batch: {labels}")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        num_examples += len(labels)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / num_examples, np.mean(losses)


# 评估函数
# 设置模型为评估模式：调用 model.eval()。
# 遍历验证数据集：获取每个批次并进行预测。
# 计算预测的准确性、精确度、召回率和 F1 分数
# 返回这些评估指标，用于衡量模型性能。
def eval_model(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    num_examples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
            num_examples += len(labels)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = correct_predictions.double() / num_examples
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    return accuracy, precision, recall, f1


# 主函数
def main():
    # 加载数据
    df = pd.read_csv('predict_climate_adaptation.csv')
    # 去除 NaN 标签
    df = df.dropna(subset=['relevance'])
    # 确保标签为整数并且是 0 或 1
    df['relevance'] = df['relevance'].astype(int)
    if df['relevance'].min() < 0 or df['relevance'].max() > 1:
        raise ValueError("标签值应该为 0 或 1")

    # 初始化 tokenizer 和模型
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 准备数据集
    dataset = ClimateDataset(df, tokenizer, max_len=512)

    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建 data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=2)

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 训练循环
    epochs = 20
    best_accuracy = 0

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device)
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_precision, val_recall, val_f1 = eval_model(model, val_data_loader, device)
        print(f'Val accuracy {val_acc}, precision {val_precision}, recall {val_recall}, F1 {val_f1}')
        print('-' * 10)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc


if __name__ == '__main__':
    main()
