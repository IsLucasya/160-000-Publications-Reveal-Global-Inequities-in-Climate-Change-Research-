import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# 加载已经训练好的模型
def load_model(model_path, device):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load(model_path))
    return model.to(device)

# 数据集类
class ClimateDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        review = str(self.data.iloc[index]['review'])
        label = int(self.data.iloc[index]['label'])

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

# 评估函数，增加 ROC AUC 计算
def eval_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]  # 获取正类的概率
            _, preds = torch.max(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)  # 计算 ROC AUC

    return accuracy, precision, recall, f1, roc_auc

# 嵌套交叉验证并保存结果
def nested_cross_validation_visualization(df, tokenizer, model_path, device, max_len=512, outer_splits=5, inner_splits=5):
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    metrics = {'Outer Fold': [], 'Inner Fold': [], 'F1': [], 'Precision': [], 'Recall': [], 'ROC AUC': []}

    for outer_fold, (train_val_idx, test_idx) in enumerate(outer_cv.split(df, df['label']), start=1):
        train_val_data, test_data = df.iloc[train_val_idx], df.iloc[test_idx]

        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)
        for inner_fold, (train_idx, val_idx) in enumerate(inner_cv.split(train_val_data, train_val_data['label']), start=1):
            # 加载模型并评估当前的验证集
            model = load_model(model_path, device)
            train_dataset = ClimateDataset(train_val_data.iloc[train_idx], tokenizer, max_len)
            val_dataset = ClimateDataset(train_val_data.iloc[val_idx], tokenizer, max_len)
            train_loader = DataLoader(train_dataset, batch_size=8)
            val_loader = DataLoader(val_dataset, batch_size=8)

            # 评估模型
            _, precision, recall, f1, roc_auc = eval_model(model, val_loader, device)
            print(f"Outer Fold {outer_fold}, Inner Fold {inner_fold}, F1: {f1}, Precision: {precision}, Recall: {recall}, ROC AUC: {roc_auc}")

            # 保存每个内层验证的结果
            metrics['Outer Fold'].append(outer_fold)
            metrics['Inner Fold'].append(inner_fold)
            metrics['F1'].append(f1)
            metrics['Precision'].append(precision)
            metrics['Recall'].append(recall)
            metrics['ROC AUC'].append(roc_auc)

    # 将所有验证结果保存到 CSV 文件
    results_df = pd.DataFrame(metrics)
    results_df.to_csv('nested_cross_validation_results.csv', index=False)
    print("Results saved to nested_cross_validation_results.csv")

    # 计算平均值和标准差
    metric_means = {metric: np.mean(values) for metric, values in metrics.items() if metric not in ['Outer Fold', 'Inner Fold']}
    metric_stds = {metric: np.std(values) for metric, values in metrics.items() if metric not in ['Outer Fold', 'Inner Fold']}

    # 输出平均值和标准差
    for metric in metric_means:
        print(f"{metric} - Mean: {metric_means[metric]:.3f}, Std: {metric_stds[metric]:.3f}")

    # 可视化各个指标的平均值和标准差
    fig, ax = plt.subplots(figsize=(8, 6))
    metric_names = list(metric_means.keys())
    means = [metric_means[metric] for metric in metric_names]
    stds = [metric_stds[metric] for metric in metric_names]

    ax.bar(metric_names, means, yerr=stds, capsize=5, color=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Cross-Validation Metrics with Mean and Standard Deviation')
    plt.show()

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_model_state.bin"
    df = pd.read_csv("气候减缓-样本相关性训练集.csv")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # 初始化 tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # 嵌套交叉验证并保存结果
    nested_cross_validation_visualization(df, tokenizer, model_path, device)

if __name__ == "__main__":
    main()
