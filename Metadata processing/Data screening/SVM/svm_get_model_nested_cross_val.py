import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 忽略不必要的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return ''


# 数据加载和预处理
training_data = pd.read_csv('气候影响-样本相关性训练集.csv')
X_text = training_data['review'].apply(clean_text)
y = training_data['label']

# 创建用于存储结果的列表
results = []

# 外部交叉验证
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for outer_fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_text, y), 1):
    X_train, X_val = X_text.iloc[train_idx], X_text.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 设置管道和参数
    tfidf = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        strip_accents='unicode',
        norm='l2'
    )

    svm = SVC(
        random_state=42,
        probability=True
    )

    pipeline = ImbPipeline([
        ('tfidf', tfidf),
        ('smote', SMOTE(random_state=42)),
        ('classifier', svm)
    ])

    param_grid = {
        'tfidf__max_features': [5000],
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto']
    }

    # 更新评分器定义
    scoring = {
        'f1': make_scorer(f1_score, average='weighted', zero_division=1),
        'precision': make_scorer(precision_score, average='weighted', zero_division=1),
        'recall': make_scorer(recall_score, average='weighted', zero_division=1),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }

    # 内部交叉验证
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring=scoring,
        refit='f1',
        n_jobs=-1
    )

    # 训练并显示进度
    print(f"正在进行外部折 {outer_fold}/5 的训练...")
    grid_search.fit(X_train, y_train)

    # 获取每个内部折的结果
    for inner_fold in range(1, 6):
        fold_results = {
            'Outer Fold': outer_fold,
            'Inner Fold': inner_fold,
            'F1': grid_search.cv_results_[f'split{inner_fold - 1}_test_f1'][0],
            'Precision': grid_search.cv_results_[f'split{inner_fold - 1}_test_precision'][0],
            'Recall': grid_search.cv_results_[f'split{inner_fold - 1}_test_recall'][0],
            'ROC AUC': grid_search.cv_results_[f'split{inner_fold - 1}_test_roc_auc'][0]
        }
        results.append(fold_results)

# 创建DataFrame并保存到CSV
results_df = pd.DataFrame(results)
results_df.to_csv('nested_cross_validation_results_svm.csv', index=False)
print("\u7ed3\u679c\u5df2\u4fdd\u5b58\u5230 nested_cross_validation_results_svm.csv")

# 保存模型
import joblib
joblib.dump(grid_search.best_estimator_, 'best_model_svm.pkl')
