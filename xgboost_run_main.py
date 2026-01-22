import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.ensemble import XGBClassifier
from EMG_data_loader import EMGDataset, get_emg_dataloader
from sklearn.metrics import accuracy_score
import xgboost as xgb

plt.rcParams['font.sans-serif'] = ['STXihei']
plt.rcParams['axes.unicode_minus'] = False

# 数据预处理函数
def preprocess_data():
    # 使用新的EMG数据加载器
    dataset = EMGDataset('Stroke Patients Data')

    # 获取所有标签
    labels = [label.item() for _, label in dataset]

    # 随机分割成训练集,验证集和测试集
    train_indices = []
    val_indices = []
    test_indices = []

    # 获得组别类型
    unique_labels = np.unique(labels)
    np.random.seed(0)

    for label in unique_labels:
        indices = np.where(np.array(labels) == label)[0]
        # 随机打乱
        np.random.shuffle(indices)
        s1 = int(len(indices) * 0.6)
        s2 = int(len(indices) * 0.8)

        # 分割数据
        train_indices.extend(indices[:s1])
        val_indices.extend(indices[s1:s2])
        test_indices.extend(indices[s2:])

    # 创建子数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

def flatten_data(dataset):
    """
    将3D数据展平为2D特征
    """
    features = []
    labels = []

    for data, label in dataset:
        # data形状为 (30, 8)，需要展平为 (240,)
        if data.dim() == 3:  # 如果是 (1, 30, 8)
            flattened = data.squeeze(0).flatten()  # 变为 (240,)
        else:  # 如果是 (30, 8)
            flattened = data.flatten()  # 变为 (240,)
        features.append(flattened.numpy())
        labels.append(label.item())

    return np.array(features), np.array(labels)

# XGBoost训练函数
def train_xgboost_model(train_dataset, val_dataset):
    # 展平数据
    X_train, y_train = flatten_data(train_dataset)
    X_val, y_val = flatten_data(val_dataset)

    # 创建XGBoost分类器
    model = XGBClassifier(
        n_estimators=200,           # 增加树的数量
        max_depth=8,                # 增加最大深度
        learning_rate=0.1,          # 学习率
        subsample=0.8,              # 子采样比例
        colsample_bytree=0.8,       # 列采样比例
        random_state=42,
        n_jobs=-1                   # 使用所有CPU核心
    )

    # 训练模型
    print("开始训练XGBoost模型...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,   # 早停策略
        verbose=True
    )

    # 预测
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # 计算准确率
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"验证集准确率: {val_acc:.4f}")

    return model, train_acc, val_acc

# XGBoost预测函数
def predict_xgboost(model, dataset):
    # 展平数据
    X, y_true = flatten_data(dataset)

    # 预测
    y_pred = model.predict(X)

    return y_pred, y_true

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    return cm

# 主函数
def main():
    # 数据预处理
    train_dataset, val_dataset, test_dataset = preprocess_data()

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 训练XGBoost模型
    model, train_acc, val_acc = train_xgboost_model(train_dataset, val_dataset)

    # 对测试集进行预测
    train_pred, train_true = predict_xgboost(model, train_dataset)
    val_pred, val_true = predict_xgboost(model, val_dataset)
    test_pred, test_true = predict_xgboost(model, test_dataset)

    # 计算准确率
    acc_train = accuracy_score(train_true, train_pred)
    acc_val = accuracy_score(val_true, val_pred)
    acc_test = accuracy_score(test_true, test_pred)

    print('训练集准确率acc_train:', acc_train)
    print('验证集准确率acc_val:', acc_val)
    print('测试集准确率acc_test:', acc_test)

    # 显示混淆矩阵
    num_classes = len(np.unique(train_true))
    label_names = [f'Class {i}' for i in range(num_classes)]

    # 训练集混淆矩阵
    cm_train = plot_confusion_matrix(train_true, train_pred, label_names, '训练集预测混淆矩阵')

    # 验证集混淆矩阵
    cm_val = plot_confusion_matrix(val_true, val_pred, label_names, '验证集预测混淆矩阵')

    # 测试集混淆矩阵
    cm_test = plot_confusion_matrix(test_true, test_pred, label_names, '测试集预测混淆矩阵')

    # 保存结果
    results = {
        'model': model,
        'train_accuracy': acc_train,
        'val_accuracy': acc_val,
        'test_accuracy': acc_test,
        'cm_train': cm_train,
        'cm_val': cm_val,
        'cm_test': cm_test
    }

    print("训练完成!")
    return results

if __name__ == "__main__":
    results = main()
