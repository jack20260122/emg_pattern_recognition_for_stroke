import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import os
import time
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class EMG_Transformer(nn.Module):
    def __init__(self, num_classes, input_dim=8, model_dim=128, num_heads=8, num_layers=3, dropout=0.1):
        super(EMG_Transformer, self).__init__()

        self.model_dim = model_dim
        self.input_dim = input_dim

        self.input_projection = nn.Linear(input_dim, model_dim)

        self.pos_encoding = PositionalEncoding(model_dim, max_len=30)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=False
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = x.squeeze(1)

        x = x.transpose(0, 1)

        x = self.input_projection(x)

        x = self.pos_encoding(x)

        x = self.transformer_encoder(x)

        x = x[0, :, :]

        x = self.classifier(x)
        return x


def load_model(model_path, num_classes=6):
    """
    加载训练好的模型权重
    """
    model = EMG_Transformer(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def read_emg_data_from_excel(file_path):
    try:

        df = pd.read_excel(file_path, header=None)

        if df.shape != (30, 8):
            raise ValueError(f"数据维度不正确，期望(30, 8)，实际{df.shape}")

        data = df.values.astype(np.float32)
        data = np.expand_dims(data, axis=(0, 1))

        return torch.tensor(data)
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None


def predict_single_sample(model, data_tensor):
    with torch.no_grad():
        output = model(data_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities, dim=1)[0]

    return predicted_class.item(), confidence.item(), probabilities.squeeze().numpy()


def visualize_prediction(predicted_class, confidence, probabilities, class_names=None):
    print("=" * 50)
    print("EMG信号分类预测结果")
    print("=" * 50)
    print(f"预测类别: {predicted_class}")
    if class_names and predicted_class < len(class_names):
        print(f"类别名称: {class_names[predicted_class]}")
    print(f"置信度: {confidence:.4f}")
    print("\n各类别概率分布:")
    for i, prob in enumerate(probabilities):
        class_name = f"类别 {i}"
        if class_names and i < len(class_names):
            class_name = class_names[i]
        bar = "█" * int(prob * 50)
        print(f"{class_name}: {prob:.4f} |{bar}")
    print("=" * 50)


def process_continuous_data_wait_new_data(file_path, model, window_size=30, step_size=1):
    try:
        print(f"开始实时处理数据，监控文件: {file_path}")
        print("按 Ctrl+C 停止处理")

        processed_rows = 0

        while True:

            df = pd.read_excel(file_path, header=None)
            data = df.values.astype(np.float32)

            total_rows, total_cols = data.shape

            if total_cols != 8:
                raise ValueError(f"数据列数不正确，期望8列，实际{total_cols}列")

            if total_rows >= processed_rows + window_size:

                start_idx = processed_rows
                end_idx = start_idx + window_size

                window_data = data[start_idx:end_idx, :]  # (30, 8)

                window_data = np.expand_dims(window_data, axis=(0, 1))  # (1, 1, 30, 8)

                data_tensor = torch.tensor(window_data)

                predicted_class, confidence, probabilities = predict_single_sample(model, data_tensor)

                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(
                    f"[{current_time}] 窗口 [{start_idx}:{end_idx}] - 预测类别: {predicted_class}, 置信度: {confidence:.4f}")

                processed_rows += step_size
            else:

                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(
                    f"[{current_time}] 等待新数据产生... (当前行数: {total_rows}, 需要: {processed_rows + window_size})")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n用户停止了实时处理")
    except Exception as e:
        print(f"处理连续数据时出错: {e}")


def main():
    MODEL_PATH = "best_test_model.pth"
    EXCEL_FILE_PATH = "123test.xlsx"
    NUM_CLASSES = 6

    CLASS_NAMES = [f'动作{i}' for i in range(NUM_CLASSES)]

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        print("请确保模型文件位于当前目录下")
        return

    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"错误: 找不到Excel数据文件 {EXCEL_FILE_PATH}")
        print("请确保Excel文件位于当前目录下")
        return

    print("正在加载模型...")
    try:
        model = load_model(MODEL_PATH, NUM_CLASSES)
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    print("\n请选择处理模式:")
    print("1. 单次预测 (默认30行数据)")
    print("2. 连续数据预测 (滑动窗口)")
    print("3. 实时连续数据预测 (每秒移动一次)")
    print("4. 实时连续数据预测 (等待新数据)")

    choice = input("请输入选择 (1, 2, 3 或 4): ").strip()

    if choice == "4":

        print("开始实时处理连续数据（等待新数据产生）...")
        process_continuous_data_wait_new_data(EXCEL_FILE_PATH, model)
    elif choice == "3":

        print("开始实时处理连续数据（每秒移动一次）...")

        try:
            processed_rows = 0
            while True:

                df = pd.read_excel(EXCEL_FILE_PATH, header=None)
                data = df.values.astype(np.float32)

                total_rows, total_cols = data.shape

                if total_cols != 8:
                    raise ValueError(f"数据列数不正确，期望8列，实际{total_cols}列")

                if total_rows >= processed_rows + 30:

                    start_idx = processed_rows
                    end_idx = start_idx + 30

                    window_data = data[start_idx:end_idx, :]  # (30, 8)

                    window_data = np.expand_dims(window_data, axis=(0, 1))  # (1, 1, 30, 8)

                    data_tensor = torch.tensor(window_data)

                    predicted_class, confidence, probabilities = predict_single_sample(model, data_tensor)

                    current_time = time.strftime("%H:%M:%S", time.localtime())
                    print(
                        f"[{current_time}] 窗口 [{start_idx}:{end_idx}] - 预测类别: {predicted_class}, 置信度: {confidence:.4f}")

                    processed_rows += 1
                else:

                    current_time = time.strftime("%H:%M:%S", time.localtime())
                    print(f"[{current_time}] 等待更多数据... (当前行数: {total_rows})")

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n用户停止了实时处理")
        except Exception as e:
            print(f"处理连续数据时出错: {e}")
    elif choice == "2":

        print("正在处理连续数据...")
        try:

            df = pd.read_excel(EXCEL_FILE_PATH, header=None)
            data = df.values.astype(np.float32)

            total_rows, total_cols = data.shape

            if total_cols != 8:
                raise ValueError(f"数据列数不正确，期望8列，实际{total_cols}列")

            predictions = []

            for start_idx in range(0, total_rows - 30 + 1, 1):
                end_idx = start_idx + 30

                window_data = data[start_idx:end_idx, :]  # (30, 8)

                window_data = np.expand_dims(window_data, axis=(0, 1))  # (1, 1, 30, 8)

                data_tensor = torch.tensor(window_data)

                predicted_class, confidence, probabilities = predict_single_sample(model, data_tensor)

                predictions.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities
                })

                print(f"窗口 [{start_idx}:{end_idx}] - 预测类别: {predicted_class}, 置信度: {confidence:.4f}")

            if predictions:
                print(f"\n总共处理了 {len(predictions)} 个窗口")

        except Exception as e:
            print(f"处理连续数据时出错: {e}")
    else:

        print("正在读取Excel数据...")
        data_tensor = read_emg_data_from_excel(EXCEL_FILE_PATH)
        if data_tensor is None:
            print("数据读取失败")
            return
        print("数据读取成功!")

        print("正在进行预测...")
        predicted_class, confidence, probabilities = predict_single_sample(model, data_tensor)

        visualize_prediction(predicted_class, confidence, probabilities, CLASS_NAMES)


if __name__ == "__main__":
    main()
    input("\n按Enter键退出...")
