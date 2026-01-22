import onnxruntime as ort
import pandas as pd
import numpy as np
import time
import os


def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session


def predict_with_onnx(session, input_data):
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_data.astype(np.float32)})
    return result[0]


def process_continuous_data_onnx(file_path, session, window_size=30, step_size=1):
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

                output = predict_with_onnx(session, window_data)
                probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
                predicted_class = np.argmax(probabilities, axis=1)[0]
                confidence = np.max(probabilities, axis=1)[0]

                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(
                    f"[{current_time}] 窗口 [{start_idx}:{end_idx}] - 预测类别: {predicted_class}, 置信度: {confidence:.4f}")

                processed_rows += step_size
            else:

                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(
                    f"[{current_time}] 等待新数据产生... (当前行数: {total_rows}, 需要: {processed_rows + window_size})")



    except KeyboardInterrupt:
        print("\n用户停止了实时处理")
    except Exception as e:
        print(f"处理连续数据时出错: {e}")


def main():
    MODEL_PATH = "emg_model.onnx"
    EXCEL_FILE_PATH = "123test.xlsx"

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到ONNX模型文件 {MODEL_PATH}")
        print("请确保模型文件位于当前目录下")
        return

    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"错误: 找不到Excel数据文件 {EXCEL_FILE_PATH}")
        print("请确保Excel文件位于当前目录下")
        return

    print("正在加载ONNX模型...")
    try:
        session = load_onnx_model(MODEL_PATH)
        print("ONNX模型加载成功!")
    except Exception as e:
        print(f"ONNX模型加载失败: {e}")
        return

    print("开始实时处理连续数据（等待新数据产生）...")
    process_continuous_data_onnx(EXCEL_FILE_PATH, session)


if __name__ == "__main__":
    main()
    input("\n按Enter键退出...")
