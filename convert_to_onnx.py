import torch
from transformer_run_main4 import EMG_Transformer


def convert_to_onnx():
    model = EMG_Transformer(num_classes=6)
    model.load_state_dict(torch.load('best_test_model.pth', map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 1, 30, 8)

    torch.onnx.export(
        model,
        dummy_input,
        "emg_model.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )


if __name__ == "__main__":
    convert_to_onnx()
