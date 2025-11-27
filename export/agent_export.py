import torch

from config.config import ExportConfig
from models.agent_hybrid import HybridCNNLSTMAttention


def export_to_onnx(
    model: HybridCNNLSTMAttention,
    export_cfg: ExportConfig,
    example_input: torch.Tensor,
    task_type: str = "classification",
):
    model.eval()
    input_names = ["input"]
    output_names = ["logits" if task_type == "classification" else "predictions"]
    dynamic_axes = {"input": {0: "batch"}, output_names[0]: {0: "batch"}}

    torch.onnx.export(
        model,
        example_input,
        export_cfg.onnx_path,
        export_params=True,
        opset_version=export_cfg.opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported ONNX model to {export_cfg.onnx_path}")
