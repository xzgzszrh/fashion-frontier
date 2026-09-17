"""Torch -> ONNX export with an immediate round-trip check.

The round-trip check is not optional ceremony. One of the project's most
expensive mistakes was trusting a deployment artefact that had silently
diverged from the trained model; the分层 verification table in
``docs/05-fpga-finn-deployment.md`` exists because of it. Every export here is
compared against the source model on the same inputs before it is written.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int] = (1, 28, 28),
    opset: int = 17,
    input_name: str = "image",
    output_name: str = "logits",
) -> Path:
    """Export a model with a dynamic batch axis."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, *input_shape)

    kwargs = dict(
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        opset_version=opset,
    )
    # torch >= 2.6 routes export through the dynamo exporter, which pulls in
    # onnxscript. The classic exporter is what this project was validated
    # against and needs no extra dependency, so pin it when the argument exists.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False

    torch.onnx.export(model.cpu(), dummy, str(output_path), **kwargs)
    onnx.checker.check_model(str(output_path))
    return output_path


@torch.inference_mode()
def compare_onnx_to_torch(
    model: nn.Module,
    onnx_path: str | Path,
    input_shape: tuple[int, int, int] = (1, 28, 28),
    batch_size: int = 8,
    tolerance: float = 1e-4,
    input_name: str = "image",
) -> dict[str, float]:
    """Run the same random inputs through torch and ORT; report the max gap."""
    x = torch.randn(batch_size, *input_shape)
    torch_out = model.cpu()(x).numpy()

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    ort_out = session.run(None, {input_name: x.numpy()})[0]

    max_abs = float(np.abs(torch_out - ort_out).max())
    result = {
        "max_abs_diff": max_abs,
        "torch_shape": list(torch_out.shape),
        "onnx_shape": list(ort_out.shape),
        "within_tolerance": bool(max_abs <= tolerance),
    }
    if not result["within_tolerance"]:
        raise AssertionError(
            f"ONNX export deviates from torch by {max_abs:.3e} "
            f"(tolerance {tolerance:.1e}). Do not ship this artefact."
        )
    return result


def describe_onnx(path: str | Path) -> dict:
    """Small structural summary used by the benchmark reports."""
    model = onnx.load(str(path))
    graph = model.graph
    ops = {}
    for node in graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    return {
        "opset": [o.version for o in model.opset_import],
        "inputs": [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in graph.input],
        "outputs": [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in graph.output],
        "op_counts": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
        "file_size_bytes": Path(path).stat().st_size,
    }
