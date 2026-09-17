import numpy as np
import pytest

from fashionfrontier.export.int8_quantize import quantize_static_int8
from fashionfrontier.export.onnx_export import compare_onnx_to_torch, describe_onnx, export_to_onnx
from fashionfrontier.models import build_model


@pytest.fixture(scope="module")
def tiny_model():
    model = build_model("tiny_fashion_cnn", variant="tinyfast_xxxs")
    model.eval()
    return model


@pytest.fixture(scope="module")
def fp32_onnx(tmp_path_factory, tiny_model):
    path = tmp_path_factory.mktemp("onnx") / "tiny_fp32.onnx"
    export_to_onnx(tiny_model, path, input_shape=(1, 28, 28))
    return path


def test_export_produces_a_loadable_dynamic_batch_graph(fp32_onnx):
    info = describe_onnx(fp32_onnx)
    assert info["inputs"][0][1][0] in ("batch", 0, 1)
    assert info["outputs"][0][1][-1] == 10
    assert "Conv" in info["op_counts"]
    assert info["file_size_bytes"] > 0


def test_onnx_roundtrip_matches_torch(tiny_model, fp32_onnx):
    report = compare_onnx_to_torch(tiny_model, fp32_onnx, input_shape=(1, 28, 28), batch_size=4)
    assert report["within_tolerance"]
    assert report["max_abs_diff"] < 1e-4


def test_int8_quantization_shrinks_the_model(fp32_onnx, tmp_path):
    calib = np.random.default_rng(0).normal(size=(32, 1, 28, 28)).astype(np.float32)
    out = tmp_path / "tiny_int8.onnx"
    report = quantize_static_int8(fp32_onnx, out, calib, quant_format="qoperator")

    assert out.exists()
    assert report.calibration_samples == 32
    assert report.int8_bytes < report.fp32_bytes
    assert report.compression_ratio > 1.0
    assert report.quant_format == "qoperator"


def test_quantized_model_still_runs(fp32_onnx, tmp_path):
    import onnxruntime as ort

    calib = np.random.default_rng(0).normal(size=(32, 1, 28, 28)).astype(np.float32)
    out = tmp_path / "tiny_int8.onnx"
    quantize_static_int8(fp32_onnx, out, calib)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    result = session.run(None, {session.get_inputs()[0].name: calib[:4]})[0]
    assert result.shape == (4, 10)
    assert np.isfinite(result).all()
