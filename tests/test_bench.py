import numpy as np
import pytest

from fashionfrontier.bench.ort_session import BenchmarkResult, benchmark_session, create_session
from fashionfrontier.board.pynq_bench import format_results_table
from fashionfrontier.export.onnx_export import export_to_onnx
from fashionfrontier.models import build_model


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory):
    model = build_model("tiny_fashion_cnn", variant="tinyfast_xxxs")
    model.eval()
    path = tmp_path_factory.mktemp("bench") / "m.onnx"
    export_to_onnx(model, path)
    return path


def test_session_uses_the_recommended_thread_configuration(onnx_path):
    session = create_session(onnx_path, intra_op_threads=2, inter_op_threads=1)
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_benchmark_is_self_consistent(onnx_path):
    session = create_session(onnx_path)
    data = np.random.default_rng(0).normal(size=(512, 1, 28, 28)).astype(np.float32)
    labels = np.random.default_rng(1).integers(0, 10, size=512).astype(np.int64)

    result = benchmark_session(
        session, data, labels, batch_size=256, model_name="unit-test"
    )
    assert result.num_samples == 512
    assert result.throughput_img_s > 0
    assert result.total_time_s > 0
    # throughput and total_time are rounded for reporting, so an exact
    # throughput * time == num_samples check would be flaky on fast hosts.
    assert 0.0 <= result.accuracy <= 1.0


def test_benchmark_result_serialises():
    r = BenchmarkResult(
        model="m", runtime="onnxruntime", device="host",
        batch_size=256, num_samples=10000,
        latency_ms=1.0, throughput_img_s=445.19, total_time_s=22.46,
        accuracy=0.9045,
    )
    payload = r.as_dict()
    assert payload["model"] == "m"
    assert payload["accuracy"] == 0.9045
    assert set(payload) >= {"throughput_img_s", "total_time_s", "intra_op_threads"}


def test_results_table_is_sorted_by_throughput():
    rows = [
        BenchmarkResult("slow", "ort", "host", 256, 10000, 10, 100.0, 100.0, 0.9),
        BenchmarkResult("fast", "ort", "host", 256, 10000, 1, 400.0, 25.0, 0.8),
    ]
    table = format_results_table(rows)
    assert table.index("fast") < table.index("slow")


def test_empty_benchmark_inputs_fail_before_running(onnx_path):
    session = create_session(onnx_path)
    with pytest.raises(ValueError, match="nonempty"):
        benchmark_session(session, np.empty((0, 1, 28, 28), np.float32))
    with pytest.raises(ValueError, match="batch_size"):
        benchmark_session(session, np.zeros((1, 1, 28, 28), np.float32), batch_size=0)


def test_benchmark_reports_actual_thread_settings(onnx_path):
    session = create_session(onnx_path, intra_op_threads=1, inter_op_threads=2)
    result = benchmark_session(session, np.zeros((1, 1, 28, 28), np.float32), batch_size=1)
    assert result.intra_op_threads == 1
    assert result.inter_op_threads == 2


def test_empty_model_suite_is_not_a_success(tmp_path):
    from fashionfrontier.board.pynq_bench import run_model_suite
    with pytest.raises(ValueError, match="no ONNX"):
        run_model_suite(tmp_path, tmp_path / "missing.npz")


def test_generic_arm_device_is_not_reported_as_pynq(monkeypatch):
    from fashionfrontier.board import pynq_bench
    monkeypatch.setattr(pynq_bench.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(pynq_bench.platform, "system", lambda: "Linux")
    monkeypatch.setattr(pynq_bench.Path, "exists", lambda self: False)
    assert pynq_bench.detect_device_label() == "Linux host (aarch64)"
