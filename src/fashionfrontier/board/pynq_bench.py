"""Board-side measurement on PYNQ-Z1 (numpy + onnxruntime only).

This module deliberately avoids torch/torchvision: the board image is a stock
PYNQ venv with Python 3.10 and onnxruntime 1.16.0, and pulling torch onto a
Cortex-A9 is not a good use of anyone's afternoon. The test set is therefore
shipped as an NPZ produced on the host (``frontier export-test-set``).

Reference numbers from the paper (Table 8, full 10 000-image test set):

=========================  ==========  =============  ==========
model                      accuracy    throughput     total time
=========================  ==========  =============  ==========
tinyplus_kd_fulltrain      92.21 %     186.8 img/s    ~53.5 s
tinyfast_xs                --          ~310 img/s     --
tinyfast_xxs               91.09 %     375.99 img/s   26.60 s
tinyfast_xxxs              90.45 %     445.19 img/s   22.46 s
=========================  ==========  =============  ==========

Measured with ``intra=2, inter=1, batch=256``. Accuracy comes from the host
test set; throughput from the board.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

from fashionfrontier.bench.ort_session import (
    DEFAULT_BATCH_SIZE,
    BenchmarkResult,
    create_session,
    input_name_of,
    results_to_json,
)


def load_test_set(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(images, labels)`` exported by ``frontier export-test-set``."""
    with np.load(npz_path) as data:
        images = np.ascontiguousarray(data["images"], dtype=np.float32)
        labels = data["labels"].astype(np.int64)
    return images, labels


def detect_device_label() -> str:
    """Best-effort identification of the machine producing the numbers.

    Reported alongside every result, because the single most common source of
    confusion in this project was throughput numbers without a device label.
    """
    machine = platform.machine()
    system = platform.system()
    board_path = Path("/proc/device-tree/model")
    if board_path.exists():
        board = board_path.read_text(errors="replace").strip("\x00\n")
        return f"{board} ({machine})"
    if system == "Darwin":
        return f"macOS host ({machine})"
    return f"{system} host ({machine})"


def run_board_benchmark(
    model_path: str | Path,
    test_set_npz: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    intra_op_threads: int = 2,
    inter_op_threads: int = 1,
    limit: int | None = None,
    notes: str = "",
) -> BenchmarkResult:
    """Run the full (or truncated) test set through one ONNX model."""
    if batch_size <= 0 or (limit is not None and limit <= 0):
        raise ValueError("batch_size and limit must be positive")
    images, labels = load_test_set(test_set_npz)
    if len(images) == 0 or len(images) != len(labels):
        raise ValueError("test images and labels must be nonempty and have equal lengths")
    if limit is not None:
        images, labels = images[:limit], labels[:limit]

    session = create_session(
        model_path,
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
    )
    name = input_name_of(session)

    # Warm up, then timed pass.
    session.run(None, {name: images[:batch_size]})
    correct = 0
    import time

    started = time.perf_counter()
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        outputs = session.run(None, {name: batch})[0]
        correct += int((outputs.argmax(axis=1) == labels[start : start + len(batch)]).sum())
    elapsed = time.perf_counter() - started

    batches = max(1, (len(images) + batch_size - 1) // batch_size)
    return BenchmarkResult(
        model=Path(model_path).stem,
        runtime="onnxruntime",
        device=detect_device_label(),
        batch_size=batch_size,
        num_samples=len(images),
        latency_ms=round(elapsed / batches * 1000.0, 3),
        throughput_img_s=round(len(images) / elapsed, 2),
        total_time_s=round(elapsed, 2),
        accuracy=round(correct / len(images), 4),
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
        notes=notes,
    )


def run_model_suite(
    model_dir: str | Path,
    test_set_npz: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_json: str | Path | None = None,
    limit: int | None = None,
) -> list[BenchmarkResult]:
    """Benchmark every ``*.onnx`` in a directory -- the four-tier ladder."""
    model_dir = Path(model_dir)
    paths = sorted(model_dir.glob("*.onnx"))
    if not paths:
        raise ValueError(f"no ONNX models found in {model_dir}")
    results = []
    for path in paths:
        results.append(
            run_board_benchmark(
                path, test_set_npz, batch_size=batch_size, limit=limit
            )
        )
    if output_json:
        results_to_json(results, output_json)
    return results


def format_results_table(results: list[BenchmarkResult]) -> str:
    header = f"{'model':<28}{'acc':>8}{'img/s':>11}{'ms/batch':>11}{'total_s':>10}"
    lines = [header, "-" * len(header)]
    for r in sorted(results, key=lambda x: -x.throughput_img_s):
        acc = f"{r.accuracy:.4f}" if r.accuracy is not None else "   --"
        lines.append(
            f"{r.model:<28}{acc:>8}{r.throughput_img_s:>11.2f}"
            f"{r.latency_ms:>11.2f}{r.total_time_s:>10.2f}"
        )
    return "\n".join(lines)


def save_board_report(results: list[BenchmarkResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": detect_device_label(),
        "python": platform.python_version(),
        "onnxruntime": __import__("onnxruntime").__version__,
        "results": [r.as_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
