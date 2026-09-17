"""Host-side throughput/latency measurement.

Not a substitute for board numbers -- it exists so that a model can be sanity
checked before the 30-minute round trip to the hardware, and so the repository
is usable without a PYNQ-Z1 at hand.

The gap between host and board is the whole point of the project: a model that
is "fast" on a laptop can collapse by two orders of magnitude on a 650 MHz
dual-core A9. Always label which machine produced a number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fashionfrontier.bench.ort_session import (
    DEFAULT_BATCH_SIZE,
    BenchmarkResult,
    create_session,
    input_name_of,
)
from fashionfrontier.board.pynq_bench import detect_device_label


@torch.inference_mode()
def benchmark_torch(
    model: nn.Module,
    data: np.ndarray,
    labels: np.ndarray | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup_batches: int = 1,
    name: str = "model",
) -> BenchmarkResult:
    import time

    model.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
    n = len(tensor)
    for start in range(0, min(n, batch_size * warmup_batches), batch_size):
        model(tensor[start : start + batch_size])

    correct = 0
    started = time.perf_counter()
    for start in range(0, n, batch_size):
        out = model(tensor[start : start + batch_size])
        if labels is not None:
            correct += int(
                (out.argmax(dim=1).numpy() == labels[start : start + out.shape[0]]).sum()
            )
    elapsed = time.perf_counter() - started
    batches = max(1, (n + batch_size - 1) // batch_size)
    return BenchmarkResult(
        model=name,
        runtime="torch-eager",
        device=detect_device_label(),
        batch_size=batch_size,
        num_samples=n,
        latency_ms=round(elapsed / batches * 1000.0, 3),
        throughput_img_s=round(n / elapsed, 2),
        total_time_s=round(elapsed, 2),
        accuracy=(round(correct / n, 4) if labels is not None else None),
        notes="host measurement, not comparable to board numbers",
    )


def benchmark_onnx(
    model_path: str | Path,
    data: np.ndarray,
    labels: np.ndarray | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    intra_op_threads: int | None = None,
) -> BenchmarkResult:
    """Measure an ONNX model on the host.

    ``intra_op_threads`` defaults to ``None`` so the host does not pretend to be
    a two-core board; pass 2 to emulate the PYNQ-Z1 setting.
    """
    import time

    kwargs = {} if intra_op_threads is None else {"intra_op_threads": intra_op_threads}
    session = create_session(model_path, **kwargs)
    name = input_name_of(session)
    data = np.ascontiguousarray(data, dtype=np.float32)
    n = len(data)

    session.run(None, {name: data[:batch_size]})
    correct = 0
    started = time.perf_counter()
    for start in range(0, n, batch_size):
        batch = data[start : start + batch_size]
        out = session.run(None, {name: batch})[0]
        if labels is not None:
            correct += int((out.argmax(axis=1) == labels[start : start + len(batch)]).sum())
    elapsed = time.perf_counter() - started
    batches = max(1, (n + batch_size - 1) // batch_size)
    return BenchmarkResult(
        model=Path(model_path).stem,
        runtime="onnxruntime",
        device=detect_device_label(),
        batch_size=batch_size,
        num_samples=n,
        latency_ms=round(elapsed / batches * 1000.0, 3),
        throughput_img_s=round(n / elapsed, 2),
        total_time_s=round(elapsed, 2),
        accuracy=(round(correct / n, 4) if labels is not None else None),
        notes="host measurement" if intra_op_threads is None else f"host, intra={intra_op_threads}",
    )
