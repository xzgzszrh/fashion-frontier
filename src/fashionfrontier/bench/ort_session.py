"""ONNX Runtime session factory and the measurement loop.

The defaults here are not guesses -- they are the configuration the paper
converged on after a parameter sweep on the real board (section 8.2):

* ``intra_op_num_threads=2`` -- uses both Cortex-A9 cores, worth **+30-50 %**
  throughput over a single thread.
* ``inter_op_num_threads=1`` -- more parallel op streams only added scheduling
  overhead on two cores.
* ``graph_optimization_level=ORT_ENABLE_ALL`` -- enables operator fusion.
* ``batch_size=256`` -- large batches keep the NEON vector lanes busy.

Session construction costs 100-300 ms. Build it once and reuse it; recreating a
session per batch dominates the measurement on this class of hardware.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

DEFAULT_INTRA_OP_THREADS = 2
DEFAULT_INTER_OP_THREADS = 1
DEFAULT_BATCH_SIZE = 256


def create_session(
    model_path: str | Path,
    intra_op_threads: int = DEFAULT_INTRA_OP_THREADS,
    inter_op_threads: int = DEFAULT_INTER_OP_THREADS,
    enable_all_optimizations: bool = True,
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if enable_all_optimizations
        else ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )


def input_name_of(session: ort.InferenceSession) -> str:
    return session.get_inputs()[0].name


@dataclass
class BenchmarkResult:
    model: str
    runtime: str
    device: str
    batch_size: int
    num_samples: int
    latency_ms: float
    throughput_img_s: float
    total_time_s: float
    accuracy: float | None = None
    intra_op_threads: int = DEFAULT_INTRA_OP_THREADS
    inter_op_threads: int = DEFAULT_INTER_OP_THREADS
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def benchmark_session(
    session: ort.InferenceSession,
    data: np.ndarray,
    labels: np.ndarray | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup_batches: int = 1,
    model_name: str = "model",
    runtime: str = "onnxruntime",
    device: str = "unknown",
    notes: str = "",
) -> BenchmarkResult:
    """Measure throughput and (optionally) accuracy over a whole dataset.

    Latency is reported per *batch*, throughput per *image*. Both matter: the
    board's batch throughput is what a production pipeline sees, while per-image
    latency is what a single interactive request sees.
    """
    name = input_name_of(session)
    data = np.ascontiguousarray(data, dtype=np.float32)
    n = len(data)

    # Warm up: the first run triggers lazy allocator and thread-pool setup.
    for start in range(0, min(n, batch_size * warmup_batches), batch_size):
        session.run(None, {name: data[start : start + batch_size]})

    correct = 0
    started = perf_counter()
    for start in range(0, n, batch_size):
        batch = data[start : start + batch_size]
        outputs = session.run(None, {name: batch})[0]
        if labels is not None:
            correct += int((outputs.argmax(axis=1) == labels[start : start + len(batch)]).sum())
    elapsed = perf_counter() - started

    per_batch_ms = (elapsed / max(1, n // batch_size + (1 if n % batch_size else 0))) * 1000.0
    return BenchmarkResult(
        model=model_name,
        runtime=runtime,
        device=device,
        batch_size=batch_size,
        num_samples=n,
        latency_ms=round(per_batch_ms, 3),
        throughput_img_s=round(n / elapsed, 2),
        total_time_s=round(elapsed, 2),
        accuracy=(round(correct / n, 4) if labels is not None else None),
        notes=notes,
    )


def sweep_thread_configs(
    model_path: str | Path,
    data: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    configs: list[tuple[int, int]] | None = None,
) -> list[BenchmarkResult]:
    """Reproduce the intra/inter thread sweep from section 8.2.

    On a two-core Cortex-A9 the sweep reliably picks (2, 1); on a host machine
    with many cores the best pair differs, which is exactly why the board
    numbers must be measured on the board.
    """
    configs = configs or [(1, 1), (2, 1), (2, 2), (4, 1)]
    results = []
    for intra, inter in configs:
        session = create_session(model_path, intra_op_threads=intra, inter_op_threads=inter)
        results.append(
            benchmark_session(
                session,
                data,
                batch_size=batch_size,
                model_name=Path(model_path).stem,
                notes=f"intra={intra},inter={inter}",
            )
        )
    return results


def results_to_json(results: list[BenchmarkResult], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([r.as_dict() for r in results], handle, indent=2)
