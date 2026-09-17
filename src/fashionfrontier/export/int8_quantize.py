"""Post-training static INT8 quantisation for the CPU-only route.

Why static INT8, and why ``QOperator``:

* Static (calibrated) INT8 gave **1.5-2x** throughput over FP32 on the
  Cortex-A9 for a controllable accuracy loss.
* ``QuantFormat.QOperator`` is another **10-20 %** faster than the
  ``QDQ`` (QuantizeLinear/DequantizeLinear) representation on this target.
  Caveat: on x64 hosts ONNX Runtime warns that QOperator can be *slower* than
  QDQ. QOperator is chosen for the ARM board, which is where the reported
  numbers come from -- if you benchmark on a laptop, use ``--format qdq``.
* Calibration size matters: **256 samples was the empirical optimum** for this
  task. Fewer left activation ranges under-covered; more gave no further gain.

Keep the first and last layers at higher precision if a model turns out to be
sensitive -- on the tiny CNN family the loss was small enough not to bother.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


@dataclass
class QuantizationReport:
    fp32_bytes: int
    int8_bytes: int
    calibration_samples: int
    quant_format: str

    @property
    def compression_ratio(self) -> float:
        return self.fp32_bytes / max(self.int8_bytes, 1)

    def as_dict(self) -> dict:
        return {
            "fp32_bytes": self.fp32_bytes,
            "int8_bytes": self.int8_bytes,
            "compression_ratio": round(self.compression_ratio, 2),
            "calibration_samples": self.calibration_samples,
            "quant_format": self.quant_format,
        }


class ArrayCalibrationReader(CalibrationDataReader):
    """Feeds a fixed in-memory NHWC-shared NCHW array to the calibrator."""

    def __init__(self, data: np.ndarray, input_name: str) -> None:
        self.data = data.astype(np.float32)
        self.input_name = input_name
        self._iter = iter([{input_name: self.data}])

    def get_next(self):
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = iter([{self.input_name: self.data}])


def quantize_static_int8(
    fp32_onnx: str | Path,
    output_onnx: str | Path,
    calibration_data: np.ndarray,
    input_name: str = "image",
    quant_format: str = "qoperator",
) -> QuantizationReport:
    """Quantise an FP32 ONNX model with static INT8 calibration.

    Args:
        fp32_onnx: source model.
        output_onnx: destination path.
        calibration_data: float32 NCHW array; 256 samples is the recommended size.
        quant_format: ``"qoperator"`` (fast on ARM) or ``"qdq"`` (portable).
    """
    fp32_onnx, output_onnx = Path(fp32_onnx), Path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    fmt = QuantFormat.QOperator if quant_format == "qoperator" else QuantFormat.QDQ
    reader = ArrayCalibrationReader(calibration_data, input_name)

    quantize_static(
        model_input=str(fp32_onnx),
        model_output=str(output_onnx),
        calibration_data_reader=reader,
        quant_format=fmt,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        # per-channel weights cost nothing at inference and recover most of the
        # accuracy that uniform per-tensor INT8 would give up.
        per_channel=True,
        reduce_range=False,
    )

    return QuantizationReport(
        fp32_bytes=fp32_onnx.stat().st_size,
        int8_bytes=output_onnx.stat().st_size,
        calibration_samples=int(calibration_data.shape[0]),
        quant_format=quant_format,
    )
