"""Quantisation-aware CNN for the FINN dataflow route (requires extras).

Install with::

    pip install -e ".[quant]"

This module is **optional** and is not needed for the CPU-only route. It exists
because the FINN path cannot be rebuilt without Brevitas/QONNX, and because the
structural constraints it encodes are the single most transferable lesson of the
project (see ``docs/04-quantization-and-finn-friendly-design.md``).

Design rules encoded here, all of them consequences of real FINN build failures:

* ``downsample_mode="stride_conv"`` -- MaxPoolNHWC is hostile to dataflow;
  stride convolution replaces it.
* ``head_type="conv"`` -- GlobalAveragePool / Flatten / MatMul are not
  FINN-friendly; a 1x1 conv head emits logits directly.
* first and last layers keep higher precision -- the head is what the FINN
  estimate reports as the accuracy bottleneck when quantised to 4 bits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # pragma: no cover - depends on the optional extra
    from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear, QuantReLU
    from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "brevitas is required for the quantisation route. Install it with:\n"
        '    pip install -e ".[quant]"\n'
        "The CPU-only route does not need it."
    ) from exc


def _weight_quant(bit_width: int):
    if bit_width >= 8:
        return Int8WeightPerTensorFloat
    return type(
        f"Int{bit_width}WeightPerTensorFloat",
        (Int8WeightPerTensorFloat,),
        {"bit_width": bit_width},
    )


def _act_quant(bit_width: int):
    if bit_width >= 8:
        return Int8ActPerTensorFloat
    return type(
        f"Int{bit_width}ActPerTensorFloat",
        (Int8ActPerTensorFloat,),
        {"bit_width": bit_width},
    )


class QuantFPGAFashionCNN(nn.Module):
    """Brevitas QAT CNN with FINN-friendly structural switches.

    Args:
        weight_bit_width / act_bit_width: 4, 2 or 1. 4W4A is the sweet spot in
            the paper (93.55 %); 2W2A degrades; 1W1A on a conv net of this size
            is a gamble.
        first_layer_bit_width / last_layer_bit_width: keep 8 for stability.
        head_type: ``"conv"`` (FINN-friendly) or ``"linear"``.
        downsample_mode: ``"stride_conv"`` (FINN-friendly) or ``"maxpool"``.
        conv_channels: stage widths; ``[24, 48, 96]`` approximates ``q_md``.
    """

    def __init__(
        self,
        conv_channels: tuple[int, ...] | list[int] = (24, 48, 96),
        weight_bit_width: int = 4,
        act_bit_width: int = 4,
        first_layer_bit_width: int = 8,
        last_layer_bit_width: int = 8,
        head_type: str = "conv",
        downsample_mode: str = "stride_conv",
        num_classes: int = 10,
        input_size: int = 28,
    ) -> None:
        super().__init__()
        if head_type not in ("conv", "linear"):
            raise ValueError("head_type must be 'conv' or 'linear'.")
        if downsample_mode not in ("stride_conv", "maxpool"):
            raise ValueError("downsample_mode must be 'stride_conv' or 'maxpool'.")

        channels = [int(c) for c in conv_channels]
        self.head_type = head_type
        self.downsample_mode = downsample_mode
        self.input_size = input_size

        self.input_quant = QuantIdentity(act_quant=_act_quant(first_layer_bit_width))

        stages: list[nn.Module] = []
        in_ch = 1
        # Two stride-2 downsampling stages, then a 1x1 conv head on a 7x7 map.
        for idx, out_ch in enumerate(channels):
            first = idx == 0
            stages += [
                QuantConv2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                    weight_quant=_weight_quant(
                        first_layer_bit_width if first else weight_bit_width
                    ),
                    output_quant=_act_quant(act_bit_width),
                    return_quant_tensor=False,
                ),
                nn.BatchNorm2d(out_ch),
                QuantReLU(act_quant=_act_quant(act_bit_width), return_quant_tensor=False),
            ]
            stages.append(self._downsample(out_ch))
            in_ch = out_ch

        self.features = nn.Sequential(*stages)
        spatial = input_size // (2 ** len(channels))

        if head_type == "conv":
            # 1x1 conv + global spatial collapse through a single-channel map:
            # emits `num_classes` logits without Flatten/MatMul/Gemm.
            self.head = nn.Sequential(
                QuantConv2d(
                    in_ch,
                    num_classes,
                    kernel_size=1,
                    weight_quant=_weight_quant(last_layer_bit_width),
                    output_quant=_act_quant(last_layer_bit_width),
                    return_quant_tensor=False,
                ),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        else:
            self.head = nn.Sequential(
                nn.Flatten(),
                QuantLinear(
                    in_ch * spatial * spatial,
                    num_classes,
                    weight_quant=_weight_quant(last_layer_bit_width),
                    return_quant_tensor=False,
                ),
            )

    def _downsample(self, channels: int) -> nn.Module:
        if self.downsample_mode == "stride_conv":
            return QuantConv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                weight_quant=_weight_quant(4),
                output_quant=_act_quant(4),
                return_quant_tensor=False,
            )
        return nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.input_quant(x)))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_finnconv_head8(
    weight_bit_width: int = 4, act_bit_width: int = 4
) -> QuantFPGAFashionCNN:
    """The ``finnconv_direct_head8`` configuration from the paper.

    FINN estimate rejected the original head (AdaptiveAvgPool + Linear). Moving
    to a 1x1 conv head cost essentially nothing in accuracy
    (93.49 % -> 93.55 %) and unblocked the dataflow build.
    """
    return QuantFPGAFashionCNN(
        conv_channels=(24, 48, 96),
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        head_type="conv",
        downsample_mode="stride_conv",
    )
