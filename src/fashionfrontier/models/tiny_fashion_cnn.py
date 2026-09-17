"""The TinyFashionCNN family -- the backbone of the CPU-only delivery route.

Architecture (identical topology across all variants, only widths change)::

    Conv(c0) -> BN -> ReLU -> MaxPool     28x28 -> 14x14
    Conv(c1) -> BN -> ReLU -> MaxPool     14x14 ->  7x7
    Flatten -> Linear(h) -> ReLU -> Dropout -> Linear(10)

Only ``conv_channels``, ``hidden_dim`` and ``dropout`` move between variants.
That was a deliberate choice: it keeps the ONNX graph -- and therefore the
board-side operator set -- unchanged while still giving a usable
speed/accuracy ladder.

Why this topology rather than something more modern: it is the smallest
structure that still clears 90% on Fashion-MNIST once distillation is applied,
and every operator in it is cheap on a dual-core Cortex-A9. See
``docs/07-cpu-only-route.md`` for the measured ladder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Width presets used in the paper. Keys match the config file names in
# ``configs/cpu/``.
TINY_FASHION_VARIANTS: dict[str, dict[str, object]] = {
    # name                     conv_channels   hidden_dim  dropout
    "tinyplus": {"conv_channels": [40, 80], "hidden_dim": 160, "dropout": 0.20},
    "tinyfast_s": {"conv_channels": [28, 56], "hidden_dim": 96, "dropout": 0.18},
    "tinyfast_xs": {"conv_channels": [24, 48], "hidden_dim": 96, "dropout": 0.18},
    "tinyfast_xxs": {"conv_channels": [22, 44], "hidden_dim": 80, "dropout": 0.18},
    "tinyfast_xxxs": {"conv_channels": [20, 40], "hidden_dim": 64, "dropout": 0.16},
}


class TinyFashionCNN(nn.Module):
    """Parameterised two-block CNN for 28x28 single-channel Fashion-MNIST."""

    def __init__(
        self,
        conv_channels: tuple[int, int] | list[int] = (32, 64),
        hidden_dim: int = 128,
        dropout: float = 0.3,
        input_size: int = 28,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if len(conv_channels) != 2:
            raise ValueError("conv_channels must contain exactly two widths.")
        c0, c1 = int(conv_channels[0]), int(conv_channels[1])
        # Two stride-2 pools halve the spatial size twice.
        if input_size % 4 != 0:
            raise ValueError("input_size must be divisible by 4 (two MaxPool stages).")
        flat_dim = c1 * (input_size // 4) * (input_size // 4)

        self.features = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=3, padding=1),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c0, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.conv_channels = (c0, c1)
        self.hidden_dim = hidden_dim
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def onnx_input_shape(self) -> tuple[int, int, int]:
        return (1, self.input_size, self.input_size)


def build_tiny_fashion_cnn(variant: str, **overrides) -> TinyFashionCNN:
    """Instantiate one of the paper's width presets by name."""
    if variant not in TINY_FASHION_VARIANTS:
        available = ", ".join(sorted(TINY_FASHION_VARIANTS))
        raise ValueError(f"Unknown variant '{variant}'. Available: {available}")
    kwargs = dict(TINY_FASHION_VARIANTS[variant])
    kwargs.update(overrides)
    return TinyFashionCNN(**kwargs)  # type: ignore[arg-type]
