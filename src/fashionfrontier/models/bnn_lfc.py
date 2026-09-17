"""BNN-PYNQ style 1W1A fully-connected network (requires extras).

Install with::

    pip install -e ".[quant]"

This is a *reconstruction*, not a revival: the original BNN-PYNQ repository
targets Python 2.7 and a long-dead FINN release. The network is rebuilt inside
the current PyTorch + Brevitas + QONNX + FINN stack so the whole loop from
training to board execution stays in one toolchain.

Results reported in the paper (section 7):

* ``paper_bnnpynq_lfc_fashion_1w1a``: test 83.16 %, best val 83.87 %
* board throughput at ``batch_size=256``: ~154 096 img/s
* 256-sample board bundle: 214/256 correct = 83.59 %

The failed sibling is documented in ``docs/09-lessons-and-negative-results.md``:
``..._w1a1_kdkl_t2_noncheat`` collapsed to ~10 % because too many risky variables
were switched on at once (binary topology + KL distillation + a cross-entropy
formulation that fights the sign activation). For 1W1A, stabilise the plain
training loop *first*, then consider distillation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # pragma: no cover - depends on the optional extra
    from brevitas.nn import QuantIdentity, QuantLinear
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "brevitas is required for the BNN route. Install it with:\n"
        '    pip install -e ".[quant]"'
    ) from exc


class BNNLFC(nn.Module):
    """1-bit weight / 1-bit activation LFC: 784-1024-1024-1024-10.

    Args:
        hidden_dims: widths of the hidden fully-connected stack.
        input_dim: flattened input size (28x28 = 784).
        num_classes: output classes.
        enable_bias: BNN-PYNQ uses bias-free FC layers; keep it off unless you
            know why you need it.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 1024, 1024),
        num_classes: int = 10,
        enable_bias: bool = False,
    ) -> None:
        super().__init__()
        dims = [input_dim, *[int(h) for h in hidden_dims], num_classes]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(
                QuantLinear(
                    dims[idx],
                    dims[idx + 1],
                    bias=enable_bias,
                    weight_bit_width=1,
                    return_quant_tensor=False,
                )
            )
            if idx < len(dims) - 2:
                # Order matters: BN -> Hardtanh -> binary activation. Putting the
                # quantiser before the normalisation was one of the instabilities
                # observed during the failed first attempt.
                layers += [
                    nn.BatchNorm1d(dims[idx + 1]),
                    nn.Hardtanh(),
                    QuantIdentity(act_bit_width=1, return_quant_tensor=False),
                ]
        self.features = nn.Sequential(nn.Flatten(), *layers)
        self.input_dim = input_dim
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_paper_lfc_1w1a() -> BNNLFC:
    """Exactly the topology used for the successful board deployment."""
    return BNNLFC(input_dim=784, hidden_dims=(1024, 1024, 1024), num_classes=10)
