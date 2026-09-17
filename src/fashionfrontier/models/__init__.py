"""Model registry.

``build_model`` resolves the ``model.name`` field of a config file. Models that
depend on Brevitas are imported lazily so that the CPU-only route works on a
plain ``pip install -e .``.
"""

from __future__ import annotations

import torch.nn as nn

from fashionfrontier.models.efficientnet import TorchvisionTeacher
from fashionfrontier.models.tiny_fashion_cnn import TinyFashionCNN, build_tiny_fashion_cnn

__all__ = [
    "TinyFashionCNN",
    "TorchvisionTeacher",
    "build_model",
    "available_models",
    "MODEL_REGISTRY",
]


def _quant_cnn(**kwargs) -> nn.Module:
    from fashionfrontier.models.quant_cnn import QuantFPGAFashionCNN

    return QuantFPGAFashionCNN(**kwargs)


def _finnconv_head8(**kwargs) -> nn.Module:
    from fashionfrontier.models.quant_cnn import build_finnconv_head8

    return build_finnconv_head8(**kwargs)


def _bnn_lfc(**kwargs) -> nn.Module:
    from fashionfrontier.models.bnn_lfc import BNNLFC

    return BNNLFC(**kwargs)


MODEL_REGISTRY: dict[str, object] = {
    "tiny_fashion_cnn": TinyFashionCNN,
    "torchvision_teacher": TorchvisionTeacher,
    "quant_fpga_cnn": _quant_cnn,
    "finnconv_head8": _finnconv_head8,
    "bnn_lfc": _bnn_lfc,
}


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def build_model(name: str, variant: str | None = None, **kwargs) -> nn.Module:
    """Instantiate a model.

    ``variant`` selects a TinyFashionCNN width preset
    (``tinyplus`` / ``tinyfast_s`` / ``tinyfast_xs`` / ``tinyfast_xxs`` /
    ``tinyfast_xxxs``) when ``name="tiny_fashion_cnn"``.
    """
    if name == "tiny_fashion_cnn" and variant:
        return build_tiny_fashion_cnn(variant, **kwargs)
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {', '.join(available_models())}"
        )
    return MODEL_REGISTRY[name](**kwargs)  # type: ignore[operator]
