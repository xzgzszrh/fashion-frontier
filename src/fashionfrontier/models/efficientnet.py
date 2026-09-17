"""EfficientNet-B0 teacher.

The teacher exists to produce *soft targets*, not to run on the board. It is
deliberately heavy: 96x96 input, ImageNet-pretrained backbone, EMA checkpoint.

Measured on this project (Table 3 of the paper):

===============================  ============  =========
experiment                       best val      test
===============================  ============  =========
efficientnet_b0_transfer_95      95.13 %       94.77 %
efficientnet_b0_finetune_95      95.20 %       94.55 %
efficientnet_b0_fulltrain_95     --            94.63 %
efficientnet_b0_teacher_96_refine 95.17 %      --
ensemble_besttest                95.32 %       94.88 %
ensemble_bestval                 95.52 %       94.74 %
===============================  ============  =========

The take-away recorded in the paper is that transfer learning bought ~3 points
over the best self-designed CNN, while everything after that (fine-tuning,
full-train, ensembling) moved the test number by less than 0.15 points. The
teacher is therefore a *one-shot investment*: build it once, export the soft
targets, then reuse them for every student.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision import models as tv_models

SUPPORTED_BACKBONES = ("efficientnet_b0", "efficientnet_v2_s", "resnet18")


def _backbone_spec(name: str):
    if name == "efficientnet_b0":
        return tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.DEFAULT
    if name == "efficientnet_v2_s":
        return tv_models.efficientnet_v2_s, tv_models.EfficientNet_V2_S_Weights.DEFAULT
    if name == "resnet18":
        return tv_models.resnet18, tv_models.ResNet18_Weights.DEFAULT
    raise ValueError(
        f"Unsupported backbone '{name}'. Supported: {', '.join(SUPPORTED_BACKBONES)}"
    )


class TorchvisionTeacher(nn.Module):
    """ImageNet-pretrained backbone with a 10-way Fashion-MNIST head.

    ``in_channels`` adapts the stem when the input is not 3-channel. For this
    project the grayscale image is replicated to 3 channels by the transform
    pipeline (``repeat_channels: 3``), so the default path is used.
    """

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        pretrained: bool = True,
        weights_path: str | None = None,
        dropout: float = 0.25,
        num_classes: int = 10,
        in_channels: int = 3,
        backbone_lr_multiplier: float = 1.0,
        classifier_lr_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        builder, default_weights = _backbone_spec(architecture)
        self.model = builder(weights=None)
        if weights_path:
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state)
        elif pretrained:
            self.model.load_state_dict(default_weights.get_state_dict(progress=True))

        self.architecture = architecture
        self.backbone_lr_multiplier = backbone_lr_multiplier
        self.classifier_lr_multiplier = classifier_lr_multiplier

        if in_channels != 3:
            self._adapt_input_layer(in_channels)

        if architecture.startswith("efficientnet"):
            in_features = self.model.classifier[-1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )
        elif architecture.startswith("resnet"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"No classifier adaptation for '{architecture}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # -- optimiser plumbing -------------------------------------------------
    def _classifier_module(self) -> nn.Module:
        if self.architecture.startswith("efficientnet"):
            return self.model.classifier
        return self.model.fc

    def get_param_groups(self, base_lr: float, **_):
        """Differential LR: backbone and head can be scaled independently."""
        if (
            self.backbone_lr_multiplier == 1.0
            and self.classifier_lr_multiplier == 1.0
        ):
            return self.parameters()
        classifier_ids = {id(p) for p in self._classifier_module().parameters()}
        backbone, head = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (head if id(p) in classifier_ids else backbone).append(p)
        return [
            {"params": backbone, "lr": base_lr * self.backbone_lr_multiplier},
            {"params": head, "lr": base_lr * self.classifier_lr_multiplier},
        ]

    # -- input adaptation ---------------------------------------------------
    def _first_conv(self) -> nn.Conv2d:
        if self.architecture.startswith("efficientnet"):
            return self.model.features[0][0]
        return self.model.conv1

    def _set_first_conv(self, layer: nn.Conv2d) -> None:
        if self.architecture.startswith("efficientnet"):
            self.model.features[0][0] = layer
        else:
            self.model.conv1 = layer

    def _adapt_input_layer(self, in_channels: int) -> None:
        old = self._first_conv()
        if old.in_channels == in_channels:
            return
        new = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=old.bias is not None,
            padding_mode=old.padding_mode,
        )
        with torch.no_grad():
            w = old.weight
            if in_channels == 1:
                lum = torch.tensor([0.2989, 0.5870, 0.1140], dtype=w.dtype).view(1, 3, 1, 1)
                new_w = (w * lum).sum(dim=1, keepdim=True)
            elif in_channels < old.in_channels:
                new_w = w[:, :in_channels] * (old.in_channels / in_channels)
            else:
                reps = math.ceil(in_channels / old.in_channels)
                new_w = w.repeat(1, reps, 1, 1)[:, :in_channels] * (
                    old.in_channels / in_channels
                )
            new.weight.copy_(new_w)
            if old.bias is not None and new.bias is not None:
                new.bias.copy_(old.bias)
        self._set_first_conv(new)
