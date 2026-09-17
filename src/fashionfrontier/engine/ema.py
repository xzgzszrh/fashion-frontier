"""Exponential moving average of model weights.

EMA was introduced with the EfficientNet-B0 teacher and kept for every student.
It matters more than its size suggests: on EB0 the best checkpoint came from the
EMA copy (95.13 % val at epoch ~9), and for the tiny students -- which train in
minutes and are therefore noisy -- EMA is what makes the reported number
reproducible across runs.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1).")
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(model_value)
                continue
            ema_value.lerp_(model_value, 1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.module.state_dict()
