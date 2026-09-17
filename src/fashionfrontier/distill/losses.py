"""Distillation losses (two modes, exactly as used in the paper).

Blend distillation::

    L = alpha * L_CE(y_hat, y) + (1 - alpha) * L_CE(y_hat, p_T)

KL distillation::

    L = alpha * L_CE(y_hat, y) + (1 - alpha) * T^2 * KL(p_S^(T) || p_T^(T))

where ``p_T`` are the teacher soft targets and ``p^(T)`` denotes a softmax at
temperature ``T``.

Which one to use: **Blend for anything 1-bit or otherwise fragile.** The paper's
only distillation disaster (a 1W1A BNN-LFC collapsing to ~10 %) used KL, and the
post-mortem conclusion was that KL injects gradient signal that fights the sign
activation. KL is fine for the 4W4A and float students, where it was used with
T=2 and T=4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy against a soft target distribution."""

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()


class BlendDistillation(nn.Module):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = alpha
        self.hard = nn.CrossEntropyLoss()
        self.soft = _SoftTargetCrossEntropy()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_probs: torch.Tensor,
    ) -> torch.Tensor:
        hard_targets = targets.argmax(dim=1) if targets.ndim > 1 else targets
        return self.alpha * self.hard(logits, hard_targets) + (
            1.0 - self.alpha
        ) * self.soft(logits, teacher_probs)


class KLDistillation(nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 2.0) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.alpha = alpha
        self.temperature = temperature
        self.hard = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_probs: torch.Tensor,
    ) -> torch.Tensor:
        hard_targets = targets.argmax(dim=1) if targets.ndim > 1 else targets
        # Teacher probs were stored already temperature-scaled by
        # ``export_soft_targets``; undo the scaling so both sides are compared at
        # the *same* temperature T inside KL.
        teacher_logits = torch.log(teacher_probs.clamp_min(1e-8)) * self.temperature
        student_log_probs = F.log_softmax(logits / self.temperature, dim=1)
        teacher_probs_T = F.softmax(teacher_logits / self.temperature, dim=1)
        kl = F.kl_div(student_log_probs, teacher_probs_T, reduction="batchmean")
        return (
            self.alpha * self.hard(logits, hard_targets)
            + (1.0 - self.alpha) * (self.temperature**2) * kl
        )


DISTILLATION_REGISTRY = {"blend": BlendDistillation, "kl": KLDistillation}


def build_distillation_loss(method: str, **kwargs) -> nn.Module:
    """Build a distillation loss by name.

    ``temperature`` is silently dropped for the blend loss: a config that sets
    it for both methods should not fail, because blend simply has no use for it.
    """
    if method not in DISTILLATION_REGISTRY:
        raise ValueError(
            f"Unknown distillation method '{method}'. "
            f"Available: {', '.join(sorted(DISTILLATION_REGISTRY))}"
        )
    if method == "blend":
        kwargs.pop("temperature", None)
    return DISTILLATION_REGISTRY[method](**kwargs)
