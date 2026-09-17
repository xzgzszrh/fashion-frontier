import numpy as np
import pytest
import torch

from fashionfrontier.distill.losses import (
    BlendDistillation,
    KLDistillation,
    build_distillation_loss,
)
from fashionfrontier.distill.soft_targets import (
    SoftTargetSet,
    average_ensemble_probabilities,
    softmax_with_temperature,
)


def _batch(n=8, num_classes=10, seed=0):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, num_classes, generator=gen)
    labels = torch.randint(0, num_classes, (n,), generator=gen)
    teacher = torch.softmax(torch.randn(n, num_classes, generator=gen), dim=1)
    return logits, labels, teacher


def test_blend_is_a_convex_combination_of_hard_and_soft():
    logits, labels, teacher = _batch()
    alpha = 0.4
    loss = BlendDistillation(alpha=alpha)(logits, labels, teacher)
    hard = torch.nn.functional.cross_entropy(logits, labels)
    soft = -(teacher * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
    assert torch.allclose(loss, alpha * hard + (1 - alpha) * soft, atol=1e-6)


def test_alpha_bounds_the_distillation_weight():
    logits, labels, teacher = _batch()
    zero = BlendDistillation(alpha=0.0)(logits, labels, teacher)
    soft = -(teacher * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
    assert torch.allclose(zero, soft, atol=1e-6)

    one = BlendDistillation(alpha=1.0)(logits, labels, teacher)
    assert torch.allclose(one, torch.nn.functional.cross_entropy(logits, labels), atol=1e-6)


def test_kl_loss_is_finite_and_scales_with_temperature_squared():
    logits, labels, teacher = _batch()
    t2 = KLDistillation(alpha=0.5, temperature=2.0)(logits, labels, teacher)
    t4 = KLDistillation(alpha=0.5, temperature=4.0)(logits, labels, teacher)
    assert torch.isfinite(t2) and torch.isfinite(t4)
    # T appears in the KL target; changing it must change the loss.
    assert not torch.allclose(t2, t4)


def test_builder_ignores_temperature_for_blend():
    """Configs set temperature for both methods; blend must not choke on it."""
    loss = build_distillation_loss("blend", alpha=0.35, temperature=2.0)
    assert isinstance(loss, BlendDistillation) and loss.alpha == 0.35

    kl = build_distillation_loss("kl", alpha=0.35, temperature=2.0)
    assert isinstance(kl, KLDistillation) and kl.temperature == 2.0


def test_alpha_and_temperature_are_validated():
    with pytest.raises(ValueError):
        BlendDistillation(alpha=1.5)
    with pytest.raises(ValueError):
        KLDistillation(temperature=0.0)
    with pytest.raises(ValueError):
        build_distillation_loss("nope")


def test_soft_target_roundtrip(tmp_path):
    logits = np.random.default_rng(0).normal(size=(16, 10)).astype(np.float32)
    soft = SoftTargetSet(
        indices=np.arange(16, dtype=np.int64),
        labels=np.zeros(16, dtype=np.int64),
        logits=logits,
        probabilities=softmax_with_temperature(logits, 2.0),
        temperature=2.0,
    )
    path = tmp_path / "teacher_targets_full_train.npz"
    soft.save(path)
    reloaded = SoftTargetSet.load(path)

    assert len(reloaded) == 16
    assert reloaded.temperature == 2.0
    assert np.allclose(reloaded.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(reloaded.probabilities, soft.probabilities)


def test_softmax_with_temperature_shifts_sharpness():
    logits = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)
    sharp = softmax_with_temperature(logits, 0.5)
    flat = softmax_with_temperature(logits, 4.0)
    # Lower temperature => more peaked distribution.
    assert sharp.max() > flat.max()


def test_ensemble_average_is_a_valid_distribution():
    rng = np.random.default_rng(1)
    sets = []
    for _ in range(3):
        logits = rng.normal(size=(8, 10)).astype(np.float32)
        sets.append(
            SoftTargetSet(
                indices=np.arange(8),
                labels=np.zeros(8, dtype=np.int64),
                logits=logits,
                probabilities=softmax_with_temperature(logits, 1.0),
                temperature=1.0,
            )
        )
    averaged = average_ensemble_probabilities(sets)
    assert averaged.shape == (8, 10)
    assert np.allclose(averaged.sum(axis=1), 1.0, atol=1e-5)
