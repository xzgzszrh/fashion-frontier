import pytest
import torch

from fashionfrontier.models import build_model
from fashionfrontier.models.tiny_fashion_cnn import (
    TINY_FASHION_VARIANTS,
    TinyFashionCNN,
    build_tiny_fashion_cnn,
)


@pytest.mark.parametrize("variant", sorted(TINY_FASHION_VARIANTS))
def test_variants_produce_correct_logits(variant):
    model = build_tiny_fashion_cnn(variant)
    out = model(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_width_ladder_is_monotonic_in_parameters():
    """The delivery ladder must actually shrink as it gets faster."""
    order = ["tinyplus", "tinyfast_xs", "tinyfast_xxs", "tinyfast_xxxs"]
    counts = [build_tiny_fashion_cnn(v).num_parameters() for v in order]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] > counts[-1] * 2


def test_expected_parameter_counts_match_results_table():
    """benchmarks/results.csv quotes these numbers; keep them honest."""
    expected = {
        "tinyplus": 658_490,
        "tinyfast_xs": 237_658,
        "tinyfast_xxs": 182_478,
        "tinyfast_xxxs": 133_714,
    }
    for variant, count in expected.items():
        assert build_tiny_fashion_cnn(variant).num_parameters() == count, variant


def test_baseline_architecture_matches_paper_description():
    """Two Conv-BN-ReLU-MaxPool blocks plus an FC hidden layer."""
    model = TinyFashionCNN(conv_channels=(32, 64), hidden_dim=128)
    convs = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]
    pools = [m for m in model.features if isinstance(m, torch.nn.MaxPool2d)]
    linears = [m for m in model.classifier if isinstance(m, torch.nn.Linear)]
    assert len(convs) == 2 and len(pools) == 2
    assert [layer.out_features for layer in linears] == [128, 10]


def test_input_size_must_be_divisible_by_four():
    with pytest.raises(ValueError):
        TinyFashionCNN(input_size=30)


def test_registry_resolves_variant_and_raises_on_unknown():
    model = build_model("tiny_fashion_cnn", variant="tinyfast_xxs")
    assert model.conv_channels == (22, 44)
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("does_not_exist")


def test_quant_models_require_the_extra():
    """The CPU-only install must not break on a Brevitas import."""
    brevitas = pytest.importorskip
    try:
        import brevitas  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"\[quant\]"):
            build_model("finnconv_head8")
    else:  # pragma: no cover - only when the extra is installed
        model = build_model("finnconv_head8")
        assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)
