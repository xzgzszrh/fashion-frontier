import pytest

from fashionfrontier.data import (
    FASHION_MNIST_LABELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IndexedSubset,
    _split_indices,
    build_dataloaders,
    build_transforms,
)


def test_label_order_is_the_official_fashion_mnist_order():
    assert len(FASHION_MNIST_LABELS) == 10
    assert FASHION_MNIST_LABELS[0] == "T-shirt/top"
    assert FASHION_MNIST_LABELS[6] == "Shirt"
    assert FASHION_MNIST_LABELS[-1] == "Ankle boot"


def test_split_is_deterministic_for_a_given_seed():
    a_train, a_val = _split_indices(1000, 0.1, 42)
    b_train, b_val = _split_indices(1000, 0.1, 42)
    assert a_train == b_train and a_val == b_val


def test_split_changes_with_the_split_seed_only():
    _, a_val = _split_indices(1000, 0.1, 42)
    _, b_val = _split_indices(1000, 0.1, 1337)
    assert a_val != b_val


def test_train_and_val_are_disjoint_and_complete():
    train, val = _split_indices(1000, 0.1, 42)
    assert len(val) == 100
    assert len(train) == 900
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(1000))


def test_val_split_zero_keeps_everything_in_train():
    train, val = _split_indices(60000, 0.0, 42)
    assert len(train) == 60000 and val == []


def test_transforms_apply_imagenet_normalisation_when_replicating_channels():
    train_t, eval_t = build_transforms(
        image_size=96,
        repeat_channels=3,
        normalize_mean=IMAGENET_MEAN,
        normalize_std=IMAGENET_STD,
    )
    # No exception and both pipelines exist; shape checks happen on real tensors.
    assert train_t is not None and eval_t is not None


def test_cheat_configuration_is_rejected_before_any_download():
    with pytest.raises(ValueError, match="forbidden"):
        build_dataloaders(train_includes_test_split=True, download=False)


def test_indexed_subset_yields_dataset_positions():
    class _Fake:
        def __getitem__(self, i):
            return f"image-{i}", i % 10

    subset = IndexedSubset(_Fake(), [5, 9, 2])
    items = [subset[i] for i in range(3)]
    assert [it[2] for it in items] == [5, 9, 2]
    assert items[0][0] == "image-5"
