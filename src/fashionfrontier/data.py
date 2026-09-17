"""Fashion-MNIST data pipeline with reproducible splits.

Two conventions from the project are encoded here and should not be relaxed:

1. **Non-cheat isolation** -- the test split is never merged into training.
   ``train_includes_test_split`` exists only so the violation can be made
   explicit and auditable; it must stay ``false`` for any reported result.
2. **Decoupled split seed** -- the train/val partition randomness uses an
   independent ``split_seed`` so that ensemble search and student comparison
   runs partition the data identically and remain comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

FASHION_MNIST_LABELS: list[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

NUM_CLASSES = len(FASHION_MNIST_LABELS)

# ImageNet normalisation constants, reused whenever a torchvision backbone is
# trained on the replicated 3-channel input.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Mean/std of Fashion-MNIST itself, used for the native 28x28 single-channel path.
FASHION_MEAN = (0.2860,)
FASHION_STD = (0.3530,)


@dataclass
class DataBundle:
    """Everything a training or evaluation run needs, with indices kept so that
    offline teacher soft targets can be aligned back to dataset positions."""

    train_loader: DataLoader
    val_loader: DataLoader | None
    test_loader: DataLoader
    train_eval_loader: DataLoader
    full_train_eval_loader: DataLoader
    train_indices: list[int]
    val_indices: list[int]
    input_shape: tuple[int, int, int]
    num_classes: int = NUM_CLASSES


def _augmentation_ops(aug: dict[str, Any] | None) -> list[Any]:
    """Translate the declarative augmentation block from a config into torchvision ops."""
    if not aug:
        return []
    ops: list[Any] = []
    padding = int(aug.get("random_crop_padding", 0))
    if padding > 0:
        ops.append(
            transforms.RandomCrop(
                size=(28, 28),
                padding=padding,
                padding_mode=aug.get("padding_mode", "reflect"),
            )
        )
    horizontal_flip = float(aug.get("horizontal_flip_prob", 0.0))
    if horizontal_flip > 0:
        ops.append(transforms.RandomHorizontalFlip(p=horizontal_flip))

    rotation = float(aug.get("rotation_degrees", 0.0))
    translate = float(aug.get("translate", 0.0))
    scale_min = float(aug.get("scale_min", 1.0))
    scale_max = float(aug.get("scale_max", 1.0))
    if rotation or translate or scale_min != 1.0 or scale_max != 1.0:
        ops.append(
            transforms.RandomAffine(
                degrees=rotation,
                translate=(translate, translate) if translate else None,
                scale=(scale_min, scale_max),
                interpolation=InterpolationMode.BILINEAR,
            )
        )
    return ops


def build_transforms(
    image_size: int = 28,
    repeat_channels: int = 1,
    normalize_mean: tuple[float, ...] | None = None,
    normalize_std: tuple[float, ...] | None = None,
    augmentation: dict[str, Any] | None = None,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform).

    ``repeat_channels=3`` replicates the single grayscale channel so an ImageNet
    pretrained backbone can be used; the resolution is then upsampled to
    ``image_size`` (96 for the EfficientNet-B0 teacher).
    """
    mean = tuple(normalize_mean) if normalize_mean else (FASHION_MEAN * repeat_channels)
    std = tuple(normalize_std) if normalize_std else (FASHION_STD * repeat_channels)
    if len(mean) != repeat_channels:
        mean = tuple(mean[:1] * repeat_channels)
    if len(std) != repeat_channels:
        std = tuple(std[:1] * repeat_channels)

    train_ops: list[Any] = list(_augmentation_ops(augmentation))

    if image_size != 28:
        train_ops.append(
            transforms.Resize(
                size=(image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        )
    if repeat_channels > 1:
        train_ops.append(transforms.Grayscale(num_output_channels=repeat_channels))

    train_ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    erasing_prob = float((augmentation or {}).get("random_erasing_prob", 0.0))
    if erasing_prob > 0:
        train_ops.append(transforms.RandomErasing(p=erasing_prob))

    eval_ops: list[Any] = []
    if image_size != 28:
        eval_ops.append(
            transforms.Resize(
                size=(image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        )
    if repeat_channels > 1:
        eval_ops.append(transforms.Grayscale(num_output_channels=repeat_channels))
    eval_ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


class IndexedSubset(Subset):
    """Subset that also yields the *dataset* position of each sample.

    Needed for distillation: the training loader is shuffled, so the cached
    teacher soft targets can only be aligned by dataset index, never by batch
    order.
    """

    def __getitem__(self, i: int):
        image, label = super().__getitem__(i)
        return image, label, self.indices[i]  # type: ignore[index]

    def __getitems__(self, indices: list[int]):
        return [self.__getitem__(int(i)) for i in indices]


def _split_indices(
    dataset_size: int, val_split: float, split_seed: int
) -> tuple[list[int], list[int]]:
    """Deterministic train/val partition driven by ``split_seed`` only."""
    if val_split <= 0:
        return list(range(dataset_size)), []
    val_size = int(dataset_size * val_split)
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    return train_indices, val_indices


def build_dataloaders(
    data_dir: str = "data",
    batch_size: int = 256,
    num_workers: int = 0,
    val_split: float = 0.1,
    seed: int = 42,
    split_seed: int | None = None,
    augmentation: dict[str, Any] | None = None,
    image_size: int = 28,
    repeat_channels: int = 1,
    normalize_mean: tuple[float, ...] | None = None,
    normalize_std: tuple[float, ...] | None = None,
    train_includes_test_split: bool = False,
    download: bool = True,
    with_sample_index: bool = False,
) -> DataBundle:
    """Build the Fashion-MNIST dataloaders used by every route in this repository.

    ``with_sample_index=True`` makes the *training* loader yield
    ``(image, label, dataset_index)`` triples. Enable it whenever distillation is
    active so cached teacher soft targets can be aligned by dataset position.
    """
    if train_includes_test_split:
        # Checked before anything is downloaded so the guard is cheap to test.
        raise ValueError(
            "train_includes_test_split=true is forbidden for reported results. "
            "It exists only to make the non-cheat guarantee auditable."
        )
    split_seed = seed if split_seed is None else split_seed
    train_transform, eval_transform = build_transforms(
        image_size=image_size,
        repeat_channels=repeat_channels,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        augmentation=augmentation,
    )

    full_train = datasets.FashionMNIST(
        root=data_dir, train=True, download=download, transform=train_transform
    )
    eval_train = datasets.FashionMNIST(
        root=data_dir, train=True, download=download, transform=eval_transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=download, transform=eval_transform
    )

    train_indices, val_indices = _split_indices(len(full_train), val_split, split_seed)

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(eval_train, val_indices) if val_indices else None
    train_eval_subset = Subset(eval_train, train_indices) if val_indices else eval_train

    pin_memory = torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_source: Any = train_subset
    if with_sample_index:
        train_source = IndexedSubset(full_train, train_indices)
    train_loader = DataLoader(train_source, shuffle=True, **loader_kwargs)
    val_loader = (
        DataLoader(val_subset, shuffle=False, **loader_kwargs)
        if val_subset is not None
        else None
    )
    # Large batches and deterministic order: used for host-side measurement and
    # for exporting teacher soft targets.
    eval_kwargs = dict(loader_kwargs)
    eval_kwargs["batch_size"] = max(batch_size, 256)
    test_loader = DataLoader(test_dataset, shuffle=False, **eval_kwargs)
    train_eval_loader = DataLoader(train_eval_subset, shuffle=False, **eval_kwargs)
    full_train_eval_loader = DataLoader(eval_train, shuffle=False, **eval_kwargs)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_eval_loader=train_eval_loader,
        full_train_eval_loader=full_train_eval_loader,
        train_indices=train_indices,
        val_indices=val_indices,
        input_shape=(repeat_channels, image_size, image_size),
    )


def sample_calibration_batch(
    data_dir: str = "data",
    num_samples: int = 256,
    image_size: int = 28,
    repeat_channels: int = 1,
    normalize_mean: tuple[float, ...] | None = None,
    normalize_std: tuple[float, ...] | None = None,
    seed: int = 0,
    download: bool = True,
) -> np.ndarray:
    """A fixed-size float32 NHWC/NCHW calibration set for post-training quantisation.

    256 was the empirically best calibration size for this task: smaller sets
    leave INT8 activation ranges under-covered, larger ones gave no further gain.
    """
    _, eval_transform = build_transforms(
        image_size=image_size,
        repeat_channels=repeat_channels,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=download, transform=eval_transform
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
    batch = torch.stack([dataset[i][0] for i in indices])
    return batch.numpy().astype(np.float32)
