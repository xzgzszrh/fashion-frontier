"""Offline teacher soft targets.

The point of this module is the *decoupling*: the teacher (96x96
EfficientNet-B0) is run once over the training set, its logits are cached to an
NPZ file, and every subsequent student -- any width, any bit-width, any route --
trains from that file without ever touching the teacher again.

That is what makes "one teacher, many scenario-specific students" practical.
Re-running the teacher per student would have dominated the compute budget.

NPZ layout (``teacher_targets_<split>.npz``)::

    indices       int64   dataset positions, aligned with the loader order
    labels        int64   ground truth
    logits        float32 teacher logits (unscaled)
    probabilities float32 softmax(logits / temperature)
    temperature   float32 the temperature used above
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SoftTargetSet:
    indices: np.ndarray
    labels: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    temperature: float

    @classmethod
    def load(cls, path: str | Path) -> "SoftTargetSet":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                indices=data["indices"].astype(np.int64),
                labels=data["labels"].astype(np.int64),
                logits=data["logits"].astype(np.float32),
                probabilities=data["probabilities"].astype(np.float32),
                temperature=float(np.atleast_1d(data["temperature"])[0]),
            )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            indices=self.indices,
            labels=self.labels,
            logits=self.logits,
            probabilities=self.probabilities,
            temperature=np.asarray([self.temperature], dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def probs_tensor(self, indices: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Look up soft targets by dataset position."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        return torch.from_numpy(self.probabilities[indices.astype(np.int64)])


class SoftTargetDataset(Dataset):
    """Wraps a torchvision dataset and appends the cached teacher distribution."""

    def __init__(self, base: Dataset, soft: SoftTargetSet, indices: list[int]) -> None:
        self.base = base
        self.soft = soft
        # dataset position -> row inside the soft target file
        self._lookup = {int(idx): row for row, idx in enumerate(soft.indices)}
        self.indices = [int(i) for i in indices if int(i) in self._lookup]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        dataset_index = self.indices[i]
        image, label = self.base[dataset_index]
        probs = torch.from_numpy(self.soft.probabilities[self._lookup[dataset_index]])
        return image, label, dataset_index, probs


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


@torch.inference_mode()
def export_soft_targets(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    output_path: str | Path,
    indices: list[int],
    temperature: float = 1.0,
) -> SoftTargetSet:
    """Run the teacher once and cache its logits."""
    model.eval()
    logits_chunks, label_chunks = [], []
    for inputs, labels in loader:
        logits_chunks.append(model(inputs.to(device)).cpu().numpy())
        label_chunks.append(labels.numpy())
    logits = np.concatenate(logits_chunks).astype(np.float32)
    labels = np.concatenate(label_chunks).astype(np.int64)
    soft = SoftTargetSet(
        indices=np.asarray(indices, dtype=np.int64),
        labels=labels,
        logits=logits,
        probabilities=softmax_with_temperature(logits, temperature),
        temperature=temperature,
    )
    soft.save(output_path)
    return soft


def average_ensemble_probabilities(sets: list[SoftTargetSet]) -> np.ndarray:
    """Uniform average of several teachers' soft distributions.

    Ensembles gave only +0.11 % test accuracy on their own (94.88 % vs 94.77 %),
    but their soft labels are smoother and higher-entropy -- which is why they
    were preferred as the distillation source.
    """
    if not sets:
        raise ValueError("Need at least one soft target set.")
    stacked = np.stack([s.probabilities for s in sets], axis=0)
    return stacked.mean(axis=0).astype(np.float32)
