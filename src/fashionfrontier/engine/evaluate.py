"""Host-side evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from fashionfrontier.data import FASHION_MNIST_LABELS


@torch.inference_mode()
def collect_predictions(
    model: nn.Module, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for inputs, labels in loader:
        logits = model(inputs.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def evaluate_predictions(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, Any]:
    accuracy = float((predictions == targets).mean())
    cm = confusion_matrix(targets, predictions, labels=list(range(len(FASHION_MNIST_LABELS))))
    report = classification_report(
        targets,
        predictions,
        target_names=FASHION_MNIST_LABELS,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        FASHION_MNIST_LABELS[i]: round(float(report[FASHION_MNIST_LABELS[i]]["recall"]), 4)
        for i in range(len(FASHION_MNIST_LABELS))
    }
    return {
        "accuracy": accuracy,
        "num_samples": int(len(targets)),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": per_class,
        "classification_report": report,
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module, loader, device: torch.device
) -> dict[str, Any]:
    predictions, targets = collect_predictions(model, loader, device)
    return evaluate_predictions(predictions, targets)
