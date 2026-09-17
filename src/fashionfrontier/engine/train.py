"""Training engine.

One ``Trainer`` covers every route in the repository: the float teacher, the
TinyFashionCNN students, and (with the quant extra installed) the Brevitas QAT
models. Differences live in the config file, not in the loop.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)
from tqdm.auto import tqdm

from fashionfrontier.data import FASHION_MNIST_LABELS, build_dataloaders
from fashionfrontier.distill.losses import build_distillation_loss
from fashionfrontier.distill.soft_targets import SoftTargetSet
from fashionfrontier.engine.ema import ModelEma
from fashionfrontier.engine.evaluate import evaluate_predictions
from fashionfrontier.models import build_model


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(cfg["learning_rate"])
    wd = float(cfg.get("weight_decay", 0.0))
    params = model.parameters()
    if hasattr(model, "get_param_groups"):
        params = model.get_param_groups(base_lr=lr)
    name = str(cfg.get("optimizer", "adamw")).lower()
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return SGD(
            params,
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            nesterov=bool(cfg.get("nesterov", True)),
            weight_decay=wd,
        )
    raise ValueError(f"Unsupported optimizer '{name}'.")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict[str, Any], steps_per_epoch: int
) -> tuple[Any, bool]:
    """Return (scheduler, step_per_batch)."""
    sched = cfg.get("scheduler") or {}
    name = str(sched.get("name", "none")).lower()
    epochs = int(cfg["epochs"])

    if name == "none":
        return None, False
    if name == "one_cycle":
        return (
            OneCycleLR(
                optimizer,
                max_lr=sched["max_lr"],
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=sched.get("pct_start", 0.15),
                anneal_strategy=sched.get("anneal_strategy", "cos"),
                div_factor=sched.get("div_factor", 10.0),
                final_div_factor=sched.get("final_div_factor", 100.0),
            ),
            True,
        )
    if name == "cosine":
        return (
            CosineAnnealingLR(optimizer, T_max=epochs, eta_min=sched.get("eta_min", 1e-5)),
            False,
        )
    if name == "warmup_cosine":
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = int(sched.get("warmup_epochs", 0)) * steps_per_epoch
        warmup_steps = min(max(0, warmup_steps), max(0, total_steps - 1))
        if warmup_steps > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=sched.get("start_factor", 0.2),
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=sched.get("eta_min", 1e-6),
            )
            return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps]), True
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=sched.get("eta_min", 1e-6)), True

    raise ValueError(f"Unsupported scheduler '{name}'.")


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """End-to-end training run driven by a single config dictionary."""

    def __init__(self, config: dict[str, Any], device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or resolve_device(config.get("device", "auto"))
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(int(config.get("seed", 42)))
        training = config["training"]
        data = config.get("data", {})

        self.bundle = build_dataloaders(
            data_dir=config.get("data_dir", "data"),
            batch_size=int(training["batch_size"]),
            num_workers=int(training.get("num_workers", 0)),
            val_split=float(training.get("val_split", 0.1)),
            seed=int(config.get("seed", 42)),
            split_seed=training.get("split_seed"),
            augmentation=training.get("augmentation"),
            image_size=int(data.get("image_size", 28)),
            repeat_channels=int(data.get("repeat_channels", 1)),
            normalize_mean=data.get("normalize_mean"),
            normalize_std=data.get("normalize_std"),
            with_sample_index=bool(config.get("distillation")),
        )

        model_cfg = dict(config["model"])
        name = model_cfg.pop("name")
        variant = model_cfg.pop("variant", None)
        self.model = build_model(name, variant=variant, **model_cfg).to(self.device)

        init_ckpt = config.get("init_checkpoint")
        if init_ckpt:
            state = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            state = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
            self.model.load_state_dict(state)

        # --- loss -----------------------------------------------------------
        self.label_smoothing = float(training.get("label_smoothing", 0.0))
        self.hard_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        distill_cfg = config.get("distillation") or {}
        self.distill_loss = None
        self.soft_targets: SoftTargetSet | None = None
        if distill_cfg:
            self.soft_targets = SoftTargetSet.load(distill_cfg["teacher_targets_path"])
            self._soft_row_of = {
                int(v): i for i, v in enumerate(self.soft_targets.indices)
            }
            self.distill_loss = build_distillation_loss(
                distill_cfg.get("method", "blend"),
                alpha=float(distill_cfg.get("alpha", 0.5)),
                temperature=float(distill_cfg.get("temperature", 2.0)),
            )
            missing = set(self.bundle.train_indices) - set(self._soft_row_of)
            if missing:
                raise ValueError(
                    f"{len(missing)} training samples have no cached teacher soft target. "
                    "Export soft targets for the same split_seed/val_split used here."
                )

        self.optimizer = build_optimizer(self.model, training)
        self.scheduler, self.step_per_batch = build_scheduler(
            self.optimizer, training, len(self.bundle.train_loader)
        )
        ema_decay = training.get("ema_decay")
        self.ema = ModelEma(self.model, decay=float(ema_decay)) if ema_decay else None
        self.ema_start_epoch = int(training.get("ema_start_epoch", 2))
        self.grad_clip = training.get("grad_clip_norm")
        self.epochs = int(training["epochs"])

        self.best_val_accuracy: float | None = None
        self.best_path = self.output_dir / "best_model.pt"
        self.history: list[dict[str, Any]] = []

    # -- soft target plumbing ------------------------------------------------
    def _teacher_probs(self, batch_positions: torch.Tensor | None) -> torch.Tensor | None:
        """Map loader batch to the matching soft-target rows.

        The train loader is shuffled, so we cannot rely on batch order. Instead
        every sample carries its dataset index and the soft target file is
        indexed by that position.
        """
        if self.soft_targets is None or batch_positions is None:
            return None
        idx = batch_positions.cpu().numpy().astype(np.int64)
        rows = np.asarray([self._soft_row_of[int(i)] for i in idx], dtype=np.int64)
        return torch.from_numpy(self.soft_targets.probabilities[rows])

    # -- epochs --------------------------------------------------------------
    def _compute_loss(self, logits, targets, teacher_probs):
        if self.distill_loss is None or teacher_probs is None:
            return self.hard_loss(logits, targets)
        return self.distill_loss(logits, targets, teacher_probs.to(logits.device))

    def _run_epoch(self, loader, train: bool, teacher_lookup: bool, desc: str) -> dict[str, float]:
        self.model.train(train)
        total_loss = total_correct = total = 0
        last_lr = self.optimizer.param_groups[0]["lr"]
        progress = tqdm(loader, desc=desc, leave=False)

        for batch in progress:
            # batches are (image, label) or (image, label, dataset_index)
            inputs, targets = batch[0], batch[1]
            positions = batch[2] if teacher_lookup and len(batch) > 2 else None
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                logits = self.model(inputs)
                loss = self._compute_loss(logits, targets, self._teacher_probs(positions))
                if train:
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=float(self.grad_clip)
                        )
                    self.optimizer.step()
                    if self.ema is not None:
                        self.ema.update(self.model)
                    if self.scheduler is not None and self.step_per_batch:
                        self.scheduler.step()
                    last_lr = self.optimizer.param_groups[0]["lr"]

            bs = inputs.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == targets).float().sum().item()
            total += bs
            progress.set_postfix(
                loss=f"{total_loss / total:.4f}", acc=f"{total_correct / total:.4f}"
            )

        return {"loss": total_loss / total, "accuracy": total_correct / total, "lr": last_lr}

    def fit(self) -> dict[str, Any]:
        """Run the full training loop and write artefacts to ``output_dir``."""
        started = perf_counter()
        train_loader = self.bundle.train_loader
        val_loader = self.bundle.val_loader

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._run_epoch(
                train_loader, train=True, teacher_lookup=True, desc=f"train {epoch}/{self.epochs}"
            )
            if self.scheduler is not None and not self.step_per_batch:
                self.scheduler.step()

            use_ema = self.ema is not None and epoch >= self.ema_start_epoch
            val_metrics = None
            source = "raw"
            ckpt_model = self.model

            if val_loader is not None:
                val_metrics = self._run_epoch(
                    val_loader, train=False, teacher_lookup=False, desc=f"valid {epoch}"
                )
                if use_ema:
                    ema_metrics = self._run_epoch(
                        val_loader,
                        train=False,
                        teacher_lookup=False,
                        desc=f"valid_ema {epoch}",
                    )
                    if ema_metrics["accuracy"] >= val_metrics["accuracy"]:
                        val_metrics, source, ckpt_model = ema_metrics, "ema", self.ema.module
            elif use_ema:
                source = "ema"
                ckpt_model = self.ema.module

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"] if val_metrics else None,
                "val_accuracy": val_metrics["accuracy"] if val_metrics else None,
                "learning_rate": train_metrics["lr"],
                "checkpoint_source": source,
            }
            self.history.append(row)

            improved = val_metrics is None or (
                self.best_val_accuracy is None
                or val_metrics["accuracy"] > self.best_val_accuracy
            )
            if val_metrics is not None:
                self.best_val_accuracy = max(
                    self.best_val_accuracy or 0.0, val_metrics["accuracy"]
                )
            if improved:
                torch.save(
                    {
                        "state_dict": ckpt_model.state_dict(),
                        "config": self.config,
                        "class_names": FASHION_MNIST_LABELS,
                        "used_ema": source == "ema",
                    },
                    self.best_path,
                )
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                + (
                    f"val_acc={val_metrics['accuracy']:.4f} source={source}"
                    if val_metrics
                    else f"source={source} (val_split=0)"
                )
            )

        pd.DataFrame(self.history).to_csv(self.output_dir / "history.csv", index=False)

        # --- final test evaluation with the best checkpoint ------------------
        checkpoint = torch.load(self.best_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        predictions, targets = self._predict(self.bundle.test_loader)
        metrics = evaluate_predictions(predictions, targets)
        metrics.update(
            {
                "experiment_name": self.config.get("experiment_name"),
                "device": str(self.device),
                "input_shape": list(self.bundle.input_shape),
                "best_val_accuracy": self.best_val_accuracy,
                "used_ema_checkpoint": bool(checkpoint.get("used_ema", False)),
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "training_seconds": round(perf_counter() - started, 2),
            }
        )
        with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        return metrics

    @torch.inference_mode()
    def _predict(self, loader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        preds, targets = [], []
        for batch in loader:
            logits = self.model(batch[0].to(self.device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(batch[1].numpy())
        return np.concatenate(preds), np.concatenate(targets)


def _loss_with_soft_targets(hard: torch.Tensor, soft: torch.Tensor | None) -> torch.Tensor:
    """Kept as an explicit helper for tests and custom loops."""
    return hard if soft is None else hard + soft


__all__ = [
    "Trainer",
    "load_config",
    "seed_everything",
    "resolve_device",
    "build_optimizer",
    "build_scheduler",
    "F",
]
