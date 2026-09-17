"""``frontier`` command line entry point.

Commands follow the delivery pipeline in order::

    frontier train              -> configs/*.yaml            (host, torch)
    frontier export-soft-targets-> teacher_targets_*.npz     (host, once)
    frontier export-onnx        -> model_fp32.onnx           (host)
    frontier quantize           -> model_int8.onnx           (host, 256-sample calibration)
    frontier export-test-set    -> fashion_mnist_test.npz    (host, for the board)
    frontier bench / suite      -> measured results          (host or PYNQ-Z1)
    frontier results            -> the authoritative table   (anywhere)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCHMARK_CSV = Path(__file__).resolve().parents[2] / "benchmarks" / "results.csv"


# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    from fashionfrontier.engine.train import Trainer, load_config

    config = load_config(args.config)
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.output_dir:
        config["output_dir"] = args.output_dir
    metrics = Trainer(config).fit()
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    return 0


def cmd_export_soft_targets(args: argparse.Namespace) -> int:
    from fashionfrontier.data import build_dataloaders
    from fashionfrontier.distill.soft_targets import export_soft_targets
    from fashionfrontier.engine.train import load_config, resolve_device
    from fashionfrontier.models import build_model

    config = load_config(args.config)
    device = resolve_device(config.get("device", "auto"))
    model_cfg = dict(config["model"])
    name = model_cfg.pop("name")
    model = build_model(name, **model_cfg).to(device)
    state = torch_load(args.checkpoint)
    model.load_state_dict(state)
    model.eval()

    bundle = build_dataloaders(
        data_dir=config.get("data_dir", "data"),
        batch_size=256,
        image_size=config.get("data", {}).get("image_size", 28),
        repeat_channels=config.get("data", {}).get("repeat_channels", 1),
        normalize_mean=config.get("data", {}).get("normalize_mean"),
        normalize_std=config.get("data", {}).get("normalize_std"),
        val_split=float(config["training"].get("val_split", 0.1)),
        split_seed=config["training"].get("split_seed"),
    )

    if args.split == "full_train":
        loader, indices = bundle.full_train_eval_loader, list(range(60000))
    elif args.split == "train":
        loader, indices = bundle.train_eval_loader, bundle.train_indices
    else:
        raise ValueError("split must be 'train' or 'full_train'.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    soft = export_soft_targets(
        model, loader, device, out, indices, temperature=args.temperature
    )
    print(f"wrote {out} rows={len(soft)} temperature={args.temperature}")
    return 0


def torch_load(path: str):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state


def cmd_export_onnx(args: argparse.Namespace) -> int:
    from fashionfrontier.export.onnx_export import (
        compare_onnx_to_torch,
        describe_onnx,
        export_to_onnx,
    )
    from fashionfrontier.models import build_model

    model_cfg = dict(args.model_kwargs or {})
    model = build_model(args.model, variant=args.variant, **model_cfg)
    state = torch_load(args.checkpoint)
    model.load_state_dict(state)
    model.eval()

    path = export_to_onnx(model, args.out, input_shape=tuple(args.input_shape), opset=args.opset)
    report = compare_onnx_to_torch(model, path, input_shape=tuple(args.input_shape))
    info = describe_onnx(path)
    print(json.dumps({"roundtrip": report, "graph": info}, indent=2))
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    from fashionfrontier.data import sample_calibration_batch
    from fashionfrontier.export.int8_quantize import quantize_static_int8

    calib = sample_calibration_batch(
        data_dir=args.data_dir,
        num_samples=args.calib_size,
        image_size=args.image_size,
        repeat_channels=args.repeat_channels,
    )
    report = quantize_static_int8(
        args.model, args.out, calib, input_name=args.input_name, quant_format=args.format
    )
    print(json.dumps(report.as_dict(), indent=2))
    return 0


def cmd_export_test_set(args: argparse.Namespace) -> int:
    import numpy as np
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )
    dataset = datasets.FashionMNIST(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    images = torch_stack([dataset[i][0] for i in range(len(dataset))])
    labels = np.asarray([dataset[i][1] for i in range(len(dataset))], dtype=np.int64)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, images=images.numpy().astype(np.float32), labels=labels)
    print(f"wrote {out} images={images.shape} labels={labels.shape}")
    return 0


def torch_stack(tensors):
    import torch

    return torch.stack(tensors)


def cmd_bench(args: argparse.Namespace) -> int:
    from fashionfrontier.board.pynq_bench import (
        format_results_table,
        run_model_suite,
        save_board_report,
    )

    results = run_model_suite(
        args.model_dir,
        args.test_set,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(format_results_table(results))
    if args.out:
        save_board_report(results, args.out)
        print(f"\nwrote {args.out}")
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    import csv

    if not BENCHMARK_CSV.exists():
        print(f"results table not found: {BENCHMARK_CSV}")
        return 1
    with BENCHMARK_CSV.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.route:
        rows = [r for r in rows if r["route"] == args.route]
    if args.only_deployed:
        rows = [r for r in rows if r["deliverable"] == "yes"]
    cols = ["route", "model", "precision", "test_accuracy", "board_throughput_img_s", "source"]
    widths = {c: max(len(c), max((len(r.get(c, "")) for r in rows), default=0)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(r.get(c, "").ljust(widths[c]) for c in cols))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from fashionfrontier.models.tiny_fashion_cnn import (
        TINY_FASHION_VARIANTS,
        build_tiny_fashion_cnn,
    )

    print(f"{'variant':<16}{'conv_channels':<16}{'hidden':>8}{'params':>12}")
    for name in TINY_FASHION_VARIANTS:
        model = build_tiny_fashion_cnn(name)
        print(
            f"{name:<16}{str(list(model.conv_channels)):<16}"
            f"{model.hidden_dim:>8}{model.num_parameters():>12,}"
        )
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frontier",
        description="End-to-end inference toolkit for PYNQ-Z1 (Fashion-MNIST).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Train a model from a YAML config.")
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export-soft-targets", help="Cache teacher logits to NPZ.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="full_train", choices=["train", "full_train"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.set_defaults(func=cmd_export_soft_targets)

    p = sub.add_parser("export-onnx", help="Export a checkpoint to ONNX with a round-trip check.")
    p.add_argument("--model", required=True)
    p.add_argument("--variant", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--input-shape", type=int, nargs=3, default=[1, 28, 28])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--model-kwargs", type=json.loads, default=None)
    p.set_defaults(func=cmd_export_onnx)

    p = sub.add_parser("quantize", help="Static INT8 post-training quantisation.")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--calib-size", type=int, default=256)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--image-size", type=int, default=28)
    p.add_argument("--repeat-channels", type=int, default=1)
    p.add_argument("--input-name", default="image")
    p.add_argument("--format", default="qoperator", choices=["qoperator", "qdq"])
    p.set_defaults(func=cmd_quantize)

    p = sub.add_parser("export-test-set", help="Dump the test set as NPZ for the board.")
    p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default="data")
    p.set_defaults(func=cmd_export_test_set)

    p = sub.add_parser("bench", help="Benchmark every ONNX model in a directory.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--test-set", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("results", help="Print the authoritative results table.")
    p.add_argument("--route", default=None)
    p.add_argument("--only-deployed", action="store_true")
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("models", help="List TinyFashionCNN variants and parameter counts.")
    p.set_defaults(func=cmd_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
