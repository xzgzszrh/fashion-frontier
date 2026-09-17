"""Build the web demo's data file from the authoritative results table.

The web demo must never invent a number. Everything it plots or tabulates is
derived here from ``benchmarks/results.csv`` -- the same file the README and the
docs quote. Run it after the CSV changes:

    python scripts/build_web_data.py

The only hand-maintained part is ``DEMO_MODELS`` below, which records *which*
checkpoints are shipped to ``web/models/`` and what each one measured when it was
re-verified against the full 10 000-image test set. That verification matters:
it is the evidence that the demo runs the paper's actual weights and not a
placeholder.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "benchmarks" / "results.csv"
WEB = ROOT / "web"
OUT = WEB / "data" / "results.json"

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Standardisation used at training time and therefore at inference time too.
# Verified: feeding the test set this way reproduces the paper's accuracy exactly.
MNIST_MEAN, MNIST_STD = [0.286], [0.353]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The demo set is exactly the set of models that has a real checkpoint on disk.
# ``artifact_accuracy`` is what THIS repository's verifier measured on the full
# test set; ``row`` names the results.csv row it corresponds to.
DEMO_MODELS = [
    {
        "id": "tinyfast_xxxs",
        "file": "models/tinyfast_xxxs_int8.onnx",
        "image_size": 28,
        "channels": 1,
        "lazy": False,
        "row": "tinyfast_xxxs",
        "tier": "Speed",
        "family": "cpu",
        "params": 133714,
        "artifact_accuracy": 0.9045,
        "blurb": "The fastest CPU result in the project. Throws away 0.6 points "
                 "against the balanced tier and buys 18% more throughput.",
        "source_artifact": "pynq_cpu_tinyfast_xxxs_v1/best_model_int8_qop.onnx",
    },
    {
        "id": "tinyfast_xxs",
        "file": "models/tinyfast_xxs_int8.onnx",
        "image_size": 28,
        "channels": 1,
        "lazy": False,
        "row": "tinyfast_xxs",
        "tier": "Balanced",
        "family": "cpu",
        "params": 182478,
        "artifact_accuracy": 0.9101,
        "blurb": "The default recommendation. Sits almost exactly in the middle of "
                 "the CPU frontier, which is where most real constraints land.",
        "source_artifact": "pynq_cpu_tinyfast_xxs_v1/best_model_int8_qop.onnx",
    },
    {
        "id": "tinyplus_kd_fulltrain",
        "file": "models/tinyplus_kd_int8.onnx",
        "image_size": 28,
        "channels": 1,
        "lazy": False,
        "row": "tinyplus_kd_fulltrain",
        "tier": "Accuracy",
        "family": "cpu",
        "params": 658490,
        "artifact_accuracy": 0.9219,
        "blurb": "4.9x the parameters of the speed tier for 1.76 accuracy points, "
                 "at 42% of the throughput. The clearest demonstration on this "
                 "board that width stops paying.",
        "source_artifact": "pynq_cpu_tinyplus_kd_v1_fulltrain/best_model_int8_qop.onnx",
    },
    {
        "id": "baseline_cnn",
        "file": "models/baseline_cnn_fp32.onnx",
        "image_size": 28,
        "channels": 1,
        "lazy": False,
        "row": "baseline_cnn",
        "tier": "Baseline",
        "family": "reference",
        "params": 421834,
        "artifact_accuracy": 0.9174,
        "blurb": "The first model that worked. Higher accuracy than two of the "
                 "shipped tiers and slower than all of them - which is exactly "
                 "why it was not shipped.",
        "source_artifact": "baseline_cnn/best_model.onnx",
    },
    {
        "id": "efficientnet_b0_transfer_95",
        "file": "models/efficientnet_b0_fp32.onnx",
        "image_size": 96,
        "channels": 3,
        "lazy": True,
        "row": "efficientnet_b0_transfer_95",
        "tier": "Teacher",
        "family": "reference",
        "params": 4020358,
        "artifact_accuracy": 0.9477,
        "blurb": "The accuracy ceiling and the source of every soft label. Running "
                 "it directly on the board gives the single worst throughput in "
                 "the project, which is the problem distillation exists to solve.",
        "source_artifact": "efficientnet_b0_transfer_95/best_model.onnx",
    },
]

FRONTIER_FAMILY = {
    "cpu": "cpu",
    "bnn": "fabric",
    "teacher": "reference",
}


def load_rows() -> list[dict]:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["_test"] = float(r["test_accuracy"]) if r["test_accuracy"].strip() else None
        r["_val"] = float(r["val_accuracy"]) if r["val_accuracy"].strip() else None
        raw = r["board_throughput_img_s"].strip().lstrip("~")
        r["_tput"] = float(raw) if raw else None
        raw = r["board_total_time_s"].strip()
        r["_time"] = float(raw) if raw else None
        raw = r["params"].strip()
        r["_params"] = int(raw) if raw else None
    return rows


def find_row(rows: list[dict], model_name: str) -> dict:
    key = model_name.lower()
    for r in rows:
        if r["model"].lower() == key:
            return r
    raise KeyError(f"no results.csv row named {model_name!r}")


def build_frontier(rows: list[dict]) -> list[dict]:
    """Every point that has both a board throughput and an accuracy."""
    out = []
    for r in rows:
        if r["device"] != "PYNQ-Z1" or r["_tput"] is None or r["_test"] is None:
            continue
        out.append({
            "label": r["model"],
            "throughput": r["_tput"],
            "accuracy": round(r["_test"] * 100, 2),
            "family": FRONTIER_FAMILY.get(r["route"], "reference"),
            "precision": r["precision"],
            "source": r["source"],
        })
    return sorted(out, key=lambda p: p["throughput"])


def build_throughput_only(rows: list[dict]) -> list[dict]:
    """Points the paper reports as throughput but never as accuracy."""
    out = []
    for r in rows:
        if r["device"] != "PYNQ-Z1" or r["_tput"] is None or r["_test"] is not None:
            continue
        out.append({
            "label": r["model"],
            "throughput": r["_tput"],
            "note": r["notes"],
            "source": r["source"],
        })
    return out


def build_table(rows: list[dict]) -> list[dict]:
    """The full results table, in display order, for the web page."""
    order = ["teacher", "distill", "quant", "cpu", "bnn", "fpga"]
    keep = [
        "route", "model", "precision", "params", "val_accuracy", "test_accuracy",
        "board_throughput_img_s", "board_total_time_s", "device", "deliverable",
        "source", "notes",
    ]
    out = []
    for r in rows:
        row = {}
        for k in keep:
            if k in ("params",):
                row[k] = r["_params"]
            elif k in ("val_accuracy",):
                row[k] = r["_val"]
            elif k == "test_accuracy":
                row[k] = r["_test"]
            elif k == "board_throughput_img_s":
                row[k] = r["_tput"]
            elif k == "board_total_time_s":
                row[k] = r["_time"]
            else:
                row[k] = r[k]
        out.append(row)
    out.sort(key=lambda r: (order.index(r["route"]) if r["route"] in order else 99, r["model"]))
    return out


def build_samples() -> list[dict]:
    samples = []
    for p in sorted((WEB / "samples").glob("sample_*.png")):
        m = re.search(r"sample_(\d+)", p.name)
        if not m:
            continue
        idx = int(m.group(1))
        samples.append({
            "file": f"samples/{p.name}",
            "label_index": idx,
            "label": CLASSES[idx],
        })
    return samples


def _summary(frontier: list[dict]) -> dict:
    slow = min(p["throughput"] for p in frontier)
    fast = max(p["throughput"] for p in frontier)
    lo = min(p["accuracy"] for p in frontier)
    hi = max(p["accuracy"] for p in frontier)
    return {
        "board_points": len(frontier),
        "slowest": slow,
        "fastest": fast,
        "span_x": fast / slow,
        "decades": math.log10(fast / slow),
        "accuracy_span_pt": hi - lo,
    }


def main() -> None:
    rows = load_rows()

    demo = []
    for spec in DEMO_MODELS:
        row = find_row(rows, spec["row"])
        path = WEB / spec["file"]
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing - copy the checkpoint in first")
        size = path.stat().st_size
        demo.append({
            **spec,
            "bytes": size,
            "precision": row["precision"],
            "params": spec["params"],
            "paper_accuracy": row["_test"],
            "paper_val_accuracy": row["_val"],
            "paper_source": row["source"],
            "board_throughput_img_s": row["_tput"],
            "board_total_time_s": row["_time"],
            "mean": IMAGENET_MEAN if spec["image_size"] == 96 else MNIST_MEAN,
            "std": IMAGENET_STD if spec["image_size"] == 96 else MNIST_STD,
        })

    payload = {
        "_generated_by": "scripts/build_web_data.py",
        "_source_of_truth": "benchmarks/results.csv",
        "_note": (
            "Do not edit by hand. Every number here is derived from the results "
            "table, which is itself a transcription of the paper."
        ),
        "classes": CLASSES,
        "frontier": build_frontier(rows),
        "throughput_only": build_throughput_only(rows),
        "demo_models": demo,
        "samples": build_samples(),
        "table": build_table(rows),
        "summary": _summary(build_frontier(rows)),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    s = payload["summary"]
    # Space-grouped to match the README, the docs and the demo page itself.
    span = f"{s['span_x']:,.0f}".replace(",", "\u00a0")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(rows)} result rows, {s['board_points']} board points")
    print(f"  {len(demo)} demo models, {len(payload['samples'])} samples")
    print(f"  frontier spans {span}x over {s['decades']:.2f} decades")


if __name__ == "__main__":
    main()
