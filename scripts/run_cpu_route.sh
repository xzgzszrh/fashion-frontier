#!/usr/bin/env bash
# Full CPU-only delivery route: teacher -> soft targets -> students -> ONNX -> INT8.
# Total: ~1-2 h on a GPU host, longer on CPU.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"

echo "==> [1/5] baseline + teacher"
"$PYTHON" -m fashionfrontier train --config configs/teacher/01_baseline_cnn.yaml
"$PYTHON" -m fashionfrontier train --config configs/teacher/02_efficientnet_b0_transfer.yaml

echo "==> [2/5] export teacher soft targets (run once, reused by every student)"
"$PYTHON" -m fashionfrontier export-soft-targets \
  --config configs/teacher/04_export_soft_targets.yaml \
  --checkpoint outputs/efficientnet_b0_transfer_95/best_model.pt \
  --out artifacts/teacher_targets_full_train.npz \
  --split full_train --temperature 1.0

echo "==> [3/5] students"
for cfg in configs/cpu/0*.yaml; do
  echo "    $cfg"
  "$PYTHON" -m fashionfrontier train --config "$cfg"
done

echo "==> [4/5] ONNX export"
VARIANTS="tinyplus tinyfast_xxs tinyfast_xxxs"
for variant in $VARIANTS; do
  case "$variant" in
    tinyplus)      ckpt="outputs/tinyplus_kd_fulltrain/best_model.pt" ;;
    tinyfast_xxs)  ckpt="outputs/tinyfast_xxs/best_model.pt" ;;
    tinyfast_xxxs) ckpt="outputs/tinyfast_xxxs/best_model.pt" ;;
    *) echo "unknown variant $variant" >&2; exit 1 ;;
  esac
  "$PYTHON" -m fashionfrontier export-onnx --model tiny_fashion_cnn --variant "$variant" \
    --checkpoint "$ckpt" --out "artifacts/${variant}_fp32.onnx"
done

echo "==> [5/5] static INT8 quantisation (256-sample calibration)"
for variant in $VARIANTS; do
  "$PYTHON" -m fashionfrontier quantize --model "artifacts/${variant}_fp32.onnx" \
    --out "artifacts/${variant}_int8.onnx" --calib-size 256
done

echo "==> done. Artifacts in artifacts/"
