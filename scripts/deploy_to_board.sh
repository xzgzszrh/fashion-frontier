#!/usr/bin/env bash
# Copy inference code, models and test tensors; reuse the board's Python environment.
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo 'usage: deploy_to_board.sh <board-host> [board-user]' >&2
  exit 1
fi
BOARD_HOST="$1"
BOARD_USER="${2:-xilinx}"
BOARD_PYTHON="${BOARD_PYTHON:-python3}"
[[ "$BOARD_HOST" =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]] || exit 2
[[ "$BOARD_USER" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_-]*$ ]] || exit 2
[[ "$BOARD_PYTHON" =~ ^[a-zA-Z0-9/_.-]+$ ]] || exit 2
cd "$(dirname "$0")/.."
shopt -s nullglob
MODELS=(artifacts/*_int8.onnx)
if [ "${#MODELS[@]}" -eq 0 ]; then MODELS=(artifacts/*.onnx); fi
if [ "${#MODELS[@]}" -eq 0 ] || [ ! -f artifacts/fashion_mnist_test.npz ]; then
  echo 'Prepare ONNX files and artifacts/fashion_mnist_test.npz first (make test-set).' >&2
  exit 1
fi
TARGET="${BOARD_USER}@${BOARD_HOST}"
REMOTE_DIR='fashion-frontier'
# Check before copying. Do not replace board-provided ARMv7 packages automatically.
ssh "$TARGET" "$BOARD_PYTHON -c 'import numpy, onnxruntime; print(onnxruntime.__version__)'"
ssh "$TARGET" "mkdir -p $REMOTE_DIR/src $REMOTE_DIR/artifacts"
scp -r src/fashionfrontier "${TARGET}:${REMOTE_DIR}/src/"
scp "${MODELS[@]}" artifacts/fashion_mnist_test.npz "${TARGET}:${REMOTE_DIR}/artifacts/"
STAMP="$(date +%Y%m%d_%H%M%S)"
ssh "$TARGET" "cd $REMOTE_DIR && PYTHONPATH=src $BOARD_PYTHON -m fashionfrontier bench --model-dir artifacts --test-set artifacts/fashion_mnist_test.npz --out board_raw_${STAMP}.json"
mkdir -p benchmarks/board_raw
scp "${TARGET}:${REMOTE_DIR}/board_raw_${STAMP}.json" "benchmarks/board_raw/${STAMP}_pynq.json"
echo "Saved benchmarks/board_raw/${STAMP}_pynq.json"
