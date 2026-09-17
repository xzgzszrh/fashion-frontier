#!/usr/bin/env bash
# Push models + test set to a PYNQ-Z1 and run the full 10000-image benchmark.
#
#   ./scripts/deploy_to_board.sh 192.168.2.99
#
# The board only needs numpy + onnxruntime (requirements/board.txt).
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: deploy_to_board.sh <board-ip> [board-user]" >&2
  exit 1
fi

BOARD_IP="$1"
BOARD_USER="${2:-xilinx}"

cd "$(dirname "$0")/.."

if [ ! -d artifacts ]; then
  echo "artifacts/ not found. Run scripts/run_cpu_route.sh first." >&2
  exit 1
fi

REMOTE_DIR="/home/${BOARD_USER}/fashion-frontier"
TARGET="${BOARD_USER}@${BOARD_IP}"

echo "==> installing board requirements on ${TARGET}"
ssh "$TARGET" "mkdir -p ${REMOTE_DIR}"
scp requirements/board.txt "${TARGET}:${REMOTE_DIR}/"
ssh "$TARGET" "pip install --quiet -r ${REMOTE_DIR}/board.txt"

echo "==> copying ONNX models and test set"
if ls artifacts/*_int8.onnx >/dev/null 2>&1; then
  scp artifacts/*_int8.onnx "${TARGET}:${REMOTE_DIR}/"
else
  scp artifacts/*.onnx "${TARGET}:${REMOTE_DIR}/"
fi
scp artifacts/fashion_mnist_test.npz "${TARGET}:${REMOTE_DIR}/"

echo "==> running benchmark on board"
STAMP="$(date +%Y%m%d_%H%M%S)"
ssh "$TARGET" "cd ${REMOTE_DIR} && python -m fashionfrontier bench \
  --model-dir . --test-set fashion_mnist_test.npz \
  --out board_raw_${STAMP}.json"

mkdir -p benchmarks/board_raw
scp "${TARGET}:${REMOTE_DIR}/board_raw_${STAMP}.json" \
    "benchmarks/board_raw/${STAMP}_pynq.json"

echo "==> results written to benchmarks/board_raw/${STAMP}_pynq.json"
echo "    Note: these are NOT merged into benchmarks/results.csv -- that file"
echo "    transcribes the paper and only changes when the paper does."
