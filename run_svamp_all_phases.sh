#!/usr/bin/env bash
set -Eeuo pipefail

# Full independent SVAMP experiment. Override MODEL with a registered model tag
# or a comma-separated list; the default runs all three Qwen backbones.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-all}"
DEVICE="${DEVICE:-cuda}"
export CCOT_DATASET=svamp
export PYTHONUNBUFFERED=1

if [[ ! -s svamp/train.jsonl || ! -s svamp/test.jsonl ]]; then
  "$PYTHON_BIN" download_dataset.py --dataset svamp
fi
"$PYTHON_BIN" verify_isolation.py --dataset svamp
"$PYTHON_BIN" -m scripts.build_splits --dataset svamp
"$PYTHON_BIN" preprocess_compress.py --dataset svamp --config S3
"$PYTHON_BIN" pipeline.py --phase 0 --dataset svamp --config S3 \
  --model "$MODEL" --device "$DEVICE" "$@"
