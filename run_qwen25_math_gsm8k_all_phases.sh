#!/usr/bin/env bash
set -euo pipefail

# Runs the full CCOT-Steering pipeline for one fixed experiment:
#   dataset: gsm8k
#   model:   qwen25_math1.5b
#   config:  S2
#
# Usage:
#   ./run_qwen25_math_gsm8k_all_phases.sh
#
# Optional:
#   DEVICE=cpu ./run_qwen25_math_gsm8k_all_phases.sh
#   PYTHON_BIN=python3 ./run_qwen25_math_gsm8k_all_phases.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="gsm8k"
CONFIG="S2"
MODEL="qwen25_math1.5b"

PIPELINE_ARGS=(--config "$CONFIG" --model "$MODEL" --dataset "$DATASET")
if [[ -n "${DEVICE:-}" ]]; then
  PIPELINE_ARGS+=(--device "$DEVICE")
fi

run_step() {
  local label="$1"
  shift
  printf '\n======================================================================\n'
  printf '%s\n' "$label"
  printf '======================================================================\n'
  "$@"
}

if [[ ! -s "$DATASET/train.jsonl" ]] || [[ ! -s "$DATASET/test.jsonl" ]]; then
  run_step "DATA: downloading/preparing gsm8k" \
    "$PYTHON_BIN" download_dataset.py --dataset "$DATASET"
else
  printf '\nDATA: using existing gsm8k/train.jsonl and gsm8k/test.jsonl\n'
fi

run_step "PREFLIGHT: verifying data isolation" \
  "$PYTHON_BIN" verify_isolation.py

run_step "PREFLIGHT: building Phase 1 compatibility cache" \
  "$PYTHON_BIN" preprocess_compress.py --dataset "$DATASET" --config "$CONFIG"

run_step "PHASE 1: Coconut training and validation for qwen25_math1.5b" \
  "$PYTHON_BIN" pipeline.py --phase 1 "${PIPELINE_ARGS[@]}"

run_step "PHASE 2: hidden-state extraction and truth-vector construction for qwen25_math1.5b" \
  "$PYTHON_BIN" pipeline.py --phase 2 "${PIPELINE_ARGS[@]}"

run_step "PHASE 3: alpha tuning and steered validation for qwen25_math1.5b" \
  "$PYTHON_BIN" pipeline.py --phase 3 "${PIPELINE_ARGS[@]}"

run_step "PHASE 4: final locked test evaluation" \
  "$PYTHON_BIN" pipeline.py --phase 4 "${PIPELINE_ARGS[@]}"

printf '\nAll qwen25_math1.5b / gsm8k phases completed successfully.\n'
