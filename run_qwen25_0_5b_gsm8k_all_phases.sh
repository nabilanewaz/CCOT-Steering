#!/usr/bin/env bash
set -euo pipefail

# Runs the full CCOT-Steering pipeline for one fixed experiment:
#   dataset: gsm8k
#   model:   qwen25_0.5b  (Qwen/Qwen2.5-0.5B)
#   config:  S2
#
# Usage:
#   ./run_qwen25_0_5b_gsm8k_all_phases.sh
#
# Optional:
#   DEVICE=cpu ./run_qwen25_0_5b_gsm8k_all_phases.sh
#   PYTHON_BIN=python3 ./run_qwen25_0_5b_gsm8k_all_phases.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="gsm8k"
CONFIG="S2"
MODEL="qwen25_0.5b"
RESULTS_DIR="results/$CONFIG/$MODEL"
export PYTHONUNBUFFERED=1
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$RESULTS_DIR/run_logs/$RUN_ID"
MASTER_LOG="$RUN_LOG_DIR/full_run.log"

mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

printf 'Run log: %s\n' "$MASTER_LOG"
printf 'Results dir: %s\n' "$RESULTS_DIR"
{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'root_dir=%s\n' "$ROOT_DIR"
  printf 'dataset=%s\n' "$DATASET"
  printf 'config=%s\n' "$CONFIG"
  printf 'model=%s\n' "$MODEL"
  printf 'device=%s\n' "${DEVICE:-cuda/default}"
  printf 'python_bin=%s\n' "$PYTHON_BIN"
  printf 'python_unbuffered=%s\n' "${PYTHONUNBUFFERED:-}"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
  printf 'git_branch=%s\n' "$(git branch --show-current 2>/dev/null || true)"
} > "$RUN_LOG_DIR/manifest.txt"
"$PYTHON_BIN" --version > "$RUN_LOG_DIR/python_version.txt" 2>&1 || true
"$PYTHON_BIN" -m pip freeze > "$RUN_LOG_DIR/pip_freeze.txt" 2>&1 || true
nvidia-smi > "$RUN_LOG_DIR/nvidia_smi.txt" 2>&1 || true

finish_run() {
  local exit_code="$?"
  printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "$RUN_LOG_DIR/manifest.txt"
  printf 'exit_code=%s\n' "$exit_code" >> "$RUN_LOG_DIR/manifest.txt"
}
trap finish_run EXIT

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

run_step "PHASE 1: Coconut training and validation for qwen25_0.5b" \
  "$PYTHON_BIN" pipeline.py --phase 1 "${PIPELINE_ARGS[@]}"

run_step "PHASE 2: hidden-state extraction and truth-vector construction for qwen25_0.5b" \
  "$PYTHON_BIN" pipeline.py --phase 2 "${PIPELINE_ARGS[@]}"

run_step "PHASE 3: alpha tuning and steered validation for qwen25_0.5b" \
  "$PYTHON_BIN" pipeline.py --phase 3 "${PIPELINE_ARGS[@]}"

run_step "PHASE 4: final locked test evaluation for qwen25_0.5b" \
  "$PYTHON_BIN" pipeline.py --phase 4 "${PIPELINE_ARGS[@]}"

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "$RUN_LOG_DIR/manifest.txt"
printf '\nAll qwen25_0.5b / gsm8k phases completed successfully.\n'
printf 'Full run log saved at: %s\n' "$MASTER_LOG"
