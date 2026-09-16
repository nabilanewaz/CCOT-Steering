#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# One-command setup and execution for the complete core experiment:
#   dataset: GSM8K
#   config:  S3 (full 60/10/30 split: 4484/747/2242; test=1319)
#   model:   Qwen/Qwen2.5-3B
#   phases:  environment setup, data setup, Phase 1, Phase 2, Phase 3, Phase 4, Phase 5
#
# Usage:
#   chmod +x run_qwen25_3b_gsm8k_all_phases.sh
#   ./run_qwen25_3b_gsm8k_all_phases.sh
#
# Optional environment variables:
#   PYTHON_BIN=python3.11       Bootstrap Python (must be >= 3.10)
#   VENV_DIR=/path/to/venv      Virtual environment (default: .venv)
#   DEVICE=cuda                 Pipeline device (default: cuda)
#   TORCH_INDEX_URL=<url>       Override the auto-detected official PyTorch wheel index
#   FORCE_PHASE4=1              Re-run Phase 4 even if final result files exist
#   RUN_PHASE5=0                Skip the Phase 5 frozen-transfer evaluation
#
# For CUDA-specific PyTorch wheels, obtain the appropriate index URL from:
#   https://pytorch.org/get-started/locally/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BOOTSTRAP_PYTHON="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
DEVICE="${DEVICE:-cuda}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
FORCE_PHASE4="${FORCE_PHASE4:-0}"
RUN_PHASE5="${RUN_PHASE5:-1}"

DATASET="gsm8k"
CONFIG="S3"
MODEL="qwen25_3b"
MODEL_ID="Qwen/Qwen2.5-3B"
RESULTS_DIR="results/$CONFIG/$MODEL"

export CCOT_DATASET="$DATASET"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="$ROOT_DIR/logs/qwen25_3b/$RUN_ID"
MASTER_LOG="$RUN_LOG_DIR/full_run.log"
MANIFEST="$RUN_LOG_DIR/manifest.txt"
mkdir -p "$RUN_LOG_DIR"

# Mirror stdout and stderr to both the terminal and a durable log.
exec > >(tee -a "$MASTER_LOG") 2>&1

CURRENT_STEP="startup"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

on_error() {
  local exit_code="$1"
  local line_number="$2"
  local failed_command="$3"
  printf '\nERROR: step=%s exit_code=%s line=%s\n' "$CURRENT_STEP" "$exit_code" "$line_number"
  printf 'Failed command: %s\n' "$failed_command"
  printf 'Inspect the full log: %s\n' "$MASTER_LOG"
}

on_exit() {
  local exit_code="$?"
  {
    printf 'finished_at=%s\n' "$(timestamp)"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'last_step=%s\n' "$CURRENT_STEP"
  } >> "$MANIFEST"
  if [[ "$exit_code" -eq 0 ]]; then
    printf '\nSUCCESS: all requested phases completed.\n'
    printf 'Full log: %s\n' "$MASTER_LOG"
  else
    printf '\nFAILED with exit code %s. Full log: %s\n' "$exit_code" "$MASTER_LOG"
  fi
}

trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
trap on_exit EXIT

run_step() {
  local label="$1"
  shift
  CURRENT_STEP="$label"
  printf '\n========================================================================\n'
  printf '[%s] %s\n' "$(timestamp)" "$label"
  printf '========================================================================\n'
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

printf 'CCOT-Steering complete runner\n'
printf 'Repository: %s\n' "$ROOT_DIR"
printf 'Model:      %s (%s)\n' "$MODEL" "$MODEL_ID"
printf 'Dataset:    %s\n' "$DATASET"
printf 'Device:     %s\n' "$DEVICE"
printf 'Run log:    %s\n' "$MASTER_LOG"

{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'started_at=%s\n' "$(timestamp)"
  printf 'root_dir=%s\n' "$ROOT_DIR"
  printf 'dataset=%s\n' "$DATASET"
  printf 'config=%s\n' "$CONFIG"
  printf 'model=%s\n' "$MODEL"
  printf 'model_id=%s\n' "$MODEL_ID"
  printf 'device=%s\n' "$DEVICE"
  printf 'venv_dir=%s\n' "$VENV_DIR"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unavailable)"
  printf 'git_branch=%s\n' "$(git branch --show-current 2>/dev/null || printf unavailable)"
} > "$MANIFEST"

if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
  printf 'ERROR: %s was not found. Install Python 3.10 or newer, or set PYTHON_BIN.\n' "$BOOTSTRAP_PYTHON"
  exit 1
fi

run_step "Checking bootstrap Python" \
  "$BOOTSTRAP_PYTHON" -c 'import sys; print("Python:", sys.version); raise SystemExit(0 if sys.version_info >= (3, 10) else "Python >= 3.10 is required")'

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  CURRENT_STEP="Creating virtual environment"
  printf '\n========================================================================\n'
  printf '[%s] %s\n' "$(timestamp)" "$CURRENT_STEP"
  printf '========================================================================\n'
  if ! "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"; then
    printf '\nUnable to create a Python virtual environment.\n'
    printf 'On Debian/Ubuntu, install the matching python3-venv package and rerun.\n'
    exit 1
  fi
else
  printf '\nEnvironment: reusing %s\n' "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"

run_step "Upgrading Python packaging tools" \
  "$PYTHON" -m pip install --upgrade pip setuptools wheel

# PyPI's newest Linux torch wheel may target a CUDA runtime newer than the
# host driver supports.  nvidia-smi reports the newest CUDA runtime supported
# by the driver, so select the newest official PyTorch wheel channel that does
# not exceed it.  An explicit TORCH_INDEX_URL always takes precedence.
if [[ -z "$TORCH_INDEX_URL" && "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null 2>&1; then
  NVIDIA_MAX_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p')"
  case "$NVIDIA_MAX_CUDA" in
    13.*|12.9|12.8) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128" ;;
    12.7|12.6)      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126" ;;
    12.5|12.4)      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124" ;;
    12.3|12.2|12.1) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" ;;
    11.9|11.8)      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118" ;;
  esac
  if [[ -n "$TORCH_INDEX_URL" ]]; then
    printf '\nPyTorch: NVIDIA driver supports CUDA %s; selected %s\n' \
      "$NVIDIA_MAX_CUDA" "$TORCH_INDEX_URL"
  else
    printf '\nWARNING: could not map NVIDIA CUDA version %q to a PyTorch wheel index.\n' \
      "$NVIDIA_MAX_CUDA"
    printf 'Set TORCH_INDEX_URL explicitly if the default PyPI wheel is incompatible.\n'
  fi
fi

if [[ -n "$TORCH_INDEX_URL" ]]; then
  TORCH_BUILD="${TORCH_INDEX_URL%/}"
  TORCH_BUILD="${TORCH_BUILD##*/}"
  INSTALLED_TORCH_VERSION="$("$PYTHON" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
  if [[ "$INSTALLED_TORCH_VERSION" == *"+$TORCH_BUILD"* ]]; then
    printf '\nPyTorch: reusing compatible torch %s\n' "$INSTALLED_TORCH_VERSION"
  else
    printf '\nPyTorch: replacing incompatible torch %s with the %s build\n' \
      "${INSTALLED_TORCH_VERSION:-not-installed}" "$TORCH_BUILD"
    run_step "Installing PyTorch from the compatible CUDA wheel index" \
      "$PYTHON" -m pip install --upgrade --force-reinstall \
        --index-url "$TORCH_INDEX_URL" 'torch>=2.2.0'
  fi
fi

run_step "Installing repository requirements" \
  "$PYTHON" -m pip install -r requirements.txt

run_step "Checking installed dependency consistency" \
  "$PYTHON" -m pip check

run_step "Validating Python packages and compute device" \
  env REQUESTED_DEVICE="$DEVICE" "$PYTHON" -u -c '
import os
import accelerate
import datasets
import matplotlib
import numpy
import peft
import scipy
import sklearn
import torch
import transformers
import yaml

requested = os.environ["REQUESTED_DEVICE"]
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"peft={peft.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"datasets={datasets.__version__}")
print(f"CUDA requested: {requested}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gib = props.total_memory / (1024 ** 3)
        print(f"GPU {index}: {props.name} ({gib:.1f} GiB)")
if requested.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit(
        "CUDA was requested but this PyTorch installation cannot access a GPU. "
        "Install the correct NVIDIA driver/PyTorch CUDA wheel, optionally using "
        "TORCH_INDEX_URL from https://pytorch.org/get-started/locally/."
    )
if not requested.startswith("cuda"):
    print("WARNING: Qwen2.5-3B full-model training on CPU will be extremely slow.")
'

if command -v nvidia-smi >/dev/null 2>&1; then
  run_step "Displaying NVIDIA driver and GPU status" nvidia-smi
else
  printf '\nGPU status: nvidia-smi is not installed or not on PATH.\n'
fi

printf '\nDisk space before the run:\n'
df -h "$ROOT_DIR" || true

run_step "Saving installed package versions" \
  "$PYTHON" -m pip freeze
"$PYTHON" -m pip freeze > "$RUN_LOG_DIR/pip_freeze.txt"

run_step "Running focused offline tests" \
  "$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v

if [[ ! -s "$DATASET/train.jsonl" || ! -s "$DATASET/test.jsonl" ]]; then
  run_step "Downloading and preparing GSM8K" \
    "$PYTHON" -u download_dataset.py --dataset "$DATASET"
else
  printf '\nDATA: using existing %s/train.jsonl and %s/test.jsonl\n' "$DATASET" "$DATASET"
fi

run_step "Verifying train/steer/validation/test isolation" \
  "$PYTHON" -u verify_isolation.py

run_step "Building the Phase 1 compatibility cache" \
  "$PYTHON" -u preprocess_compress.py --dataset "$DATASET" --config "$CONFIG"

PIPELINE_ARGS=(
  --config "$CONFIG"
  --model "$MODEL"
  --dataset "$DATASET"
  --device "$DEVICE"
)

run_step "PHASE 1/4: Coconut training and validation (30 epochs)" \
  "$PYTHON" -u pipeline.py --phase 1 "${PIPELINE_ARGS[@]}"

run_step "PHASE 2/4: Hidden-state extraction and truth-vector construction" \
  "$PYTHON" -u pipeline.py --phase 2 "${PIPELINE_ARGS[@]}"

run_step "PHASE 3/4: Alpha tuning, steered validation, and configuration lock" \
  "$PYTHON" -u pipeline.py --phase 3 "${PIPELINE_ARGS[@]}"

run_step "Pre-Phase-4 isolation verification" \
  "$PYTHON" -u verify_isolation.py

FINAL_MODEL_RESULT="results/final/${MODEL}_test.json"
FINAL_SUMMARY="results/final/summary_test.json"
FINAL_ARGS=(--dataset "$DATASET" --model "$MODEL")
if [[ "$FORCE_PHASE4" == "1" ]]; then
  FINAL_ARGS+=(--overwrite)
fi
run_step "PHASE 4/4: Locked final evaluation on D_test" \
  "$PYTHON" -u evaluate_final.py "${FINAL_ARGS[@]}"

if [[ "$RUN_PHASE5" == "1" ]]; then
  if [[ ! -s "svamp/train.jsonl" || ! -s "svamp/test.jsonl" ]]; then
    run_step "OPTIONAL PHASE 5: Downloading SVAMP" \
      "$PYTHON" -u -c 'from download_dataset import download_svamp; download_svamp()'
  fi
  run_step "OPTIONAL PHASE 5: Frozen GSM8K-to-SVAMP transfer evaluation" \
    env CCOT_DATASET=svamp "$PYTHON" -u evaluate_final.py \
      --dataset svamp --training-dataset gsm8k \
      --results-dir results/final_svamp_transfer \
      --model "$MODEL"
fi

CURRENT_STEP="complete"
printf '\n========================================================================\n'
printf 'Qwen2.5-3B experiment completed successfully.\n'
printf 'Phase results: %s\n' "$RESULTS_DIR"
printf 'Final results: results/final/\n'
if [[ "$RUN_PHASE5" == "1" ]]; then
  printf 'Transfer results: results/final_svamp_transfer/\n'
fi
printf 'Full terminal log: %s\n' "$MASTER_LOG"
printf '========================================================================\n'
