# RunPod Trial — `qwen25_math1.5b` + GSM8K

All commands below assume you are inside a RunPod terminal (bash).  
Target model: `Qwen/Qwen2.5-Math-1.5B`  
Dataset: GSM8K

---

## Pod Selection

| Spec | Minimum | Recommended |
|------|---------|-------------|
| GPU VRAM | 16 GB | 24 GB (RTX 3090 / RTX 4090 / A10G) |
| RAM | 32 GB | 64 GB |
| Disk (container) | 30 GB | 50 GB |
| Disk (volume / persistent) | 40 GB | 80 GB |

> Mount a **Network Volume** at `/workspace` so checkpoints survive pod restarts.  
> Use the **RunPod PyTorch** template (CUDA 12.1, PyTorch 2.x already installed).

---

## 0. One-time Pod Setup

Run these once when the pod is first started.

```bash
# ── 0-A. Set working directory ──────────────────────────────────────────────
cd /workspace

# ── 0-B. Clone the repo ──────────────────────────────────────────────────────
git clone https://github.com/nabilanewaz/CCOT-Steering.git
cd CCOT-Steering

# ── 0-C. Install dependencies ────────────────────────────────────────────────
pip install -r requirements.txt

# ── 0-D. Set HuggingFace cache to the persistent volume ──────────────────────
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
mkdir -p /workspace/hf_cache

# ── 0-E. (Optional) HuggingFace token — only needed if gated models are used
# Qwen2.5-Math-1.5B is public, so this step can be skipped.
# huggingface-cli login --token hf_YOUR_TOKEN_HERE

# ── 0-F. Pre-download the model weights ──────────────────────────────────────
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-1.5B',
    torch_dtype='auto', trust_remote_code=True)
print('Model cached.')
"
```

---

## 1. Download Dataset

```bash
python download_dataset.py --dataset gsm8k
```

Expected output: `gsm8k/train.jsonl` and `gsm8k/test.jsonl`

---

## 2. Verify Data Isolation

```bash
python verify_isolation.py
```

All three checks (D_train / D_steer / D_val vs D_test) must pass before proceeding.

---

## 3. Build Compression Cache

Compresses D_train reasoning traces at ratios 0.5–0.9 using LLMLingua-2.  
Writes 5 files to `cache/S2/`.

```bash
python preprocess_compress.py --dataset gsm8k
```

Expected runtime: ~20–40 min on GPU for the 1.5B model.  
Safe to re-run — already-built ratio files are skipped.

---

## 4. Phase 1 — Fine-tuning + Evaluation

Trains one CoT checkpoint and five CCoT checkpoints, then evaluates all on D_val.

```bash
python pipeline.py --phase 1 --model qwen25_math1.5b --dataset gsm8k
```

Expected outputs:
- `checkpoints/S2/qwen25_math1.5b/cot/`
- `checkpoints/S2/qwen25_math1.5b/ccot_R5/` … `ccot_R9/`
- `results/S2/qwen25_math1.5b/phase1_val.json`

Expected runtime: ~1–2 hours (CoT: 1 epoch; CCoT: 3 epochs × 5 ratios).

---

## 5. Phase 2 — Truth Vector Extraction

Collects hidden states from D_steer, runs probing, computes DoM and cPCA vectors.

```bash
python pipeline.py --phase 2 --model qwen25_math1.5b --dataset gsm8k
```

Expected outputs:
- `vectors/S2/qwen25_math1.5b/ccot_dom.pt`
- `vectors/S2/qwen25_math1.5b/base_dom.pt`
- `vectors/S2/qwen25_math1.5b/ccot_cpca_r10.pt`
- `vectors/S2/qwen25_math1.5b/phase2_meta.json`

Expected runtime: ~20–40 min.

---

## 6. Phase 3 — Alpha Tuning + Steered Evaluation

Runs the λ sweep, learns α*, evaluates all steered conditions on D_val.

```bash
python pipeline.py --phase 3 --model qwen25_math1.5b --dataset gsm8k
```

Expected outputs:
- `results/S2/qwen25_math1.5b/phase3_val.json`
- `results/S2/qwen25_math1.5b/phase3_best_config.yaml`
- `vectors/S2/qwen25_math1.5b/ccot_alpha_star.pt`
- `vectors/S2/qwen25_math1.5b/ccot_lambda_sweep.json`
- `plots/S2/qwen25_math1.5b/` (loss curves, λ heatmap)
- `configs/selected.yaml` (auto-written at end of Phase 3)

Expected runtime: ~1–2 hours.

---

## 7. Verify Isolation Before Opening D_test

```bash
python verify_isolation.py
```

All checks must pass. D_test has not been touched yet.

---

## 8. Phase 4 — Final Evaluation on D_test

**Run once only.** D_test is opened here and nowhere else.

```bash
python evaluate_final.py --models qwen25_math1.5b --results-dir results/final_qwen_trial
```

Or equivalently via pipeline:

```bash
python pipeline.py --phase 4 --model qwen25_math1.5b --dataset gsm8k
```

Expected outputs in `results/final_qwen_trial/`:
- `qwen25_math1.5b_test.json`
- `summary_test.json`
- `qwen25_math1.5b_diagnostics.json`
- `plots/qwen25_math1.5b/` — 8 per-model plots
- `tables/` — per-condition accuracy, CI, flip matrix, mechanism gain tables

Expected runtime: ~30–60 min.

---

## 9. (Optional) SVAMP Transfer

Evaluates the frozen GSM8K vectors on SVAMP test set. Requires Phase 4 to have completed.

```bash
python download_dataset.py --dataset svamp
python pipeline.py --phase 5 --model qwen25_math1.5b
python scripts/print_transfer_summary.py results/final_svamp_transfer
```

---

## Full Pipeline in One Shot

```bash
cd /workspace/CCOT-Steering
export HF_HOME=/workspace/hf_cache TRANSFORMERS_CACHE=/workspace/hf_cache

python download_dataset.py --dataset gsm8k && \
python verify_isolation.py && \
python preprocess_compress.py --dataset gsm8k && \
python pipeline.py --phase 1 --model qwen25_math1.5b --dataset gsm8k && \
python pipeline.py --phase 2 --model qwen25_math1.5b --dataset gsm8k && \
python pipeline.py --phase 3 --model qwen25_math1.5b --dataset gsm8k && \
python verify_isolation.py && \
python evaluate_final.py --models qwen25_math1.5b --results-dir results/final_qwen_trial
```

---

## Persistence Notes

| Path | What to keep |
|------|--------------|
| `/workspace/hf_cache/` | Downloaded model weights — survives pod restart if on volume |
| `/workspace/CCOT-Steering/checkpoints/` | LoRA adapters from Phase 1 |
| `/workspace/CCOT-Steering/vectors/` | Truth vectors from Phase 2 |
| `/workspace/CCOT-Steering/results/` | All evaluation results |
| `/workspace/CCOT-Steering/cache/` | Compressed reasoning cache from Phase 3 |

> If you stop the pod, all files inside the **container disk** are lost. Mount all output paths to the **Network Volume** (`/workspace`) before starting.

---

## Resuming After a Pod Restart

All phases are **resumable** — existing checkpoint/result files are skipped automatically.

```bash
cd /workspace/CCOT-Steering
export HF_HOME=/workspace/hf_cache TRANSFORMERS_CACHE=/workspace/hf_cache
# re-run whichever phase was interrupted; it will pick up where it left off
python pipeline.py --phase 1 --model qwen25_math1.5b --dataset gsm8k
```

---

## Checking GPU

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 1e9, 'GB')"
```
