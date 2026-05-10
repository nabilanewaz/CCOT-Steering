# Execution Commands — CCoT Steering Pipeline

All commands are run from the **project root** (`d:\Thesis\CCOT-Steering`).  
Dataset used throughout: **GSM8K** (replace `gsm8k` with `svamp` or `prontoqa` for other datasets).  
GPU is assumed (`cuda`). Append `--device cpu` to any `pipeline.py` command if needed.

---

## 0. Environment

```bash
pip install -r requirements.txt
```

If `llmlingua` is needed separately:

```bash
pip install llmlingua
```

---

## 1. Download Dataset

Downloads the raw dataset and writes `gsm8k/train.jsonl` and `gsm8k/test.jsonl`.  
Run once per dataset. Persists the active dataset choice to `configs/active_dataset.txt`.

```bash
python download_dataset.py --dataset gsm8k
```

For SVAMP or ProntoQA transfer experiments:

```bash
python download_dataset.py --dataset svamp
python download_dataset.py --dataset prontoqa
```

---

## 2. Verify Data Isolation (pre-flight check)

Confirms D\_train / D\_steer / D\_val have zero overlap with D\_test by item ID.  
Run once after downloading, and again before Phase 4.

```bash
python verify_isolation.py
```

---

## 3. Build Offline Compression Cache

Compresses D\_train reasoning traces at all five ratios (0.5–0.9) using LLMLingua-2.  
Writes `cache/S2/compressed_R5.jsonl` … `cache/S2/compressed_R9.jsonl`.  
Run once. Safe to re-run — already-built files are skipped.

```bash
python preprocess_compress.py
```

To target a specific dataset explicitly:

```bash
python preprocess_compress.py --dataset gsm8k
```

---

## 4. Phase 1 — Fine-tuning (CoT + CCoT) and Evaluation

Trains one CoT checkpoint and five CCoT checkpoints (one per ratio) per backbone.  
Evaluates all on D\_val and writes `results/S2/<model>/phase1_val.json`.

**All four backbones (recommended):**

```bash
python pipeline.py --phase 1
```

**Single backbone:**

```bash
python pipeline.py --phase 1 --model llama32_3b
python pipeline.py --phase 1 --model phi2
python pipeline.py --phase 1 --model qwen25_3b
python pipeline.py --phase 1 --model qwen25_math1.5b
```

Checkpoint output: `checkpoints/S2/<model>/cot/` and `checkpoints/S2/<model>/ccot_R{5..9}/`

---

## 5. Phase 2 — Truth Vector Extraction

Collects hidden states from D\_steer, runs probing, computes DoM and cPCA vectors,  
and also produces the shuffled-label control vectors.  
Writes to `vectors/S2/<model>/`.

**All four backbones:**

```bash
python pipeline.py --phase 2
```

**Single backbone:**

```bash
python pipeline.py --phase 2 --model llama32_3b
```

Key output files per model:
- `vectors/S2/<model>/ccot_dom.pt` — CCoT DoM vector
- `vectors/S2/<model>/base_dom.pt` — CoT DoM vector
- `vectors/S2/<model>/ccot_cpca_r10.pt` — CCoT cPCA subspace
- `vectors/S2/<model>/phase2_meta.json` — best probe layer, probe accuracy

---

## 6. Phase 3 — α-Tuning and Steered Evaluation

Runs the λ sweep, learns α\* via AdamW, evaluates all steered conditions on D\_val  
(including control conditions), and selects the best steered config per backbone.  
Updates `configs/selected.yaml`.

**All four backbones:**

```bash
python pipeline.py --phase 3
```

**Single backbone:**

```bash
python pipeline.py --phase 3 --model llama32_3b
```

Key output files per model:
- `results/S2/<model>/phase3_val.json` — per-condition D\_val accuracy
- `results/S2/<model>/phase3_best_config.yaml` — selected condition for this backbone
- `vectors/S2/<model>/ccot_alpha_star.pt` — learned α\*
- `vectors/S2/<model>/ccot_alpha_history.json` — training loss curves per epoch
- `vectors/S2/<model>/ccot_lambda_sweep.json` — λ grid search results
- `plots/S2/<model>/loss_curves_*.png` — L\_ans / L\_align / L\_mag curves
- `plots/S2/<model>/lambda_heatmap_*.png` — λ sweep heatmap

After Phase 3 completes, `configs/selected.yaml` is written automatically with `winning_config: S2`.

---

## 7. Phase 4 — Final Evaluation on D\_test

**Run once. D\_test is opened exactly once here and nowhere else.**  
Uses the locked configs from `configs/selected.yaml`.  
Writes results to `results/final/`.

```bash
python pipeline.py --phase 4
```

Or invoke `evaluate_final.py` directly (same effect — `pipeline.py` delegates to it):

```bash
python evaluate_final.py
```

Custom results directory:

```bash
python evaluate_final.py --results-dir results/final_gsm8k
```

Key output files:
- `results/final/summary_test.json` — cross-model summary with bootstrap CIs
- `results/final/<model>_test.json` — per-model per-condition results

---

## 8. Full Pipeline in One Command

Runs phases 1 → 2 → 3 → 4 sequentially for all backbones:

```bash
python pipeline.py --phase 0
```

Alternatively, the simplified sweep runner (equivalent for phases 1–4):

```bash
python scripts/run_sweep.py
```

With an explicit dataset:

```bash
python scripts/run_sweep.py --dataset gsm8k
```

---

## 9. Phase 5 — SVAMP Transfer Evaluation (optional)

Evaluates the **frozen GSM8K** steering vectors on SVAMP D\_test.  
No re-tuning of v\_truth or α\* is performed.  
Requires Phase 4 to have completed first (`configs/selected.yaml` must exist).

```bash
python pipeline.py --phase 5
```

Results are written to `results/final_svamp_transfer/` (separate from Phase 4 output).

Print the transfer summary table:

```bash
python scripts/print_transfer_summary.py results/final_svamp_transfer
```

---

## 10. Standalone Utilities

**Re-run split selection only** (reads existing Phase 3 val results):

```bash
python compare_all.py
```

**Build splits to disk** (for inspection — not required by the pipeline):

```bash
python scripts/build_splits.py --out configs/splits
```

**Verify isolation again before Phase 4:**

```bash
python verify_isolation.py
```

---

## Execution Order Summary

```
1. python download_dataset.py --dataset gsm8k
2. python verify_isolation.py
3. python preprocess_compress.py
4. python pipeline.py --phase 1        # train CoT + CCoT, eval on D_val
5. python pipeline.py --phase 2        # extract truth vectors
6. python pipeline.py --phase 3        # tune alpha, steered eval on D_val
7. python verify_isolation.py          # confirm isolation before opening D_test
8. python pipeline.py --phase 4        # final eval on D_test (once only)
9. python pipeline.py --phase 5        # (optional) SVAMP transfer
```

---

## Common Options

| Flag | Values | Applies to |
|---|---|---|
| `--dataset` | `gsm8k` \| `svamp` \| `prontoqa` | all entry points |
| `--phase` | `0`–`5` | `pipeline.py` |
| `--model` | `llama32_3b` \| `phi2` \| `qwen25_3b` \| `qwen25_math1.5b` | `pipeline.py` |
| `--device` | `cuda` \| `cpu` | `pipeline.py` |
| `--results-dir` | any path | `evaluate_final.py` |

All phases are **resumable** — if a checkpoint or result file already exists the step is skipped automatically.
