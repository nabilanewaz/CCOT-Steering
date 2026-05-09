# CCOT-Steering: Steering Continuous Reasoning via Latent Intervention

An end-to-end framework for extracting and steering latent reasoning in language models using inference-time intervention. Four phases: training baselines, extracting truth vectors, tuning steering intensity, and final evaluation.

---

## Quick Start

### 0. Environment & Data

```bash
# Install dependencies
pip install -r requirements.txt

# Download GSM8K (7,473 train + 1,319 test)
python download_gsm8k.py

# Verify data isolation (no train/val/test overlap)
python verify_isolation.py
```

### 1. Pre-process: Build Compression Cache

```bash
# Pre-compute TokenSkip compression at ratios [0.5, 0.6, 0.7, 0.8, 0.9]
# Runs once, outputs to cache/compressed/R{5,6,7,8,9}.jsonl
python preprocess_compress.py
```

### 2. Full Pipeline (all Phases 1–4)

```bash
# Train, extract, tune, and evaluate all split configs × all models
python pipeline.py --phase 0

# Or run phase-by-phase
python pipeline.py --phase 1                    # train (≈2h per model)
python pipeline.py --phase 2                    # extract vectors (≈3h per model)
python pipeline.py --phase 3                    # α-tune + steered eval (≈3h per model)
python pipeline.py --phase 4                    # final D_test eval (≈1h total)
```

### 3. Selective Runs (one split + one model)

```bash
python pipeline.py --phase 1 --config S2 --model llama32_3b
python pipeline.py --phase 2 --config S2 --model llama32_3b
python pipeline.py --phase 3 --config S2 --model llama32_3b
```

### 4. View Results

- Per-config metrics: `results/{S1,S2,S3,S4}/{model}/phase{1,2,3}_val.json`
- Winning config: `configs/selected.yaml`
- Final D_test results: `results/final/{model}_test.json` + `summary_test.json`

---

## Project Structure

```
project/
├── configs/
│   ├── protocol.yaml           # master hyperparameters
│   └── selected.yaml           # written at end of Phase 3 — winning config
├── data/gsm8k/
│   ├── train.jsonl             # 7,473 examples — split pool
│   └── test.jsonl              # 1,319 examples (locked until Phase 4)
├── cache/compressed/           # pre-computed TokenSkip traces
│   ├── R5.jsonl  ├── R6.jsonl  ├── R7.jsonl  ├── R8.jsonl  └── R9.jsonl
├── checkpoints/{S1,S2,S3,S4}/{model}/
│   ├── cot/                    # Stage 1: CoT LoRA adapter
│   ├── ccot_R5/ ├── ccot_R6/ ├── ccot_R7/ ├── ccot_R8/ └── ccot_R9/
├── vectors/{S1,S2,S3,S4}/{model}/
│   ├── ccot_dom.pt             # Source A: DoM vector
│   ├── ccot_cpca_r10.pt        # Source A: cPCA subspace [d, r_final]
│   ├── base_dom.pt             # Source B: DoM vector
│   ├── base_cpca_r10.pt        # Source B: cPCA subspace
│   ├── {source}_alpha_star.pt  # tuned steering intensity α*
│   └── phase2_meta.json        # probe scores + layer metadata
├── results/
│   ├── {S1,S2,S3,S4}/{model}/
│   │   ├── phase1_val.json     # 12 conditions on D_val
│   │   ├── phase2_probe_scores.json
│   │   └── phase3_val.json     # steered 52-condition grid on D_val
│   └── final/                  # Phase 4 D_test results (run once)
│       ├── llama32_3b_test.json
│       ├── phi2_test.json
│       ├── qwen25_3b_test.json
│       ├── qwen25_math1.5b_test.json
│       └── summary_test.json
├── phase1/                     # training: CoT + CCoT
├── phase2/                     # extraction: hidden states → vectors
├── phase3/                     # tuning: LearnableAlpha + steered eval
├── scripts/                    # build_splits, selection, run_sweep
├── utils/                      # data guards, metrics, helpers
├── download_gsm8k.py           # fetch dataset
├── preprocess_compress.py      # build compression cache
├── verify_isolation.py         # check data isolation invariants
├── pipeline.py                 # master orchestrator (wrapper)
├── evaluate_final.py           # Phase 4: only script opening test.jsonl
├── compare_all.py              # read results/, print summary tables
├── requirements.txt
└── README.md
```

---

## Phase Breakdown

### Phase 1: Training Baselines (Stages 1 & 2)

**Stage 1: CoT Fine-Tuning**

Train base model to generate full reasoning chains on D_train.

```bash
# Called automatically by pipeline.py or scripts/run_sweep.py
# Manual: python -m phase1 --stage cot --config S1 --model llama32_3b
```

- Adapter saved → `checkpoints/{config}/{model}/cot/`
- Time: ~2h per model (A100)

**Stage 2: CCoT Fine-Tuning (per ratio)**

Train base model on compressed reasoning (via pre-computed TokenSkip traces) for each ratio.

```bash
# Called automatically; manual:
# python -m phase1 --stage ccot --config S1 --model llama32_3b --ratio 0.7
```

- Adapters saved → `checkpoints/{config}/{model}/ccot_R{5,6,7,8,9}/`
- Time: ~8h for all 5 ratios (A100)

**Phase 1 Evaluation**

Run 12 conditions on D_val: no_cot, full_cot, trimmed_cot (per ratio), ccot (per ratio).

- Results saved → `results/{config}/{model}/phase1_val.json`
- Accuracy, token count, latency per condition
- Time: ~1h (A100)

### Phase 2: Truth Vector Extraction

**Overview**

Collect hidden states from positive (correct reasoning) and negative (incorrect) examples. Score layers via logistic regression. Extract DoM and cPCA vectors weighted by probe accuracy.

**Two Sources**

1. **Source A (CCoT)**: hidden states from the best CCoT checkpoint (R ratio with highest mechanism gain)
2. **Source B (CoT)**: hidden states from the full CoT checkpoint

**Per Source**

1. Collect H⁺ and H⁻ from D_steer (×15 rollouts per question)
2. Score all layers (logistic regression on held-out split)
3. Select layers above threshold (mean + 0.5×std)
4. Compute per-layer DoM and cPCA
5. Merge via weighted SVD (weights = probe_acc × eigenvalue × directional_agreement)
6. Compare DoM vs cPCA on held-out split → pick winner

**Saved Artifacts**

- `vectors/{config}/{model}/{source}_dom.pt` — [d]
- `vectors/{config}/{model}/{source}_cpca_r10.pt` — [d, r_final]
- `vectors/{config}/{model}/phase2_meta.json` — probe scores + layer metadata

**Manual Run**

```bash
# Called by pipeline.py; manual:
# python -m phase2 --config S1 --model llama32_3b
```

Time: ~3h per model (A100)

### Phase 3: Inference-Time Steering (LearnableAlpha Tuning)

**Overview**

For each vector source, tune α (steering intensity) on D_val using differentiable loss:

$$\mathcal{L} = \text{NLL}_{\text{answer}} + \lambda_a \cdot \text{align\_loss} + \lambda_m \cdot \text{mag\_penalty}$$

Then evaluate all 8 conditions × all 5 ratios × 2 sources on D_val (52 total).

**Conditions**

1. No CoT (baseline)
2. Full CoT (upper bound)
3. CCoT (unsteered)
4. Trimmed CoT (token budget matched)
5. Random Noise (control)
6. CCoT + DoM (steering with 1-D vector)
7. CCoT + cPCA (steering with r-D subspace)
8. Trimmed + DoM (steering on trimmed baseline)

**Selection**

Per model, pick the single best (source, ratio, method) that maximizes Wilson CI lower bound on D_val accuracy.

- Best config → `results/{config}/{model}/phase3_best_config.yaml`
- α* saved → `vectors/{config}/{model}/{source}_alpha_star.pt`

**Manual Run**

```bash
# Called by pipeline.py; manual:
# python -m phase3 --config S1 --model llama32_3b
```

Time: ~3h per model (A100)

### Split Selection

After Phase 3 for all configs × all models, compute mean Wilson CI lower bound
on steered accuracy across models. Pick split S1/S2/S3/S4 with highest mean.

```bash
# Called by pipeline.py; manual:
python compare_all.py
# Writes: configs/selected.yaml (winning_config + per-model phase3_best)
```

### Phase 4: Final Evaluation on D_test

**The Only Script That Opens test.jsonl**

Run all 8 conditions for each model using locked Phase 3 configs. Compute:

- Per-condition accuracy, token count, latency, latent metrics
- Flip matrices (pairwise comparison: improvements vs degradations)
- Full net-gain grid (all pairs, all conditions)
- α sweep diagnostic (paper figure: accuracy + truth alignment + trajectory coherence)

**Guard**

`utils/data.py` raises `RuntimeError` if `load_test_set()` is called from any script other than `evaluate_final.py`.

```bash
# Called by pipeline.py; manual:
python evaluate_final.py
```

Results → `results/final/{model}_test.json` + `summary_test.json`

Time: ~1h (A100 for all 4 models)

---

## Configuration

### configs/protocol.yaml

Master configuration file. Sets:

- Train/steer/val split ratios (S1–S4)
- Compression ratios [0.5, 0.6, 0.7, 0.8, 0.9]
- Model IDs (Llama, Phi-2, Qwen)
- Phase 1: LoRA, training epochs, batch size
- Phase 2: n_rollouts, layer threshold, cPCA rank
- Phase 3: α_max, α_lr, λ_a (alignment loss weight), λ_m (magnitude penalty)
- Phase 4: output directory

Modify ratios and hyperparameters before running the full pipeline.

---

## Checkpoint Resume

All phases check if output exists before running:

```python
if not already_done(output_path):
    # run phase
else:
    print("output exists — skipping")
```

If interrupted, restart the same command — it resumes from the last completed step.

To force re-run, delete the checkpoint:

```bash
rm -rf checkpoints/S2/llama32_3b/ccot_R7/
python pipeline.py --phase 1 --config S2 --model llama32_3b
```

---

## Expected Compute

| Task | Per Model | All 4 Models |
|------|-----------|--------------|
| Phase 1: CoT training | 2h | 8h |
| Phase 1: CCoT training (5 ratios) | 8h | 32h |
| Phase 1: Eval (D_val) | 1h | 4h |
| Phase 2: Extraction (2 sources) | 3h | 12h |
| Phase 3: α-tuning + steered eval | 3h | 12h |
| Phase 4: Final eval (D_test) | 1h | 4h |
| **Total (1 config)** | **18h** | **72h** |
| **Total (4 configs)** | **70h** | **280h** |

---

## Key Invariants

### 1. Data Isolation

No train/steer/val examples appear in test set.

```bash
python verify_isolation.py  # checks before each run
```

### 2. Test Set Access

Only `evaluate_final.py` may call `utils.data.load_test_set()`.

Guard implemented in [utils/data.py](utils/data.py#L6).

### 3. Deterministic Results

Set `seed=42` in [configs/protocol.yaml](configs/protocol.yaml) and seed all RNGs in scripts.

Ensures reproducible splits and comparable results across runs.

---

## Reporting Tables (Phase 4)

After `python evaluate_final.py`:

1. **Latent Metrics** (sanity check — run first)
   - Trajectory coherence & truth alignment across conditions
   - Confirms steering equation is mechanically correct

2. **Accuracy** (main result)
   - Per condition × per model

3. **Efficiency** (token count + latency)
   - Reasoning token mean/std, actual ratio, latency p50/p95

4. **Primary Flip Matrix** (CCoT → CCoT + cPCA)
   - Net gain, improvement rate, degradation rate

5. **Mechanism Gain** (CCoT vs Trimmed CoT)
   - Validates TokenSkip compression

6. **Direction Specificity** (Steered vs Random Noise)
   - Confirms gain is specific to truth direction, not random perturbation

7. **Full Net-Gain Grid** (8×8 pairwise)
   - Complete pairwise comparison across all conditions

---

## Troubleshooting

### Out of Memory

Set `batch_size` and `grad_accum` smaller in [configs/protocol.yaml](configs/protocol.yaml).

### Slow Model Loading

Models are loaded and frozen for each phase. Consider caching across phases (advanced).

### Missing Checkpoints

After Phase 1 training, verify:

```bash
find checkpoints/ -name "adapter_config.json"
```

Each should exist for every config × model × stage combination.

### Missing Vectors

After Phase 2, verify:

```bash
find vectors/ -name "*.pt"  # should have dom.pt and cpca_*.pt per source
```

---

## Extending the Framework

The codebase is modular:

- **New evaluation condition?** Add to `evaluate_final.py` conditions list
- **New steering method?** Implement in `phase3/hooks.py` and wire into Phase 3 loop
- **Different base models?** Update `MODEL_ID_MAP` in [scripts/run_sweep.py](scripts/run_sweep.py)
- **Different dataset?** Replace data loaders in [utils/data.py](utils/data.py)

---

## References

- Phase 1 training: [phase1/train.py](phase1/train.py)
- Phase 2 extraction: [phase2/run.py](phase2/run.py)
- Phase 3 tuning: [phase3/alpha.py](phase3/alpha.py), [phase3/evaluate.py](phase3/evaluate.py)
- Phase 4 evaluation: [evaluate_final.py](evaluate_final.py)
- Full orchestration: [scripts/run_sweep.py](scripts/run_sweep.py)
