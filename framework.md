# CCOT-Steering: Framework Documentation

## What This Project Does

CCOT-Steering investigates whether **inference-time activation steering** can recover the reasoning quality lost when a language model is fine-tuned on *compressed* chain-of-thought (CCoT). The core hypothesis: the model learns what correct reasoning looks like even on short chains, and the "truth direction" is latent in its hidden states — we can nudge it back at generation time without retraining.

The pipeline runs in 5 phases: train baselines → extract truth vectors → tune steering intensity → evaluate on held-out validation → lock and test.

---

## Project Structure

```
CCOT-Steering/
├── pipeline.py                  # Master entry point (Phases 1–5)
├── evaluate_final.py            # Phase 4: only script that opens test.jsonl
├── preprocess_compress.py       # Offline compression cache builder
├── download_dataset.py
├── verify_isolation.py          # Data isolation integrity check
│
├── configs/
│   ├── protocol.yaml            # Master hyperparameters
│   ├── selected.yaml            # Written after Phase 3; locked for Phase 4
│   └── active_dataset.txt       # Persisted dataset choice
│
├── phase1/                      # Training baselines
│   ├── train.py                 # CoT & CCoT LoRA fine-tuning
│   ├── evaluate.py              # 12-condition evaluation on D_val
│   ├── compress.py              # Compression cache builder
│   ├── format.py                # Prompt formatting utilities
│   └── inference.py             # Model loading & greedy generation
│
├── phase2/                      # Truth vector extraction
│   ├── run.py                   # Full pipeline: collect → probe → DoM → cPCA
│   ├── collect.py               # Hidden state collection (H+, H−)
│   ├── probe.py                 # Layer scoring via logistic regression
│   ├── dom.py                   # Difference-of-Means vectors
│   ├── cpca.py                  # Contrastive PCA subspaces
│   ├── balance.py               # Stratified H+/H− balancing
│   ├── loaders.py               # Model/tokenizer loading utilities
│   ├── config.py                # Per-backbone Phase 2 hyperparameters
│   └── compare.py               # DoM vs cPCA comparison
│
├── phase3/                      # Inference-time steering
│   ├── evaluate.py              # 52-condition grid on D_val
│   ├── alpha.py                 # LearnableAlpha tuning
│   ├── hooks.py                 # Forward hooks (DoM / cPCA / noise)
│   ├── select.py                # Best config selection (Wilson CI)
│   ├── lambda_sweep.py          # λ_a × λ_m grid search
│   └── plots.py                 # Loss curves, heatmaps
│
├── scripts/
│   ├── build_splits.py          # Data split builder (S1–S4)
│   ├── selection.py             # Config winner selection
│   ├── run_sweep.py             # Alternative pipeline orchestrator
│   └── print_transfer_summary.py
│
├── utils/
│   ├── dataset_paths.py         # Dataset selection logic
│   └── data.py                  # Test set loader (guarded access)
│
├── checkpoints/{config}/{model}/
│   ├── cot/                     # Phase 1 Stage 1 LoRA adapter
│   └── ccot_R{5–9}/             # Phase 1 Stage 2 LoRA adapters
│
├── vectors/{config}/{model}/
│   ├── ccot_dom.pt              # DoM truth vector [d]
│   ├── ccot_cpca_r*.pt          # cPCA subspace [d, r]
│   ├── ccot_alpha_star.pt       # Learned steering intensity (Phase 3)
│   ├── shuffled_{dom,cpca}.pt   # Shuffled-label control vectors
│   └── phase2_meta.json         # Layer metadata & probe scores
│
├── results/{config}/{model}/
│   ├── phase1_val.json          # Phase 1 condition results
│   ├── phase3_val.json          # Phase 3 condition results
│   ├── phase3_val_interim.json  # Per-condition saves (resume support)
│   └── phase3_best_config.yaml  # Selected winning condition
│
├── results/final/
│   ├── summary_test.json        # Cross-model aggregated results
│   └── {model}_test.json        # Per-model D_test results
│
├── cache/{config}/
│   └── compressed_R{5–9}.jsonl  # Pre-computed TokenSkip compressions
│
└── {gsm8k,svamp,prontoqa}/
    ├── train.jsonl              # Split pool (D_train + D_steer + D_val)
    └── test.jsonl               # Locked test set (Phase 4 only)
```

---

## Data Splits

The training pool (`train.jsonl`) is divided into three **disjoint** subsets:

| Split | Symbol | Purpose | S2 ratio |
|-------|--------|---------|---------|
| Training set | D_train | Phase 1 fine-tuning | 60% |
| Steering set | D_steer | Phase 2 hidden state collection | 10% |
| Validation set | D_val | Phase 3 α-tuning + condition selection | 30% |
| Test set | D_test | Phase 4 final evaluation (locked) | separate file |

**Data isolation invariant**: D_test is never opened outside `evaluate_final.py`. Enforced by `utils/data.py` which raises `RuntimeError` if called from any other script. `verify_isolation.py` checks all disjoint constraints at setup time.

The sweep configs S1–S4 vary the split ratios to find the most stable data split; the winner is recorded in `configs/selected.yaml` after Phase 3.

---

## Models

| Tag | Model ID |
|-----|---------|
| `llama32_3b` | meta-llama/Llama-3.2-3B |
| `phi2` | microsoft/phi-2 |
| `qwen25_3b` | Qwen/Qwen2.5-3B |
| `qwen25_math1.5b` | Qwen/Qwen2.5-Math-1.5B |

All models are fine-tuned with LoRA (r=16, α=32, dropout=0.05). Per-backbone hyperparameters (LR, epochs, batch size) are in `phase2/config.py` and `phase1/train.py`.

---

## Phase 0: Data Setup

**Run once before any training.**

```bash
python download_dataset.py --dataset gsm8k
python verify_isolation.py
python preprocess_compress.py        # builds cache/ — takes ~1h on GPU
```

`preprocess_compress.py` runs LLMLingua-2 offline over D_train at each compression ratio (0.5, 0.6, 0.7, 0.8, 0.9) and saves the results as `cache/S2/compressed_R{5–9}.jsonl`. This cache is used in Phase 1 Stage 2 so compression does not happen at training time.

**Dataset format**: each row in `train.jsonl` / `test.jsonl`:
```json
{"id": "gsm8k_0042", "question": "...", "answer": "reasoning chain\n####numeric_answer"}
```

---

## Phase 1: Training Baselines

**Goal**: produce two kinds of fine-tuned adapters for each model — one trained on full reasoning (CoT), five trained on compressed reasoning (CCoT at ratios 0.5–0.9). Then evaluate all conditions on D_val to understand the accuracy–compression tradeoff before any steering.

### Stage 1: CoT Fine-Tuning (`phase1/train.py::train_cot`)

- Load frozen base model + apply LoRA
- Dataset: `CoTDataset` — format `"Question\n\n{reasoning}\n\nAnswer: {answer}"`
- Objective: next-token prediction over full reasoning chain
- Saves: `checkpoints/{config}/{model}/cot/adapter_config.json`

### Stage 2: CCoT Fine-Tuning (`phase1/train.py::train_ccot`)

For each ratio r ∈ {0.5, 0.6, 0.7, 0.8, 0.9}:
- Load compressed reasoning from `cache/{config}/compressed_R{5–9}.jsonl`
- Dataset: `CCoTDataset` — format `"Question\n\n[compress:R]\n{compressed_chain}\n\nAnswer: {answer}"`
- Train separate LoRA adapter; always 3 epochs (hardness override)
- Saves: `checkpoints/{config}/{model}/ccot_R{5–9}/adapter_config.json`

### Phase 1 Evaluation (`phase1/evaluate.py`)

Runs 12 conditions on D_val:

| # | Condition | Description |
|---|-----------|-------------|
| 1 | `no_cot` | Base model, direct answer (no reasoning) |
| 2 | `full_cot` | CoT checkpoint, full reasoning chain |
| 3–7 | `ccot_R5`–`ccot_R9` | CCoT checkpoints at each ratio |
| 8–12 | `trimmed_R5`–`trimmed_R9` | Full CoT chain token-capped to match CCoT budget |

The trimmed condition isolates the effect of *compression quality* vs *token count*: if CCoT beats trimmed, the model actually learned from compressed reasoning, not just from seeing fewer tokens.

**Metrics**: accuracy, mean reasoning tokens, actual compression ratio (tokens/full_cot_tokens), latency (s/example), answer extraction rate.

**Output**: `results/{config}/{model}/phase1_val.json`

---

## Phase 2: Truth Vector Extraction

**Goal**: collect hidden states from correct vs incorrect rollouts of the fine-tuned models, then extract the "truth direction" — the linear subspace that distinguishes correct from incorrect reasoning.

### Step 1: Collect Hidden States (`phase2/collect.py`)

For each question in D_steer:
- Run the frozen model N=10 times (temperature=1.0) → stochastic rollouts
- Classify each rollout as correct (H+) or incorrect (H−) by answer comparison
- Register forward hooks on all transformer layers
- At each layer, capture the hidden state at the **reasoning–answer boundary**:
  - Boundary detection looks for `</think>`, `\n\nAnswer:`, or model-specific separators
  - `h[:, boundary_idx, :]` — the representation at that token position

**Stratified balancing** (`phase2/balance.py`):
- Group questions by difficulty quartile (fraction of correct rollouts)
- Undersample the larger class within each bucket
- Ensures H+ and H− counts are approximately equal across difficulty levels
- Prevents easy questions from dominating the direction

**Output**: `vectors/{config}/{model}/{source}_hstates_cache.pt`

### Step 2: Layer Scoring (`phase2/probe.py`)

For each layer L:
- Fit a logistic regression probe on 80% of (H+_L, H−_L)
- Evaluate on held-out 20% → `probe_acc[L]`
- Gate check: at least one layer must exceed 55% accuracy
  - If no layer passes, raises `RuntimeError` — indicates insufficient signal in hidden states

**Output**: `dict[layer → accuracy]` — used to weight vectors in later steps.

### Step 3: DoM — Difference of Means (`phase2/dom.py`)

For each layer L:
```
v_L = (mean(H+_L) − mean(H−_L)) / ||mean(H+_L) − mean(H−_L)||
```

Best layer selection: pick L* = argmax(probe_acc[L]), return v_truth = v_L*.

Simple but effective. The unit-normalized direction points from "incorrect reasoning" toward "correct reasoning" in activation space.

**Output**: `vectors/{config}/{model}/ccot_dom.pt` — shape [d]

### Step 4: cPCA — Contrastive PCA (`phase2/cpca.py`)

Finds a subspace where H+ has **high variance** while H− has **low variance**:

```
C_contrast = C_pos − β · C_neg       (β = 0.5 by default)
```

Eigenvectors of C_contrast with largest eigenvalues span the subspace.

Three computational variants (model-dependent):
- **full**: full eigendecomposition (Llama, Qwen2.5-3B, Phi-2)
- **shrunk**: Ledoit-Wolf covariance shrinkage (Qwen2.5-Math-1.5B, small n)
- **randomized**: randomized SVD (speed option for large d)

Layer selection (`cpca.py::select_layers`):
- Keep only layers where `probe_acc[L] ≥ mean(probe_acc) + 0.5 · std(probe_acc)`
- Contiguity filter: keep layers within ±3 of the median selected layer
- Prevents scattering across uninformative layers

Weighted merge (`cpca.py::weighted_subspace_merge`):
- Combine r=3 eigenvectors per selected layer, weighted by `probe_acc × eigenvalue × directional_agreement`
- Final subspace U_truth has shape [d, r_final] where r_final=10 by default

**Output**: `vectors/{config}/{model}/ccot_cpca_r10.pt` — shape [d, 10]

### Step 5: Shuffled-Label Control

Same pipeline repeated with H+ and H− labels randomly shuffled. The resulting vectors should have near-zero cosine similarity to the real v_truth. Used in Phase 3 as a specificity baseline.

**Output**: `vectors/{config}/{model}/ccot_shuffled_{dom,cpca}.pt`

### Phase 2 Outputs Summary

```
vectors/{config}/{model}/
├── ccot_dom.pt              # [d] DoM truth vector
├── ccot_cpca_r10.pt         # [d, 10] cPCA subspace
├── ccot_shuffled_dom.pt     # [d] shuffled control (DoM)
├── ccot_shuffled_cpca.pt    # [d, 10] shuffled control (cPCA)
└── phase2_meta.json         # probe scores, selected layers, best layer
```

---

## Phase 3: Inference-Time Steering

**Goal**: learn the optimal steering intensity α for each (method, ratio) combination on D_val, run all conditions, and select the single best steered config for Phase 4.

### How Steering Works (`phase3/hooks.py`)

At generation time, a forward hook intercepts the hidden state at layer L* and the boundary token position, then adds a direction-aligned perturbation:

**DoM hook**:
```
h' = h + α · σ_h · v̂
```
where v̂ = v_truth / ||v_truth||, σ_h = ||h|| / √d (layer-norm normalization)

**cPCA hook**:
```
h' = h + α · σ_h · U U^T ĥ
```
where ĥ = h / ||h||, U is the cPCA subspace — projects normalized h onto the truth subspace and adds it back

**Noise hook** (control):
```
h' = h + α · σ_h · ε     (ε ~ random unit vector, fresh per call)
```
Tests whether any perturbation helps, not a specific direction.

**Dtype handling**: U is stored as float32; hidden states may be bfloat16. `U_ = U.to(h_t.dtype)` is applied before the matmul to prevent `RuntimeError: float != c10::BFloat16`.

### α-Tuning (`phase3/alpha.py`)

`LearnableAlpha` module with sigmoid reparameterization:
```
α = α_max · sigmoid(θ)      (α_max = 50, so α ∈ (0, 50))
```
θ is initialized so α_0 ≈ 1.0.

**Loss function** (3 terms):
```
L = L_ans + λ_a · L_align + λ_m · L_mag
```

| Term | Formula | Purpose |
|------|---------|---------|
| L_ans | NLL of gold answer tokens (teacher-forced) | directly optimize answer correctness |
| L_align | 1 − cos(h_steered, v_truth) | penalize diverging from truth direction |
| L_mag | (‖δ‖ / ‖h_orig‖)² | prevent RMSNorm norm collapse from large perturbations |

λ_m is model-dependent: 0.005 for Phi-2 (LayerNorm), 0.01 for Llama/Qwen (RMSNorm).

**Optimization**: AdamW, lr=5e-2, max_epochs=5, early stopping patience=3. Data split: 90% tune / 10% early-stop.

Learned α* is saved as `vectors/{config}/{model}/{source}_alpha_star.pt`.

### 52-Condition Evaluation Grid

Phase 3 evaluates every combination:

| Condition | Description |
|-----------|-------------|
| `no_cot` | Base model, direct answer |
| `full_cot` | CoT checkpoint, full chain |
| `ccot_R*` (×5) | CCoT checkpoint at each ratio |
| `trimmed_R*` (×5) | CoT chain token-capped to match CCoT budget |
| `noise_ccot_R*` (×5) | CCoT + random perturbation (control) |
| `noise_trimmed_R*` (×5) | Trimmed + random perturbation |
| `dom_ccot_R*` (×5) | CCoT + DoM steering |
| `dom_trimmed_R*` (×5) | Trimmed + DoM steering |
| `cpca_ccot_R*` (×5) | CCoT + cPCA steering |
| `cpca_trimmed_R*` (×5) | Trimmed + cPCA steering |

**Per-condition save & resume**: after each condition completes, results are flushed to `phase3_val_interim.json`. If the process crashes and is restarted, all completed conditions are loaded back and skipped. Budget computation (~10h) is also cached to `phase3_budgets_R6.json`.

**Flip rate**:
```
flip_rate = (# wrong→right vs CCoT baseline) / total examples
```
Measures how many examples were *fixed* by steering — directional improvement, penalizing regressions.

### Best Config Selection (`phase3/select.py`)

For each model, pick the condition that maximizes accuracy (tiebreaker: flip_rate). Confidence interval via **Wilson score lower bound** (95%) to account for finite-sample uncertainty — favors conditions that are consistently good, not just lucky on a small subset.

**Output**: `results/{config}/{model}/phase3_best_config.yaml`, `vectors/{config}/{model}/ccot_alpha_star.pt`

### Split Config Winner (`scripts/selection.py`)

Across S1–S4 split configs, compute mean Wilson CI lower bound per config averaged over all models. The config with highest mean lower bound wins and is recorded in `configs/selected.yaml` — locked for Phase 4.

---

## Phase 4: Final Evaluation on D_test (`evaluate_final.py`)

**Goal**: evaluate all conditions on the locked test set using the config and α* selected in Phase 3. This is the only phase that opens `test.jsonl`.

### Conditions

| # | Condition | Notes |
|---|-----------|-------|
| 1 | `no_cot` | Base model, direct answer |
| 2 | `full_cot` | CoT checkpoint, full reasoning |
| 3 | `ccot_R*` | Best ratio from Phase 3 selection |
| 4 | `trimmed_cot` | CoT capped to match CCoT token budget |
| 5 | `noise_ccot` | CCoT + random direction (control) |
| 6 | `dom_ccot` | CCoT + DoM steering at α* |
| 7 | `cpca_ccot` | CCoT + cPCA steering at α* |
| 8 | `trimmed_dom` | Trimmed CoT + DoM steering |

### Alpha Sweep

Phase 4 also runs a sweep of α ∈ [0, 2·α*] separately for DoM and cPCA, producing:
- `results/final/alpha_sweep_dom.png`
- `results/final/alpha_sweep_cpca.png`

This reveals the true optimal α on unseen test data and validates the Phase 3 tuning.

### Metrics

**Per-condition**:
- Accuracy (normalized answer match)
- Mean ± std reasoning tokens
- Actual compression ratio (vs full_cot tokens)
- Latency: mean, p50, p95 (seconds/example)
- Answer extraction rate

**Latent metrics** (steered conditions only):
- Trajectory coherence: mean cos(h_t, h_{t+1}) at injection layer
- Truth alignment: mean cos(h_t, v_truth)

**Flip matrices** (for each condition pair A→B):
- F00: both wrong, F10: A wrong → B right (improvement), F01: A right → B wrong (degradation), F11: both right
- Net gain = improvement rate − degradation rate

**Bootstrap CIs**: 1,000 resamples, 95% percentile confidence intervals on all accuracy figures.

### Output Tables

1. Per-condition accuracy + compression efficiency
2. Latent metric sanity check (trajectory coherence, truth alignment)
3. Efficiency metrics (tokens, latency percentiles)
4. Primary flip matrix: CCoT → CCoT+cPCA
5. Mechanism gain: CCoT vs Trimmed CoT (isolates learning effect from token-count effect)
6. Direction specificity: steered vs noise (confirms gain is direction-specific)
7. Full 8×8 pairwise net-gain grid

**Output files**: `results/final/summary_test.json`, `results/final/{model}_test.json`

---

## Phase 5: Transfer Evaluation (Optional)

**Goal**: test whether steering learned on GSM8K generalizes to SVAMP (a harder, differently-distributed math benchmark) without any retraining.

Uses frozen vectors, checkpoints, and α* from Phase 4 (no retuning). Results go to `results/final_svamp_transfer/`.

---

## Running the Pipeline

```bash
# Full pipeline (all phases, all models)
python pipeline.py --phase 0 --dataset gsm8k

# Individual phases
python pipeline.py --phase 1                           # Train
python pipeline.py --phase 2                           # Extract vectors
python pipeline.py --phase 3                           # Steer + tune
python pipeline.py --phase 4                           # Test eval
python pipeline.py --phase 5                           # SVAMP transfer

# Specific model/config
python pipeline.py --phase 1 --model qwen25_math1.5b --config S2
python pipeline.py --phase 2 --model qwen25_math1.5b

# Checkpoint resume: re-running any phase skips completed steps automatically
```

**Checkpoint logic**: every step checks whether its output file exists and skips if found. Safe to re-run after any interruption.

---

## Key Algorithmic Concepts

### Why CCoT + Steering?

CCoT fine-tuning makes the model generate shorter reasoning chains, which reduces inference cost. But compression degrades accuracy because the model has less space to work through the problem. The hypothesis: the model's *internal representation* of correct reasoning is mostly intact even on compressed chains — we just need to nudge the hidden states toward the truth direction at the right layer.

### Boundary Injection

Steering is applied at the **reasoning–answer boundary** (the token position where the model transitions from chain-of-thought to the final answer). This is the highest-information moment in the forward pass for answer accuracy. Injecting earlier would corrupt the reasoning structure; injecting later is too late to influence the answer.

### Layer Selection

Not all layers carry useful truth information. Layers are scored by logistic probe accuracy (how well linear separation works between H+ and H−). Only layers that exceed the probe gate (55%) are used. For cPCA, an additional contiguity constraint ensures the selected layers are clustered, not scattered.

### Contrastive PCA vs DoM

DoM is a single direction. cPCA is a subspace (10 dimensions). cPCA can capture multi-dimensional structure in reasoning correctness that a single vector misses, but it requires more data to estimate reliably and is more sensitive to the covariance structure. Phase 4 evaluates both and reports which is better for each model.

### LearnableAlpha Reparameterization

Directly optimizing α can lead to pathological values (α → ∞ or α → 0). The sigmoid reparameterization `α = α_max · sigmoid(θ)` naturally constrains α to (0, α_max) without any gradient clipping, and the 3-term loss prevents the optimizer from "solving" the alignment term by making the perturbation arbitrarily large.

---

## Compute Estimates

| Phase | Per model | 4 models |
|-------|-----------|---------|
| Phase 1: CoT (Stage 1) | ~2h | ~8h |
| Phase 1: CCoT (Stage 2, 5 ratios) | ~8h | ~32h |
| Phase 1: Evaluation | ~1h | ~4h |
| Phase 2: Vector extraction | ~3h | ~12h |
| Phase 3: Steering grid (D_val=300) | ~1.5h | ~6h |
| Phase 4: Final test eval | ~1h | ~4h |
| **Total (single split config)** | ~17h | ~66h |

All estimates on a single A100 80GB GPU.

---

## Dependencies

| Library | Used For |
|---------|---------|
| `transformers` | Model loading, LoRA, generation |
| `peft` | LoRA adapter training |
| `torch` | All tensor operations |
| `llmlingua` | CCoT compression cache (Phase 0) |
| `sklearn` | Logistic probe, StandardScaler |
| `numpy` | Eigendecomposition (cPCA) |
| `yaml` | Config reading/writing |
| `scipy` | Wilson CI, bootstrap |
| `matplotlib` | Phase 3/4 plots |
