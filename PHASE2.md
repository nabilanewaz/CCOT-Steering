# Phase 2: Truth Vector Extraction

> **Full-experiment implementation (2026-09-16):** See [FULL_EXPERIMENT.md](FULL_EXPERIMENT.md) for the current executable contract. S3 now means 60/10/30 (4,484/747/2,242 on GSM8K); all validation conditions use 2,242 examples. Phase 1 remains Coconut with `ccot_L3/L4/L6` checkpoints and `phase1_best_latent.json`, so the historical `R6`/compression-ratio examples below map to the selected `L` budget. The skip set is empty. Steering uses generated-token positions for both cached and uncached decoding, with no oracle reasoning prefix. Alpha tuning/diagnostic subsets remain as specified; test-set alpha tuning is disabled. Historical results and audit notes below are not results of the new full run.

Phase 2 extracts the "truth direction" from the frozen model's hidden states — the linear subspace that separates correct from incorrect reasoning activations. This direction is what Phase 3 injects at inference time to steer the model toward better answers.

---

## Files

| File | Role |
|------|------|
| `run.py` | Full pipeline orchestrator (9 steps) |
| `collect.py` | Hidden state collection with stochastic rollouts |
| `probe.py` | Layer scoring via logistic regression |
| `dom.py` | Difference-of-Means vector computation |
| `cpca.py` | Contrastive PCA subspace computation |
| `balance.py` | Stratified H+/H− sample balancing |
| `loaders.py` | Model/tokenizer loading, boundary detection |
| `config.py` | Per-backbone Phase 2 hyperparameters |
| `compare.py` | DoM vs cPCA comparison and best-source selection |
| `collect_heads.py` | Phase 2.5: per-attention-head activation collection (ITI) |
| `probe_heads.py` | Phase 2.5: per-head probing + top-K selection (ITI) |

---

## Prerequisites

- Phase 1 complete: `checkpoints/{config}/{model}/ccot_R{best}/adapter_config.json` exists
- `phase1_val.json` exists (to identify the best CCoT ratio)
- D_steer split available (10% of training pool)

---

## Overview: 9 Steps

`run_phase2_source` runs 9 numbered steps for each (model, source) pair:

```
STEP 1 — Collect hidden states (H+, H−)
STEP 2 — Logistic probe: score all layers
STEP 3 — Method A: Difference-of-Means (DoM)
STEP 4 — Control: shuffled-label DoM
STEP 5 — cPCA layer selection
STEP 6 — Method B: cPCA sweep
STEP 7 — Weighted subspace merge
STEP 8 — Method comparison (DoM vs cPCA)
STEP 9 — Control: shuffled-label cPCA
```

---

## Sources

Phase 2 runs on two sources (the model whose hidden states are collected):

| Source tag | Checkpoint | Prompt format |
|------------|-----------|---------------|
| `ccot` | `checkpoints/{config}/{model}/ccot_R{best}/` | `Question: ...\n\n[compress:{ratio}]\n` |
| `base` | `checkpoints/{config}/{model}/cot/` | `Question: ...\n\nReasoning:` (disabled by default) |

The best CCoT ratio is chosen by `pick_best_ccot_ratio`: reads `phase1_val.json`, returns the ratio with highest `ccot_acc − trimmed_acc` (mechanism gain). Defaults to R=6 (ratio 0.6) if the file is missing.

Source B (CoT) is disabled by default (`run_source_b=False`) — enable by passing `run_source_b=True` to `run_phase2_all_sources`.

---

## Step 1: Hidden State Collection (`collect.py`)

### What happens

For each question in D_steer:
1. Run the frozen model N=10 times with temperature=1.0 → stochastic rollouts
2. Classify each rollout: compare extracted answer to ground truth → H+ (correct) or H− (incorrect)
3. A forward hook captures a hidden state at every transformer layer simultaneously, at the position(s) determined by the active extraction mode (below)
4. When the mode needs it, the boundary index is detected from the generated sequence:
   - `find_boundary_idx_ccot`: looks for `</think>`, `####`, or `\n\nAnswer:`
   - `find_boundary_idx_base`: looks for similar markers in the CoT format

### Extraction modes (`collect_hidden_states(..., extraction=..., gen_window=...)`)

`collect.py` supports three extraction modes, selected by the `extraction` argument (**`'mean_gen'` is the current default**, passed from `run_phase2_source`):

| Mode | What is captured | Notes |
|------|-------------------|-------|
| `'mean_gen'` **(default)** | Mean hidden state over the first `gen_window` (default 20) generated tokens: `h[:, prompt_len:prompt_len+gen_window, :].mean(dim=1)` | Targets the model's *active reasoning state* rather than a single boundary encoding. Rollouts shorter than 3 generated tokens are skipped. |
| `'first_gen'` | Hidden state at the single first generated token | Cheapest option; rarely used. |
| `'boundary'` | Hidden state at `boundary_idx_fn(out_ids, tokenizer)` — the original single-token boundary position | The original extraction mode (see "Why the boundary token?" below); rollouts where the boundary can't be located are skipped. |

Internally, `_register_all_hooks` supports two hook keys — `boundary_idx` (single position) and `boundary_range` (a `[start, end)` window, averaged) — so `mean_gen` and `boundary`/`first_gen` share the same hook machinery; only which key is set per rollout differs. Hooks are no-ops during `model.generate()` itself — the extraction position is only captured on the follow-up teacher-forced forward pass (`model(out_ids)`) run once per rollout.

### Why the boundary token? (original `'boundary'` mode rationale)

The boundary is the last token of the reasoning chain before the model commits to an answer. It concentrates the maximum reasoning-relevant signal: all prior reasoning has been processed, but the answer hasn't been generated yet. Injecting later is too late; injecting earlier corrupts the reasoning. **In practice this mode produced a null steering result end-to-end** (see Phase 3/4 §"Extraction and injection alignment") — pooling over the first `gen_window` generated tokens (`'mean_gen'`) is what the current default extracts instead, because injecting a vector characterized at a single prefill-time boundary position did not transfer to influencing token-by-token generation.

### Stratified balancing (`balance.py`)

Without balancing, easy questions (high fraction of correct rollouts) dominate H+ and hard questions dominate H−. This biases the direction toward question-difficulty features rather than reasoning-quality features.

Fix: group questions into difficulty quartiles by `frac_correct`, then undersample the larger class within each bucket to produce balanced H+ and H− per layer.

### Minimum samples gate

Only layers with ≥ `min_samples` hidden states (default 200) in both H+ and H− are kept. Layers with insufficient data are dropped entirely.

### Caching

Hidden states are cached to `{vectors_dir}/{source}_hstates_cache.pt` after the first collection run. Subsequent runs load the cache directly (collection is the most expensive step, ~1–2h).

---

## Step 2: Layer Scoring (`probe.py`)

### What happens

`run.py` calls `score_all_layers_both`, which runs **two** probe families per layer and keeps the better of the two:

1. **Logistic regression** (`score_all_layers`): for each layer L with sufficient samples, concatenate H+_L and H−_L into X (L2-row-normalized), labels y = [1, ..., 0, ...]; stratified 80/20 split; `StandardScaler` fit on train; `LogisticRegression(max_iter=1000, C=1.0)` fit on train; accuracy on held-out 20%.
2. **MLP** (`score_all_layers_nn`): same preprocessing, then a grid search over `hidden_layer_sizes ∈ {(64,), (128,64)}` × `learning_rate_init ∈ {1e-3, 5e-3, 1e-2}` × `alpha ∈ {1e-4, 1e-3}` (`MLPClassifier`, early stopping on a 15% validation split, patience 20).

`layer_scores[L] = max(LR_acc, MLP_acc)` for each layer, and the diagnostics record `probe_method: 'max(LR, MLP)'` plus which of the two won per layer. The MLP pass roughly doubles Step 2's wall-clock time but can surface non-linear separability the logistic probe misses.

### Gate check

If no layer exceeds `PROBE_GATE = 0.55` (55% accuracy), a `RuntimeError` is raised:
```
Probe gate failed: no layer exceeded 55%. Best was layer 13 at 0.541.
```
This indicates the hidden states carry insufficient linear signal — the model may not have learned meaningful reasoning representations, or D_steer is too small.

### Output

`dict[int → float]` — layer → held-out probe accuracy. Used to weight all downstream vector computations.

---

## Step 3: Method A — Difference of Means (`dom.py`)

### Formula

For each layer L:
```
v_L = (mean(H+_L) − mean(H−_L)) / ‖mean(H+_L) − mean(H−_L)‖
```

Unit-normalized direction pointing from "incorrect reasoning" toward "correct reasoning" in the layer-L activation space.

### Best layer selection

```
L* = argmax(layer_scores[L])
v_truth = v_{L*}
```

The layer with the highest probe accuracy is most informative about reasoning correctness.

### Output

Saved to `{vectors_dir}/{source}_dom.pt`:
```python
{'v_truth': tensor([d]), 'best_layer': int, 'model_tag': str, 'source': str}
```

### Multi-layer DoM (`save_multilayer_dom_vectors`, saved alongside best-layer DoM)

In addition to the single best-layer vector above, `run.py` always also saves the **top-3** layers by probe accuracy as a separate multi-layer payload — this is what Phase 3's `multilayer_dom_*` condition injects simultaneously (one DoM vector per layer, not one vector copied to three layers):

```python
# {vectors_dir}/{source}_multilayer_dom.pt
{
    'top_layers':    [L1, L2, L3],                 # top-3 by probe accuracy, descending
    'layer_vectors':  {L1: tensor([d]), L2: ..., L3: ...},
    'layer_scores':   {L1: float, L2: float, L3: float},
    'top_k': 3, 'method': 'multilayer_dom', 'model_tag': str, 'source': str,
}
```

This file is saved whenever Step 3 runs — including the early-exit path when cPCA finds no usable subspace (Step 6 below) — so multi-layer DoM does not depend on cPCA succeeding.

---

## Step 4: Control — Shuffled-Label DoM

Same computation as Step 3, but H+ and H− labels are randomly shuffled before taking the mean difference.

Expected: near-zero cosine similarity to the real v_truth.

If the shuffled vector has high cosine similarity to v_truth, the direction is capturing something structural (e.g. question difficulty, token position) rather than reasoning correctness. Used in Phase 3 to confirm direction specificity.

Output: `{vectors_dir}/{source}_shuffled_dom.pt`

---

## Per-Backbone Phase 2 Configuration (`config.py`)

All of Steps 1–7's tunable parameters are set per backbone by `get_model_config(model_tag)` — there is no single global default used for every model:

| Model | N (rollouts) | cPCA variant | `r_per_layer` | `r_final` | β | `threshold_multiplier` | `min_samples` |
|-------|:---:|---|:---:|:---:|:---:|:---:|:---:|
| qwen25_3b | 10 | full | 3 | 10 | 0.5 | 0.5 | 200 |
| **qwen25\_math1.5b** | **10** | **shrunk** | **3** | **8** | **0.5** | **0.4** | **200** |
| qwen25_0.5b$^{\dagger}$ | 10 | full | 3 | 10 | 0.5 | 0.5 | 200 |
| *(fallback default)* | 10 | full | 3 | 10 | 0.5 | 0.5 | 200 |

`qwen25_math1.5b` is the only backbone with a completed end-to-end run (see `phase1/PHASE1.md`), and it deliberately deviates from the fallback defaults on two parameters: `r_final=8` (not 10) and `threshold_multiplier=0.4` (not 0.5). Anything citing "the default" cPCA rank or layer threshold for the reported results should use these values, not the fallback-default column. $^{\dagger}$`qwen25_0.5b` has no dedicated entry in `MODEL_PHASE2_CONFIG` and falls back to `_DEFAULT_CONFIG`, shown above — not a value chosen for a 0.5B model. Llama-3.2-3B and Phi-2 have been dropped from the backbone roster; Qwen2.5-0.5B was added to match Architecture B's (Coconut) registered set.

---

## Step 5: cPCA Layer Selection (`cpca.py::select_layers`)

Not all layers should be included in the cPCA computation. Selection criteria:

1. **Probe threshold**: `probe_acc[L] ≥ mean(all scores) + threshold_multiplier × std(all scores)`
   - `threshold_multiplier` is per-backbone (table above) — 0.4 for `qwen25_math1.5b`, 0.5 fallback default
   - Keeps only above-average informative layers

2. **Contiguity filter**: keep layers within ±3 of the median selected layer
   - Prevents scattered selection across early, middle, and late layers
   - Assumes the "truth signal" is concentrated in a contiguous band

---

## Step 6: Method B — cPCA Sweep (`cpca.py::run_cpca_sweep`)

### Objective

Find a subspace where H+ has **high variance** and H− has **low variance**:

```
C_contrast = C_pos − β · C_neg
```

Eigenvectors of C_contrast with the largest eigenvalues define the subspace.

### Sweep

For each selected layer, a grid search over k ∈ {3, 5, 8, 10} (subspace rank) and β ∈ {0.1, 0.5, 1.0, 2.0} selects the (k, β) pair with the highest classification accuracy on the held-out probe set.

### Three computational variants

Selected in `config.py` per backbone:

| Variant | When used | How |
|---------|-----------|-----|
| `full` | Qwen2.5-3B, Qwen2.5-0.5B (fallback default) | Full eigendecomposition of C_contrast |
| `shrunk` | Qwen2.5-Math-1.5B | Ledoit-Wolf covariance shrinkage before eigendecomposition (better for small n, large d) |
| `randomized` | Large d, speed needed | Randomized SVD via `sklearn.utils.extmath.randomized_svd` |

---

## Step 7: Weighted Subspace Merge (`cpca.py::weighted_subspace_merge`)

Combines per-layer subspaces into a single U_truth [d, r_final] where r_final is per-backbone (10 by default, 8 for `qwen25_math1.5b` — see the config table above).

**Per-layer weight**:
```
w_L = probe_acc[L] × mean_eigenvalue_L × directional_agreement_L
```
where `directional_agreement_L = max(0, cos(v_truth, U_L[:, 0]))` — alignment between the DoM direction and the leading eigenvector of layer L's subspace.

**Merge**: form a weighted block matrix of all per-layer eigenvectors, then take the top r_final left singular vectors via SVD.

This ensures that layers which are both informative (high probe accuracy) and coherent with the DoM direction contribute more to the final subspace.

Output: `{vectors_dir}/{source}_cpca_r{r_final}.pt` (e.g. `ccot_cpca_r8.pt` for `qwen25_math1.5b`, `ccot_cpca_r10.pt` for the other backbones):
```python
{
    'U_truth': tensor([d, r_final]),
    'selected_layers': [int, ...],
    'model_tag': str,
    'source': str,
    'r_final': int,   # per-backbone — see config table above
}
```

---

## Step 8: Method Comparison (`compare.py`)

Evaluates both DoM and cPCA on a held-out subset of D_steer:
- DoM: classify via cosine similarity to v_truth
- cPCA: classify via projection magnitude onto U_truth

The method with higher accuracy on this held-out set is the `winner` (stored in `phase2_meta.json`). Both vectors are saved regardless — Phase 3 evaluates both.

---

## Step 9: Control — Shuffled-Label cPCA

Recomputes cPCA with shuffled labels. Expected: U_shuffled has no useful subspace structure. Used in Phase 3 as direction-specificity control D.

Output: `{vectors_dir}/{source}_shuffled_cpca_r{r_final}.pt`

---

## Phase 2.5: ITI Head Extraction (optional, standalone)

Run **after** Phase 2 completes (needs `phase2_meta.json` for `best_ccot_ratio`). This is a separate, opt-in stage — `pipeline.py --phase 2` does **not** run it automatically:

```bash
python scripts/run_iti_phase25.py --model qwen25_math1.5b --config S2 --dataset gsm8k
```

Extracts per-attention-**head** directions (rather than per-layer directions) as an alternative steering target, following Inference-Time Intervention (Li et al. 2023). Implemented in `phase2/run.py::run_iti_phase2`, which calls two new modules:

### Step A — Collect per-head activations (`collect_heads.py::collect_head_activations`)

For each D_steer question, `N_rollouts` (default 10) rollouts are sampled at temperature 0.8, max 128 new tokens. For each rollout:
1. A forward **pre**-hook on every layer's `self_attn.o_proj` captures its *input* — the concatenated per-head attention output `[batch, seq_len, num_heads × head_dim]` — **before** it is mixed by the output projection.
2. The reasoning/answer boundary position `b_idx` is located the same way as Step 1's `'boundary'` mode (`find_boundary_idx_ccot`/`_base`).
3. A teacher-forced forward pass on the full generated sequence re-triggers the hooks so activations at `b_idx` can be read back.
4. The boundary-position activation is reshaped to `[num_heads, head_dim]` and each head's slice is appended to `H_pos[(layer, head)]` or `H_neg[(layer, head)]` depending on rollout correctness.

Only `(layer, head)` pairs with at least `min_samples` (default 30) examples in **both** classes are kept. Cached to `{vectors_dir}/{source}_head_hstates_cache.pt`.

### Step B — Per-head probing and top-K selection (`probe_heads.py::probe_all_heads`)

For every valid `(layer, head)` pair: balance classes, L2-normalize, fit a `LogisticRegression` (80/20 split) to get `head_scores[(L,h)]`; compute a DoM-style unit direction `head_directions[(L,h)]` in that head's `head_dim`-dimensional subspace; compute `head_sigmas[(L,h)]` as the standard deviation of all (H+ ∪ H−) activations projected onto that direction (Li et al. 2023, Eq. 5 — this is the per-head analogue of Phase 3's `σ_h`). The `top_k` (default 48) pairs by probe accuracy are kept.

**On the reported `qwen25_math1.5b` run, the single best attention head reached only 53.8% held-out probe accuracy** — below the 55% gate that whole-layer probes must clear in Step 2 — indicating no individual head cleanly separates correct from incorrect compressed reasoning (see Phase 3/4 discussion of ITI's negative result).

### Output

`{vectors_dir}/{source}_iti_heads.pt`:
```python
{
    'model_tag': str, 'source_tag': str, 'top_k': int,
    'top_heads':       [(layer, head), ...],       # sorted by probe acc, descending
    'head_scores':     {(layer, head): float, ...},
    'head_directions': {(layer, head): tensor([head_dim]), ...},
    'head_sigmas':     {(layer, head): float, ...},
}
```

Phase 3's ITI condition (`iti_{rtag}_{source}`) loads this file directly; see `phase3/PHASE3.md`.

---

## Diagnostic Output

`{source}_diagnostics.json` — comprehensive per-step statistics:

```json
{
  "probe":           {"best_layer": 13, "best_score": 0.612, "layers_passing_gate": [...]},
  "dom":             {"best_layer": 13, "per_layer_cos_to_best": {...}},
  "layer_selection": {"selected_layers": [11, 12, 13, 14], "n_selected": 4},
  "cpca_sweep":      {"13": {"best_k": 5, "best_beta": 0.5, "best_acc": 0.634}},
  "subspace_merge":  {"r_final": 10, "U_shape": [2048, 10], "layer_weights": {...}},
  "method_comparison": {"dom_acc": 0.612, "cpca_acc": 0.634, "winner": "cpca"},
  "step_times_s":    {"collection": 4821, "probe": 12, "dom": 0.3, ...}
}
```

---

## Metadata File

`phase2_meta.json` is read by Phase 3 to know which layer to inject at and which vector to load:

```json
{
  "model_tag": "qwen25_math1.5b",
  "best_ccot_ratio": 6,
  "ccot_best_layer": 13,
  "base_best_layer": 11,
  "ccot_selected_layers": [11, 12, 13, 14],
  "ccot_layer_scores": {"11": 0.571, "12": 0.598, "13": 0.612, ...},
  "ccot_max_probe_score": 0.612,
  "ccot_method_accs": {"dom": 0.612, "cpca": 0.634},
  "ccot_winner_method": "cpca",
  "best_source": "ccot",
  "best_method": "cpca",
  "ccot_r_final": 8
}
```
(`ccot_r_final` reflects the per-backbone config — 8 for `qwen25_math1.5b`, 10 for the other three; see the config table above. The example above matches the actual `qwen25_math1.5b` run rather than the generic fallback default.)

---

## All Output Files

```
vectors/{config}/{model}/
├── ccot_hstates_cache.pt           # Raw H+/H− per layer (cached)
├── ccot_dom.pt                     # [d] best-layer DoM truth vector
├── ccot_multilayer_dom.pt          # top-3 layers' DoM vectors (always saved alongside ccot_dom.pt)
├── ccot_cpca_r{r_final}.pt         # [d, r_final] cPCA subspace (r_final=8 for qwen25_math1.5b, else 10)
├── ccot_shuffled_dom.pt            # [d] shuffled DoM control
├── ccot_shuffled_cpca_r{r_final}.pt # [d, r_final] shuffled cPCA control
├── ccot_diagnostics.json           # Per-step statistics
├── ccot_head_hstates_cache.pt      # Phase 2.5 only: raw per-(layer,head) H+/H− (cached)
├── ccot_iti_heads.pt               # Phase 2.5 only: top-K head directions/sigmas (ITI)
├── base_dom.pt                     # [d] CoT-source DoM (if run_source_b=True)
├── base_multilayer_dom.pt          # (if run_source_b=True)
├── base_cpca_r{r_final}.pt         # [d, r_final] CoT-source cPCA (if run_source_b=True)
├── base_diagnostics.json           # (if run_source_b=True)
└── phase2_meta.json                # Layer metadata + probe scores (read by Phase 3)
```

---

## Checkpoint Resume

Hidden state collection is cached. If `{source}_hstates_cache.pt` already exists, Step 1 is skipped:

```python
if os.path.exists(hstates_cache):
    _cache = torch.load(hstates_cache, map_location='cpu')
    H_pos, H_neg = _cache['H_pos'], _cache['H_neg']
else:
    H_pos, H_neg = collect_hidden_states(...)
    torch.save({'H_pos': H_pos, 'H_neg': H_neg}, hstates_cache)
```

The full Phase 2 run is also guarded by `pipeline.py`: if `ccot_dom.pt` already exists in `vectors_dir`, the entire phase is skipped.

---

## Trigger

```bash
python pipeline.py --phase 2
python pipeline.py --phase 2 --model qwen25_math1.5b --config S2
```

---

## What Phase 3 Uses

- `ccot_dom.pt` → v_truth for the single-layer DoM steering hook
- `ccot_multilayer_dom.pt` → top-3 layer vectors for the multi-layer DoM steering hook (and its MLP-sublayer extension)
- `ccot_cpca_r{r_final}.pt` → U_truth for cPCA steering hook
- `ccot_shuffled_dom.pt`, `ccot_shuffled_cpca_r{r_final}.pt` → control conditions
- `ccot_iti_heads.pt` (Phase 2.5, if generated) → top-K head directions/sigmas for the ITI steering hook
- `phase2_meta.json` → `ccot_best_layer` (injection layer L*), `best_ccot_ratio` (which CCoT checkpoint to load), `ccot_r_final`
