# Phase 2: Truth Vector Extraction

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
3. A forward hook captures `h[:, boundary_idx, :]` at every transformer layer simultaneously
4. The boundary index is detected from the generated sequence:
   - `find_boundary_idx_ccot`: looks for `</think>`, `####`, or `\n\nAnswer:`
   - `find_boundary_idx_base`: looks for similar markers in the CoT format

### Why the boundary token?

The boundary is the last token of the reasoning chain before the model commits to an answer. It concentrates the maximum reasoning-relevant signal: all prior reasoning has been processed, but the answer hasn't been generated yet. Injecting later is too late; injecting earlier corrupts the reasoning.

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

For each layer L with sufficient samples:
1. Concatenate H+_L and H−_L into X; labels y = [1, ..., 0, ...]
2. Stratified 80/20 split
3. `StandardScaler` fit on train, applied to both
4. `LogisticRegression(max_iter=1000, C=1.0)` fit on train
5. Accuracy on held-out 20% → `layer_scores[L]`

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

---

## Step 4: Control — Shuffled-Label DoM

Same computation as Step 3, but H+ and H− labels are randomly shuffled before taking the mean difference.

Expected: near-zero cosine similarity to the real v_truth.

If the shuffled vector has high cosine similarity to v_truth, the direction is capturing something structural (e.g. question difficulty, token position) rather than reasoning correctness. Used in Phase 3 to confirm direction specificity.

Output: `{vectors_dir}/{source}_shuffled_dom.pt`

---

## Step 5: cPCA Layer Selection (`cpca.py::select_layers`)

Not all layers should be included in the cPCA computation. Selection criteria:

1. **Probe threshold**: `probe_acc[L] ≥ mean(all scores) + threshold_multiplier × std(all scores)`
   - `threshold_multiplier = 0.5` by default
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
| `full` | Llama, Qwen2.5-3B, Phi-2 | Full eigendecomposition of C_contrast |
| `shrunk` | Qwen2.5-Math-1.5B | Ledoit-Wolf covariance shrinkage before eigendecomposition (better for small n, large d) |
| `randomized` | Large d, speed needed | Randomized SVD via `sklearn.utils.extmath.randomized_svd` |

---

## Step 7: Weighted Subspace Merge (`cpca.py::weighted_subspace_merge`)

Combines per-layer subspaces into a single U_truth [d, r_final] where r_final=10.

**Per-layer weight**:
```
w_L = probe_acc[L] × mean_eigenvalue_L × directional_agreement_L
```
where `directional_agreement_L = max(0, cos(v_truth, U_L[:, 0]))` — alignment between the DoM direction and the leading eigenvector of layer L's subspace.

**Merge**: form a weighted block matrix of all per-layer eigenvectors, then take the top r_final left singular vectors via SVD.

This ensures that layers which are both informative (high probe accuracy) and coherent with the DoM direction contribute more to the final subspace.

Output: `{vectors_dir}/{source}_cpca_r10.pt`:
```python
{
    'U_truth': tensor([d, 10]),
    'selected_layers': [int, ...],
    'model_tag': str,
    'source': str,
    'r_final': 10,
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

Output: `{vectors_dir}/{source}_shuffled_cpca_r10.pt`

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
  "ccot_r_final": 10
}
```

---

## All Output Files

```
vectors/{config}/{model}/
├── ccot_hstates_cache.pt           # Raw H+/H− per layer (cached)
├── ccot_dom.pt                     # [d] DoM truth vector
├── ccot_cpca_r10.pt                # [d, 10] cPCA subspace
├── ccot_shuffled_dom.pt            # [d] shuffled DoM control
├── ccot_shuffled_cpca_r10.pt       # [d, 10] shuffled cPCA control
├── ccot_diagnostics.json           # Per-step statistics
├── base_dom.pt                     # [d] CoT-source DoM (if run_source_b=True)
├── base_cpca_r10.pt                # [d, 10] CoT-source cPCA (if run_source_b=True)
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

- `ccot_dom.pt` → v_truth for DoM steering hook
- `ccot_cpca_r10.pt` → U_truth for cPCA steering hook
- `ccot_shuffled_dom.pt`, `ccot_shuffled_cpca_r10.pt` → control conditions
- `phase2_meta.json` → `ccot_best_layer` (injection layer L*), `best_ccot_ratio` (which CCoT checkpoint to load)
