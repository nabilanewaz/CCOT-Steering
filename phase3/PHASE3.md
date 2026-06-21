# Phase 3: Inference-Time Steering

Phase 3 takes the truth vectors from Phase 2 and applies them at generation time by injecting a scaled perturbation into the model's hidden states. It learns the optimal perturbation magnitude α on D_val, then evaluates every condition in a full grid to find the best steered configuration for Phase 4.

---

## Files

| File | Role |
|------|------|
| `evaluate.py` | Full condition grid runner + per-condition save/resume |
| `alpha.py` | `LearnableAlpha` module + gradient-based α tuning |
| `hooks.py` | Forward hooks that apply DoM / cPCA / noise perturbations |
| `select.py` | Best config selection using Wilson CI lower bound |
| `lambda_sweep.py` | λ_a × λ_m grid search for loss hyperparameters |
| `plots.py` | Loss curves, α diagnostic plot, λ heatmap |

---

## Prerequisites

- Phase 2 complete: `vectors/{config}/{model}/ccot_dom.pt`, `ccot_cpca_r10.pt`, `phase2_meta.json`
- Phase 1 complete: `checkpoints/{config}/{model}/cot/` and `ccot_R{best}/` adapters

---

## How Steering Works (`hooks.py`)

A PyTorch forward hook intercepts the hidden state tensor at a specific layer and token position during generation, then adds a direction-aligned perturbation:

### DoM hook (`make_dom_hook`)

```
h' = h + α · σ_h · v̂
```

- v̂ = v_truth / ‖v_truth‖ (unit-normalized DoM direction)
- σ_h = ‖h_t‖ / √d (layer-norm scale estimate; keeps the perturbation proportional to the activation magnitude)
- Applied only at `boundary_idx` — the token where reasoning transitions to answer

### cPCA hook (`make_cpca_hook`)

```
h' = h + α · σ_h · UU^T ĥ
```

- ĥ = h_t / ‖h_t‖ (normalized current hidden state)
- U = cPCA subspace [d, r]
- `UU^T ĥ` projects ĥ onto the truth subspace and adds it back — moves h toward the region where H+ was concentrated
- **Dtype safety**: `U_ = U.to(h_t.dtype)` is applied before the matmul to handle bfloat16 models where `U` is stored as float32

### Noise hook (`make_noise_hook`, control)

```
h' = h + α · σ_h · ε     (ε ~ random unit vector, fresh each call)
```

Tests whether any perturbation at the right layer improves accuracy, regardless of direction. If noise ≈ DoM, the gain is from perturbation size, not direction.

### Negative hooks (controls)

`alpha = -alpha` in `make_dom_hook` / `make_cpca_hook`. Steers in the opposite direction — should hurt accuracy if the direction is correct.

### Boundary index

Phase 3 uses a probe-generate pass to locate the boundary:
1. Generate a short sequence (max 128 tokens, greedy, no_grad)
2. Scan for `</think>`, `####`, or `\n\nAnswer:` in the generated token IDs
3. Use that position as `boundary_idx` for the actual steered generation

---

## α Tuning (`alpha.py::tune_alpha`)

### Why α matters

A perturbation that is too small has no effect. Too large collapses the RMSNorm activations. α needs to be calibrated to the scale of the model's hidden states.

### `LearnableAlpha` module

Sigmoid reparameterization keeps α in a valid range without explicit clipping:

```python
α = α_max · sigmoid(θ)       # α_max = 50.0
```

θ initialization: `log(1 / (α_max − 1)) ≈ −3.89` → α₀ ≈ 1.0

Gradient flows through θ → α → the perturbation δ → the loss.

### Three-term loss

```
L = L_ans + λ_a · L_align + λ_m · L_mag
```

| Term | Formula | Purpose |
|------|---------|---------|
| L_ans | NLL of gold answer tokens (teacher-forced) | directly minimize answer prediction error |
| L_align | 1 − cos(h_steered, v_truth) | prevent α from growing in a direction that diverges from v_truth |
| L_mag | (‖δ‖ / ‖h_orig‖)² | prevent norm collapse: large δ relative to the original hidden state distorts downstream attention |

λ_m is model-dependent:
- Phi-2 (LayerNorm): `λ_m = 0.005` — more tolerant of norm changes
- Llama, Qwen (RMSNorm): `λ_m = 0.01` — stricter norm preservation

### Optimization

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| LR | 5e-2 |
| Max epochs | 5 |
| Early stopping patience | 5 |
| Tune/ES split | 90% / 10% of D_alpha |
| D_alpha cap | min(50, len(D_val)) |

α is a single scalar — gradient tuning on thousands of examples is redundant. 50 examples is enough to estimate the loss landscape.

### λ sweep (before α tuning)

A grid search over λ_a × λ_m is run first on min(200, len(D_val)) examples with max_epochs=2:
- Result cached to `{vectors_dir}/{source}_lambda_sweep.json`
- Selected (λ_a, λ_m) used for the full α tuning run

### Output

`{vectors_dir}/{source}_alpha_star.pt` — scalar tensor, the learned α*

Also saved:
- `{results_dir}/{source}_alpha_history.json` — per-epoch loss breakdown
- `{results_dir}/{source}_loss_curves.png` — training + ES loss plot
- `{results_dir}/{source}_lambda_heatmap.png` — λ grid accuracy heatmap

---

## Full Condition Grid (`evaluate.py::run_phase3_evaluation`)

### Condition naming

```
{method}_{R{ratio}}_{source}
```

Examples: `dom_R6_ccot`, `cpca_R6_ccot`, `noise_R6_ccot`, `shuf_dom_R6_ccot`

### Complete condition list

Fixed (no ratio):
- `no_cot` — base model, direct answer
- `full_cot` — CoT adapter, full chain

Per ratio (currently `RATIOS = [0.6]`, one ratio for fast iteration):
- `trimmed_R6` — CoT adapter, token-capped
- `ccot_R6` — CCoT adapter baseline

Per ratio × per source (sources = `ccot`, `base`):
- `noise_R6_{source}` — random direction control
- `dom_R6_{source}` — DoM steering
- `cpca_R6_{source}` — cPCA steering (skipped if U_cpca missing)
- `shuf_dom_R6_{source}` — shuffled DoM control (ctrl-A)
- `neg_dom_R6_{source}` — negative DoM (ctrl-B)
- `neg_cpca_R6_{source}` — negative cPCA (ctrl-C)
- `shuf_cpca_R6_{source}` — shuffled cPCA control (ctrl-D)

Per ratio only:
- `trimmed_dom_R6` — trimmed CoT + DoM (CoT adapter + base DoM vector)

### `ConditionResult` dataclass

| Field | Meaning |
|-------|---------|
| `condition` | Condition name string |
| `model_tag` | Model identifier |
| `ratio` | Compression ratio (None for no_cot / full_cot) |
| `vector_source` | `'ccot'` or `'base'` or None |
| `vector_method` | `'dom'`, `'cpca'`, `'noise'`, `'shuf_dom'`, etc. |
| `alpha` | α value used |
| `accuracy` | Fraction of correct answers |
| `flip_rate` | Fraction of CCoT-wrong examples that became correct |
| `reasoning_tokens` | Mean tokens in generated chain |
| `actual_ratio` | `reasoning_tokens / full_cot_mean_tokens` |
| `latency_sec` | Mean wall-clock seconds per example |
| `answer_found_rate` | Fraction with extractable answer |
| `n_examples` | Number of D_val examples evaluated |

### Flip rate

```
flip_rate = #{i : ccot_correct[i]=False AND condition_correct[i]=True} / #{i : ccot_correct[i]=False}
```

Measures how many previously-wrong CCoT examples were fixed by steering. Penalizes conditions that degrade many correct examples while fixing few.

### Token budgets

Per-example budgets are computed from the CoT model's actual output lengths, cached to `phase3_budgets_R6.json`. Loading from cache avoids ~10h of recompute on resume.

---

## Per-Condition Save & Resume

After every condition, results are flushed to disk:

```python
def _append_result(r, results, results_dir):
    results.append(r)
    path = os.path.join(results_dir, 'phase3_val_interim.json')
    with open(path, 'w') as f:
        json.dump([asdict(rr) for rr in results], f, indent=2)
```

On the next run, if `phase3_val_interim.json` exists, all completed conditions are loaded back:

```python
if os.path.exists(_interim_path):
    _saved = json.load(...)
    _done_conds = {r.condition for r in _saved}
    print(f"  [RESUME] {len(results)} conditions already done: ...")
```

Every condition block is guarded:
```python
if 'dom_R6_ccot' not in _done_conds:
    # run condition
else:
    print("  [RESUME] skipping")
```

The `ccot_correct` list (used for flip rate) is also cached per ratio to `phase3_ccot_correct_R6.json`.

---

## Diagnostic Alpha Sweep

After all conditions, a sweep of α ∈ {0, 0.1, 0.5, 1, 2, 5, 10, 20, 50} is run on 50 D_val examples to show the full accuracy-vs-α curve. Results saved to:
- `{results_dir}/alpha_diagnostic.json`
- `{results_dir}/alpha_diagnostic.png`

This reveals whether α* is near the peak of the curve.

---

## Best Config Selection (`select.py`)

After `run_phase3_evaluation` completes, `select_best_steered_config` picks the single best steered condition:

```python
best = max(steered, key=lambda r: (r['accuracy'], r['flip_rate']))
```

Only `vector_method in ('dom', 'cpca')` conditions are considered (not noise or controls).

**Wilson score lower bound** (95% CI, z=1.96):
```
p̂ = accuracy
centre = (p̂ + z²/2n) / (1 + z²/n)
margin = z · sqrt(p̂(1−p̂)/n + z²/4n²) / (1 + z²/n)
wilson_lower = centre − margin
```

This penalizes conditions that look good on few examples vs conditions that are consistently strong.

### Output

`{results_dir}/phase3_best_config.yaml`:
```yaml
model_tag: qwen25_math1.5b
best_condition: dom_R6_ccot
ratio: 0.6
vector_source: ccot
vector_method: dom
alpha_star: 2.3
steered_accuracy: 0.712
ccot_accuracy: 0.694
flip_rate: 0.063
wilson_lower_95: 0.685
```

`pipeline.py` then calls `_update_selected_phase3_best` to write this into `configs/selected.yaml` under `phase3_best.{model_tag}`.

---

## Config Selection Across Models (`scripts/selection.py`)

After all models complete Phase 3, `select_best_config` reads all `steered_val.json` files and picks the split config (S1/S2/...) with the highest mean Wilson CI lower bound across models. The winner is written to `configs/selected.yaml` as `winning_config`.

---

## All Output Files

```
results/{config}/{model}/
├── phase3_val_interim.json         # Per-condition save (resume support)
├── phase3_val.json                 # Final full results (written at end)
├── steered_val.json                # Best steered summary (read by selection.py)
├── phase3_diagnostics.json         # Full metadata + timing
├── phase3_best_config.yaml         # Selected winning condition
├── phase3_budgets_R6.json          # Cached per-example token budgets
├── phase3_ccot_correct_R6.json     # Cached CCoT correctness list (for flip rate)
├── {source}_lambda_sweep.json      # λ grid results
├── {source}_lambda_heatmap.png     # λ heatmap
├── {source}_alpha_history.json     # Per-epoch α training log
├── {source}_loss_curves.png        # Training/ES loss plot
└── alpha_diagnostic.json + .png    # α sweep curve

vectors/{config}/{model}/
└── {source}_alpha_star.pt          # Learned α* (scalar tensor)
```

---

## Trigger

```bash
python pipeline.py --phase 3
python pipeline.py --phase 3 --model qwen25_math1.5b --config S2
```

Phase 3 with D_val capped to 300 examples (set in `pipeline.py`):
```python
run_phase3_evaluation(
    ...
    D_val=D_val[:300],
    max_new_tokens=128,
)
```

---

## What Phase 4 Uses

Phase 4 reads `configs/selected.yaml` which contains:
- `winning_config` → which S*/config directory to use
- `phase3_best.{model_tag}.best_condition` → which steered condition to run on D_test
- `phase3_best.{model_tag}.alpha_star` → α* for DoM/cPCA hooks
- `phase3_best.{model_tag}.ratio` → which CCoT checkpoint to load

Phase 4 does its own dual alpha sweep (DoM + cPCA separately) over a wider range on D_test.
