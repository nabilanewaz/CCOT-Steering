# Phase 3: Inference-Time Steering

> **Full-experiment implementation (2026-09-16):** See [FULL_EXPERIMENT.md](FULL_EXPERIMENT.md) for the current executable contract. S3 now means 60/10/30 (4,484/747/2,242 on GSM8K); all validation conditions use 2,242 examples. Phase 1 remains Coconut with `ccot_L3/L4/L6` checkpoints and `phase1_best_latent.json`, so the historical `R6`/compression-ratio examples below map to the selected `L` budget. The skip set is empty. Steering uses generated-token positions for both cached and uncached decoding, with no oracle reasoning prefix. Alpha tuning/diagnostic subsets remain as specified; test-set alpha tuning is disabled. Historical results and audit notes below are not results of the new full run.

Phase 3 takes the truth vectors from Phase 2 and applies them at generation time by injecting a scaled perturbation into the model's hidden states. It learns the optimal perturbation magnitude α on D_val, then evaluates every condition in a full grid to find the best steered configuration for Phase 4.

---

## Files

| File | Role |
|------|------|
| `evaluate.py` | Full condition grid runner + per-condition save/resume |
| `alpha.py` | `LearnableAlpha` module + gradient-based α tuning |
| `hooks.py` | Forward hooks that apply DoM / cPCA / noise / multi-layer / MLP / ITI perturbations |
| `select.py` | Best config selection using Wilson CI lower bound |
| `lambda_sweep.py` | λ_a × λ_m grid search for loss hyperparameters |
| `plots.py` | Loss curves, α diagnostic plot, λ heatmap |

> **Note on scope**: this document was originally written for the boundary-extraction / boundary-injection / single-layer pipeline. That configuration produced a null steering result on the locked test set (steered accuracy statistically indistinguishable from a random-direction control). The sections below have been updated to describe the **current** code, which defaults to mean-generated extraction (Phase 2) + generation-only injection (this phase) and adds multi-layer DoM, an MLP-sublayer extension, and ITI as additional conditions. Where the two eras differ, both are described and labeled.

---

## Prerequisites

- Phase 2 complete: `vectors/{config}/{model}/ccot_dom.pt`, `ccot_multilayer_dom.pt`, `ccot_cpca_r{r_final}.pt`, `phase2_meta.json`
- Phase 1 complete: `checkpoints/{config}/{model}/cot/` and `ccot_R{best}/` adapters
- Optional: Phase 2.5 complete (`ccot_iti_heads.pt`) for the ITI condition — if absent, `iti_{rtag}_{source}` is simply skipped, everything else still runs

---

## How Steering Works (`hooks.py`)

A PyTorch forward hook intercepts the hidden state tensor at a specific layer during generation, then adds a direction-aligned perturbation. The base perturbation forms (DoM/cPCA/noise) are unchanged from the original design; what changed is **which token positions the hook fires at** — see "Injection mode" below.

### DoM hook (`make_dom_hook`)

```
h' = h + α · σ_h · v̂
```

- v̂ = v_truth / ‖v_truth‖ (unit-normalized DoM direction)
- σ_h = ‖h_t‖ / √d (layer-norm scale estimate; keeps the perturbation proportional to the activation magnitude)

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

### Injection mode: `gen_only` (current default) vs. legacy boundary injection

Every hook factory (`make_dom_hook`, `make_cpca_hook`, `make_noise_hook`) now takes a `gen_only: bool = True` argument:

| Mode | Positions the hook actually perturbs |
|------|----------------------------------------|
| `gen_only=True` **(default)** | Only during autoregressive decoding steps, i.e. when the hook fires with `seq_len == 1` (position 0 of that single-token forward pass). **Never** during the prefill pass that encodes the prompt/probe-generated prefix. |
| `gen_only=False` (legacy) | Also perturbs `boundary_idx` (and up to `n_pre` tokens before it) during the prefill pass that re-encodes the sequence up to the boundary — the original design. |

This mirrors the Phase 2 extraction-mode change: `gen_only=True` aligns injection with `mean_gen` extraction (both act on generated tokens, not a single prefill-time boundary position), whereas `gen_only=False` paired with `extraction='boundary'` was the original, jointly-null-performing combination.

### Multi-layer DoM hook (`run_with_multihook`)

Registers `make_dom_hook` independently at **each of the top-3 probe-scoring layers** (from `{source}_multilayer_dom.pt`, Phase 2) simultaneously — each layer uses its own layer-local DoM vector, not one vector copied to three layers. This is the `multilayer_dom_{rtag}_{source}` condition, and it is Phase 3's best-performing configuration on D_val.

### MLP-sublayer extension (in progress)

`run_with_multihook` accepts an optional `mlp_hook_pairs` argument: the same multi-layer DoM vectors, hooked additionally onto each selected layer's `layer.mlp` output (not just the full transformer-block output), motivated by Geva et al. (2021) — MLP sublayers act as key-value memories. This is the `multilayer_dom_mlp_{rtag}_{source}` condition; implemented, but evaluation is not yet complete/reported.

### ITI hook (`make_iti_hook_for_layer`, `run_with_iti`)

Unlike the hooks above (which perturb the full residual-stream output of a transformer block), ITI registers a forward **pre**-hook on each selected layer's `self_attn.o_proj` **input** — the concatenated per-head attention output, before it is mixed by the output projection — and modifies only the selected heads' `head_dim`-dimensional sub-vectors:
```
head_out' = head_out + α · σ_head · v̂_head        (only for heads in top_heads)
```
where `v̂_head` and `σ_head` come from `{source}_iti_heads.pt` (Phase 2.5, `PHASE2.md`). This is the `iti_{rtag}_{source}` condition. Because the perturbation happens before `o_proj`, unselected heads and the eventual MLP sublayer are untouched — only the selected heads' contribution to the residual stream changes.

### Boundary index (probe-generate pass)

Whenever a hook needs `boundary_idx` (multi-layer/MLP/ITI conditions, and any `gen_only=False` condition), Phase 3 locates it with a probe-generate pass before the real steered generation:
1. Generate a short sequence (max 128 tokens, greedy, no_grad)
2. Scan for `</think>`, `####`, or `\n\nAnswer:` in the generated token IDs
3. Use that position as `boundary_idx` for the actual steered generation

For plain DoM/cPCA/noise conditions with `gen_only=True`, `boundary_idx` is still computed this way but is only used to decide *whether* prefill positions would have been touched under the legacy mode — under the default `gen_only=True` it has no effect on the actual injected positions.

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
| Early stopping patience | 5 (code: `es_patience`; `configs/protocol.yaml` documents 3 — check which was actually used for a given run) |
| Tune/ES split | 90% / 10% of D_alpha |
| D_alpha cap | min(50, len(D_val)) — **note: this is D_val, not D_steer** |

α is a single scalar — gradient tuning on thousands of examples is redundant. 50 examples is enough to estimate the loss landscape.

> **Flag — "grid search on D_steer" claim not found in code.** Some project write-ups (`phase3_overview.md`, the EMNLP-style paper draft) describe the reported multi-layer-DoM/cPCA/noise α (0.8198) as "tuned by grid search directly against D_steer accuracy," distinct from this gradient-based method. As of this audit, `_tune_and_save_alpha` (above) is the **only** α-tuning code path found in the repository, it uses the gradient/AdamW/sigmoid method described here, and it tunes on a subset of **D_val** (capped at 50), not D_steer. Either a separate grid-search script exists outside this repo/was run ad hoc and not checked in, or the "grid search / D_steer" description is inaccurate and the 0.8198 value in fact came from this same `tune_alpha` path. Confirm against the actual run artifacts before diagramming this as two distinct tuning procedures.

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
- `dom_R6_{source}` — single-layer DoM steering
- `cpca_R6_{source}` — cPCA steering (skipped if U_cpca missing)
- `multilayer_dom_R6_{source}` — top-3-layer DoM steering, simultaneous (skipped unless `{source}_multilayer_dom.pt` exists) — **best result on D_val**
- `multilayer_dom_mlp_R6_{source}` — multi-layer DoM + MLP-sublayer hooks at the same layers (same file dependency as above; implemented, evaluation in progress)
- `iti_R6_{source}` — ITI head-level steering (skipped unless `{source}_iti_heads.pt` exists — i.e. only after Phase 2.5 has been run)
- `shuf_dom_R6_{source}` — shuffled DoM control (ctrl-A)
- `neg_dom_R6_{source}` — negative DoM (ctrl-B)
- `neg_cpca_R6_{source}` — negative cPCA (ctrl-C)
- `shuf_cpca_R6_{source}` — shuffled cPCA control (ctrl-D)

Per ratio only:
- `trimmed_dom_R6` — trimmed CoT + DoM (CoT adapter + base DoM vector)

### `SKIP_CONDITIONS` — fast-iteration toggle, not a fixed condition list

`evaluate.py` defines `SKIP_CONDITIONS: frozenset` at module level; any condition name in it is treated as already-done and never (re-)run. As currently checked in:
```python
SKIP_CONDITIONS = frozenset({
    'no_cot', 'full_cot', 'trimmed_R6', 'noise_R6_ccot', 'noise_R6_base',
})
```
This does **not** mean those conditions have no results — it means a prior run already computed them (they're sitting in `phase3_val_interim.json`, which is loaded and preserved on every run regardless of `SKIP_CONDITIONS`), and this setting is a speed toggle to avoid re-running them while iterating on newer conditions (multi-layer DoM, MLP, ITI). Comment in code: *"Conditions to skip entirely (for fast testing). Set to frozenset() to run all."* Before diagramming "what Phase 3 runs," check this set's value at the commit that produced the results being diagrammed — it changes over time and is not itself part of the experimental design.

### `ConditionResult` dataclass

| Field | Meaning |
|-------|---------|
| `condition` | Condition name string |
| `model_tag` | Model identifier |
| `ratio` | Compression ratio (None for no_cot / full_cot) |
| `vector_source` | `'ccot'` or `'base'` or None |
| `vector_method` | `'dom'`, `'cpca'`, `'multilayer_dom'`, `'multilayer_dom_mlp'`, `'iti'`, `'noise'`, `'shuf_dom'`, `'neg_dom'`, `'neg_cpca'`, `'shuf_cpca'`, or `None` (no_cot/full_cot/trimmed/ccot) |
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

After all conditions, `_run_diagnostic_sweep` runs a sweep of α ∈ {0, 0.1, 0.5, 1, 2, 5, 10, 20, 50} for source=`ccot` DoM on `n_sub=50` D_val examples (default) to show the full accuracy-vs-α curve. Results saved to:
- `{results_dir}/alpha_diagnostic.json`
- `{results_dir}/alpha_diagnostic.png`

This reveals whether α* is near the peak of the curve.

> **Data-integrity note**: a saved sweep table for `qwen25_math1.5b` reports accuracies as exact multiples of one percentage point (e.g. 63%, 65%, 58%). At `n=50`, achievable accuracies are multiples of two percentage points (0, 2, 4, ... 100%) — values like 63% or 65% are not reachable at exactly 50 examples. Either that particular saved sweep was produced with a different `n_sub` (100 fits every observed value), or the accuracy was rounded/aggregated in a way that isn't a simple `correct/n_sub`. Check `n_sub` and the aggregation in the sweep artifact itself before treating any reported peak α from this sweep as final.

### ITI Alpha Sweep (`_eval_iti_alpha_sweep`, separate code path)

ITI has its own diagnostic sweep, run once per source **before** the main condition loop (only if `{source}_iti_heads.pt` exists — i.e. Phase 2.5 has run): α ∈ {0.5, 1, 2, 5, 10, 15, 20} on `n_sub=50` D_val examples, greedy-decoding the actual `run_with_iti` steered generation (not a hook-based single alpha value). Saved to `{results_dir}/iti_alpha_diagnostic_{source}.json`, and its `best_alpha` (default 5.0 if unavailable) is what the `iti_{rtag}_{source}` condition actually uses — **not** the same `α*` as DoM/cPCA/multi-layer DoM.

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
├── phase3_val_interim.json         # Per-condition save (resume support; accumulates across runs/SKIP_CONDITIONS changes)
├── phase3_val.json                 # Final full results (written at end)
├── steered_val.json                # Best steered summary (read by selection.py) — best over dom/cpca/multilayer_dom/multilayer_dom_mlp/iti
├── phase3_diagnostics.json         # Full metadata + timing
├── phase3_best_config.yaml         # Selected winning condition
├── phase3_budgets_R6.json          # Cached per-example token budgets
├── phase3_ccot_correct_R6.json     # Cached CCoT correctness list (for flip rate)
├── {source}_lambda_sweep.json      # λ grid results
├── {source}_lambda_heatmap.png     # λ heatmap
├── {source}_alpha_history.json     # Per-epoch α training log
├── {source}_loss_curves.png        # Training/ES loss plot
├── alpha_diagnostic.json + .png    # α sweep curve (DoM, source=ccot, n=50)
└── iti_alpha_diagnostic_{source}.json  # ITI-specific α sweep (if Phase 2.5 ran)

vectors/{config}/{model}/
└── {source}_alpha_star.pt          # Learned α* (scalar tensor) — used by DoM/cPCA/multi-layer DoM, not ITI
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
