# Phase 4: Final Evaluation on D_test

Phase 4 is the single locked evaluation on the held-out test set. It runs once, uses only configuration decided before this phase (the Phase 3 winner), and produces the final accuracy numbers that constitute the experiment's results.

The entry point is `evaluate_final.py`. It is the **only script that opens `test.jsonl`**.

---

## Files

| File | Role |
|------|------|
| `evaluate_final.py` | All Phase 4 logic: conditions, metrics, flip matrices, CIs, plots |
| `configs/selected.yaml` | Written by Phase 3 — locked config for Phase 4 |
| `utils/data.py` | Guarded `load_test_set()` — raises RuntimeError if called from anywhere else |

---

## Prerequisites

- `configs/selected.yaml` exists (written by Phase 3 + `scripts/selection.py`)
- For each model: `vectors/{winner}/{model}/ccot_dom.pt`, `ccot_cpca_r*.pt`, `{source}_alpha_star.pt`
- For each model: `checkpoints/{winner}/{model}/cot/` and `ccot_R{best}/` adapters

---

## How it is triggered

```bash
python pipeline.py --phase 4 --dataset gsm8k
```

`pipeline.py` delegates entirely to `evaluate_final.py` as a subprocess:
```python
subprocess.run([sys.executable, 'evaluate_final.py'], env=phase4_subprocess_env())
```

`phase4_subprocess_env()` propagates the `CCOT_DATASET` environment variable so the subprocess knows which dataset is active.

Direct invocation also works:
```bash
python evaluate_final.py
python evaluate_final.py --dataset gsm8k --results-dir results/final
```

---

## What `configs/selected.yaml` contains

```yaml
winning_config: S2
phase3_best:
  qwen25_math1.5b:
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
  llama32_3b:
    ...
```

Every value here was decided on D_val and is not changed for Phase 4. The `alpha_star`, `ratio`, `vector_source`, and `vector_method` are used directly to configure the steering hooks.

---

## Conditions Evaluated

Phase 4 runs 8 conditions per model on the full D_test:

| # | Condition | Model loaded | Steering |
|---|-----------|-------------|---------|
| 1 | `no_cot` | frozen base | none |
| 2 | `full_cot` | cot/ adapter | none |
| 3 | `ccot_R*` | ccot_R{best}/ adapter | none |
| 4 | `trimmed_R*` | cot/ adapter | none (token-capped) |
| 5 | `noise_R*_{source}` | ccot_R{best}/ adapter | random direction at α* |
| 6 | `dom_R*_{source}` | ccot_R{best}/ adapter | DoM vector at α* |
| 7 | `cpca_R*_{source}` | ccot_R{best}/ adapter | cPCA subspace at α* |
| 8 | `trimmed_dom_R*` | cot/ adapter | DoM at α* (token-capped) |

The ratio and source come from `configs/selected.yaml`. Conditions 5–7 use the same hooks as Phase 3 (`make_noise_hook`, `make_dom_hook`, `make_cpca_hook`).

---

## Alpha Sweep on D_test

After conditions are evaluated, Phase 4 runs a sweep of α ∈ {0, 0.1, 0.5, 1, 2, 5, 10, 20, 50} **separately for DoM and cPCA** on a 50-example subset of D_test:

```python
dom_sweep  = run_alpha_sweep_test(..., method='dom')
cpca_sweep = run_alpha_sweep_test(..., method='cpca')
```

Each sweep point records accuracy, truth alignment, and trajectory coherence. This reveals the true optimal α on unseen data and validates whether the α* tuned on D_val (Phase 3) was near the peak.

Output plots: `alpha_sweep_dom.png`, `alpha_sweep_cpca.png`

---

## Metrics

### Per example (`ExampleResult`)

| Field | Meaning |
|-------|---------|
| `correct` | Answer matches gold (normalized) |
| `answer_found` | An answer string was extractable |
| `reasoning_tokens` | Token count of generated reasoning span |
| `total_tokens` | Total tokens generated (including answer) |
| `latency_sec` | Wall-clock seconds for this example |
| `traj_coherence` | Mean cos(h_t, h_{t+1}) at injection layer |
| `truth_align` | Mean cos(h_t, v_truth) at injection layer |

### Per condition (`FinalMetrics`)

| Field | Meaning |
|-------|---------|
| `accuracy` | Fraction correct |
| `n_correct` / `n_total` | Raw counts |
| `reasoning_tokens_mean/std/min/max` | Token distribution |
| `actual_ratio_mean` | `reasoning_tokens / full_cot_tokens` |
| `latency_mean/std/p50/p95` | Latency distribution |
| `answer_found_rate` | Extraction success rate |
| `trajectory_coherence` | Mean across examples (steered conditions only) |
| `truth_alignment` | Mean cos to v_truth (steered conditions only) |
| `ci_lower_95` / `ci_upper_95` | Bootstrap CI bounds (filled after `compute_condition_cis`) |

---

## Latent Metrics

Recorded for every steered condition (DoM, cPCA, noise) by attaching a read-only hook at layer L*:

**Trajectory coherence**: are consecutive hidden states at L* pointing in a consistent direction?
```
traj_coherence = mean( cos(h_t, h_{t+1}) )   over generation steps at L*
```
High coherence = stable directional flow. Low coherence = the perturbation is causing erratic jumps.

**Truth alignment**: is the hidden state aligned with the truth direction?
```
truth_alignment = mean( cos(h_t, v_truth) )   over generation steps at L*
```
Steered conditions should have higher truth alignment than unsteered CCoT. Random noise should have alignment ≈ 0.

These two metrics are a sanity check: if `dom_ccot` improves accuracy but has low truth alignment, the improvement may be spurious.

---

## Flip Matrices

A `FlipMatrix` records all four possible transitions between two conditions for each example:

| Cell | Meaning |
|------|---------|
| F00 | Both conditions correct (stable correct) |
| F01 | Condition A correct → B wrong (degradation) |
| F10 | Condition A wrong → B correct (improvement) |
| F11 | Both conditions wrong (stable wrong) |

Derived metrics:
- **Improvement rate**: F10 / (F10 + F11) — of examples A got wrong, how many did B fix?
- **Degradation rate**: F01 / (F00 + F01) — of examples A got right, how many did B break?
- **Net gain**: F10 − F01 — number of examples where B is strictly better

Key flip matrices computed:
1. CCoT → CCoT+DoM (primary claim: steering improves over baseline)
2. CCoT → CCoT+cPCA (primary claim: cPCA version)
3. CCoT → Random Noise (direction specificity: is noise as good as DoM?)
4. Full CoT → CCoT (compression cost)
5. Trimmed → CCoT (mechanism gain: does CCoT training help beyond just fewer tokens?)
6. CCoT → Trimmed+DoM (does steering work on trimmed chains too?)

Full 8×8 net-gain pairwise grid also saved as a heatmap.

---

## Bootstrap Confidence Intervals

All accuracy values are accompanied by 95% bootstrap CIs (1,000 resamples, percentile method):

**Per-condition CI**:
```python
point, lower, upper = bootstrap_ci(correct_array, n_bootstrap=1000, seed=0)
```
Resamples the binary correct/wrong array 1,000 times; takes 2.5th and 97.5th percentiles.

**Paired CI on difference** (Δ = acc_B − acc_A):
```python
point, lower, upper, significant = bootstrap_ci_difference(results_a, results_b)
```
Uses the **same resampled indices for both conditions** on each bootstrap iteration. This accounts for question-level correlation (both conditions see the same test examples). `significant = True` iff the CI excludes 0.

Key paired CIs:
- `dom_vs_ccot` — steered vs unsteered (primary claim)
- `cpca_vs_ccot` — cPCA steered vs unsteered
- `dom_vs_noise` — DoM vs random direction (direction specificity)
- `dom_vs_full_cot` — efficiency claim (steered CCoT vs full reasoning)
- `ccot_vs_full_cot` — compression cost
- `trimmed_vs_full_cot` — trimmed baseline vs full

---

## Output Files

### Per-model

```
results/final/{model_tag}/
├── {model_tag}_test.json              # All condition metrics + flip matrices + CIs
├── alpha_diagnostic.json + .png       # Phase 3 diagnostic α sweep
├── alpha_sweep_dom.json + .png        # Phase 4 α sweep (DoM)
├── alpha_sweep_cpca.json + .png       # Phase 4 α sweep (cPCA)
├── accuracy_ci.png                    # Bar chart with 95% CI error bars
├── mechanism_gain.png                 # Gain vs CCoT baseline bar chart
├── flip_matrix_heatmap.png            # F00/F01/F10/F11 heatmap
├── latent_metrics.png                 # Trajectory coherence + truth alignment
├── latency.png                        # Mean + p95 latency per condition
├── token_efficiency.png               # Scatter: tokens vs accuracy
└── net_gain_grid.png                  # 8×8 pairwise net-gain heatmap
```

### Cross-model summary

```
results/final/
├── summary_test.json                  # All models aggregated
├── cross_model_accuracy.png           # Grouped bar chart across models
└── cross_model_mechanism_gain.png     # Heatmap: gain over CCoT per model
```

### Text tables (also written to `.txt` files alongside JSON)

```
results/final/{model_tag}/
├── tables_accuracy_table.txt
├── tables_latent_metrics_table.txt
├── tables_efficiency_table.txt
├── tables_flip_matrices.txt
├── tables_mechanism_gain.txt
├── tables_specificity.txt
├── tables_flip_grid.txt
├── tables_confidence_intervals.txt
├── tables_alpha_sweep_dom.txt
├── tables_alpha_sweep_cpca.txt
└── tables_paired_cis.txt
```

---

## The `{model_tag}_test.json` structure

```json
{
  "model_tag": "qwen25_math1.5b",
  "winning_config": "S2",
  "dataset": "gsm8k",
  "n_test": 1319,
  "conditions": ["no_cot", "full_cot", "ccot", "trimmed_cot", ...],
  "metrics": {
    "no_cot":   {"accuracy": 0.412, "n_correct": 544, ...},
    "full_cot": {"accuracy": 0.781, ...},
    "dom_R6_ccot": {"accuracy": 0.718, "truth_alignment": 0.312, ...},
    ...
  },
  "flip_matrices": [
    {"condition_a": "ccot_R6", "condition_b": "dom_R6_ccot",
     "F00": 623, "F01": 48, "F10": 67, "F11": 581,
     "improvement_rate": 0.103, "degradation_rate": 0.071, "net_gain": 19},
    ...
  ],
  "flip_grid": {
    "model_tag": "qwen25_math1.5b",
    "conditions": ["no_cot", ...],
    "net_gain": {"no_cot": {"full_cot": -284, ...}, ...}
  },
  "condition_cis": {
    "dom_R6_ccot": {"accuracy": 0.718, "ci_lower": 0.693, "ci_upper": 0.743, "half_width": 0.025}
  },
  "paired_cis": {
    "dom_vs_ccot": {"delta": 0.024, "ci_lower": 0.003, "ci_upper": 0.045, "significant": true}
  },
  "alpha_sweeps": {
    "dom":  [{"alpha": 0.0, "accuracy": 0.694, ...}, {"alpha": 2.3, "accuracy": 0.718, ...}],
    "cpca": [...]
  }
}
```

---

## Phase 5: SVAMP Transfer

Phase 5 reuses all Phase 4 infrastructure to evaluate on SVAMP without any retraining:

```bash
python pipeline.py --phase 5
# or:
python evaluate_final.py --dataset svamp --results-dir results/final_svamp_transfer
```

Vectors and checkpoints are read from `vectors/{winner}/{model}/` (GSM8K pipeline). Only the test set changes. Results go to `results/final_svamp_transfer/` so the GSM8K results in `results/final/` are preserved.

---

## Invariants

- `evaluate_final.py` is the **only** file that calls `load_test_set()`
- No hyperparameter may be changed between Phase 3 and Phase 4
- `evaluate_final.py` is designed to be run exactly once per dataset (Phase 4 for GSM8K, Phase 5 for SVAMP)
- If a result file already exists, Phase 4 will overwrite it — there is no skip guard (unlike Phases 1–3). This is intentional: final results should always be reproducible from the locked config.
