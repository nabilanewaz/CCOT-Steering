# Latent Guidance Research Pipeline
### Full Step-by-Step Guide: What to Do & How to Do It

---

## Overview

**Research Goal:** Apply a model-agnostic *latent guidance* (steering) method across multiple continuous/compressed Chain-of-Thought (CoT) architectures to improve reasoning accuracy at inference time — without changing model weights.

**Target Venue:** EMNLP 2026

**Architectures Under Study:**

| Method | Type | Key Characteristic |
|---|---|---|
| Coconut | Continuous latent loop | Reasoning = recurrent hidden-state vectors |
| CCoT | Continuous latent | Hidden-state encoded CoT |
| CODI | Diffusion-based latent | Reasoning via denoising process |
| TokenSkip | Compressed text | Shorter discrete token chain (no latent loop) |

---

## PHASE 0 — Setup & Data Splits (GATE: Must complete before anything else)

### 0.1 Lock Your Data Splits

Split GSM8K into **four non-overlapping sets** before writing any code. Never shuffle after this point.

```
D_train   →  CCoT/Coconut model fine-tuning
D_steer   →  Extract the correctness vector (H+, H-)
D_val     →  All hyperparameter tuning (α, λ, layer selection)
D_test    →  Final evaluation — open ONCE, never iterate on it
```

**How to do it:**

```python
import json
import random

random.seed(42)

with open("gsm8k_train.jsonl") as f:
    data = [json.loads(l) for l in f]

random.shuffle(data)
n = len(data)

D_train = data[:int(n * 0.50)]
D_steer = data[int(n * 0.50):int(n * 0.65)]
D_val   = data[int(n * 0.65):int(n * 0.80)]
D_test  = data[int(n * 0.80):]

# Save and seal D_test — do not load again until final evaluation
```

> **GATE CHECK:** All four splits saved to disk before any training begins. D_test is sealed.

---

## PHASE 1 — Train Compressed CoT Models

### 1.1 Train Each Architecture

For each of the four methods (Coconut, CCoT, CODI, TokenSkip), train or fine-tune on `D_train`.

**Coconut — Curriculum Training:**

```
Stage 1: Train with full text CoT
Stage 2: Replace last N reasoning tokens with latent tokens → train
Stage 3: Increase N progressively until all reasoning is latent
```

Follow the curriculum from Hao et al. (2024). Target: ~34% on GSM8K (reported baseline).

**CCoT:** Fine-tune using the procedure from arxiv 2412.13171.

**CODI:** Fine-tune using the diffusion-based training from arxiv 2502.21074.

**TokenSkip:** Apply LLMLingua-2 compression to full CoT chains, then fine-tune on compressed versions.

### 1.2 Verify Phase 1 Baseline (GATE)

**For each architecture, on `D_val`, verify:**

```
Compressed CoT accuracy  >  No-CoT accuracy
```

If any model fails this check → its training has not converged → do not run steering experiments on it.

**Report:**

- Training loss curves (per stage for Coconut curriculum)
- Side-by-side accuracy on a held-out 200-question slice:
  - No-CoT baseline
  - Text CoT baseline
  - Compressed CoT (your trained model)

> **GATE CHECK:** All architectures beat no-CoT before proceeding.

---

## PHASE 2 — Extract the Correctness Vector

### 2.1 Collect H+ and H− Hidden States

Using `D_steer`, run each model **10 times per question** (stochastic decoding).

- **H+** = hidden states from runs where the final numeric answer is correct
- **H−** = hidden states from runs where the final numeric answer is wrong

**Where to collect (per architecture):**

| Architecture | Collection Point |
|---|---|
| Coconut / CCoT | At each latent step `h_t` in the reasoning loop |
| CODI | At the final denoising step before answer decoding |
| TokenSkip | At the last token of the compressed text (boundary token) |

```python
H_plus  = []   # hidden states from correct runs
H_minus = []   # hidden states from incorrect runs

for question in D_steer:
    for run in range(10):
        output, hidden_state = model.run_stochastic(question)
        predicted_answer = extract_answer(output)
        if predicted_answer == question["answer"]:
            H_plus.append(hidden_state)
        else:
            H_minus.append(hidden_state)
```

### 2.2 Linear Probe Separability Test (GATE)

**Before computing any vector**, check whether H+ and H− are actually separable in the hidden space.

**Why:** If the hidden states look identical for correct and incorrect runs, there is no meaningful direction to extract — steering would be pure noise.

**How:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

X = np.array(H_plus + H_minus)
y = np.array([1]*len(H_plus) + [0]*len(H_minus))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

probe = LogisticRegression(max_iter=1000)
probe.fit(X_train, y_train)
accuracy = probe.score(X_test, y_test)
print(f"Probe accuracy at this layer: {accuracy:.3f}")
```

**Run this across all layers.** Plot probe accuracy by layer number.

- **If probe accuracy < 60%** at all layers → hidden states do not separate correctness → investigate model training before continuing.
- **If probe accuracy ≥ 65%** at some layer → proceed with vector extraction from the best layer.
- **Report:** A figure showing probe accuracy vs. layer number. This is itself a publishable result.

> **GATE CHECK:** Probe accuracy ≥ 65% at the chosen layer for each architecture.

### 2.3 Compute the Correctness Vector

Run **both** extraction methods and compare on `D_val`.

**Method A — Difference of Means (DoM):**

```python
import numpy as np

H_plus_array  = np.stack(H_plus)   # shape: (N+, d)
H_minus_array = np.stack(H_minus)  # shape: (N-, d)

v_truth = H_plus_array.mean(axis=0) - H_minus_array.mean(axis=0)
v_hat   = v_truth / np.linalg.norm(v_truth)   # unit vector
```

**Method B — Contrastive PCA (cPCA), k ∈ {1, 2, 5}:**

```python
from sklearn.decomposition import PCA

# Contrastive covariance: high variance in H+, low variance in H-
cov_plus  = np.cov(H_plus_array.T)
cov_minus = np.cov(H_minus_array.T)
C_contrast = cov_plus - cov_minus

eigenvalues, eigenvectors = np.linalg.eigh(C_contrast)
# Take top-k eigenvectors (columns of eigenvectors, sorted descending)
U_truth = eigenvectors[:, -k:]   # shape: (d, k)
```

**Select best configuration on `D_val` only.** Do not touch `D_test` yet.

---

## PHASE 3 — Steering at Inference

### 3.1 The Steering Equation

**For Coconut / CCoT (multi-step injection):**

```
h'_t = h_t + α · γ^t · v̂_truth
```

Where:
- `h_t` = hidden state at reasoning step t
- `α` = steering strength (learned/tuned on D_val)
- `γ` = decay factor (0 < γ ≤ 1), tapering intervention over steps
- `v̂_truth` = unit correctness vector

**For CODI (diffusion — specify injection timestep explicitly):**

```
h'_τ = h_τ + α · v̂_truth
```

Where `τ` is the final denoising step before answer generation. Discuss in the paper how this differs from residual stream injection.

**For TokenSkip (single injection only — no loop):**

```
h'_boundary = h_boundary + α · v̂_truth
```

γ_t is irrelevant here (t = 1 only). State this explicitly in the paper.

**For cPCA subspace (replace v̂ with subspace projection):**

```
h'_t = h_t + α · U_truth · U_truth^T · ĥ_t
```

### 3.2 Learn α on D_val

Freeze everything: base model weights, LoRA, embeddings, v̂_truth. Only learn `θ_α` (the parameter controlling α).

**Loss function (three-term):**

```
L = L_ans + λ_a · L_align + λ_m · L_mag
```

- `L_ans` = answer correctness loss (cross-entropy on final answer)
- `L_align` = cosine alignment regularizer toward v̂_truth
- `L_mag` = magnitude regularizer (prevent α from exploding)

```python
import torch
import torch.nn as nn

theta_alpha = nn.Parameter(torch.tensor(0.0))
alpha = torch.sigmoid(theta_alpha)   # constrained to (0, 1)

optimizer = torch.optim.Adam([theta_alpha], lr=1e-3)

for batch in D_val_loader:
    h_steered = h + alpha * v_hat
    
    loss_ans   = cross_entropy(model_head(h_steered), targets)
    loss_align = 1 - F.cosine_similarity(h_steered, v_hat.unsqueeze(0))
    loss_mag   = alpha ** 2   # penalize large alpha
    
    loss = loss_ans + lambda_a * loss_align.mean() + lambda_m * loss_mag
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

alpha_star = torch.sigmoid(theta_alpha).item()
```

> **Note:** `L_align` is optimized during training — do NOT report cosine alignment as a faithfulness metric in the paper. Call it "latent alignment" and acknowledge it was a training objective.

---

## PHASE 4 — Control Baselines (Run on D_val)

Run **all five conditions** before touching D_test:

| Condition | What to do | What it rules out |
|---|---|---|
| No intervention | α = 0, no steering | Establishes the floor |
| Random noise | Inject a random unit vector instead of v̂ | "Any perturbation helps" |
| Shuffled-label vector | Compute v̂ with H+/H− labels randomly swapped | "The extraction process itself creates a useful artifact" |
| Negative direction | Inject −v̂ (or −U·U^T·ĥ for cPCA) | Confirms the direction is meaningful |
| Truth vector (v̂_truth) | Your actual method | The real signal |

**How to compute the shuffled-label vector:**

```python
import random

all_states = H_plus + H_minus
labels = [1]*len(H_plus) + [0]*len(H_minus)

shuffled_labels = labels.copy()
random.shuffle(shuffled_labels)

H_plus_shuffled  = [s for s, l in zip(all_states, shuffled_labels) if l == 1]
H_minus_shuffled = [s for s, l in zip(all_states, shuffled_labels) if l == 0]

v_shuffled = np.mean(H_plus_shuffled, axis=0) - np.mean(H_minus_shuffled, axis=0)
v_shuffled /= np.linalg.norm(v_shuffled)
```

**Negative direction for cPCA subspace:**

```python
# Option 1 (recommended): Negate the projection
h_steered = h - alpha * U_truth @ U_truth.T @ h_hat

# Option 2: Shuffled subspace (same as shuffled-label vector but for cPCA)
# Compute C_contrast with shuffled labels, take top-k eigenvectors
```

---

## PHASE 5 — Flip Matrix Analysis (Run on D_val)

Do not report accuracy as a single number. Report the full flip matrix for each α value:

|  | Predicted Correct (after) | Predicted Wrong (after) |
|---|---|---|
| **Was Correct (before)** | Right → Right (good) | Right → Wrong (**bad**) |
| **Was Wrong (before)** | Wrong → Right (**target**) | Wrong → Wrong (no change) |

```python
def flip_matrix(before, after):
    """
    before, after: lists of 1s/0s (1=correct, 0=wrong)
    """
    rr = sum(b==1 and a==1 for b, a in zip(before, after))
    rw = sum(b==1 and a==0 for b, a in zip(before, after))
    wr = sum(b==0 and a==1 for b, a in zip(before, after))
    ww = sum(b==0 and a==0 for b, a in zip(before, after))
    return {"Right→Right": rr, "Right→Wrong": rw, "Wrong→Right": wr, "Wrong→Wrong": ww}
```

Report this for each architecture × each α value. This reveals whether accuracy gains at moderate α are reversed by regressions at high α.

---

## PHASE 6 — Causal Evidence: Activation Patching

**Why needed:** Showing accuracy goes up after steering is correlation. Activation patching tests causation.

**How it works:**

For each question in `D_steer`:
1. Collect one correct trace → hidden state `h+` at the boundary layer
2. Collect one incorrect trace → hidden state `h−` at the boundary layer
3. **Surgically replace** `h−` with `h+` in the incorrect trace (everything else unchanged)
4. Measure: does the model now output the correct answer?

```python
def activation_patch(model, question, h_plus, layer_idx, position_idx):
    """
    Run the model on `question` but replace the hidden state
    at (layer_idx, position_idx) with h_plus mid-forward-pass.
    Returns: model output after patching.
    """
    def hook_fn(module, input, output):
        output[0][:, position_idx, :] = h_plus
        return output

    hook = model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        output = model(question)
    hook.remove()
    return output

patch_flip_rate   = # fraction of wrong→right from exact patching
steer_flip_rate   = # fraction of wrong→right from your vector steering
```

**Report:** Compare `patch_flip_rate` vs. `steer_flip_rate`. The ratio shows how much causal information your linear vector preserves vs. the full exact patch.

---

## PHASE 7 — Statistical Significance

**Run on D_val first, then D_test.**

### Bootstrap 95% Confidence Intervals

You run the model **once** on all questions. The 1000 iterations are pure Python (milliseconds, no GPU).

```python
import numpy as np

def bootstrap_ci(correct_array, n_bootstrap=1000, confidence=0.95):
    """
    correct_array: list of 1s (correct) and 0s (wrong), one per question
    Returns: (point_estimate, lower_bound, upper_bound)
    """
    correct_array = np.array(correct_array)
    n = len(correct_array)
    point_estimate = correct_array.mean()

    boot_accuracies = []
    for _ in range(n_bootstrap):
        sample_indices = np.random.randint(0, n, size=n)
        boot_accuracies.append(correct_array[sample_indices].mean())

    boot_accuracies = np.array(boot_accuracies)
    alpha_ci = 1 - confidence
    lower = np.percentile(boot_accuracies, 100 * alpha_ci / 2)
    upper = np.percentile(boot_accuracies, 100 * (1 - alpha_ci / 2))

    return point_estimate, lower, upper


def bootstrap_ci_difference(results_a, results_b, n_bootstrap=1000):
    """
    Tests if (accuracy_b - accuracy_a) is significantly > 0.
    If the CI excludes 0, the difference is statistically significant.
    """
    results_a = np.array(results_a)
    results_b = np.array(results_b)
    n = len(results_a)

    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        diff = results_b[idx].mean() - results_a[idx].mean()
        boot_diffs.append(diff)

    boot_diffs = np.array(boot_diffs)
    lower = np.percentile(boot_diffs, 2.5)
    upper = np.percentile(boot_diffs, 97.5)
    point = results_b.mean() - results_a.mean()

    return point, lower, upper


# Example usage
unsteered = [1, 0, 1, 1, 0, ...]   # 1319 entries
steered   = [1, 0, 1, 1, 1, ...]   # same order, same questions

pt_u, lo_u, hi_u = bootstrap_ci(unsteered)
pt_s, lo_s, hi_s = bootstrap_ci(steered)

diff, lo_d, hi_d = bootstrap_ci_difference(unsteered, steered)
print(f"Unsteered: {pt_u:.3f} [{lo_u:.3f}, {hi_u:.3f}]")
print(f"Steered:   {pt_s:.3f} [{lo_s:.3f}, {hi_s:.3f}]")
print(f"Gain: {diff:+.3f} [{lo_d:+.3f}, {hi_d:+.3f}]")

if lo_d > 0:
    print("Statistically significant")
else:
    print("NOT statistically significant — overlapping intervals")
```

**Interpreting the result:**
- If the 95% CI on the difference **excludes 0** → statistically significant gain
- If the CI **includes 0** → the gain could be random chance → cannot claim improvement

---

## PHASE 8 — Final D_test Evaluation (GATE: Open Once)

Only open `D_test` **after** all of the following are locked on `D_val`:
- Best architecture per method confirmed
- Best layer and vector extraction method selected (DoM vs. cPCA, k value)
- `α*` learned and fixed
- All baselines run
- Flip matrix analyzed

**Run once. Record. Do not iterate.**

Report for each architecture:
- Accuracy with 95% bootstrap CI
- Flip matrix (all four cells)
- Token count comparison (compressed vs. full CoT)
- Inference latency

---

## PHASE 9 — Transfer Evaluation (SVAMP)

Take `v̂_truth` computed from `D_steer` (GSM8K). Apply it directly to SVAMP. Do not re-tune `α` on SVAMP.

This tests whether the correctness direction generalizes beyond the training distribution.

```python
# Load SVAMP
svamp = load_dataset("svamp")

# Apply the same v̂_truth and α* from GSM8K
# No retuning allowed
results_svamp = []
for question in svamp["test"]:
    output = steer_and_generate(model, question, v_hat=v_truth, alpha=alpha_star)
    results_svamp.append(check_answer(output, question["answer"]))

pt, lo, hi = bootstrap_ci(results_svamp)
print(f"SVAMP transfer accuracy: {pt:.3f} [{lo:.3f}, {hi:.3f}]")
```

**Both outcomes are publishable:**
- Transfer works → method is architecturally robust
- Transfer fails → direction is distribution-specific → reveals something interesting about correctness representations

---

## Naming Conventions (Critical for Paper)

| ❌ Wrong name | ✅ Correct name | Reason |
|---|---|---|
| "Truth vector" | "Correctness vector" or "successful-reasoning direction" | Your vector encodes answer correctness on GSM8K, not universal truth |
| "Faithful reasoning" | "Latent alignment" | Cosine alignment is a training objective — calling it faithfulness is circular |
| "Latent chain-of-thought" (for TokenSkip) | "Compressed text CoT" | TokenSkip has no latent loop; the reasoning is still discrete text |

---

## Master Checklist

### Phase 0 — Setup
- [ ] **[GATE]** D_train / D_steer / D_val / D_test splits locked, seeds recorded
- [ ] D_test sealed — not loaded again until Phase 8

### Phase 1 — Model Training
- [ ] Coconut trained with curriculum (3 stages)
- [ ] CCoT trained
- [ ] CODI trained
- [ ] TokenSkip compression + fine-tuning done
- [ ] **[GATE]** All architectures beat no-CoT baseline on D_val
- [ ] Training curves saved and reported

### Phase 2 — Vector Extraction
- [ ] H+/H− collected from D_steer (10 runs per question per architecture)
- [ ] **[GATE]** Linear probe accuracy ≥ 65% at chosen layer — probe accuracy by layer plotted
- [ ] DoM vector computed
- [ ] cPCA subspace computed (k ∈ {1, 2, 5})

### Phase 3 — Steering
- [ ] Steering equation implemented per architecture (multi-step for Coconut/CCoT, single-step for TokenSkip, diffusion-aware for CODI)
- [ ] α* learned on D_val (frozen everything except θ_α)
- [ ] Best config selected on D_val only

### Phase 4 — Baselines
- [ ] No intervention baseline run
- [ ] Random noise baseline run
- [ ] Shuffled-label vector computed and tested
- [ ] Negative direction tested (−v̂ or negated projection for cPCA)
- [ ] All five conditions compared on D_val

### Phase 5 — Flip Matrix
- [ ] Flip matrix reported for each architecture × each α value on D_val
- [ ] Right→Wrong regression identified and discussed

### Phase 6 — Causal Evidence
- [ ] Activation patching experiment run on D_steer
- [ ] Patch flip rate vs. steer flip rate reported

### Phase 7 — Statistics
- [ ] Bootstrap 95% CIs computed for every condition (1000 resamples)
- [ ] CI on accuracy difference computed
- [ ] Significance confirmed before reporting any "improvement"

### Phase 8 — D_test
- [ ] **[GATE]** Best config locked from D_val before opening D_test
- [ ] **[GATE]** D_test evaluated exactly once
- [ ] Accuracy, flip matrix, token count, latency — all reported with CIs

### Phase 9 — Transfer
- [ ] SVAMP evaluation run with α* from GSM8K (no retuning)
- [ ] Transfer result reported (positive or negative — both are valid)

---

## Key Equations Summary

| Equation | Formula |
|---|---|
| DoM vector | `v_truth = mean(H+) − mean(H−)` |
| Unit vector | `v̂ = v_truth / ‖v_truth‖` |
| Steering (Coconut/CCoT) | `h'_t = h_t + α · γ^t · v̂_truth` |
| Steering (TokenSkip) | `h'_boundary = h_boundary + α · v̂_truth` |
| Steering (cPCA) | `h'_t = h_t + α · U · U^T · ĥ_t` |
| Negative direction (cPCA) | `h'_t = h_t − α · U · U^T · ĥ_t` |
| Three-term loss | `L = L_ans + λ_a · L_align + λ_m · L_mag` |

---

## Timeline Estimate

| Phase | Estimated Time |
|---|---|
| Phase 0 — Data splits | 1 day |
| Phase 1 — Training all 4 architectures | 2–4 weeks (GPU-heavy) |
| Phase 2 — Vector extraction + probe | 3–5 days |
| Phase 3 — Steering implementation | 1 week |
| Phase 4–5 — Baselines + flip matrix | 3–5 days |
| Phase 6 — Activation patching | 2–3 days |
| Phase 7 — Statistics | 1 day (CPU only) |
| Phase 8 — D_test evaluation | 1 day |
| Phase 9 — Transfer (SVAMP) | 2–3 days |
| **Total** | **~8–10 weeks** |

> **Realistic target:** Given EMNLP 2026 submissions close around May–June 2026, a workshop submission while targeting main conference 2027 is the safest path. A workshop appearance at EMNLP 2026 also gives early community feedback.

---

*Pipeline prepared based on the research framework and review discussion. All phases must be completed in order. Gates are hard prerequisites — do not skip them.*
