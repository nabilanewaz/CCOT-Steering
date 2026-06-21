# Phase 1: Training Baselines

Phase 1 produces the fine-tuned model checkpoints that all later phases depend on. It trains two adapter families per backbone — one on full chain-of-thought (CoT) and five on compressed chain-of-thought (CCoT) at different compression ratios — then evaluates all of them on D_val to measure the accuracy–compression tradeoff before any steering is applied.

---

## Files

| File | Role |
|------|------|
| `train.py` | CoT and CCoT LoRA fine-tuning |
| `evaluate.py` | 12-condition evaluation on D_val |
| `compress.py` | Compression cache builder (offline, called by `preprocess_compress.py`) |
| `format.py` | Prompt string formatting for both dataset types |
| `inference.py` | Model loading, greedy generation, answer extraction |

---

## Prerequisites

Before Phase 1 runs, the compression cache must exist:

```
cache/{config}/compressed_R5.jsonl   # ratio 0.5
cache/{config}/compressed_R6.jsonl   # ratio 0.6
cache/{config}/compressed_R7.jsonl   # ratio 0.7
cache/{config}/compressed_R8.jsonl   # ratio 0.8
cache/{config}/compressed_R9.jsonl   # ratio 0.9
```

These are built offline by `preprocess_compress.py` using LLMLingua-2 on D_train. Each row:
```json
{"id": "gsm8k_0042", "compressed": "shortened reasoning text", "actual_ratio": 0.61}
```

---

## Stage 1: CoT Fine-Tuning (`train_cot`)

### What it does

Trains a LoRA adapter on full chain-of-thought reasoning chains.

### Input

- Frozen base model (Llama-3.2-3B, Phi-2, Qwen2.5-3B, or Qwen2.5-Math-1.5B)
- D_train examples — each has `question` and `answer` (reasoning####numeric_answer)

### Dataset format (`CoTDataset`)

```
Question: {question}

{full_reasoning_chain}

Answer: {numeric_answer}
```

The entire string is tokenized; `labels` = `input_ids` (next-token prediction over the full sequence).

### LoRA configuration

| Parameter | Value |
|-----------|-------|
| r | 16 |
| lora_alpha | 32 |
| dropout | 0.05 |
| bias | none |
| target modules (Llama/Qwen) | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| target modules (Phi-2) | q_proj, v_proj, k_proj, dense, fc1, fc2 |

### Per-backbone hyperparameters

| Model | LR | Batch | Grad accum | Eff. batch | CoT epochs |
|-------|----|-------|------------|------------|------------|
| llama32_3b | 2e-4 | 4 | 4 | 16 | 3 |
| phi2 | 1e-4 | 4 | 4 | 16 | 3 |
| qwen25_3b | 2e-4 | 4 | 4 | 16 | 3 |
| qwen25_math1.5b | 2e-4 | 8 | 2 | 16 | 1 |

Qwen2.5-Math-1.5B uses 1 epoch for CoT because math pretraining means the model converges fast. All others use 3 epochs.

LR schedule: cosine decay with 5% warmup. fp16 enabled when CUDA is available.

### Output

```
checkpoints/{config}/{model}/cot/
├── adapter_config.json
├── adapter_model.safetensors   (or adapter_model.bin)
└── training_info.json          (hyperparams + per-epoch loss log)
```

---

## Stage 2: CCoT Fine-Tuning (`train_ccot`)

### What it does

Trains five separate LoRA adapters — one per compression ratio — on compressed reasoning chains. The adapter learns to generate the abbreviated form of a reasoning chain.

### Input

- Frozen base model (same as Stage 1)
- D_train examples matched against the pre-built compression cache by `id`
- Ratios ∈ {0.5, 0.6, 0.7, 0.8, 0.9}

### Dataset format (`CCoTDataset`)

```
Question: {question}

[compress:{ratio}]
{compressed_reasoning_chain}

Answer: {numeric_answer}
```

Items in D_train that have no matching cache entry are silently dropped. If the resulting dataset is empty, a `ValueError` is raised.

### Epoch override

CCoT always uses **3 epochs regardless of model tag** (`CCOT_EPOCHS = 3`). Compressed reasoning is a harder generation target — the model needs more passes even when it would converge in 1 epoch on full CoT.

### Output

```
checkpoints/{config}/{model}/ccot_R5/    # ratio 0.5
checkpoints/{config}/{model}/ccot_R6/    # ratio 0.6
checkpoints/{config}/{model}/ccot_R7/    # ratio 0.7
checkpoints/{config}/{model}/ccot_R8/    # ratio 0.8
checkpoints/{config}/{model}/ccot_R9/    # ratio 0.9
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── training_info.json
```

---

## Phase 1 Evaluation (`run_phase1_evaluation`)

### Goal

Measure accuracy, token usage, and latency for every condition on D_val. The results inform Phase 2 (which CCoT ratio to use for hidden-state collection) and are the baseline reference for all later phases.

### 12 Conditions

| # | Condition tag | Model | Description |
|---|--------------|-------|-------------|
| 1 | `no_cot` | frozen base | Direct answer, no reasoning |
| 2 | `full_cot` | cot/ adapter | Full reasoning chain |
| 3 | `trimmed_cot_R9` | cot/ adapter | CoT chain truncated to match R=0.9 CCoT token budget |
| 4 | `trimmed_cot_R8` | cot/ adapter | Same at R=0.8 |
| 5 | `trimmed_cot_R7` | cot/ adapter | Same at R=0.7 |
| 6 | `trimmed_cot_R6` | cot/ adapter | Same at R=0.6 |
| 7 | `trimmed_cot_R5` | cot/ adapter | Same at R=0.5 |
| 8 | `ccot_R9` | ccot_R9/ adapter | CCoT at ratio 0.9 |
| 9 | `ccot_R8` | ccot_R8/ adapter | CCoT at ratio 0.8 |
| 10 | `ccot_R7` | ccot_R7/ adapter | CCoT at ratio 0.7 |
| 11 | `ccot_R6` | ccot_R6/ adapter | CCoT at ratio 0.6 |
| 12 | `ccot_R5` | ccot_R5/ adapter | CCoT at ratio 0.5 |

### Why trimmed CoT?

Trimmed CoT uses the CoT adapter but stops generating at the same token budget the corresponding CCoT model would use. This creates a matched comparison: if CCoT is more accurate than trimmed CoT at the same length, the model actually learned from compressed reasoning — not just from generating fewer tokens. This is the **mechanism gain** check.

### Token budget computation

For each example in D_val, the budget for ratio r is:
```
budget_i = max(1, round(r × full_cot_tokens_i))
```
This is computed per-example from the CoT model's actual output length, so the comparison is tight.

### Metrics per condition (`ConditionMetrics`)

| Field | Meaning |
|-------|---------|
| `accuracy` | Fraction of correct answers (normalized comparison) |
| `reasoning_tokens` | Mean tokens in the reasoning span |
| `actual_ratio` | `reasoning_tokens / full_cot_mean_tokens` |
| `latency_sec` | Mean wall-clock seconds per example |
| `answer_found_rate` | Fraction where an answer string was extracted |

Answer normalization strips whitespace, converts to lowercase, and handles numeric equivalence (e.g. "4.0" == "4").

### Output

```
results/{config}/{model}/phase1_val.json
```

A JSON list of 12 `ConditionMetrics` records. Example:
```json
[
  {"condition": "no_cot",          "accuracy": 0.412, "reasoning_tokens": 0.0,  ...},
  {"condition": "full_cot",        "accuracy": 0.781, "reasoning_tokens": 187.3,...},
  {"condition": "trimmed_cot_R7",  "accuracy": 0.743, "reasoning_tokens": 131.2,...},
  {"condition": "ccot_R7",         "accuracy": 0.762, "reasoning_tokens": 129.8,...},
  ...
]
```

### Mechanism gain table

Printed after evaluation (and computed by `print_comparison_table`):

```
R=0.7: CCoT=0.762  Trimmed=0.743  Gain=+0.019  -> CCoT better
R=0.6: CCoT=0.751  Trimmed=0.739  Gain=+0.012  -> CCoT better
```

A positive gain means the CCoT adapter genuinely learned the compressed reasoning task, not just shorter outputs.

---

## Checkpoint Resume

Every step checks if the output directory already contains `adapter_config.json` before training. Safe to re-run after any interruption:

```python
if not _done(os.path.join(cot_out, 'adapter_config.json')):
    train_cot(...)
else:
    print("CoT checkpoint exists — skipping")
```

Phase 1 evaluation is similarly guarded by `phase1_val.json`.

---

## Trigger

Phase 1 is invoked via the master pipeline:

```bash
python pipeline.py --phase 1
python pipeline.py --phase 1 --model qwen25_math1.5b --config S2
```

Or directly for development:

```python
from phase1.train import train_cot, train_ccot
from phase1.evaluate import run_phase1_evaluation
```

---

## What Phase 2 Uses

`phase2/run.py::pick_best_ccot_ratio` reads `phase1_val.json` and selects the ratio with the highest mechanism gain (CCoT acc − Trimmed CoT acc). This becomes the CCoT checkpoint used for hidden state collection in Phase 2.
