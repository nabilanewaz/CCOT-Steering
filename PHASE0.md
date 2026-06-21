# Phase 0: Data Setup

Phase 0 is run **once** before any training begins. It downloads the dataset, verifies that no data leaks exist between splits and the test set, and builds the offline compression cache that Phase 1 Stage 2 (CCoT training) requires.

---

## Files

| File | Role |
|------|------|
| `download_dataset.py` | Download GSM8K / SVAMP / ProntoQA from HuggingFace |
| `verify_isolation.py` | Assert D_train/D_steer/D_val ∩ D_test = ∅ for all split configs |
| `preprocess_compress.py` | Compress D_train reasoning traces at all ratios → JSONL cache |
| `phase1/compress.py` | Compression utilities (called by both `preprocess_compress.py` and `pipeline.py`) |
| `scripts/build_splits.py` | Build S1–S4 in-memory splits from the train pool |
| `utils/dataset_paths.py` | Dataset selection logic (env var → CLI → persisted file → default) |

---

## Step 0.1: Download Dataset

```bash
python download_dataset.py --dataset gsm8k
```

Downloads the dataset from HuggingFace and writes two files:

```
gsm8k/train.jsonl    # official train split (used as pool for D_train + D_steer + D_val)
gsm8k/test.jsonl     # official test split  (NEVER opened until Phase 4)
```

Each line is a JSON object:
```json
{"id": "gsm8k_0", "question": "Natalia sold ...", "answer": "Natalia sold 48/2 = <<48/2=24>>24 ...\n####72"}
```

The `answer` field has the format `{reasoning_chain}\n####\n{numeric_answer}`. All answer extraction throughout the pipeline splits on `####` to get the numeric answer.

### Supported datasets

| ID | Source | Task |
|----|--------|------|
| `gsm8k` | Grade School Math 8K (Cobbe et al.) | Math word problems |
| `svamp` | SVAMP (Patel et al.) | Challenge math word problems |
| `prontoqa` | ProntoQA | Synthetic logical reasoning (True/False) |

All three use the same JSONL format and the same pipeline. GSM8K is the primary dataset; SVAMP is used for Phase 5 transfer evaluation.

---

## Step 0.2: Dataset Selection

The active dataset is tracked by `utils/dataset_paths.py`. Resolution order (highest to lowest priority):

1. Environment variable `CCOT_DATASET=gsm8k`
2. CLI flag `--dataset gsm8k` (on supported entry points)
3. Persisted file `configs/active_dataset.txt`
4. Interactive prompt (if stdout is a TTY)
5. Default: `gsm8k`

Once set interactively, the choice is written to `configs/active_dataset.txt` and reused in subsequent runs without prompting.

Key functions:
- `init_project_dataset(cli_dataset, interactive)` — call at the start of any entry point
- `get_train_pool_path()` → `"{dataset_id}/train.jsonl"`
- `get_test_path()` → `"{dataset_id}/test.jsonl"`
- `phase4_subprocess_env()` → `os.environ` copy with `CCOT_DATASET` set (used when `pipeline.py` spawns `evaluate_final.py` as a subprocess)

---

## Step 0.3: Build Splits (`scripts/build_splits.py`)

`build_all_splits(pool_path, seed=42)` reads `train.jsonl`, shuffles with a fixed seed, and produces four split configurations:

| Config | D_train | D_steer | D_val | Note |
|--------|---------|---------|-------|------|
| S1 | 70% | 10% | 20% | More training data |
| S2 | 60% | 10% | 30% | Balanced (default) |
| S3 | 60% | 20% | 20% | More steering data |
| S4 | 50% | 20% | 30% | More evaluation data |

For GSM8K (7473 train examples):

| Config | D_train | D_steer | D_val |
|--------|---------|---------|-------|
| S1 | ~5231 | ~747 | ~1495 |
| S2 | ~4484 | ~747 | ~2242 |
| S3 | ~4484 | ~1495 | ~1494 |
| S4 | ~3737 | ~1495 | ~2241 |

**All four configs are built from the same shuffled pool** with the same seed=42. The winning config (S2 by default) is selected at the end of Phase 3 based on D_val performance, and locked in `configs/selected.yaml`.

### Disjointness guarantee

D_train and D_steer are always disjoint slices of the pool (train = `pool[:n_tr]`, steer = `pool[n_tr : n_tr + n_st]`, val = remainder). D_test comes from a separate file and is never part of the pool.

---

## Step 0.4: Verify Isolation (`verify_isolation.py`)

```bash
python verify_isolation.py
```

Checks two invariants for all four split configs:

1. **No D_test leakage**: `D_train ∩ D_test = ∅`, `D_steer ∩ D_test = ∅`, `D_val ∩ D_test = ∅`
2. **No intra-split leakage**: `D_train ∩ D_steer = ∅`

Comparison is by `id` field. Exits with code 1 and a clear message if any check fails:
```
ISOLATION CHECK FAILED:
  ✗ LEAKAGE: S2/D_train overlaps D_test on 3 id(s)
```

On success:
```
All isolation checks passed. (4 configs, 29892 total split examples, 1319 test ids checked)
```

**Run this before Phase 2 and before Phase 4.** Phase 2 uses D_steer; Phase 4 uses D_test. Leakage at either point invalidates the experiment.

---

## Step 0.5: Build Compression Cache (`preprocess_compress.py`)

```bash
python preprocess_compress.py              # S2 config only (default)
python preprocess_compress.py --all        # all four split configs
python preprocess_compress.py --config S1  # specific config
```

### What it does

For each compression ratio r ∈ {0.5, 0.6, 0.7, 0.8, 0.9}:
1. Iterate over every example in D_train
2. Extract the reasoning chain: `item['answer'].split('####')[0].strip()`
3. Compress with LLMLingua-2 (`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`)
4. Write a JSONL record

Output format per line:
```json
{
  "id":             "gsm8k_42",
  "compressed":     "shortened reasoning text",
  "actual_ratio":   0.61,
  "target_ratio":   0.6,
  "original_len":   87,
  "compressed_len": 53
}
```

### LLMLingua-2 compression settings

```python
compressor.compress_prompt(
    reasoning,
    rate=ratio,
    force_tokens=["\n", "."],    # never drop newlines or sentence-ending periods
    drop_consecutive=True,       # clean up redundant whitespace
)
```

`force_tokens=["\n", "."]` preserves the logical structure of the reasoning chain — line breaks and sentence endings are the skeleton of step-by-step math reasoning and must not be dropped.

### Output files

```
cache/S2/
├── compressed_R5.jsonl   # ratio 0.5 (~50% of original tokens)
├── compressed_R6.jsonl   # ratio 0.6
├── compressed_R7.jsonl   # ratio 0.7
├── compressed_R8.jsonl   # ratio 0.8
└── compressed_R9.jsonl   # ratio 0.9
```

### Resume behavior

Each ratio is checked independently. If `compressed_R6.jsonl` already exists, it is skipped:
```
Cache R6 exists (cache/S2/compressed_R6.jsonl) — skipping
```

Safe to re-run after interruption — only missing ratios are computed.

### Why offline?

LLMLingua-2 is a ~560M parameter XLM-RoBERTa model. Running it at training time for every example on every epoch would add hours per training run. The offline cache pays the cost once and stores the result as simple JSONL, which Phase 1 Stage 2 reads in milliseconds.

### Time estimate

~20–40 minutes on GPU for all 5 ratios on ~4500 examples (S2 D_train). CPU is ~3–5× slower.

---

## Execution Order

```bash
# 1. Download dataset
python download_dataset.py --dataset gsm8k

# 2. Verify isolation
python verify_isolation.py

# 3. Build compression cache (GPU recommended)
python preprocess_compress.py

# 4. Confirm files exist
ls gsm8k/
# train.jsonl  test.jsonl

ls cache/S2/
# compressed_R5.jsonl  compressed_R6.jsonl  ...  compressed_R9.jsonl

# Now Phase 1 can run:
python pipeline.py --phase 1 --dataset gsm8k
```

---

## Directory State After Phase 0

```
gsm8k/
├── train.jsonl              # 7473 examples (pool for all splits)
└── test.jsonl               # 1319 examples (locked until Phase 4)

cache/S2/
├── compressed_R5.jsonl      # 4484 records (D_train size for S2)
├── compressed_R6.jsonl
├── compressed_R7.jsonl
├── compressed_R8.jsonl
└── compressed_R9.jsonl

configs/
└── active_dataset.txt       # "gsm8k"
```

---

## Data Isolation Invariant (enforced throughout)

D_test is never opened by any script except `evaluate_final.py`. This is enforced at two levels:

1. **`verify_isolation.py`** checks before the run that no IDs overlap
2. **`utils/data.py::load_test_set()`** raises `RuntimeError` if called from outside `evaluate_final.py`:
   ```python
   # utils/data.py
   caller = inspect.stack()[1].filename
   if not caller.endswith('evaluate_final.py'):
       raise RuntimeError("D_test may only be opened from evaluate_final.py")
   ```

Any accidental import of `load_test_set` in another module will fail immediately, not silently.
