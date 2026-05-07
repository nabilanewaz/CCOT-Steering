# CCOT-Steering

Phase 1 is implemented in the `phase1/` package and can be run directly as a module.
Phase 2 truth-vector extraction is implemented in the `phase2/` package and is
wired into the sweep runner.

Quick run (place `gsm8k/train.jsonl` in `gsm8k/`):

1. Build splits (saves JSONL folds to `configs/splits` plus `splits_meta.json`):

```bash
python scripts/build_splits.py --pool gsm8k/train.jsonl --out configs/splits
```

2. Run the full Phase 1 + Phase 2 pipeline, including offline compression,
CoT training, CCoT training, the 12-condition Phase 1 evaluation grid, and
truth-vector extraction from both sources (CCoT checkpoint and CoT checkpoint):

```bash
python -m phase1
```

Optional: run Phase 2 only for one model/config when checkpoints already exist:

```bash
python -m phase2 --model-tag llama32_3b --base-model-id meta-llama/Llama-3.2-3B --checkpoints-dir checkpoints/S1/llama32_3b --steer-jsonl configs/splits/S1_D_steer.jsonl --vectors-dir vectors/S1/llama32_3b --results-dir results/S1/llama32_3b
```

3. Compute selection from `results/` and write `configs/selected.yaml`:

```bash
python scripts/selection.py --results results
```

4. Final evaluation (only script allowed to load `gsm8k/test.jsonl`):

```bash
python evaluate_final.py
```

Guard: `utils/data.py` enforces that `gsm8k/test.jsonl` may only be loaded from `evaluate_final.py`.

Dependencies are listed in `requirements.txt`.
