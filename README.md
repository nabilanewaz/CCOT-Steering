# CCOT-Steering

Sweep utilities added: `utils/data.py`, `scripts/build_splits.py`, `scripts/run_sweep.py`, `scripts/selection.py`, and `evaluate_final.py`.

Quick run (place `gsm8k/train.jsonl` in `gsm8k/`):

1. Build splits (saves JSONL folds to `configs/splits`):

```bash
python scripts/build_splits.py --pool gsm8k/train.jsonl --out configs/splits
```

2. Run the sweep orchestrator (this is a scaffold; replace placeholders with actual training/eval steps):

```bash
python scripts/run_sweep.py
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
