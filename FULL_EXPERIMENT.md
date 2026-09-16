# Full experiment execution contract

The full experiment uses S3 with seed 42 and a 60/10/30 allocation of the training pool. On GSM8K this is **4,484 training, 747 steering, and 2,242 validation examples**, plus the separate **1,319-example test set**. No main evaluation condition is capped to 300 examples. Split manifests record ordered dataset fingerprints, and isolation checks compare normalized question text.

## Phase 1 compatibility

The `phase1/` implementation is unchanged. Its curriculum, optimizer, special-token initialization, checkpoint export, and latent-budget evaluation remain intact. Shared data loading now supplies the full training and validation allocations. Phase 2 consumes its selected `ccot_L3`, `ccot_L4`, or `ccot_L6` checkpoint and `phase1_best_latent.json`; it does not substitute the historical ratio-conditioned architecture described in parts of PHASE2.md/PHASE3.md.

Use Phase 1 checkpoints trained on the matching seeded training allocation. Historical 300-example runs are not full-data training runs. Existing Phase 1 validation selections must also be regenerated on the new validation allocation before extraction. Do not copy old S2 artifacts into S3 and treat them as a new run.

## Independent SVAMP experiment

SVAMP is also supported through all four phases with the unchanged Phase 1 implementation. Its existing 700-example training pool is divided into **420 training / 70 steering / 210 validation** examples, with **all 300 test examples** reserved for final evaluation. The 60/10/30 percentages apply to the training pool, not the combined train and test sets. The local files pass the normalized-question isolation audit.

SVAMP uses separate artifacts:

| Artifact | SVAMP location |
|---|---|
| Split files and manifest | `configs/svamp/splits/` |
| Checkpoints | `checkpoints/svamp/S3/<model>/` |
| Vectors | `vectors/svamp/S3/<model>/` |
| Validation results | `results/svamp/S3/<model>/` |
| Locked selection | `configs/svamp/selected.yaml` |
| Final test results | `results/svamp/final/` |
| Compatibility cache | `cache/svamp/S3/` |

GSM8K retains its existing artifact locations. Explicit `--dataset` takes precedence over `CCOT_DATASET`. Every main SVAMP validation condition uses all 210 examples. The prescribed probe gate and minimum class counts are unchanged; with only 70 steering questions, a run may fail the contrastive-sample or probe gate, which is reported rather than silently weakening the protocol.

```bash
# Full independent experiment on all three Qwen backbones:
bash run_svamp_all_phases.sh --with-iti --source-b

# Or choose one backbone:
MODEL=qwen25_math1.5b bash run_svamp_all_phases.sh --with-iti --source-b

# Continue individual phases using SVAMP artifacts:
python pipeline.py --phase 2 --dataset svamp --config S3 --model qwen25_math1.5b --with-iti --source-b
python pipeline.py --phase 3 --dataset svamp --config S3 --model qwen25_math1.5b --with-iti --source-b
python pipeline.py --phase 4 --dataset svamp --config S3 --model qwen25_math1.5b

# Separate frozen GSM8K-to-SVAMP transfer experiment:
python evaluate_final.py --dataset svamp --training-dataset gsm8k --model qwen25_math1.5b
```

`evaluate_final.py --dataset svamp` defaults to independently trained SVAMP artifacts. The explicit `--training-dataset gsm8k` flag selects transfer artifacts, and `pipeline.py --phase 5` supplies that flag automatically. Independent and transfer results have separate output directories and cannot be merged into one summary.

## Phase 2

The frozen selected checkpoint samples ten rollouts per steering question. Default extraction averages the first 20 generated-token states, rejects rollouts shorter than three tokens, balances classes within difficulty buckets, and requires 200 examples per class. Logistic and MLP probes score layers; no score above 0.55 stops extraction.

Outputs include best-layer DoM, top-three layer-local DoM vectors, shuffled DoM, cPCA parameter sweeps and weighted merged subspaces, shuffled cPCA, and diagnostics. A separate 20% activation holdout is reserved for the final DoM/cPCA comparison. These are activation-level splits, not question-grouped probe holdouts. The math backbone uses shrunk cPCA, rank 8 and threshold multiplier 0.4; other registered Qwen backbones use full cPCA, rank 10 and multiplier 0.5. If no cPCA subspace survives, DoM remains available and cPCA conditions are omitted explicitly in metadata.

CoT-source extraction and ITI head extraction remain opt-in as specified. ITI collects boundary head outputs before the attention output projection, then saves top-head directions and projected standard deviations.

## Phase 3

Every enabled condition evaluates all 2,242 validation examples. The grid includes direct-answer, full CoT, token-matched trimmed CoT, Coconut, noise, DoM, multi-layer DoM, multi-layer plus MLP DoM, shuffled and negative controls, available cPCA conditions, and ITI when extracted. CoT-source vectors also enable the trimmed-CoT plus DoM condition.

Prompts contain the question and model-format markers, without supplied gold reasoning. Residual interventions use local norm scaling, `alpha * ||h|| / sqrt(d)`, with no step decay. ITI uses its saved per-head projected standard deviation. Interventions affect generated-token positions, excluding the prompt and latent prefill loop. Position tracking supports both cached token decoding and Coconut's uncached generation; `seq_len == 1` alone is insufficient for Coconut.

The prescribed tuning subsets are retained: up to 200 validation examples for the lambda grid, 50 for gradient alpha tuning with a 90/10 tuning/early-stopping split, and 50 for diagnostics. These subsets tune parameters; they do not reduce the main condition grid. ITI selects its own alpha on validation examples. The best eligible steering method is selected using Wilson lower bound, then accuracy and flip rate; controls are excluded.

Per-condition outputs include generated text, predictions, correctness, question hashes, token counts, and timing. Checkpoint, vector, protocol, and dataset identities guard resume caches. Legacy Phase 2/3 artifacts must be regenerated.

## Final evaluation

Phase 4 uses frozen validation-selected settings on the entire test set, including the available controls for the selected vector source. It performs no alpha sweep or selection on test data. Final files contain per-example outputs, bootstrap confidence intervals, flip matrices, and dataset provenance. Sequential model evaluations preserve other models in the summary; repeating an already evaluated model requires `evaluate_final.py --overwrite`. Legacy final results require a new output directory.

SVAMP transfer reuses frozen GSM8K checkpoints/vectors and evaluates its separate 300-example test partition in `results/final_svamp_transfer`. It does not tune on SVAMP test examples.

## Commands

Run from the repository root in an environment with `requirements.txt` installed:

```bash
python -m scripts.build_splits --dataset gsm8k
python verify_isolation.py --dataset gsm8k

# Full run, including unchanged Phase 1, for one backbone:
python pipeline.py --phase 0 --dataset gsm8k --config S3 --model qwen25_math1.5b --device cuda --with-iti --source-b

# Continue after compatible Phase 1 training and validation:
python pipeline.py --phase 2 --dataset gsm8k --config S3 --model qwen25_math1.5b --device cuda --with-iti --source-b
python pipeline.py --phase 3 --dataset gsm8k --config S3 --model qwen25_math1.5b --device cuda --with-iti --source-b
python pipeline.py --phase 4 --dataset gsm8k --config S3 --model qwen25_math1.5b
python pipeline.py --phase 5 --model qwen25_math1.5b
```

Omit `--with-iti` or `--source-b` to use the documented core defaults. `--model all` uses Qwen2.5-0.5B, Qwen2.5-3B, and Qwen2.5-Math-1.5B. Legacy Llama/Phi identifiers remain accepted explicitly. Run all model jobs sequentially in this checkout because selection files are shared.

## Verification boundary

Local checks cover full-data counts/isolation, extraction windows, balancing, probe gates, per-layer vectors, head scales, steering positions, bfloat16 hooks, hook cleanup, real Coconut forward/generation behavior, nonzero alpha gradients, and method selection. These are code-level regression tests, not a reduced experimental result. This checkout has no `checkpoints/` or `vectors/` directory, so full model extraction, training, and evaluation have not been executed here. Experimental accuracy and runtime remain unmeasured for this implementation.
