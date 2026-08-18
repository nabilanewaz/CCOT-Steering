# GSM8K Dataset Configuration Change

## Purpose

The experiment is now fixed to exactly 300 unique GSM8K examples for each data
role used by the pipeline: training, steering-vector extraction, validation, and
final test evaluation. The full downloaded GSM8K files remain unchanged; the
pipeline selects deterministic subsets at runtime.

## Previous And Current Configuration

| Data role | Pipeline use | Previous effective configuration | Current configuration |
|---|---|---:|---:|
| `D_train` | Phase 1 Coconut training | 60% of 7,473 train rows = 4,484 | 300 |
| `D_steer` | Phase 2 hidden-state collection | 20% of 7,473 train rows = 1,495 | 300 |
| `D_val` | Phase 1 validation and Phase 3 tuning/evaluation | Remaining train rows = 1,494 | 300 |
| `D_test` | Phase 4 locked final evaluation | Full official test split = 1,319 | 300 |

The previous `D_val` count was 1,494 in the running code because Python's
`round()` produced 4,484 training examples and 1,495 steering examples, leaving
1,494 rows. Some old fallback metadata incorrectly listed `D_val` as 1,495; that
fallback has also been corrected to 300.

## Current Selection Rules

- Master setting: `samples_per_phase: 300` in `configs/protocol.yaml`.
- Seed: `42`, unchanged from the previous protocol.
- `D_train`, `D_steer`, and `D_val` are built from the official GSM8K train
  pool after one seed-42 shuffle.
- The shuffled positions are `0:300` for `D_train`, `300:600` for `D_steer`,
  and `600:900` for `D_val`.
- The three train-pool roles are disjoint and require at least 900 source rows.
- `D_test` is selected independently from the official GSM8K test file by a
  seed-42 shuffle followed by the first 300 rows.
- The official test file remains guarded: only `evaluate_final.py` may load it
  for model evaluation.
- Isolation checks compare normalized question text and validate the same
  seeded 300-example test slice used by Phase 4.

## Phase-Level Effect

| Phase | Dataset boundary | Previous | Current |
|---|---|---:|---:|
| Preprocessing | Phase 1 cache input (`D_train`) | 4,484 | 300 |
| Phase 1 training | Unique `D_train` examples received | 4,484 | 300 |
| Phase 1 evaluation | Unique `D_val` examples | 1,494 | 300 |
| Phase 2 | Unique `D_steer` questions | 1,495 | 300 |
| Phase 2 per source/model | Rollouts (`questions x 20`) | 29,900 | 6,000 |
| Phase 3 main evaluation | Unique `D_val` examples | 1,494 | 300 |
| Phase 3 lambda sweep | Validation examples | 200 | 300 |
| Phase 3 generated-answer alpha validation | Validation examples | 100 | 300 |
| Phase 3 diagnostic alpha sweep | Validation examples | 50 | 300 |
| Phase 4 main evaluation | Unique `D_test` examples | 1,319 | 300 |
| Phase 4 diagnostic alpha sweep | Test examples | 100 | 300 |

Phase 1 now optimizes on all 300 `D_train` examples. Curriculum monitoring uses
the separate, disjoint 300-example `D_val`; no rows are removed from `D_train`
and no extra GSM8K examples are introduced.

## Settings Unchanged By The Dataset Reduction

The dataset-size change does not alter the remaining protocol settings:

- Split/config identifier: `S2`
- Compression ratios: `0.5, 0.6, 0.7, 0.8, 0.9`
- LoRA settings: rank `16`, alpha `32`, dropout `0.05`
- Protocol training settings: 3 epochs, batch size `4`, gradient accumulation
  `4`, learning rate `2e-4`, maximum sequence length `512`
- Phase 2 rollouts: `20` per steering question
- Phase 2 temperature: `1.0`
- cPCA defaults: beta `0.5`, 3 components per layer, 10 final components
- Phase 3 alpha maximum/init/learning rate: `50.0 / 1.0 / 0.05`
- Phase 3 regularizers: `lambda_a=0.1`, `lambda_m=0.01`
- Phase 3 early-stopping fraction: `0.1`

Phase 1 was subsequently aligned with the Coconut mechanism and curriculum,
with its run length reduced to 30 epochs at learning rate `1e-4` and effective batch size 128. Those
method changes are documented in `PHASE1_COCONUT_PAPER_AUDIT.md`; they are
separate from the dataset reduction. The Qwen2.5-Math Phase 2 minimum H+/H-
sample threshold remains 300.

## Enforcement And Resume Safety

- Phase 1 training and evaluation reject lists that are not exactly 300 rows.
- Phase 2 extraction and Phase 3 evaluation do the same at their public entry
  points; their standalone JSONL loaders select exactly 300 rows.
- Phase 4 asserts that the guarded test loader returned exactly 300 rows.
- Phase 1 compatibility caches are reused only when their ordered IDs match
  the current 300-row `D_train`.
- Phase 1 training metadata now records `n_phase_examples`.
- Phase 2 metadata and hidden-state cache metadata now record `n_steer` and the
  rollout count.
- Phase 3 run, lambda-sweep, and alpha-validation metadata now record `n_val`.
- Phase 4 result provenance records `n_test`.
- Pipeline readiness checks reject old Phase 2 or Phase 3 artifacts whose
  counts do not match the current 300-example inputs.
- Phase 1 always delegates resume decisions to its metadata-aware trainer, so
  full-data checkpoints cannot be accepted only because checkpoint files exist.

Existing artifacts produced with the old counts should not be manually copied
into a new 300-example run. The pipeline will treat count-aware artifacts as
stale and regenerate them.
