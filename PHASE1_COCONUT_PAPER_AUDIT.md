# Phase 1 Coconut Paper Audit

## Scope

This audit compares Phase 1 with the local paper
[`Coconut (Chain of Continuous Thought).pdf`](Coconut%20(Chain%20of%20Continuous%20Thought).pdf)
and the authors' public
[Coconut implementation](https://github.com/facebookresearch/coconut).
The paper is the authority for the requested training schedule. The public code
is used to verify implementation details such as dataset construction, recurrent
hidden-state substitution, optimizer resets, and special-token initialization.

The experiment remains a reduced-cost steering study, not an exact reproduction
of the paper. The corrections below make the Coconut mechanism and curriculum
faithful within the fixed 300-example protocol.

## Base Coconut Method

For a sequence containing a latent placeholder, Coconut does not process that
placeholder as an ordinary vocabulary embedding. It takes the previous
transformer hidden state and feeds it directly as the next input embedding. With
multiple continuous thoughts this creates recurrent transformer passes between
the start- and end-latent markers.

For GSM8K, the paper uses curriculum parameter `c=2`:

1. Stage 0 trains ordinary, fully visible chain-of-thought.
2. At Stage `k`, the first `k` textual reasoning steps are removed and
   replaced by `k * c` continuous thoughts.
3. GSM8K uses three latent curriculum stages, producing 2, 4, and 6 continuous
   thoughts.
4. A final stage retains six continuous thoughts and removes every remaining
   textual reasoning step, so the supervised target is the answer.
5. The paper runs the initial stage for 6 epochs, each latent stage for 3 epochs,
   and the final stage through epoch 50. This experiment stops at epoch 30.
6. The optimizer is reset at every curriculum transition. The paper reports
   learning rate `1e-4` and effective batch size `128`.

## Previous Versus Corrected Configuration

| Area | Base paper/reference | Before this audit | Corrected implementation |
|---|---|---|---|
| Stage 0 data | Question + all visible reasoning steps + answer | Kept the first reasoning step as a permanent skeleton, inserted latent boundary markers, and masked the skeleton from loss | Full visible CoT; no latent markers; question masked, all reasoning and answer tokens supervised |
| Latent curriculum | Replace the first `k` reasoning steps | Preserved step 1 and started replacement from step 2 | Removes steps from the front exactly: Stage 1 removes step 1, Stage 2 removes steps 1-2, Stage 3 removes steps 1-3 |
| Continuous thoughts | `c=2`; 2, 4, then 6 thoughts | `c=2`, but counts were tied to the shortened post-skeleton chain | Exactly 2, 4, and 6 latent placeholders for Stages 1-3 |
| Final stage | Six thoughts, no textual reasoning | Permanent first-step skeleton remained visible | Six recurrent thoughts and answer only |
| Epoch schedule | 6 + 3 + 3 + 3, then final stage through epoch 50 | Hard cap of 20 epochs, with early-stopping machinery | Fixed 30 epochs; transitions at epochs 0, 6, 9, 12, and 15; no early stop |
| Learning rate | `1e-4` | Usually `1e-5`; later stages used half-rate, and Qwen-Math used `3e-6` | Constant `1e-4` in every stage |
| Effective batch | 128 | 64 for most backbones and 128 for Qwen-Math | 128 for every backbone through gradient accumulation |
| Optimizer transitions | Reset at every stage | Reset at transitions, but also recreated a warmup/decay scheduler | AdamW reset at every transition; no scheduler |
| Partial accumulation | Apply all examples | The final incomplete accumulation group in each epoch was discarded | Final partial group is rescaled and stepped |
| Train/validation data | Separate train and validation sets | Split the 300-row `D_train` internally into 270 train and 30 monitor rows | Optimizes on all 300 `D_train` rows and validates on all 300 disjoint `D_val` rows |
| Stage validation | Evaluate the current curriculum behavior | Every epoch was evaluated with the final six-latent prompt | Validation prompt now matches the active stage |
| CoT checkpoint | Stage 0 CoT model | `cot/` was an alias of the same latent checkpoint as every CCoT condition | Best Stage 0 checkpoint is exported to `cot/` |
| CCoT checkpoint | Fully latent model | A Stage 3 partially verbal checkpoint could be selected | Only fully latent Stage 4 checkpoints are eligible |
| Saved-model inference | Reconstruct recurrent hidden-state feedback | Reloaded as plain `AutoModelForCausalLM`; `<|latent|>` was treated as a normal token | Metadata reconstructs the `Coconut` wrapper in Phases 1-4 |
| Answer generation | Continue from substituted input embeddings | First answer token used recurrence, then generation restarted from literal token IDs | Every generated token continues from the substituted embedding sequence |
| New-token initialization | Public code copies the embedding for `<<` | Copied the embedding for `The` | Copies `<<` as in the reference implementation |
| Checkpoint reuse | Reuse only method-compatible artifacts | Old count-aware artifacts could still satisfy the Phase 1 check | Curriculum version, schedule, counts, stages, LR, batch size, and CoT checkpoint are all required |

## Corrected Phase 1 Flow

| Epochs | Stage | Training sequence after the question | Supervised targets |
|---:|---:|---|---|
| 1-6 | 0 | All textual reasoning steps, then answer | All reasoning steps and answer |
| 7-9 | 1 | 2 continuous thoughts, textual steps 2 onward, then answer | Remaining reasoning and answer |
| 10-12 | 2 | 4 continuous thoughts, textual steps 3 onward, then answer | Remaining reasoning and answer |
| 13-15 | 3 | 6 continuous thoughts, textual steps 4 onward, then answer | Remaining reasoning and answer |
| 16-30 | 4 | 6 continuous thoughts, then answer | Answer only |

At each latent position, the preceding last hidden state replaces the placeholder
embedding. Gradients remain attached across latent passes during training.
Question tokens, latent boundary tokens, and continuous-thought positions are
masked from cross-entropy loss.

After training:

- `_coconut_phase1_cot_best/` supplies the visible-CoT `cot/` alias.
- `_coconut_phase1_best/` supplies all `ccot_L3/`, `ccot_L4/`, and
  `ccot_L6/` compatibility aliases.
- Coconut metadata causes every CCoT alias to reload with recurrent latent
  execution.
- Phase 1 evaluation chooses the best downstream latent budget on `D_val`.
- Phases 2-4 use that locked budget while retaining recurrent Coconut execution.

## Intentional Deviations From The Paper

These are experiment-design choices and were not changed:

| Paper setup | This experiment | Reason |
|---|---|---|
| 50 training epochs | 30 training epochs | Reduced runtime requested for the experiment |
| GPT-2 backbones | Qwen2.5, Qwen2.5-Math, Llama 3.2, and Phi-2 | The research compares steering across the selected model families |
| Large synthetic GSM8K training set described in the paper | Deterministic 300-example subset of the official GSM8K train split | Required cost cap |
| Paper validation/test sizes | `D_val=300` and sealed `D_test=300` | Exact 300-example role protocol |
| Distributed effective batch 128 | Per-device batch 1 with gradient accumulation 128 | Fits the available execution setup |
| Paper's final six-thought GSM8K setting | Downstream evaluation also tests 3 and 4 latent tokens | This is the steering experiment's latent-budget ablation |
| Exact paper reproduction objective | Method-faithful Phase 1 inside a steering pipeline | The downstream truth-vector and intervention phases are new research |

Because of the smaller dataset and different backbones, matching the paper's
reported accuracy is not expected. The valid claim is that Phase 1 now follows
the paper's Coconut mechanism and curriculum, subject to the explicit
experimental deviations above.

## Pre-Run Gates

Before a final run, the pipeline now requires:

- exactly 300 rows in `D_train` and exactly 300 rows in `D_val`;
- all 30 Phase 1 epochs completed;
- Stage 0 CoT and Stage 4 fully latent checkpoints present;
- recurrent Coconut metadata on CCoT checkpoints;
- a recorded effective batch size of 128, learning rate `1e-4`, `c=2`, and
  six final latent tokens;
- stale Phase 1 evaluation artifacts removed after retraining.

The focused offline checks are in
[`tests/test_phase1_coconut.py`](tests/test_phase1_coconut.py). They verify
stage construction, epoch transitions, final-stage masking, and that generation
continues from substituted latent embeddings instead of literal latent-token
IDs.
