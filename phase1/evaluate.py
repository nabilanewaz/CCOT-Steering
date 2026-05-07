import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch

from phase1.inference import (
    load_base_frozen,
    load_finetuned,
    run_no_cot,
    run_cot,
    run_ccot,
    run_trimmed_cot,
    compute_per_example_budgets,
    normalize_answer,
)

RATIOS = [0.9, 0.8, 0.7, 0.6, 0.5]


@dataclass
class ConditionMetrics:
    condition:         str
    model_tag:         str
    ratio:             Optional[float]
    accuracy:          float
    reasoning_tokens:  float   # mean tokens in reasoning span
    actual_ratio:      float   # mean_tokens / full_cot_mean_tokens
    latency_sec:       float   # mean wall-clock seconds per example
    answer_found_rate: float   # fraction where an answer string was extracted


def evaluate_condition(
    model,
    tokenizer,
    dataset: list,
    device: str,
    condition_name: str,
    model_tag: str,
    ratio: float = None,
    is_trimmed: bool = False,
    budgets: list = None,
    full_cot_mean_tokens: float = None,
) -> ConditionMetrics:
    correct = 0
    found = 0
    reasoning_lengths = []
    latencies = []

    for i, item in enumerate(dataset):
        t0 = time.time()

        if condition_name == 'no_cot':
            pred, reasoning = run_no_cot(model, tokenizer, item, device)
        elif is_trimmed:
            budget = budgets[i] if budgets else max(10, int((ratio or 1.0) * 100))
            pred, reasoning = run_trimmed_cot(model, tokenizer, item, budget, device)
        elif ratio is not None:
            pred, reasoning = run_ccot(model, tokenizer, item, ratio, device)
        else:
            pred, reasoning = run_cot(model, tokenizer, item, device)

        latencies.append(time.time() - t0)

        if pred is not None:
            found += 1
            gold = item['answer'].split('####')[1].strip()
            if normalize_answer(pred) == normalize_answer(gold):
                correct += 1

        r_ids = tokenizer.encode(reasoning or '', add_special_tokens=False)
        reasoning_lengths.append(len(r_ids))

    mean_tokens = float(np.mean(reasoning_lengths))
    actual_ratio = (mean_tokens / full_cot_mean_tokens
                    if full_cot_mean_tokens else 0.0)

    return ConditionMetrics(
        condition=condition_name,
        model_tag=model_tag,
        ratio=ratio,
        accuracy=correct / len(dataset),
        reasoning_tokens=mean_tokens,
        actual_ratio=actual_ratio,
        latency_sec=float(np.mean(latencies)),
        answer_found_rate=found / len(dataset),
    )


def run_phase1_evaluation(
    model_tag: str,
    base_model_id: str,
    D_val: list,
    device: str,
    checkpoints_dir: str,
    results_dir: str,
) -> list[ConditionMetrics]:
    """
    Runs all 12 evaluation conditions for one backbone:
      1  No CoT (frozen base)
      2  Full CoT
      3–7  Trimmed CoT at R ∈ {0.9, 0.8, 0.7, 0.6, 0.5}
      8–12 CCoT at R ∈ {0.9, 0.8, 0.7, 0.6, 0.5}

    Saves results to {results_dir}/phase1_val.json.
    """
    results: list[ConditionMetrics] = []

    header = f"Phase 1 evaluation: {model_tag}  ({len(D_val)} val examples)"
    print(f"\n{'='*len(header)}\n{header}\n{'='*len(header)}")

    # ── 1. No CoT ──────────────────────────────────────────────────────────────
    print("\n[1/12] No CoT (frozen base)...")
    base_model, tok = load_base_frozen(base_model_id, device)
    results.append(evaluate_condition(
        base_model, tok, D_val, device,
        condition_name='no_cot', model_tag=model_tag,
    ))
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 2. Full CoT ────────────────────────────────────────────────────────────
    print("\n[2/12] Full CoT...")
    cot_dir = os.path.join(checkpoints_dir, 'cot')
    cot_model, tok = load_finetuned(cot_dir, device)
    full_cot_metrics = evaluate_condition(
        cot_model, tok, D_val, device,
        condition_name='full_cot', model_tag=model_tag,
    )
    results.append(full_cot_metrics)
    full_cot_mean_tokens = full_cot_metrics.reasoning_tokens

    # ── 3–7. Trimmed CoT ───────────────────────────────────────────────────────
    for step, ratio in enumerate(RATIOS, start=3):
        tag = f'trimmed_cot_R{int(ratio * 10)}'
        print(f"\n[{step}/12] Trimmed CoT R={ratio}  (computing per-example budgets)...")
        budgets = compute_per_example_budgets(cot_model, tok, D_val, device, ratio)
        results.append(evaluate_condition(
            cot_model, tok, D_val, device,
            condition_name=tag,
            model_tag=model_tag,
            ratio=ratio,
            is_trimmed=True,
            budgets=budgets,
            full_cot_mean_tokens=full_cot_mean_tokens,
        ))

    del cot_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 8–12. CCoT ─────────────────────────────────────────────────────────────
    for step, ratio in enumerate(RATIOS, start=8):
        tag = f'ccot_R{int(ratio * 10)}'
        print(f"\n[{step}/12] CCoT R={ratio}...")
        ccot_dir = os.path.join(checkpoints_dir, f'ccot_R{int(ratio * 10)}')
        ccot_model, tok = load_finetuned(ccot_dir, device)
        results.append(evaluate_condition(
            ccot_model, tok, D_val, device,
            condition_name=tag,
            model_tag=model_tag,
            ratio=ratio,
            full_cot_mean_tokens=full_cot_mean_tokens,
        ))
        del ccot_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'phase1_val.json')
    with open(out_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nPhase 1 results saved → {out_path}")
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_comparison_table(results: list[ConditionMetrics]) -> None:
    print(f"\n{'Condition':<30} {'Accuracy':>10} {'Tokens':>10} "
          f"{'Actual R':>10} {'Latency':>10}")
    print('─' * 72)
    for m in results:
        print(f"{m.condition:<30} {m.accuracy:>10.3f} "
              f"{m.reasoning_tokens:>10.1f} {m.actual_ratio:>10.3f} "
              f"{m.latency_sec:>10.2f}s")

    print("\n── Mechanism Gain (CCoT − Trimmed CoT at same token budget) ──")
    res_map = {m.condition: m for m in results}
    for ratio in RATIOS:
        key_ccot    = f'ccot_R{int(ratio * 10)}'
        key_trimmed = f'trimmed_cot_R{int(ratio * 10)}'
        if key_ccot not in res_map or key_trimmed not in res_map:
            continue
        acc_ccot    = res_map[key_ccot].accuracy
        acc_trimmed = res_map[key_trimmed].accuracy
        gain = acc_ccot - acc_trimmed
        label = ("CCoT better" if gain > 0.01 else
                 "Trimmed better" if gain < -0.01 else "Roughly equal")
        print(f"  R={ratio}: CCoT={acc_ccot:.3f}  Trimmed={acc_trimmed:.3f}  "
              f"Gain={gain:+.3f}  -> {label}")
