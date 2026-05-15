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
    run_ccot,
    normalize_answer,
)

LATENT_TOKEN_COUNTS = [3, 4, 6]


@dataclass
class ConditionMetrics:
    condition:         str
    model_tag:         str
    latent_tokens:     Optional[int]
    accuracy:          float
    reasoning_tokens:  float   # mean tokens in reasoning span
    latency_sec:       float   # mean wall-clock seconds per example
    answer_found_rate: float   # fraction where an answer string was extracted
    selected:          bool = False


def evaluate_condition(
    model,
    tokenizer,
    dataset: list,
    device: str,
    condition_name: str,
    model_tag: str,
    latent_tokens: int = None,
    preview_examples: int = 0,
) -> tuple[ConditionMetrics, list[dict]]:
    correct = 0
    found = 0
    reasoning_lengths = []
    latencies = []
    predictions = []

    for i, item in enumerate(dataset):
        t0 = time.time()

        if condition_name == 'no_cot':
            pred, reasoning = run_no_cot(model, tokenizer, item, device)
        elif latent_tokens is not None:
            pred, reasoning = run_ccot(model, tokenizer, item, latent_tokens, device)
        else:
            raise ValueError("Only no_cot and latent-token CCoT conditions are supported.")

        latencies.append(time.time() - t0)

        gold = item['answer'].split('####')[1].strip()
        ok = False
        if pred is not None:
            found += 1
            ok = normalize_answer(pred) == normalize_answer(gold)
            if ok:
                correct += 1

        r_ids = tokenizer.encode(reasoning or '', add_special_tokens=False)
        reasoning_lengths.append(len(r_ids))
        record = {
            "condition": condition_name,
            "id": item.get("id", i),
            "question": item.get("question"),
            "gold": gold,
            "prediction": pred,
            "correct": ok,
            "reasoning_tokens": len(r_ids),
            "reasoning": reasoning,
        }
        if latent_tokens is not None:
            record["latent_tokens"] = latent_tokens
        predictions.append(record)

        if i < preview_examples:
            print("\n" + "-" * 72)
            print(f"[{condition_name}] example={i + 1} latent_tokens={latent_tokens}")
            print(f"Q: {item.get('question')}")
            print(f"Pred: {pred} | Gold: {gold} | Correct: {ok}")
            if reasoning:
                print("Reasoning:")
                print(reasoning[:1200])

    mean_tokens = float(np.mean(reasoning_lengths))

    return ConditionMetrics(
        condition=condition_name,
        model_tag=model_tag,
        latent_tokens=latent_tokens,
        accuracy=correct / len(dataset),
        reasoning_tokens=mean_tokens,
        latency_sec=float(np.mean(latencies)),
        answer_found_rate=found / len(dataset),
    ), predictions


def run_phase1_evaluation(
    model_tag: str,
    base_model_id: str,
    D_val: list,
    device: str,
    checkpoints_dir: str,
    results_dir: str,
) -> list[ConditionMetrics]:
    """
    Run the Phase 1 latent-token sweep.

    The sweep evaluates CCoT with latent-token counts {3, 4, 6}, selects the
    best count by validation accuracy, then evaluates a compact comparison set:
    no_cot and ccot_L{best}. All metrics and predictions are saved in results_dir.
    """
    os.makedirs(results_dir, exist_ok=True)
    latent_results: list[ConditionMetrics] = []
    comparison_results: list[ConditionMetrics] = []
    all_predictions: list[dict] = []

    header = f"Phase 1 evaluation: {model_tag}  ({len(D_val)} val examples)"
    print(f"\n{'='*len(header)}\n{header}\n{'='*len(header)}")

    # ── Latent-token sweep ────────────────────────────────────────────────────
    for step, latent_tokens in enumerate(LATENT_TOKEN_COUNTS, start=1):
        tag = f'ccot_L{latent_tokens}'
        ccot_dir = os.path.join(checkpoints_dir, tag)
        print(f"\n[{step}/{len(LATENT_TOKEN_COUNTS)}] Latent CCoT: {latent_tokens} tokens")
        print(f"Checkpoint: {ccot_dir}")
        ccot_model, tok = load_finetuned(ccot_dir, device)
        metrics, predictions = evaluate_condition(
            ccot_model, tok, D_val, device,
            condition_name=tag,
            model_tag=model_tag,
            latent_tokens=latent_tokens,
            preview_examples=2,
        )
        latent_results.append(metrics)
        all_predictions.extend(predictions)
        print(
            f"[phase1][{model_tag}] {tag}: "
            f"acc={metrics.accuracy:.4f} tokens={metrics.reasoning_tokens:.1f} "
            f"answer_found={metrics.answer_found_rate:.3f}"
        )
        del ccot_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = max(
        latent_results,
        key=lambda m: (m.accuracy, -abs((m.latent_tokens or 0) - 4), m.latent_tokens or 0),
    )
    for metrics in latent_results:
        metrics.selected = metrics.condition == best.condition

    latent_path = os.path.join(results_dir, 'phase1_latent_sweep.json')
    with open(latent_path, 'w') as f:
        json.dump([asdict(r) for r in latent_results], f, indent=2)

    pred_path = os.path.join(results_dir, 'phase1_latent_predictions.jsonl')
    with open(pred_path, 'w', encoding='utf-8') as f:
        for record in all_predictions:
            f.write(json.dumps(record) + "\n")

    best_path = os.path.join(results_dir, 'phase1_best_latent.json')
    with open(best_path, 'w') as f:
        json.dump(asdict(best), f, indent=2)

    print("\n" + "=" * 72)
    print(f"Best latent-token setting: {best.condition}")
    print(f"latent_tokens={best.latent_tokens}  accuracy={best.accuracy:.4f}")
    print(f"Saved latent sweep -> {latent_path}")
    print(f"Saved latent predictions -> {pred_path}")
    print(f"Saved best latent metadata -> {best_path}")

    # ── Compact comparison: no_cot vs best CCoT ──────────────────────────────
    print("\n[comparison] No CoT (frozen base)...")
    base_model, tok = load_base_frozen(base_model_id, device)
    no_cot_metrics, no_cot_predictions = evaluate_condition(
        base_model, tok, D_val, device,
        condition_name='no_cot',
        model_tag=model_tag,
    )
    comparison_results.append(no_cot_metrics)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[comparison] Best CCoT ({best.condition})...")
    best_dir = os.path.join(checkpoints_dir, best.condition)
    ccot_model, tok = load_finetuned(best_dir, device)
    best_metrics, best_predictions = evaluate_condition(
        ccot_model, tok, D_val, device,
        condition_name=best.condition,
        model_tag=model_tag,
        latent_tokens=best.latent_tokens,
    )
    best_metrics.selected = True
    comparison_results.append(best_metrics)
    del ccot_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_path = os.path.join(results_dir, 'phase1_val.json')
    with open(out_path, 'w') as f:
        json.dump([asdict(r) for r in comparison_results], f, indent=2)

    comparison_pred_path = os.path.join(results_dir, 'phase1_comparison_predictions.jsonl')
    with open(comparison_pred_path, 'w', encoding='utf-8') as f:
        for record in no_cot_predictions + best_predictions:
            f.write(json.dumps(record) + "\n")

    print(f"\nPhase 1 comparison saved -> {out_path}")
    print(f"Phase 1 comparison predictions saved -> {comparison_pred_path}")
    return latent_results + comparison_results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_comparison_table(results: list[ConditionMetrics]) -> None:
    latent_rows = [m for m in results if m.condition.startswith('ccot_L')]
    comparison_rows = (
        [m for m in results if m.condition == 'no_cot']
        + [m for m in results if m.selected and m.condition.startswith('ccot_L')]
    )

    print(f"\n{'Latent Sweep':<30} {'Latents':>8} {'Accuracy':>10} {'Tokens':>10} {'Latency':>10}")
    print('-' * 74)
    seen = set()
    for m in latent_rows:
        if m.condition in seen:
            continue
        seen.add(m.condition)
        marker = ' *' if m.selected else ''
        print(f"{m.condition + marker:<30} {m.latent_tokens or '-':>8} "
              f"{m.accuracy:>10.3f} {m.reasoning_tokens:>10.1f} "
              f"{m.latency_sec:>10.2f}s")

    print(f"\n{'Comparison':<30} {'Latents':>8} {'Accuracy':>10} {'Tokens':>10} {'Latency':>10}")
    print('-' * 74)
    printed = set()
    for m in comparison_rows:
        key = (m.condition, m.latent_tokens)
        if key in printed:
            continue
        printed.add(key)
        print(f"{m.condition:<30} {m.latent_tokens or '-':>8} "
              f"{m.accuracy:>10.3f} {m.reasoning_tokens:>10.1f} "
              f"{m.latency_sec:>10.2f}s")
