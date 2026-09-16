"""Frozen stochastic rollouts and generated-token or boundary extraction."""
from collections import defaultdict
from typing import Callable

import torch
from tqdm.auto import tqdm

from phase1.inference import extract_answer, normalize_answer
from phase2.balance import difficulty_bucket, stratified_balance
from phase2.loaders import forward_for_boundary_hooks, get_transformer_layers


def _register_all_hooks(model) -> tuple[list, dict]:
    captured = {}
    handles = []

    def make_hook(layer_index):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            if "boundary_range" in captured:
                start, end = captured["boundary_range"]
                if end <= hidden.shape[1]:
                    captured[layer_index] = hidden[:, start:end].float().mean(1).detach().cpu()
            elif "boundary_idx" in captured:
                index = captured["boundary_idx"]
                if 0 <= index < hidden.shape[1]:
                    captured[layer_index] = hidden[:, index].detach().float().cpu()
        return hook

    for index, layer in enumerate(get_transformer_layers(model)):
        handles.append(layer.register_forward_hook(make_hook(index)))
    return handles, captured


def extraction_position(out_ids, prompt_len, tokenizer, boundary_fn, extraction, gen_window):
    generated_length = out_ids.shape[1] - prompt_len
    if extraction == "mean_gen":
        if generated_length < 3:
            return None
        return {"boundary_range": (prompt_len, min(out_ids.shape[1], prompt_len + gen_window))}
    if extraction == "first_gen":
        return {"boundary_idx": prompt_len} if generated_length else None
    if extraction == "boundary":
        try:
            return {"boundary_idx": boundary_fn(out_ids, tokenizer)}
        except ValueError:
            return None
    raise ValueError(f"Unknown extraction mode: {extraction}")


def collect_hidden_states(
    model, tokenizer, D_steer: list, N: int, device: str,
    boundary_idx_fn: Callable, source_tag: str, prompt_fn: Callable = None,
    min_samples: int = 200, extraction: str = "mean_gen", gen_window: int = 20,
) -> tuple[dict, dict, dict]:
    if extraction not in {"mean_gen", "first_gen", "boundary"} or gen_window < 1:
        raise ValueError("Invalid extraction mode or generated-token window")
    model.eval()
    prompt_fn = prompt_fn or (lambda item: item["question"])
    positives, negatives = defaultdict(list), defaultdict(list)
    pos_buckets, neg_buckets = defaultdict(list), defaultdict(list)
    handles, captured = _register_all_hooks(model)
    used_questions = skipped_rollouts = 0
    try:
        for item in tqdm(D_steer, desc=f"collect[{source_tag}]", unit="question"):
            encoded = tokenizer(prompt_fn(item), return_tensors="pt").to(device)
            prompt_len = encoded["input_ids"].shape[1]
            gold = normalize_answer(item["answer"].split("####", 1)[1].strip())
            question_rows = []
            for rollout in range(N):
                captured.clear()
                with torch.no_grad():
                    out_ids = model.generate(
                        **encoded, do_sample=True, temperature=1.0,
                        max_new_tokens=256, pad_token_id=tokenizer.pad_token_id,
                    )
                position = extraction_position(
                    out_ids, prompt_len, tokenizer, boundary_idx_fn, extraction, gen_window,
                )
                if position is None:
                    skipped_rollouts += 1
                    continue
                captured.update(position)
                with torch.no_grad():
                    forward_for_boundary_hooks(model, out_ids)
                text = tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=True)
                prediction = extract_answer(text)
                correct = prediction is not None and normalize_answer(prediction) == gold
                states = {layer: value.squeeze(0) for layer, value in captured.items()
                          if isinstance(layer, int)}
                if states:
                    question_rows.append((correct, states))
            if not question_rows:
                continue
            used_questions += 1
            fraction = sum(correct for correct, states in question_rows) / len(question_rows)
            bucket = difficulty_bucket(fraction)
            for correct, states in question_rows:
                destination = positives if correct else negatives
                buckets = pos_buckets if correct else neg_buckets
                for layer, value in states.items():
                    destination[layer].append(value)
                    buckets[layer].append(bucket)
    finally:
        for handle in handles:
            handle.remove()
    positives, negatives = stratified_balance(positives, negatives, pos_buckets, neg_buckets)
    H_pos, H_neg, per_layer = {}, {}, {}
    for layer in range(len(get_transformer_layers(model))):
        pos_rows, neg_rows = positives.get(layer, []), negatives.get(layer, [])
        included = min(len(pos_rows), len(neg_rows)) >= min_samples
        per_layer[str(layer)] = {
            "h_pos": len(pos_rows), "h_neg": len(neg_rows), "included": included,
        }
        if included:
            H_pos[layer] = torch.stack(pos_rows)
            H_neg[layer] = torch.stack(neg_rows)
    diagnostics = {
        "source": source_tag, "n_questions": len(D_steer), "rollouts_per_question": N,
        "total_rollouts": len(D_steer) * N, "questions_used": used_questions,
        "skipped_rollouts": skipped_rollouts, "extraction": extraction,
        "gen_window": gen_window, "min_samples": min_samples,
        "layers_included": sorted(H_pos), "per_layer": per_layer,
    }
    return H_pos, H_neg, diagnostics
