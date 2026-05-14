from collections import defaultdict
from typing import Callable

import torch
from tqdm.auto import tqdm

from phase1.inference import extract_answer, normalize_answer
from phase2.loaders import get_transformer_layers
from phase2.balance import (
    stratified_balance, check_balance,
    difficulty_bucket, IMBALANCE_THRESHOLD,
)


def _register_all_hooks(model) -> tuple[list, dict]:
    """
    Register a forward hook on every transformer layer.
    The hook captures hidden[boundary_idx] only if 'boundary_idx' is set in
    the shared `captured` dict, so it is a no-op during model.generate().
    """
    layers = get_transformer_layers(model)
    captured: dict = {}
    handles: list = []

    def make_hook(L: int):
        def hook(module, input, output):
            if 'boundary_idx' not in captured:
                return
            bidx = captured['boundary_idx']
            h = output[0]
            # Decoder layers may return (B, S, D) or (S, D) depending on transformers version.
            if h.dim() == 3:
                if bidx < h.shape[1]:
                    captured[L] = h[:, bidx, :].detach().cpu()
            elif h.dim() == 2 and bidx < h.shape[0]:
                captured[L] = h[bidx, :].detach().cpu()
        return hook

    for L, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(L)))

    return handles, captured


def collect_hidden_states(
    model,
    tokenizer,
    D_steer: list,
    N: int,
    device: str,
    boundary_idx_fn: Callable,
    source_tag: str,
    prompt_fn: Callable = None,
    min_samples: int = 200,
) -> tuple[dict, dict]:
    """
    Run the frozen model N times per D_steer question (temperature=1.0),
    classify each rollout as correct/incorrect, and accumulate per-layer
    hidden states at the reasoning-boundary token.

    Returns:
        H_pos: dict[layer -> Tensor (n+, d)]
        H_neg: dict[layer -> Tensor (n-, d)]
    """
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    H_plus_raw: dict = defaultdict(list)
    H_minus_raw: dict = defaultdict(list)
    bucket_pos:  dict = defaultdict(list)   # difficulty bucket per H+ sample
    bucket_neg:  dict = defaultdict(list)   # difficulty bucket per H- sample

    handles, captured = _register_all_hooks(model)

    if prompt_fn is None:
        prompt_fn = lambda item: item['question']

    n_used = n_skipped = 0
    n_questions = len(D_steer)
    print(
        f"  Collecting [{source_tag}]: {n_questions} steer questions × {N} rollouts "
        f"(each rollout: generate + full forward for hooks). Progress updates every question.",
        flush=True,
    )

    pbar = tqdm(
        enumerate(D_steer),
        total=n_questions,
        desc=f"collect[{source_tag}]",
        unit="q",
        dynamic_ncols=True,
    )
    for item_idx, item in pbar:
        prompt = prompt_fn(item)
        input_enc = tokenizer(prompt, return_tensors='pt').to(device)
        gold = normalize_answer(item['answer'].split('####')[1].strip())

        # Per-question local buffers — only merged into the global pool if
        # this question produces contrast (at least one correct AND one incorrect rollout).
        item_h_pos: dict = defaultdict(list)
        item_h_neg: dict = defaultdict(list)
        item_n_pos = item_n_neg = 0

        for r_idx in range(N):
            # Clear state from the previous rollout
            captured.pop('boundary_idx', None)
            for L in range(num_layers):
                captured.pop(L, None)

            pbar.set_postfix(
                roll=f"{r_idx + 1}/{N}",
                used=n_used,
                skip=n_skipped,
                refresh=True,
            )

            # Sampling rollout — hooks are no-ops here (boundary_idx not set)
            with torch.no_grad():
                out_ids = model.generate(
                    **input_enc,
                    do_sample=True,
                    temperature=1.0,
                    max_new_tokens=256,
                    pad_token_id=tokenizer.pad_token_id,
                )

            try:
                bidx = boundary_idx_fn(out_ids, tokenizer)
            except ValueError:
                continue

            # Set boundary, then re-run full forward pass to trigger hooks
            captured['boundary_idx'] = bidx
            with torch.no_grad():
                model(out_ids)

            pred = extract_answer(
                tokenizer.decode(out_ids[0], skip_special_tokens=True)
            )
            is_correct = (normalize_answer(pred) == gold) if pred is not None else False

            if is_correct:
                item_n_pos += 1
            else:
                item_n_neg += 1

            for L in range(num_layers):
                if L in captured:
                    h = captured[L].squeeze(0)  # [d]
                    if is_correct:
                        item_h_pos[L].append(h)
                    else:
                        item_h_neg[L].append(h)

        # Skip questions with no contrast (always right or always wrong)
        if item_n_pos == 0 or item_n_neg == 0:
            n_skipped += 1
            n_pos = len(H_plus_raw[0]) if 0 in H_plus_raw else 0
            n_neg = len(H_minus_raw[0]) if 0 in H_minus_raw else 0
            pbar.set_postfix(used=n_used, skip=n_skipped, Hp=n_pos, Hm=n_neg, refresh=True)
            continue

        n_used += 1
        bkt = difficulty_bucket(item_n_pos / (item_n_pos + item_n_neg))
        for L in range(num_layers):
            H_plus_raw[L].extend(item_h_pos[L])
            H_minus_raw[L].extend(item_h_neg[L])
            bucket_pos[L].extend([bkt] * len(item_h_pos[L]))
            bucket_neg[L].extend([bkt] * len(item_h_neg[L]))

        n_pos = len(H_plus_raw[0]) if 0 in H_plus_raw else 0
        n_neg = len(H_minus_raw[0]) if 0 in H_minus_raw else 0
        pbar.set_postfix(used=n_used, skip=n_skipped, Hp=n_pos, Hm=n_neg, refresh=True)

    for h in handles:
        h.remove()

    # ── Class balance check and stratified undersampling ─────────────────────
    ratio, n_pos_total, n_neg_total = check_balance(H_plus_raw, H_minus_raw)
    print(f"\nClass balance  [source: {source_tag}]  "
          f"H+={n_pos_total}  H-={n_neg_total}  ratio={ratio:.2f}x")
    if ratio > IMBALANCE_THRESHOLD:
        print(f"  Imbalance > {IMBALANCE_THRESHOLD}x "
              f"— stratified undersampling of H-")
        H_plus_raw, H_minus_raw = stratified_balance(
            H_plus_raw, H_minus_raw, bucket_pos, bucket_neg,
        )

    # ── Stack tensors and apply min_samples filter ────────────────────────────
    H_pos, H_neg = {}, {}
    for L in range(num_layers):
        n_pos = len(H_plus_raw[L])
        n_neg = len(H_minus_raw[L])
        if n_pos >= min_samples and n_neg >= min_samples:
            H_pos[L] = torch.stack(H_plus_raw[L])   # [n+, d]
            H_neg[L] = torch.stack(H_minus_raw[L])  # [n-, d]
        else:
            print(f"  Layer {L:02d}: skipped "
                  f"(H+={n_pos}, H-={n_neg}, min={min_samples})")

    print(f"\nCollected hidden states from {len(H_pos)} / {num_layers} layers  "
          f"[source: {source_tag}]  "
          f"questions used={n_used}  skipped={n_skipped} (no contrast)")
    return H_pos, H_neg
