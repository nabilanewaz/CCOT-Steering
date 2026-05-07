from collections import defaultdict
from typing import Callable

import torch

from phase1.inference import extract_answer, normalize_answer
from phase2.loaders import get_transformer_layers


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
            if bidx < h.shape[1]:
                captured[L] = h[:, bidx, :].detach().cpu()
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

    handles, captured = _register_all_hooks(model)

    if prompt_fn is None:
        prompt_fn = lambda item: item['question']

    for item_idx, item in enumerate(D_steer):
        prompt = prompt_fn(item)
        input_enc = tokenizer(prompt, return_tensors='pt').to(device)
        gold = normalize_answer(item['answer'].split('####')[1].strip())

        for _ in range(N):
            # Clear state from the previous rollout
            captured.pop('boundary_idx', None)
            for L in range(num_layers):
                captured.pop(L, None)

            # Sampling rollout — hooks are no-ops here (boundary_idx not set)
            with torch.no_grad():
                out_ids = model.generate(
                    **input_enc,
                    do_sample=True,
                    temperature=1.0,
                    max_new_tokens=256,
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

            for L in range(num_layers):
                if L in captured:
                    h = captured[L].squeeze(0)  # [d]
                    if is_correct:
                        H_plus_raw[L].append(h)
                    else:
                        H_minus_raw[L].append(h)

        if (item_idx + 1) % 50 == 0:
            n_pos = len(H_plus_raw[0]) if 0 in H_plus_raw else 0
            n_neg = len(H_minus_raw[0]) if 0 in H_minus_raw else 0
            print(f"  [{item_idx+1}/{len(D_steer)}]  "
                  f"H+ per layer ~{n_pos}  H- per layer ~{n_neg}")

    for h in handles:
        h.remove()

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
          f"[source: {source_tag}]")
    return H_pos, H_neg
