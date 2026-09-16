"""Collect attention head outputs before the attention output projection."""
from collections import defaultdict

import torch
from tqdm.auto import tqdm

from phase1.inference import extract_answer, normalize_answer
from phase2.loaders import forward_for_boundary_hooks, get_transformer_layers


def attention_output_projection(layer):
    attention = getattr(layer, "self_attn", None)
    for name in ("o_proj", "dense"):
        projection = getattr(attention, name, None)
        if projection is not None:
            return projection
    raise ValueError("Attention module has no supported output projection")


def collect_head_activations(
    model, tokenizer, D_steer, device, prompt_fn, boundary_idx_fn,
    N_rollouts=10, min_samples=30,
):
    num_heads = int(model.config.num_attention_heads)
    captured = {}
    positives, negatives = defaultdict(list), defaultdict(list)
    handles = []

    def make_hook(layer_index):
        def hook(module, inputs):
            if "boundary_idx" not in captured:
                return
            hidden = inputs[0]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            boundary = captured["boundary_idx"]
            if 0 <= boundary < hidden.shape[1]:
                captured[layer_index] = hidden[0, boundary].detach().float().cpu().reshape(num_heads, -1)
        return hook

    model.eval()
    try:
        for layer_index, layer in enumerate(get_transformer_layers(model)):
            handles.append(attention_output_projection(layer).register_forward_pre_hook(make_hook(layer_index)))
        for item in tqdm(D_steer, desc="collect[ITI]", unit="question"):
            encoded = tokenizer(prompt_fn(item), return_tensors="pt").to(device)
            gold = normalize_answer(item["answer"].split("####", 1)[1].strip())
            for rollout in range(N_rollouts):
                captured.clear()
                with torch.no_grad():
                    output = model.generate(**encoded, do_sample=True, temperature=0.8,
                                            max_new_tokens=128, pad_token_id=tokenizer.pad_token_id)
                try:
                    captured["boundary_idx"] = boundary_idx_fn(output, tokenizer)
                except ValueError:
                    continue
                with torch.no_grad():
                    forward_for_boundary_hooks(model, output)
                text = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                prediction = extract_answer(text)
                correct = prediction is not None and normalize_answer(prediction) == gold
                destination = positives if correct else negatives
                for layer, states in captured.items():
                    if isinstance(layer, int):
                        for head in range(num_heads):
                            destination[(layer, head)].append(states[head])
    finally:
        for handle in handles:
            handle.remove()
    keys = [key for key in positives
            if min(len(positives[key]), len(negatives[key])) >= min_samples]
    return ({key: torch.stack(positives[key]) for key in keys},
            {key: torch.stack(negatives[key]) for key in keys})
