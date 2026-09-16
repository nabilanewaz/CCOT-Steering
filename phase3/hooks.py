"""Locally scaled steering aligned to generated positions for cached and Coconut decoding."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from phase2.collect_heads import attention_output_projection
from phase2.loaders import get_transformer_layers
from phase3.hook_utils import first_hidden, replace_first_hidden

_GENERATION = ContextVar("steering_generation", default=None)


@contextmanager
def generation_scope(model, prompt_length):
    state = {"prompt_length": prompt_length, "start": 0, "next_position": 0}
    token = _GENERATION.set(state)
    backbone = getattr(model, "base_causallm", model)

    def before_forward(module, inputs, kwargs):
        values = kwargs.get("inputs_embeds")
        if values is None:
            values = kwargs.get("input_ids")
        if values is None and inputs:
            values = inputs[0]
        length = values.shape[1] if values is not None else 1
        positions = kwargs.get("position_ids")
        if positions is None:
            positions = kwargs.get("cache_position")
        if positions is not None:
            start = int(positions.reshape(-1)[0])
        elif length > 1:
            start = 0
        else:
            start = state["next_position"]
        state["start"] = start
        state["next_position"] = max(state["next_position"], start + length)

    handle = backbone.register_forward_pre_hook(before_forward, with_kwargs=True)
    try:
        yield state
    finally:
        handle.remove()
        _GENERATION.reset(token)


def intervention_positions(hidden, boundary_idx=0, gen_only=True, n_pre=0):
    sequence_length = hidden.shape[-2]
    state = _GENERATION.get()
    offset = state["start"] if state else 0
    if gen_only:
        if state is None:
            return [0] if sequence_length == 1 else []
        start = max(0, state["prompt_length"] - offset)
        return list(range(start, sequence_length))
    start = max(0, boundary_idx - n_pre - offset)
    end = min(sequence_length, boundary_idx - offset + 1)
    return list(range(start, max(start, end)))


def _residual_hook(boundary_idx, alpha, device, transform, gen_only, n_pre):
    def hook(module, inputs, output):
        hidden = first_hidden(output)
        positions = intervention_positions(hidden, boundary_idx, gen_only, n_pre)
        if not positions:
            return output
        selected = hidden[..., positions, :].float()
        sigma = selected.norm(dim=-1, keepdim=True) / selected.shape[-1] ** 0.5
        delta = alpha * sigma * transform(selected)
        modified = hidden.clone()
        modified[..., positions, :] = (selected + delta).to(hidden.dtype)
        return replace_first_hidden(output, modified)
    return hook


def make_dom_hook(boundary_idx, v_truth, alpha, device, gen_only=True, n_pre=0):
    direction = torch.nn.functional.normalize(v_truth.to(device).float(), dim=-1)
    return _residual_hook(boundary_idx, alpha, device, lambda hidden: direction, gen_only, n_pre)


def make_cpca_hook(boundary_idx, U_truth, alpha, device, gen_only=True, n_pre=0):
    basis = U_truth.to(device).float()
    def projection(hidden):
        unit = torch.nn.functional.normalize(hidden, dim=-1)
        return (unit @ basis) @ basis.T
    return _residual_hook(boundary_idx, alpha, device, projection, gen_only, n_pre)


def make_noise_hook(boundary_idx, alpha, device, gen_only=True, n_pre=0):
    def noise(hidden):
        return torch.nn.functional.normalize(torch.randn_like(hidden), dim=-1)
    return _residual_hook(boundary_idx, alpha, device, noise, gen_only, n_pre)


def make_iti_hook_for_layer(layer_index, payload, alpha, device, boundary_idx=0, gen_only=True):
    selected = [tuple(key) for key in payload["top_heads"] if key[0] == layer_index]
    num_heads = int(payload["num_heads"])

    def hook(module, inputs):
        hidden = inputs[0]
        positions = intervention_positions(hidden, boundary_idx, gen_only)
        if not positions:
            return
        modified = hidden.clone()
        head_dim = hidden.shape[-1] // num_heads
        if hidden.shape[-1] % num_heads:
            raise ValueError("Attention output width is not divisible by num_heads")
        for key in selected:
            head = key[1]
            direction = torch.nn.functional.normalize(payload["head_directions"][key].to(device).float(), dim=-1)
            if direction.numel() != head_dim:
                raise ValueError("ITI head direction has incompatible dimension")
            delta = alpha * float(payload["head_sigmas"][key]) * direction
            start, end = head * head_dim, (head + 1) * head_dim
            modified[..., positions, start:end] = (
                hidden[..., positions, start:end].float() + delta
            ).to(hidden.dtype)
        return (modified, *inputs[1:])
    return hook


@contextmanager
def installed_hooks(model, hook_pairs=(), mlp_hook_pairs=(), iti_payload=None, alpha=0.0, device="cpu"):
    layers = get_transformer_layers(model)
    handles = []
    try:
        for layer_index, hook in hook_pairs:
            handles.append(layers[layer_index].register_forward_hook(hook))
        for layer_index, hook in mlp_hook_pairs:
            handles.append(layers[layer_index].mlp.register_forward_hook(hook))
        if iti_payload is not None:
            for layer_index in sorted({key[0] for key in iti_payload["top_heads"]}):
                projection = attention_output_projection(layers[layer_index])
                handles.append(projection.register_forward_pre_hook(
                    make_iti_hook_for_layer(layer_index, iti_payload, alpha, device),
                ))
        yield
    finally:
        for handle in handles:
            handle.remove()


def condition_hooks(model, method, artifacts, alpha, device):
    layer = artifacts["dom"]["best_layer"]
    direction = artifacts["dom"]["v_truth"]
    pairs, mlp_pairs, iti = [], [], None
    if method in ("dom", "neg_dom", "shuf_dom"):
        vector = artifacts["shuffled_dom"]["v_shuffled"] if method == "shuf_dom" else direction
        scale = -alpha if method == "neg_dom" else alpha
        pairs = [(layer, make_dom_hook(0, vector, scale, device))]
    elif method in ("cpca", "neg_cpca", "shuf_cpca"):
        key = "shuffled_cpca" if method == "shuf_cpca" else "cpca"
        basis_key = "U_shuffled" if method == "shuf_cpca" else "U_truth"
        basis = artifacts[key][basis_key]
        injection_layer = artifacts["cpca"]["selected_layers"][0]
        scale = -alpha if method == "neg_cpca" else alpha
        pairs = [(injection_layer, make_cpca_hook(0, basis, scale, device))]
    elif method == "noise":
        pairs = [(layer, make_noise_hook(0, alpha, device))]
    elif method in ("multilayer_dom", "multilayer_dom_mlp"):
        payload = artifacts["multilayer_dom"]
        pairs = [(index, make_dom_hook(0, payload["layer_vectors"][index], alpha, device))
                 for index in payload["top_layers"]]
        if method == "multilayer_dom_mlp":
            mlp_pairs = [(index, make_dom_hook(0, payload["layer_vectors"][index], alpha, device))
                         for index in payload["top_layers"]]
    elif method == "iti":
        iti = artifacts["iti"]
    elif method is not None:
        raise ValueError(f"Unsupported steering method: {method}")
    return installed_hooks(model, pairs, mlp_pairs, iti, alpha, device)


def get_injection_layer(vectors_dir, source):
    import os
    payload = torch.load(os.path.join(vectors_dir, f"{source}_dom.pt"), map_location="cpu", weights_only=False)
    return int(payload["best_layer"])


def run_with_multihook(model, tokenizer, prompt, hook_pairs, device, max_new_tokens=256,
                       mlp_hook_pairs=(), iti_payload=None, alpha=0.0):
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = encoded["input_ids"].shape[1]
    with generation_scope(model, prompt_length), installed_hooks(
        model, hook_pairs, mlp_hook_pairs, iti_payload, alpha, device,
    ), torch.no_grad():
        output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)


def run_with_hook(model, tokenizer, prompt, layer_star, hook_fn, device, max_new_tokens=256):
    return run_with_multihook(model, tokenizer, prompt, [(layer_star, hook_fn)], device, max_new_tokens)


def run_with_iti(model, tokenizer, prompt, payload, alpha, device, max_new_tokens=256):
    return run_with_multihook(model, tokenizer, prompt, [], device, max_new_tokens,
                             iti_payload=payload, alpha=alpha)
