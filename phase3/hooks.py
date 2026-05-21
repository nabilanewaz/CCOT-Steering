"""Inference-time steering hooks for Phase 3 evaluation (spec §3.5)."""
import torch

from phase2.loaders import get_transformer_layers
from phase3.hook_utils import (
    boundary_state,
    first_hidden,
    replace_first_hidden,
    write_boundary_state,
)


# ── Hook factories ─────────────────────────────────────────────────────────────

def make_dom_hook(boundary_idx: int, v_truth: torch.Tensor,
                  alpha: float, device: str):
    """h' = h + alpha * sigma_h * v_hat  (DoM direction, spec §3.5)"""
    v = (v_truth / (v_truth.norm() + 1e-8)).to(device).float()

    def hook(module, input, output):
        h = first_hidden(output)
        h_t = boundary_state(h, boundary_idx)
        if h_t is None:
            return output
        h_float = h_t.float()
        sigma = h_float.norm(dim=-1, keepdim=True) / (h_float.shape[-1] ** 0.5)
        h_out = write_boundary_state(h, boundary_idx, h_float + alpha * sigma * v)
        return replace_first_hidden(output, h_out)

    return hook


def make_cpca_hook(boundary_idx: int, U_truth: torch.Tensor,
                   alpha: float, device: str):
    """h' = h + alpha * sigma_h * U U^T h_hat  (cPCA subspace, spec §3.5)"""
    U = U_truth.to(device).float()

    def hook(module, input, output):
        h = first_hidden(output)
        h_t = boundary_state(h, boundary_idx)
        if h_t is None:
            return output
        h_float = h_t.float()
        sigma = h_float.norm(dim=-1, keepdim=True) / (h_float.shape[-1] ** 0.5)
        h_hat = h_float / (h_float.norm(dim=-1, keepdim=True) + 1e-8)
        if h_hat.dim() == 1:
            proj = U @ (U.T @ h_hat)
        else:
            proj = (U @ (U.T @ h_hat.T)).T
        h_out = write_boundary_state(h, boundary_idx, h_float + alpha * sigma * proj)
        return replace_first_hidden(output, h_out)

    return hook


def make_noise_hook(boundary_idx: int, alpha: float, device: str):
    """Random unit-vector control (spec §3.5).
    A fresh random direction is sampled per call — this is intentional:
    the condition tests whether ANY perturbation helps, not a specific direction."""

    def hook(module, input, output):
        h = first_hidden(output)
        h_t = boundary_state(h, boundary_idx)
        if h_t is None:
            return output
        h_float = h_t.float()
        sigma = h_float.norm(dim=-1, keepdim=True) / (h_float.shape[-1] ** 0.5)
        noise = torch.randn(h_float.shape, device=device, dtype=h_float.dtype)
        noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-8)
        h_out = write_boundary_state(h, boundary_idx, h_float + alpha * sigma * noise)
        return replace_first_hidden(output, h_out)

    return hook


# ── Injection-layer lookup ─────────────────────────────────────────────────────

def get_injection_layer(vectors_dir: str, source: str) -> int:
    """
    Load the Phase 2 cPCA file and return the top probe-score layer.
    selected_layers[0] is the top-probe-score layer (sorted by save_subspace).
    """
    import glob as _glob, os, torch as _torch
    files = _glob.glob(os.path.join(vectors_dir, f'{source}_cpca_r*.pt'))
    if not files:
        raise FileNotFoundError(
            f"No cPCA file for source={source} in {vectors_dir}"
        )
    data = _torch.load(sorted(files)[-1], map_location='cpu')
    sel  = data.get('selected_layers', [])
    if not sel:
        raise ValueError(f"selected_layers is empty in {sorted(files)[-1]}")
    return sel[0]


# ── Generation helper ──────────────────────────────────────────────────────────

def run_with_hook(
    model,
    tokenizer,
    prompt: str,
    layer_star: int,
    hook_fn,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    """Greedy-decode with hook_fn registered at layer_star. Returns decoded text."""
    layers = get_transformer_layers(model)
    handle = layers[layer_star].register_forward_hook(hook_fn)
    try:
        enc = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][enc['input_ids'].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)
    finally:
        handle.remove()
