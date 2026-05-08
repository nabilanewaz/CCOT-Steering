"""Inference-time steering hooks for Phase 3 evaluation (spec §3.5)."""
import torch

from phase2.loaders import get_transformer_layers


# ── Hook factories ─────────────────────────────────────────────────────────────

def make_dom_hook(boundary_idx: int, v_truth: torch.Tensor,
                  alpha: float, device: str):
    """h' = h + alpha * sigma_h * v_hat  (DoM direction, spec §3.5)"""
    v = (v_truth / (v_truth.norm() + 1e-8)).to(device)

    def hook(module, input, output):
        h = output[0].clone()
        if boundary_idx >= h.shape[1]:
            return output
        h_t   = h[:, boundary_idx, :]
        sigma = h_t.norm(dim=-1, keepdim=True) / (h_t.shape[-1] ** 0.5)
        h[:, boundary_idx, :] = h_t + alpha * sigma * v
        return (h,) + output[1:]

    return hook


def make_cpca_hook(boundary_idx: int, U_truth: torch.Tensor,
                   alpha: float, device: str):
    """h' = h + alpha * sigma_h * U U^T h_hat  (cPCA subspace, spec §3.5)"""
    U = U_truth.to(device)

    def hook(module, input, output):
        h = output[0].clone()
        if boundary_idx >= h.shape[1]:
            return output
        h_t   = h[:, boundary_idx, :]
        sigma = h_t.norm(dim=-1, keepdim=True) / (h_t.shape[-1] ** 0.5)
        h_hat = h_t / (h_t.norm(dim=-1, keepdim=True) + 1e-8)
        proj  = (U @ (U.T @ h_hat.T)).T
        h[:, boundary_idx, :] = h_t + alpha * sigma * proj
        return (h,) + output[1:]

    return hook


def make_noise_hook(boundary_idx: int, alpha: float, device: str):
    """Random unit-vector control (spec §3.5).
    A fresh random direction is sampled per call — this is intentional:
    the condition tests whether ANY perturbation helps, not a specific direction."""

    def hook(module, input, output):
        h = output[0].clone()
        if boundary_idx >= h.shape[1]:
            return output
        h_t   = h[:, boundary_idx, :]
        sigma = h_t.norm(dim=-1, keepdim=True) / (h_t.shape[-1] ** 0.5)
        noise = torch.randn(h_t.shape, device=device)
        noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-8)
        h[:, boundary_idx, :] = h_t + alpha * sigma * noise
        return (h,) + output[1:]

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
