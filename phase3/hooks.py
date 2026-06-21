"""Inference-time steering hooks for Phase 3 evaluation (spec §3.5)."""
import torch

from phase2.loaders import get_transformer_layers
from phase2.collect_heads import get_head_config


# ── Hook factories ─────────────────────────────────────────────────────────────

def make_dom_hook(boundary_idx: int, v_truth: torch.Tensor,
                  alpha: float, device: str, n_pre: int = 0,
                  gen_only: bool = True):
    """h' = h + alpha * sigma_h * v_hat at each generated token.

    gen_only=True (default): steer only during autoregressive generation
    (seq_len==1), not during prefill. This aligns injection with the
    mean_gen collection position (reasoning tokens, not prompt boundary).
    gen_only=False: legacy behaviour — also steer at boundary_idx during prefill.
    """
    v = (v_truth / (v_truth.norm() + 1e-8)).to(device)
    positions = list(range(max(0, boundary_idx - n_pre), boundary_idx + 1))

    def hook(module, input, output):
        _is_tuple = isinstance(output, tuple)
        h = (output[0] if _is_tuple else output).clone()
        seq_len = h.shape[1]
        if seq_len == 1:
            steer_positions = [0]
        elif gen_only:
            steer_positions = []
        else:
            steer_positions = [p for p in positions if p < seq_len]
        for pos in steer_positions:
            h_t   = h[:, pos, :]
            sigma = h_t.norm(dim=-1, keepdim=True) / (h_t.shape[-1] ** 0.5)
            h[:, pos, :] = h_t + alpha * sigma * v
        if _is_tuple:
            return (h,) + output[1:]
        return h

    return hook


def make_cpca_hook(boundary_idx: int, U_truth: torch.Tensor,
                   alpha: float, device: str, n_pre: int = 0,
                   gen_only: bool = True):
    """h' = h + alpha * sigma_h * U U^T h_hat. See make_dom_hook for gen_only docs."""
    U = U_truth.to(device)
    positions = list(range(max(0, boundary_idx - n_pre), boundary_idx + 1))

    def hook(module, input, output):
        _is_tuple = isinstance(output, tuple)
        h = (output[0] if _is_tuple else output).clone()
        seq_len = h.shape[1]
        if seq_len == 1:
            steer_positions = [0]
        elif gen_only:
            steer_positions = []
        else:
            steer_positions = [p for p in positions if p < seq_len]
        for pos in steer_positions:
            h_t   = h[:, pos, :]
            sigma = h_t.norm(dim=-1, keepdim=True) / (h_t.shape[-1] ** 0.5)
            h_hat = h_t / (h_t.norm(dim=-1, keepdim=True) + 1e-8)
            U_    = U.to(h_t.dtype)
            proj  = (U_ @ (U_.T @ h_hat.T)).T
            h[:, pos, :] = h_t + alpha * sigma * proj
        if _is_tuple:
            return (h,) + output[1:]
        return h

    return hook


def make_noise_hook(boundary_idx: int, alpha: float, device: str, n_pre: int = 0,
                    gen_only: bool = True):
    """Random unit-vector control. Fresh direction per call, per position.
    See make_dom_hook for gen_only docs."""
    positions = list(range(max(0, boundary_idx - n_pre), boundary_idx + 1))

    def hook(module, input, output):
        _is_tuple = isinstance(output, tuple)
        h = (output[0] if _is_tuple else output).clone()
        seq_len = h.shape[1]
        if seq_len == 1:
            steer_positions = [0]
        elif gen_only:
            steer_positions = []
        else:
            steer_positions = [p for p in positions if p < seq_len]
        for pos in steer_positions:
            h_t   = h[:, pos, :]
            sigma = h_t.norm(dim=-1, keepdim=True) / (h_t.shape[-1] ** 0.5)
            noise = torch.randn(h_t.shape, device=device)
            noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-8)
            h[:, pos, :] = h_t + alpha * sigma * noise
        if _is_tuple:
            return (h,) + output[1:]
        return h

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


# ── ITI: attention-head-level steering ────────────────────────────────────────

def make_iti_hook_for_layer(
    layer_idx: int,
    selected_heads: list,          # [head_idx, ...] for this layer
    head_directions: dict,         # (layer, head) -> v_truth Tensor(head_dim,)
    head_sigmas: dict,             # (layer, head) -> sigma float
    alpha: float,
    num_heads: int,
    head_dim: int,
    device: str,
) -> object:
    """
    Returns a PRE-hook for o_proj that modifies concatenated attention head outputs
    BEFORE the output projection.

    ITI (Li et al. 2023) steers at this exact site — between MHA and MLP — and only
    touches selected heads.  The MLP can then process the (gently perturbed) attention
    output as normal, preserving math-reasoning and fluency.

    sigma (precomputed offline) = std dev of projections onto v_truth across training
    data, exactly as in Li et al. Eq. 5.  Use register_forward_pre_hook so the
    modification flows through o_proj's weight matrix (not returned as the output).
    """
    # Pre-move direction vectors to avoid per-step dtype/device casting
    vecs = {}
    for h_idx in selected_heads:
        key = (layer_idx, h_idx)
        if key in head_directions:
            vecs[h_idx] = head_directions[key].to(device)

    def pre_hook(module, input):
        # input is a 1-tuple: (x,) where x: [batch, seq_len, num_heads * head_dim]
        x = input[0]
        seq_len = x.shape[1]
        positions = [0] if seq_len == 1 else list(range(seq_len))

        x = x.clone()
        x_heads = x.view(x.shape[0], seq_len, num_heads, head_dim)

        for pos in positions:
            for h_idx in selected_heads:
                if h_idx not in vecs:
                    continue
                key = (layer_idx, h_idx)
                v = vecs[h_idx].to(x.dtype)
                sigma = head_sigmas.get(key, 1.0)
                h_t = x_heads[:, pos, h_idx, :]     # [batch, head_dim]
                x_heads[:, pos, h_idx, :] = h_t + alpha * sigma * v

        return (x_heads.view(x.shape[0], seq_len, -1),)

    return pre_hook


def run_with_iti(
    model,
    tokenizer,
    prompt: str,
    top_heads: list,                # [(layer_idx, head_idx), ...]
    head_directions: dict,
    head_sigmas: dict,
    alpha: float,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Generate with ITI-style per-head attention steering.
    Hooks fire on each selected layer's self_attn.o_proj INPUT
    (between MHA and MLP), touching only the selected heads.
    """
    layers = get_transformer_layers(model)
    num_heads, head_dim = get_head_config(model)

    # Group selected heads by layer
    heads_by_layer: dict[int, list] = {}
    for (l, h) in top_heads:
        heads_by_layer.setdefault(l, []).append(h)

    handles = []
    for l, sel_heads in heads_by_layer.items():
        pre_hook_fn = make_iti_hook_for_layer(
            l, sel_heads, head_directions, head_sigmas,
            alpha, num_heads, head_dim, device,
        )
        h = layers[l].self_attn.o_proj.register_forward_pre_hook(pre_hook_fn)
        handles.append(h)

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
        for h in handles:
            h.remove()


def run_with_multihook(
    model,
    tokenizer,
    prompt: str,
    layer_hook_pairs: list,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    """Greedy-decode with hooks registered at multiple layers simultaneously.

    layer_hook_pairs: list of (layer_idx: int, hook_fn) tuples.
    """
    layers  = get_transformer_layers(model)
    handles = [layers[L].register_forward_hook(hfn) for L, hfn in layer_hook_pairs]
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
        for h in handles:
            h.remove()
