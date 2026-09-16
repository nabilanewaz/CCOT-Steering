"""Learnable alpha and gradient-based tuning for Phase 3."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase2.loaders import (
    find_boundary_idx_ccot,
    forward_for_boundary_hooks,
    get_transformer_layers,
)
from phase1.inference import latent_prompt, extract_reasoning_span
from phase3.hook_utils import (
    boundary_state,
    first_hidden,
    replace_first_hidden,
    write_boundary_state,
)

_LAMBDA_M = {
    'phi2': 0.005,         # LayerNorm — more tolerant of large perturbations
}
_LAMBDA_M_DEFAULT = 0.01   # Llama, Qwen (RMSNorm)


class LearnableAlpha(nn.Module):
    """Alpha ∈ (0, alpha_max) via sigmoid reparameterisation (spec §3.4.1).

    θ init: log(1 / (alpha_max − 1)) ≈ −3.89 → α₀ ≈ 1.0
    """

    def __init__(self, alpha_max: float = 50.0, alpha_init: float = 1.0):
        super().__init__()
        self.alpha_max = alpha_max
        alpha_init = float(min(max(alpha_init, 1e-4), alpha_max - 1e-4))
        theta_init = math.log(alpha_init / (alpha_max - alpha_init))
        self.theta = nn.Parameter(torch.tensor(theta_init))

    def forward(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.theta)

    @property
    def value(self) -> float:
        return self.forward().item()


def tune_alpha(
    model,
    tokenizer,
    D_val_tune: list,
    v_truth: torch.Tensor,      # [d] unit-normalised DoM vector
    layer_star: int,
    device: str,
    model_tag: str = '',
    latent_tokens: int = 4,
    lambda_a: float = 0.1,
    lambda_m: float = None,     # None → per-backbone default from _LAMBDA_M
    max_epochs: int = 5,
    es_patience: int = 5,
    lr: float = 5e-2,
    prompt_fn=None,
    boundary_fn=None,
) -> tuple:
    """
    Gradient-descent alpha tuning on 90% of D_val_tune (10% early-stopping).
    Three-term loss: L_ans + lambda_a * L_align + lambda_m * L_mag.

    L_ans  — NLL of gold answer tokens (teacher-forced after steered boundary).
    L_align — 1 − cos(h_steered, v_truth): penalises directional misalignment.
    L_mag  — (‖δ‖ / ‖h_orig‖)²: prevents RMSNorm norm collapse.

    Returns:
        alpha_star — learned α* as a detached scalar tensor
        history    — list of per-epoch dicts with keys:
                     epoch, L_ans, L_align, L_mag, total_train, es_loss, alpha
    """
    if lambda_m is None:
        lambda_m = _LAMBDA_M.get(model_tag, _LAMBDA_M_DEFAULT)
    if prompt_fn is None:
        prompt_fn = lambda item: latent_prompt(item['question'], latent_tokens)
    if boundary_fn is None:
        boundary_fn = find_boundary_idx_ccot

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    v = (v_truth / (v_truth.norm() + 1e-8)).to(device).float()

    alpha_module = LearnableAlpha(alpha_max=50.0, alpha_init=1.0).to(device)
    optimizer    = torch.optim.AdamW(alpha_module.parameters(), lr=lr)

    if len(D_val_tune) < 2:
        raise ValueError("Alpha tuning requires at least two validation examples")
    n_tune = max(1, int(len(D_val_tune) * 0.9))
    D_tune = D_val_tune[:n_tune]
    D_es   = D_val_tune[n_tune:]

    layers       = get_transformer_layers(model)
    target_layer = layers[layer_star]

    cache: dict = {}


    def steer_hook(module, input, output):
        h = first_hidden(output)
        start = cache['prompt_length']
        if h.shape[-2] <= start:
            return output
        h_t = h[..., start:, :]
        h_float = h_t.float()
        sigma = h_float.detach().norm(dim=-1, keepdim=True) / (h_float.shape[-1] ** 0.5)
        alpha = alpha_module()
        delta = alpha * sigma * v        # grad flows through alpha_module -> delta
        cache['h_steered'] = h_float + delta  # grad through delta -> alpha
        cache['h_orig']    = h_float.detach() # reference norm (no grad needed)
        cache['delta']     = delta        # keep grad so L_mag regularises alpha
        h_out = h.clone()
        h_out[..., start:, :] = cache['h_steered'].to(h.dtype)
        return replace_first_hidden(output, h_out)


    prepared = {}

    def _compute_losses(item, grad: bool):
        """Returns (loss, L_ans, L_align, L_mag, boundary_fallback)."""
        cache.clear()
        q_prompt = prompt_fn(item)
        ans_text = item['answer'].split('####')[1].strip()

        q_enc = tokenizer(q_prompt, return_tensors='pt').to(device)
        cache['prompt_length'] = q_enc['input_ids'].shape[1]
        if q_prompt not in prepared:
            with torch.no_grad():
                gen_ids = model.generate(
                    **q_enc, do_sample=False, max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(gen_ids[0, cache['prompt_length']:], skip_special_tokens=True)
            reasoning = extract_reasoning_span(generated)
            suffix = tokenizer.encode(reasoning + "\n#### ", add_special_tokens=False, return_tensors="pt").to(device)
            prepared[q_prompt] = torch.cat([q_enc['input_ids'], suffix], dim=1)
        gen_ids = prepared[q_prompt]
        boundary_fallback = False

        a_ids    = tokenizer(ans_text, return_tensors='pt',
                             add_special_tokens=False).input_ids.to(device)
        full_ids = torch.cat([gen_ids, a_ids], dim=1)
        labels   = full_ids.clone()
        labels[:, :gen_ids.shape[1]] = -100  # mask reasoning positions

        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            handle = target_layer.register_forward_hook(steer_hook)
            try:
                # Keep decoder-layer positions global for Coconut; its cached
                # path emits local chunks that cannot be indexed by b.
                out = forward_for_boundary_hooks(
                    model, input_ids=full_ids, labels=labels
                )
            finally:
                handle.remove()
            L_ans = out.loss

            h_s = cache.get('h_steered')
            L_align = (
                1.0 - F.cosine_similarity(
                    h_s.float(), v.unsqueeze(0), dim=-1
                ).clamp(-1.0, 1.0).mean()
                if h_s is not None
                else torch.tensor(0.0, device=device)
            )

            delta  = cache.get('delta')
            h_orig = cache.get('h_orig')
            L_mag = (
                (delta.float().norm(dim=-1) / (h_orig.float().norm(dim=-1) + 1e-8)).pow(2).mean()
                if delta is not None and h_orig is not None
                else torch.tensor(0.0, device=device)
            )

            loss = L_ans + lambda_a * L_align + lambda_m * L_mag
            if grad and not loss.requires_grad:
                raise RuntimeError("Alpha has no gradient path through generated-token intervention")
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite alpha tuning loss")

        return loss, L_ans.item(), L_align.item(), L_mag.item(), boundary_fallback

    def _mean(lst):
        return sum(lst) / max(len(lst), 1)

    best_es_loss = float('inf')
    best_theta   = alpha_module.theta.detach().clone()
    patience     = es_patience
    history      = []
    train_log_every = max(1, len(D_tune) // 10)
    es_log_every = max(1, len(D_es) // 5) if D_es else 1
    print(
        f"  [α-tune] start: train={len(D_tune)} es={len(D_es)} "
        f"epochs={max_epochs} λ_a={lambda_a} λ_m={lambda_m}",
        flush=True,
    )

    for epoch in range(max_epochs):
        print(f"  [α-tune] epoch {epoch + 1}/{max_epochs} train begin", flush=True)
        ep_loss, ep_la, ep_lal, ep_lm, ep_grad = [], [], [], [], []
        ep_boundary_total = 0
        ep_boundary_fallback = 0
        for item_idx, item in enumerate(D_tune, start=1):
            optimizer.zero_grad()
            loss, la, lal, lm, bf = _compute_losses(item, grad=True)
            loss.backward()
            grad_norm = (
                float(alpha_module.theta.grad.detach().abs().item())
                if alpha_module.theta.grad is not None else 0.0
            )
            optimizer.step()
            ep_loss.append(loss.item())
            ep_la.append(la)
            ep_lal.append(lal)
            ep_lm.append(lm)
            ep_grad.append(grad_norm)
            ep_boundary_total += 1
            ep_boundary_fallback += int(bf)
            if item_idx == 1 or item_idx % train_log_every == 0 or item_idx == len(D_tune):
                print(
                    f"    [α-tune] epoch {epoch + 1}/{max_epochs} "
                    f"train {item_idx}/{len(D_tune)} "
                    f"loss={_mean(ep_loss):.4f} α={alpha_module.value:.4f}",
                    flush=True,
                )

        if D_es:
            es_vals = []
            print(f"  [α-tune] epoch {epoch + 1}/{max_epochs} ES begin", flush=True)
            for item_idx, item in enumerate(D_es, start=1):
                es_vals.append(_compute_losses(item, grad=False)[0].item())
                if item_idx == 1 or item_idx % es_log_every == 0 or item_idx == len(D_es):
                    print(
                        f"    [α-tune] epoch {epoch + 1}/{max_epochs} "
                        f"ES {item_idx}/{len(D_es)} loss={_mean(es_vals):.4f}",
                        flush=True,
                    )
            es_loss = _mean(es_vals)
        else:
            es_loss = _mean(ep_loss)

        history.append({
            'epoch':       epoch + 1,
            'L_ans':       _mean(ep_la),
            'L_align':     _mean(ep_lal),
            'L_mag':       _mean(ep_lm),
            'total_train': _mean(ep_loss),
            'es_loss':     es_loss,
            'alpha':       alpha_module.value,
            'grad_norm_theta': _mean(ep_grad),
            'boundary_fallback_rate': (
                ep_boundary_fallback / max(ep_boundary_total, 1)
            ),
            'mean_delta_norm_ratio': _mean([x ** 0.5 for x in ep_lm]),
        })

        print(f"  [α-tune] epoch {epoch + 1}/{max_epochs}  "
              f"train={_mean(ep_loss):.4f}  es={es_loss:.4f}  "
              f"L_ans={_mean(ep_la):.4f}  L_align={_mean(ep_lal):.4f}  "
              f"L_mag={_mean(ep_lm):.4f}  α={alpha_module.value:.4f}",
              flush=True)

        if es_loss < best_es_loss:
            best_es_loss = es_loss
            best_theta   = alpha_module.theta.detach().clone()
            patience     = es_patience
        else:
            patience -= 1
            if patience == 0:
                print(f"  Early stopping at epoch {epoch + 1}", flush=True)
                break

    alpha_module.theta.data = best_theta
    alpha_star = alpha_module().detach()
    print(f"  Learned α* = {alpha_star.item():.4f}", flush=True)
    return alpha_star, history
