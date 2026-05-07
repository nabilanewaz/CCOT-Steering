"""Learnable alpha and gradient-based tuning for Phase 3."""
import math

import torch
import torch.nn as nn

from phase2.loaders import get_transformer_layers


class LearnableAlpha(nn.Module):
    """Alpha ∈ (0, alpha_max) via sigmoid reparameterisation (spec §3.4.1)."""

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
    ratio: float = 0.7,
    lambda_a: float = 0.1,
    lambda_m: float = 0.01,
    max_epochs: int = 5,
    es_patience: int = 3,
    lr: float = 5e-2,
) -> torch.Tensor:
    """
    Gradient-descent alpha tuning on 90% of D_val_tune (10% early-stopping).
    Three-term loss: L_ans + lambda_a * L_align + lambda_m * L_mag.

    L_ans  — NLL of gold answer tokens (teacher-forced after steered boundary).
    L_align — 1 - cos(h_steered, v_truth): penalises directional misalignment.
    L_mag  — (||delta|| / ||h_orig||)^2: prevents RMSNorm shattering.

    Returns the learned alpha* as a detached scalar tensor.
    """
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    v = (v_truth / (v_truth.norm() + 1e-8)).to(device)

    alpha_module = LearnableAlpha(alpha_max=50.0, alpha_init=1.0).to(device)
    optimizer    = torch.optim.AdamW(alpha_module.parameters(), lr=lr)

    n_tune = int(len(D_val_tune) * 0.9)
    D_tune = D_val_tune[:n_tune]
    D_es   = D_val_tune[n_tune:]

    layers       = get_transformer_layers(model)
    target_layer = layers[layer_star]

    # Mutable cache populated by the hook
    cache: dict = {}

    def steer_hook(module, input, output):
        h   = output[0]
        b   = cache.get('boundary_idx', 0)
        if b >= h.shape[1]:
            return output
        h_t   = h[:, b, :]
        sigma = h_t.detach().norm(dim=-1, keepdim=True) / (h_t.shape[-1] ** 0.5)
        alpha = alpha_module()
        delta = alpha * sigma * v
        cache['h_steered'] = h_t + delta
        cache['h_orig_d']  = h_t.detach()
        cache['delta_d']   = delta.detach()
        h_out          = h.clone()
        h_out[:, b, :] = cache['h_steered']
        return (h_out,) + output[1:]

    handle = target_layer.register_forward_hook(steer_hook)

    def _compute_losses(item, grad: bool):
        q_prompt = (f"Question: {item['question']}\n\n[compress:{ratio}]\n")
        ans_text  = item['answer'].split('####')[1].strip()

        # Probe-generate with current alpha to get steered reasoning
        q_enc = tokenizer(q_prompt, return_tensors='pt').to(device)
        cache['boundary_idx'] = max(0, q_enc['input_ids'].shape[1] - 1)
        with torch.no_grad():
            gen_ids = model.generate(
                **q_enc, do_sample=False, max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Teacher-forced: [steered reasoning | gold answer]
        a_ids    = tokenizer(ans_text, return_tensors='pt',
                             add_special_tokens=False).input_ids.to(device)
        full_ids = torch.cat([gen_ids, a_ids], dim=1)
        labels   = full_ids.clone()
        labels[:, :gen_ids.shape[1]] = -100  # mask reasoning positions

        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out   = model(input_ids=full_ids, labels=labels)
            L_ans = out.loss

            h_s   = cache.get('h_steered')
            if h_s is not None:
                L_align = (
                    1.0 - torch.nn.functional.cosine_similarity(
                        h_s, v.unsqueeze(0), dim=-1
                    ).clamp(-1.0, 1.0).mean()
                )
            else:
                L_align = torch.tensor(0.0, device=device)

            delta  = cache.get('delta_d')
            h_orig = cache.get('h_orig_d')
            if delta is not None and h_orig is not None:
                L_mag = (
                    (delta.norm(dim=-1) / (h_orig.norm(dim=-1) + 1e-8))
                    .pow(2).mean()
                )
            else:
                L_mag = torch.tensor(0.0, device=device)

            loss = L_ans + lambda_a * L_align + lambda_m * L_mag

        return loss

    best_es_loss = float('inf')
    best_theta   = alpha_module.theta.detach().clone()
    patience     = es_patience

    for epoch in range(max_epochs):
        epoch_losses = []
        for item in D_tune:
            optimizer.zero_grad()
            loss = _compute_losses(item, grad=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Early-stopping evaluation
        if D_es:
            es_loss = sum(
                _compute_losses(item, grad=False).item() for item in D_es
            ) / len(D_es)
        else:
            es_loss = sum(epoch_losses) / len(epoch_losses)

        train_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  [α-tune] epoch {epoch + 1}/{max_epochs}  "
              f"train={train_loss:.4f}  es={es_loss:.4f}  "
              f"alpha={alpha_module.value:.4f}")

        if es_loss < best_es_loss:
            best_es_loss = es_loss
            best_theta   = alpha_module.theta.detach().clone()
            patience     = es_patience
        else:
            patience -= 1
            if patience == 0:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    handle.remove()
    alpha_module.theta.data = best_theta
    alpha_star = alpha_module().detach()
    print(f"  Learned α* = {alpha_star.item():.4f}")
    return alpha_star
