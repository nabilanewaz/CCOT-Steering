import os

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.utils.extmath import randomized_svd as rsvd


# ── Layer selection ───────────────────────────────────────────────────────────

def select_layers(
    layer_scores: dict[int, float],
    multiplier: float = 0.5,
    contiguity_window: int = 3,
) -> list[int]:
    """
    Select layers above (mean + multiplier*std) probe accuracy, then keep
    only those within contiguity_window of the median selected layer.
    """
    layers = sorted(layer_scores.keys())
    scores = torch.tensor([layer_scores[L] for L in layers], dtype=torch.float32)
    threshold = (scores.mean() + multiplier * scores.std()).item()

    print(f"\nProbe threshold: {threshold:.3f}  "
          f"(mean={scores.mean():.3f}, std={scores.std():.3f})")

    candidates = [L for L in layers if layer_scores[L] >= threshold]
    if not candidates:
        print("Warning: no layers above threshold — falling back to top-5")
        candidates = sorted(layer_scores, key=layer_scores.get, reverse=True)[:5]

    median_L  = sorted(candidates)[len(candidates) // 2]
    selected  = [L for L in candidates if abs(L - median_L) <= contiguity_window]
    print(f"Selected {len(selected)} layers: {selected}")
    return selected


# ── cPCA variants ─────────────────────────────────────────────────────────────

def cpca_full(
    H_pos_L: torch.Tensor,
    H_neg_L: torch.Tensor,
    r: int,
    beta: float = 0.5,
    **_,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full eigendecomposition cPCA. Suitable for d <= 2560 (Phi-2, Qwen2.5-3B).
    Returns (U_L [d, r], lam_L [r]).
    """
    mu_pos, mu_neg = H_pos_L.mean(0), H_neg_L.mean(0)
    H_pos_c = H_pos_L - mu_pos
    H_neg_c = H_neg_L - mu_neg

    C_pos = (H_pos_c.T @ H_pos_c) / max(H_pos_c.shape[0] - 1, 1)
    C_neg = (H_neg_c.T @ H_neg_c) / max(H_neg_c.shape[0] - 1, 1)
    C_contrast = C_pos - beta * C_neg

    eigenvalues, eigenvectors = torch.linalg.eigh(C_contrast)
    pos_mask = eigenvalues > 0
    ev_pos   = eigenvalues[pos_mask]
    vec_pos  = eigenvectors[:, pos_mask]

    r = min(r, len(ev_pos))
    if r == 0:
        raise ValueError("No positive eigenvalues in cpca_full.")
    top_idx = ev_pos.argsort(descending=True)[:r]
    return vec_pos[:, top_idx], ev_pos[top_idx]


def cpca_shrunk(
    H_pos_L: torch.Tensor,
    H_neg_L: torch.Tensor,
    r: int,
    beta: float = 0.5,
    **_,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    cPCA with Ledoit-Wolf shrinkage. Use when n/d < 1.5 (Qwen2.5-Math-1.5B).
    Returns (U_L [d, r], lam_L [r]).
    """
    mu_pos, mu_neg = H_pos_L.mean(0), H_neg_L.mean(0)
    H_pos_c = (H_pos_L - mu_pos).numpy().astype(np.float32)
    H_neg_c = (H_neg_L - mu_neg).numpy().astype(np.float32)

    lw_pos = LedoitWolf().fit(H_pos_c)
    lw_neg = LedoitWolf().fit(H_neg_c)

    C_pos = torch.tensor(lw_pos.covariance_, dtype=torch.float32)
    C_neg = torch.tensor(lw_neg.covariance_, dtype=torch.float32)
    C_contrast = C_pos - beta * C_neg

    eigenvalues, eigenvectors = torch.linalg.eigh(C_contrast)
    pos_mask = eigenvalues > 0
    ev_pos   = eigenvalues[pos_mask]
    vec_pos  = eigenvectors[:, pos_mask]

    r = min(r, len(ev_pos))
    if r == 0:
        raise ValueError("No positive eigenvalues in cpca_shrunk.")
    top_idx = ev_pos.argsort(descending=True)[:r]
    return vec_pos[:, top_idx], ev_pos[top_idx]


def cpca_randomized(
    H_pos_L: torch.Tensor,
    H_neg_L: torch.Tensor,
    r: int,
    beta_rank: int = None,
    **_,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate cPCA via randomized SVD. Use for d >= 3072 (Llama 3.2-3B).
    Projects out dominant H- directions from H+ instead of forming d×d matrices.
    Returns (U_L [d, r], lam_L [r] singular values).
    """
    if beta_rank is None:
        beta_rank = r * 2

    mu_pos, mu_neg = H_pos_L.mean(0), H_neg_L.mean(0)
    H_pos_c = (H_pos_L - mu_pos).numpy().astype(np.float32)
    H_neg_c = (H_neg_L - mu_neg).numpy().astype(np.float32)

    # Dominant directions of incorrect-reasoning space
    _, _, Vt_neg = rsvd(H_neg_c, n_components=beta_rank, random_state=42)
    V_neg = torch.tensor(Vt_neg.T, dtype=torch.float32)  # [d, beta_rank]

    # Project H- directions out of H+
    H_pos_t   = torch.tensor(H_pos_c, dtype=torch.float32)
    H_pos_res = H_pos_t - H_pos_t @ V_neg @ V_neg.T      # [n+, d]

    # Top-r directions of contrastive residual
    _, S, Vt = rsvd(H_pos_res.numpy(), n_components=r, random_state=42)
    U_L   = torch.tensor(Vt.T, dtype=torch.float32)      # [d, r]
    lam_L = torch.tensor(S,    dtype=torch.float32)      # [r]
    return U_L, lam_L


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_eigenspectrum(lam_L: torch.Tensor, layer_idx: int, r_max: int = 10):
    total = lam_L.sum().item()
    if total == 0:
        print(f"\nLayer {layer_idx}: all-zero eigenspectrum — check your data.")
        return
    print(f"\nLayer {layer_idx} — explained variance:")
    cumulative = 0.0
    for i, val in enumerate(lam_L[:r_max]):
        pct = val.item() / total * 100
        cumulative += pct
        bar = '▓' * int(pct / 2)
        print(f"  PC{i+1}: {pct:5.1f}%  cumul {cumulative:5.1f}%  {bar}")


# ── Subspace merge ────────────────────────────────────────────────────────────

def weighted_subspace_merge(
    subspaces: dict[int, tuple[torch.Tensor, torch.Tensor]],
    layer_scores: dict[int, float],
    dom_vectors: dict[int, torch.Tensor],
    v_global: torch.Tensor,
    r_final: int,
) -> torch.Tensor:
    """
    Merge per-layer cPCA subspaces into one [d, r_final] orthonormal subspace.
    Weight = probe_accuracy × mean_eigenvalue × directional_agreement.
    """
    print("\nLayer weights for subspace merge:")
    weighted_cols = []

    for L, (U_L, lam_L) in sorted(subspaces.items()):
        probe_w = layer_scores.get(L, 0.5)
        eig_w   = lam_L.mean().item()
        dir_w   = max(0.0, torch.dot(dom_vectors[L], v_global).item())
        weight  = probe_w * eig_w * dir_w
        weighted_cols.append(U_L * weight)
        print(f"  Layer {L:02d}: probe={probe_w:.3f}  eig={eig_w:.4f}  "
              f"dir={dir_w:.3f}  -> weight={weight:.6f}")

    if not weighted_cols:
        raise ValueError("No subspaces available to merge.")

    W = torch.cat(weighted_cols, dim=1)            # [d, r_per_layer * k]
    U_final, S, _ = torch.linalg.svd(W, full_matrices=False)
    U_truth_final = U_final[:, :r_final]           # [d, r_final]

    total_S = S.sum().item()
    cumulative = 0.0
    print(f"\nMerge SVD singular values (top {min(10, len(S))}):")
    for i, s in enumerate(S[:10]):
        pct = s.item() / total_S * 100
        cumulative += pct
        print(f"  SV{i+1}: {pct:5.1f}%  cumul {cumulative:5.1f}%")

    return U_truth_final


# ── Steering hook ─────────────────────────────────────────────────────────────

def make_subspace_hook(boundary_idx: int, U_truth: torch.Tensor,
                       alpha: float, device: str):
    """
    Returns a hook that applies subspace-based steering at inference time:
        h' = h + alpha * sigma_h * U U^T h_hat
    """
    U = U_truth.to(device)

    def hook(module, input, output):
        h = output[0].clone()
        h_t   = h[:, boundary_idx, :]
        sigma = h_t.norm(dim=-1, keepdim=True) / (h_t.shape[-1] ** 0.5)
        h_hat = h_t / (h_t.norm(dim=-1, keepdim=True) + 1e-8)
        proj  = (U @ (U.T @ h_hat.T)).T
        h[:, boundary_idx, :] = h_t + alpha * sigma * proj
        return (h,) + output[1:]

    return hook


# ── Persistence ───────────────────────────────────────────────────────────────

def save_subspace(
    U_truth: torch.Tensor,
    selected_layers: list[int],
    model_tag: str,
    source: str,
    r_final: int,
    beta: float,
    vectors_dir: str,
) -> str:
    os.makedirs(vectors_dir, exist_ok=True)
    path = os.path.join(vectors_dir, f"{source}_cpca_r{r_final}.pt")
    torch.save({
        'U_truth':         U_truth,
        'method':          'threshold_cpca_weighted_merge',
        'selected_layers': selected_layers,
        'r_final':         r_final,
        'beta':            beta,
        'model_tag':       model_tag,
        'source':          source,
    }, path)
    print(f"Saved subspace → {path}  shape={tuple(U_truth.shape)}")
    return path
