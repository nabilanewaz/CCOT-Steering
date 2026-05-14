import os

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd as rsvd

K_SWEEP    = [1, 2, 5, 10]     # per-layer rank candidates
BETA_SWEEP = [0.3, 0.5, 0.7]   # contrastive weight candidates


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
    beta: float = 0.5,
    beta_rank: int = None,
    **_,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate cPCA via randomized SVD. Use for d >= 3072 (Llama 3.2-3B).
    Projects out dominant H- directions from H+ instead of forming d×d matrices.
    Returns (U_L [d, r], lam_L [r] singular values).

    beta controls how many H- directions are projected out:
        beta_rank = max(r, round(r * 4 * beta))
        β=0.3 → ~1.2r (weak suppression)  β=0.7 → ~2.8r (strong suppression)
    """
    if beta_rank is None:
        beta_rank = max(r, round(r * 4 * beta))

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


# ── cPCA sweep (k × β grid) ──────────────────────────────────────────────────

def sweep_cpca_layer(
    H_pos_L: torch.Tensor,
    H_neg_L: torch.Tensor,
    cpca_fn,
    k_vals: list = K_SWEEP,
    beta_vals: list = BETA_SWEEP,
) -> dict:
    """
    Run cPCA for every (k, β) in k_vals × beta_vals at a single layer.
    Returns {(k, beta): (U [d,k], lam [k])}.
    """
    results = {}
    for beta in beta_vals:
        for k in k_vals:
            try:
                U, lam = cpca_fn(H_pos_L, H_neg_L, r=k, beta=beta)
                results[(k, beta)] = (U, lam)
            except Exception as exc:
                print(f"      k={k}  β={beta:.1f}: failed ({exc})")
    return results


def select_best_cpca(
    sweep_results: dict,
    H_pos_L: torch.Tensor,
    H_neg_L: torch.Tensor,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """
    Evaluate each (k, β) by stratified 80/20 held-out probe accuracy on the
    projected space. Returns (U_best [d,k], lam_best [k], best_k, best_beta).
    """
    X = torch.cat([H_pos_L, H_neg_L]).numpy().astype(np.float32)
    y = np.array([1] * len(H_pos_L) + [0] * len(H_neg_L))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scores = {}
    for (k, beta), (U, _) in sweep_results.items():
        U_np = U.numpy()
        sc   = StandardScaler()
        pr_tr = sc.fit_transform(X_tr @ U_np)
        pr_te = sc.transform(X_te @ U_np)
        probe = LogisticRegression(max_iter=500, C=1.0)
        probe.fit(pr_tr, y_tr)
        scores[(k, beta)] = float(accuracy_score(y_te, probe.predict(pr_te)))

    best_key = max(scores, key=scores.get)

    print(f"\n    {'k':>4}  {'β':>5}  {'probe_acc':>10}")
    print(f"    {'─' * 24}")
    for k in K_SWEEP:
        for beta in BETA_SWEEP:
            if (k, beta) not in scores:
                continue
            mark = ' *' if (k, beta) == best_key else ''
            print(f"    {k:>4}  {beta:>5.1f}  {scores[(k, beta)]:>10.3f}{mark}")

    best_k, best_beta = best_key
    U_best, lam_best  = sweep_results[best_key]
    best_acc = scores[best_key]
    print(f"    Best: k={best_k}  β={best_beta:.1f}  acc={best_acc:.3f}")
    return U_best, lam_best, best_k, best_beta, best_acc


def run_cpca_sweep(
    H_pos: dict,
    H_neg: dict,
    selected_layers: list[int],
    cpca_fn,
    k_vals: list = K_SWEEP,
    beta_vals: list = BETA_SWEEP,
) -> dict:
    """
    Run the (k, β) grid sweep at every selected layer, pick the best per layer.
    Returns {L: (U_best [d, best_k], lam_best [best_k], best_k, best_beta, best_acc)}.
    """
    layer_results = {}
    for L in selected_layers:
        if L not in H_pos:
            continue
        n_pos = H_pos[L].shape[0]
        n_neg = H_neg[L].shape[0] if L in H_neg else 0
        print(f"\n  cPCA sweep at layer {L}  (H+={n_pos}  H-={n_neg})")
        sweep = sweep_cpca_layer(H_pos[L], H_neg[L], cpca_fn, k_vals, beta_vals)
        if not sweep:
            print(f"    All (k, β) combinations failed — skipping layer {L}.")
            continue
        U_best, lam_best, best_k, best_beta, best_acc = select_best_cpca(
            sweep, H_pos[L], H_neg[L],
        )
        analyze_eigenspectrum(lam_best, L)
        layer_results[L] = (U_best, lam_best, best_k, best_beta, best_acc)
    return layer_results


# ── Subspace merge ────────────────────────────────────────────────────────────

def weighted_subspace_merge(
    subspaces: dict[int, tuple[torch.Tensor, torch.Tensor]],
    layer_scores: dict[int, float],
    dom_vectors: dict[int, torch.Tensor],
    v_global: torch.Tensor,
    r_final: int,
) -> tuple[torch.Tensor, dict]:
    """
    Merge per-layer cPCA subspaces into one [d, r_final] orthonormal subspace.
    Weight = probe_accuracy × mean_eigenvalue × directional_agreement.
    Returns (U_truth_final [d, r_final], layer_weights {L: float}).
    """
    print("\nLayer weights for subspace merge:")
    weighted_cols = []
    layer_weights: dict = {}

    for L, (U_L, lam_L) in sorted(subspaces.items()):
        probe_w = layer_scores.get(L, 0.5)
        eig_w   = lam_L.mean().item()
        dir_w   = max(0.0, torch.dot(dom_vectors[L], v_global).item())
        weight  = probe_w * eig_w * dir_w
        weighted_cols.append(U_L * weight)
        layer_weights[L] = weight
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

    return U_truth_final, layer_weights


# ── Shuffled-label cPCA control ───────────────────────────────────────────────

def compute_shuffled_cpca(
    H_pos: dict,
    H_neg: dict,
    selected_layers: list,
    cpca_fn,
    dom_vectors: dict,
    layer_scores: dict,
    v_truth: torch.Tensor,
    r_final: int,
    seed: int = 42,
) -> 'torch.Tensor | None':
    """
    Shuffled-label cPCA control: at each selected layer, pool H+/H− and apply
    a random per-layer permutation (preserving class sizes), then recompute cPCA
    and merge subspaces with the real probe-accuracy weights.

    Used as a Phase 3 control baseline — the resulting subspace should be no
    more useful than random noise if the true cPCA captures a meaningful direction.
    Per-layer seeds: seed + L (deterministic, reproducible).
    """
    H_shuf_pos: dict = {}
    H_shuf_neg: dict = {}

    for L in selected_layers:
        if L not in H_pos or L not in H_neg:
            continue
        H_all = torch.cat([H_pos[L], H_neg[L]])   # [n_pos + n_neg, d]
        n_pos = H_pos[L].shape[0]

        g = torch.Generator()
        g.manual_seed(seed + L)
        perm = torch.randperm(H_all.shape[0], generator=g)
        H_all = H_all[perm]

        H_shuf_pos[L] = H_all[:n_pos]
        H_shuf_neg[L] = H_all[n_pos:]

    if not H_shuf_pos:
        return None

    shuf_results = run_cpca_sweep(H_shuf_pos, H_shuf_neg, selected_layers, cpca_fn)
    if not shuf_results:
        return None

    subspaces_shuf = {L: (U, lam) for L, (U, lam, _, _, _) in shuf_results.items()}
    U_shuffled, _ = weighted_subspace_merge(
        subspaces_shuf, layer_scores, dom_vectors, v_truth, r_final
    )

    cos_vals = []
    for L, (U_L, _) in subspaces_shuf.items():
        if L in dom_vectors:
            cos = (dom_vectors[L].unsqueeze(0) @ U_L).norm().item()
            cos_vals.append(cos)
    if cos_vals:
        print(f"\n  Shuffled cPCA: mean |v·U_col| = "
              f"{sum(cos_vals) / len(cos_vals):.4f}  "
              f"(near-zero expected for random labels)")

    return U_shuffled


def save_shuffled_subspace(
    U_shuffled: torch.Tensor,
    model_tag: str,
    source: str,
    r_final: int,
    vectors_dir: str,
) -> str:
    os.makedirs(vectors_dir, exist_ok=True)
    path = os.path.join(vectors_dir, f'{source}_shuffled_cpca_r{r_final}.pt')
    torch.save({
        'U_shuffled': U_shuffled,
        'method':     'shuffled_label_cpca',
        'model_tag':  model_tag,
        'source':     source,
        'r_final':    r_final,
    }, path)
    print(f"Saved shuffled cPCA -> {path}  shape={tuple(U_shuffled.shape)}")
    return path


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
    layer_scores: dict = None,
    sweep_meta: dict = None,
) -> str:
    """
    sweep_meta: optional {L: {'k': int, 'beta': float}} from run_cpca_sweep,
                storing the best (k, β) chosen per layer.
    """
    os.makedirs(vectors_dir, exist_ok=True)
    path = os.path.join(vectors_dir, f"{source}_cpca_r{r_final}.pt")

    if layer_scores:
        sel = sorted(selected_layers,
                     key=lambda L: layer_scores.get(L, 0.0), reverse=True)
    else:
        sel = selected_layers

    payload = {
        'U_truth':         U_truth,
        'method':          'threshold_cpca_weighted_merge',
        'selected_layers': sel,
        'r_final':         r_final,
        'beta':            beta,
        'model_tag':       model_tag,
        'source':          source,
        'layer_scores':    {str(L): layer_scores.get(L, 0.0)
                            for L in sel} if layer_scores else {},
    }
    if sweep_meta:
        payload['sweep_meta'] = {str(L): m for L, m in sweep_meta.items()}

    torch.save(payload, path)
    print(f"Saved subspace -> {path}  shape={tuple(U_truth.shape)}")
    return path
