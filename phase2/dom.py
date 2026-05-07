import os

import torch


def compute_per_layer_dom(H_pos: dict, H_neg: dict) -> dict[int, torch.Tensor]:
    """Unit-normalized difference-of-means vector at every available layer."""
    dom_vectors: dict[int, torch.Tensor] = {}
    for L in H_pos:
        mu_pos = H_pos[L].mean(dim=0)
        mu_neg = H_neg[L].mean(dim=0)
        v_raw  = mu_pos - mu_neg
        dom_vectors[L] = v_raw / (v_raw.norm() + 1e-8)
    return dom_vectors


def compute_global_dom(
    dom_vectors: dict[int, torch.Tensor],
    layer_scores: dict[int, float],
) -> torch.Tensor:
    """
    Probe-accuracy-weighted SVD over per-layer DoM vectors.
    Returns the top left singular vector [d], unit-normalized.
    """
    layers = sorted(dom_vectors.keys())
    cols = [dom_vectors[L] * layer_scores.get(L, 0.5) for L in layers]
    M = torch.stack(cols, dim=1)   # [d, num_layers]

    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    v_truth = U[:, 0]
    v_truth = v_truth / (v_truth.norm() + 1e-8)

    print("\nPer-layer cosine alignment with global DoM vector:")
    for L in layers:
        alignment = torch.dot(dom_vectors[L], v_truth).item()
        print(f"  Layer {L:02d}: {alignment:+.3f}")

    return v_truth


def compute_global_dom_filtered(
    dom_vectors: dict[int, torch.Tensor],
    layer_scores: dict[int, float],
    selected_layers: list[int],
) -> torch.Tensor:
    """Same as compute_global_dom but restricted to selected_layers only."""
    filtered = {L: dom_vectors[L] for L in selected_layers if L in dom_vectors}
    return compute_global_dom(filtered, layer_scores)


def save_dom_vector(
    v_truth: torch.Tensor,
    model_tag: str,
    source: str,
    vectors_dir: str,
) -> str:
    os.makedirs(vectors_dir, exist_ok=True)
    path = os.path.join(vectors_dir, f"{source}_dom.pt")
    torch.save({
        'v_truth':   v_truth,
        'method':    'multi_layer_dom',
        'model_tag': model_tag,
        'source':    source,
    }, path)
    print(f"Saved DoM vector → {path}  shape={tuple(v_truth.shape)}")
    return path
