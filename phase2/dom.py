import os

import torch


def compute_per_layer_dom(H_pos: dict, H_neg: dict) -> dict[int, torch.Tensor]:
    """Unit-normalised DoM vector at every layer: (mean H+ − mean H−) / norm."""
    dom_vectors: dict[int, torch.Tensor] = {}
    for L in H_pos:
        mu_pos = H_pos[L].mean(dim=0)
        mu_neg = H_neg[L].mean(dim=0)
        v_raw  = mu_pos - mu_neg
        dom_vectors[L] = v_raw / (v_raw.norm() + 1e-8)
    return dom_vectors


def compute_best_layer_dom(
    dom_vectors: dict[int, torch.Tensor],
    layer_scores: dict[int, float],
) -> tuple[torch.Tensor, int]:
    """
    Select the layer with the highest probe accuracy and return its
    unit-normalised DoM vector as v_truth.

    Returns:
        v_truth:  Tensor [d]  unit-normalised DoM at best probe layer
        best_L:   int         layer index selected
    """
    if not dom_vectors:
        raise ValueError("No DoM vectors — nothing to select from.")

    # Restrict to layers that have both a DoM vector and a probe score
    candidates = [L for L in dom_vectors if L in layer_scores]
    if not candidates:
        candidates = list(dom_vectors.keys())

    best_L   = max(candidates, key=lambda L: layer_scores.get(L, 0.0))
    v_truth  = dom_vectors[best_L]   # already unit-normalised

    probe_acc = layer_scores.get(best_L, float('nan'))
    print(f"\nMethod A — Best-Layer DoM:")
    print(f"  Selected layer: L={best_L}  probe_acc={probe_acc:.3f}")
    print(f"  v_truth norm  : {v_truth.norm().item():.6f}")

    print(f"\n  Per-layer cosine alignment with v_truth (L={best_L}):")
    for L in sorted(dom_vectors.keys()):
        cos = torch.dot(dom_vectors[L], v_truth).item()
        bar = '█' * int(abs(cos) * 20)
        tag = ' <-- best' if L == best_L else ''
        print(f"    Layer {L:02d}: {cos:+.3f}  {bar}{tag}")

    return v_truth, best_L


def report_cross_source_alignment(
    v_a: torch.Tensor,
    v_b: torch.Tensor,
    best_L_a: int,
    best_L_b: int,
) -> float:
    """
    Report cosine similarity between Source A and Source B DoM directions.
    A high value (> 0.8) means both sources agree on the truth direction.
    """
    cos = torch.dot(v_a.float(), v_b.float()).item()
    label = ('strong agreement' if cos > 0.8
             else 'moderate agreement' if cos > 0.5
             else 'weak / divergent')
    print(f"\nCross-source DoM alignment:")
    print(f"  Source A (L={best_L_a}) · Source B (L={best_L_b}) = {cos:+.4f}  [{label}]")
    return cos


def compute_shuffled_dom(
    H_pos: dict,
    H_neg: dict,
    best_layer: int,
    v_truth: torch.Tensor,
    seed: int = 42,
) -> tuple[torch.Tensor, dict]:
    """
    Control baseline: pool H+ and H- at best_layer, randomly permute labels
    (keeping class sizes constant), recompute DoM.  Returns unit-normalised
    v_shuffled [d].

    Reports:
      - raw norm before normalisation (near-zero norm = labels carry no signal)
      - cosine alignment with v_truth (near zero expected for random labels)
    """
    H_p = H_pos[best_layer]   # [n+, d]
    H_n = H_neg[best_layer]   # [n-, d]
    n_pos = H_p.shape[0]

    g = torch.Generator()
    g.manual_seed(seed)
    H_all  = torch.cat([H_p, H_n])                        # [n++n-, d]
    perm   = torch.randperm(H_all.shape[0], generator=g)
    H_all  = H_all[perm]

    mu_shuf_pos = H_all[:n_pos].mean(dim=0)
    mu_shuf_neg = H_all[n_pos:].mean(dim=0)
    v_raw       = mu_shuf_pos - mu_shuf_neg
    raw_norm    = v_raw.norm().item()
    v_shuffled  = v_raw / (raw_norm + 1e-8)

    true_raw_norm = (H_p.mean(0) - H_n.mean(0)).norm().item()
    cos = torch.dot(v_shuffled.float(), v_truth.float()).item()
    print(f"\nControl — Shuffled-Label DoM  (layer L={best_layer}):")
    print(f"  raw norm          : {raw_norm:.6f}")
    print(f"  true DoM raw norm : {true_raw_norm:.6f}")
    print(f"  cos(v_shuffled, v_truth): {cos:+.4f}  "
          f"({'unexpectedly aligned — check data' if abs(cos) > 0.3 else 'near-zero as expected'})")

    stats = {
        'layer':            best_layer,
        'shuffled_raw_norm': raw_norm,
        'true_raw_norm':     true_raw_norm,
        'cos_with_truth':    cos,
    }
    return v_shuffled, stats


def save_shuffled_vector(
    v_shuffled: torch.Tensor,
    model_tag: str,
    source: str,
    vectors_dir: str,
    best_layer: int = None,
) -> str:
    os.makedirs(vectors_dir, exist_ok=True)
    path = os.path.join(vectors_dir, f"{source}_shuffled_dom.pt")
    payload = {
        'v_shuffled': v_shuffled,
        'method':     'shuffled_label_dom',
        'model_tag':  model_tag,
        'source':     source,
    }
    if best_layer is not None:
        payload['best_layer'] = best_layer
    torch.save(payload, path)
    print(f"Saved shuffled DoM -> {path}  shape={tuple(v_shuffled.shape)}")
    return path


def save_dom_vector(
    v_truth: torch.Tensor,
    model_tag: str,
    source: str,
    vectors_dir: str,
    best_layer: int = None,
) -> str:
    os.makedirs(vectors_dir, exist_ok=True)
    path = os.path.join(vectors_dir, f"{source}_dom.pt")
    payload = {
        'v_truth':   v_truth,
        'method':    'best_layer_dom',
        'model_tag': model_tag,
        'source':    source,
    }
    if best_layer is not None:
        payload['best_layer'] = best_layer
    torch.save(payload, path)
    print(f"Saved DoM vector -> {path}  shape={tuple(v_truth.shape)}")
    return path
