"""Per-head probing for ITI: probe each (layer, head) pair, select top-K, compute sigmas."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


def probe_all_heads(
    H_pos: dict,
    H_neg: dict,
    top_k: int = 48,
) -> tuple[list, dict, dict, dict]:
    """
    Train a linear (logistic) probe on each (layer, head) pair.
    Returns:
        top_heads:       list of (layer_idx, head_idx) sorted by probe acc descending
        head_scores:     dict[(layer, head) -> float]
        head_directions: dict[(layer, head) -> Tensor(head_dim,)]  unit-norm DoM vector
        head_sigmas:     dict[(layer, head) -> float]  std dev along truth direction
    """
    head_scores: dict = {}
    head_directions: dict = {}
    head_sigmas: dict = {}

    all_keys = sorted(H_pos.keys())
    total = len(all_keys)

    for i, key in enumerate(all_keys):
        H_p = H_pos[key].numpy().astype(np.float32)
        H_n = H_neg[key].numpy().astype(np.float32)

        n_min = min(len(H_p), len(H_n))
        rng = np.random.default_rng(42)
        H_p = H_p[rng.choice(len(H_p), n_min, replace=False)]
        H_n = H_n[rng.choice(len(H_n), n_min, replace=False)]

        X_raw = np.vstack([H_p, H_n])
        y = np.array([1] * n_min + [0] * n_min)

        X = normalize(X_raw, norm='l2')
        n_tr = int(0.8 * len(X))
        X_tr, X_te = X[:n_tr], X[n_tr:]
        y_tr, y_te = y[:n_tr], y[n_tr:]

        probe = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        probe.fit(X_tr, y_tr)
        head_scores[key] = float(probe.score(X_te, y_te))

        # DoM direction in head space
        mu_pos = torch.from_numpy(H_pos[key].float().mean(0).numpy())
        mu_neg = torch.from_numpy(H_neg[key].float().mean(0).numpy())
        v_raw = mu_pos - mu_neg
        v_norm = v_raw / (v_raw.norm() + 1e-8)
        head_directions[key] = v_norm

        # Sigma: std dev of projections of all training data onto truth direction
        H_all = torch.cat([
            H_pos[key].float(),
            H_neg[key].float(),
        ])
        projections = H_all @ v_norm  # [n]
        head_sigmas[key] = float(projections.std())

        if (i + 1) % 100 == 0:
            print(f"  Probed {i+1}/{total} (layer,head) pairs…")

    top_heads = sorted(head_scores, key=head_scores.get, reverse=True)[:top_k]

    _print_head_probe_summary(top_heads, head_scores)

    return top_heads, head_scores, head_directions, head_sigmas


def _print_head_probe_summary(top_heads: list, head_scores: dict) -> None:
    if not top_heads:
        print("  No heads selected.")
        return
    best = top_heads[0]
    print(f"\n  ITI head selection complete:")
    print(f"  Top-{len(top_heads)} heads.  Best: L={best[0]:02d} H={best[1]:02d}  "
          f"acc={head_scores[best]:.4f}")
    print(f"  {'Layer':>5}  {'Head':>4}  {'Probe acc':>10}")
    for key in top_heads[:20]:
        print(f"    {key[0]:02d}     {key[1]:02d}    {head_scores[key]:.4f}")
    if len(top_heads) > 20:
        print(f"    … ({len(top_heads) - 20} more)")
    tail = top_heads[-1]
    print(f"  Worst selected: L={tail[0]:02d} H={tail[1]:02d}  acc={head_scores[tail]:.4f}")
