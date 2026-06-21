import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, normalize

PROBE_GATE = 0.55   # minimum per-layer accuracy required in at least one layer


def _build_xy(H_pos: dict, H_neg: dict, L: int):
    H_p, H_n = H_pos[L], H_neg[L]
    n_min = min(len(H_p), len(H_n))

    # Undersample larger class to 1:1 (fixed seed → same examples across all layers)
    rng   = np.random.default_rng(42)
    p_idx = rng.choice(len(H_p), size=n_min, replace=False)
    n_idx = rng.choice(len(H_n), size=n_min, replace=False)

    X_raw = np.vstack([H_p[p_idx].numpy().astype(np.float32),
                       H_n[n_idx].numpy().astype(np.float32)])
    y = np.array([1] * n_min + [0] * n_min)

    X_norm = normalize(X_raw, norm='l2')

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te


# ── Logistic regression probe ──────────────────────────────────────────────────

def score_all_layers(
    H_pos: dict,
    H_neg: dict,
    gate: float = PROBE_GATE,
) -> dict[int, float]:
    """
    Fit a stratified 80/20 logistic probe (L2-normalised inputs) on each layer.
    Returns dict[layer -> held-out accuracy].
    """
    layer_scores: dict[int, float] = {}

    for L in sorted(H_pos.keys()):
        X_tr, X_te, y_tr, y_te = _build_xy(H_pos, H_neg, L)
        probe = LogisticRegression(max_iter=1000, C=1.0)
        probe.fit(X_tr, y_tr)
        layer_scores[L] = float(accuracy_score(y_te, probe.predict(X_te)))

    _report(layer_scores, gate, label='LR')
    _gate_check(layer_scores, gate)
    return layer_scores


# ── MLP probe (sklearn MLPClassifier) ─────────────────────────────────────────

# Grid: hidden_layer_sizes × learning_rate_init × alpha (L2 reg / weight decay)
_HIDDEN_GRID = ((64,), (128, 64))
_LR_GRID     = (1e-3, 5e-3, 1e-2)
_ALPHA_GRID  = (1e-4, 1e-3)


def score_all_layers_nn(
    H_pos: dict,
    H_neg: dict,
    gate: float = PROBE_GATE,
) -> dict[int, float]:
    """
    Train MLP probes on each layer, grid-searching over architecture, lr, and alpha.

    Architectures: (64,) and (128, 64) — 1 or 2 hidden layers
    Preprocessing: L2 row-normalisation → StandardScaler
    Grid search  : hidden ∈ {(64,),(128,64)}  ×  lr ∈ {1e-3,5e-3,1e-2}  ×  alpha ∈ {1e-4,1e-3}
    Overfitting  : early_stopping=True (15 % val split, patience=20 epochs)
    """
    layer_scores: dict[int, float] = {}

    for L in sorted(H_pos.keys()):
        X_tr, X_te, y_tr, y_te = _build_xy(H_pos, H_neg, L)

        best_acc = 0.0
        for hidden in _HIDDEN_GRID:
            for lr in _LR_GRID:
                for alpha in _ALPHA_GRID:
                    mlp = MLPClassifier(
                        hidden_layer_sizes=hidden,
                        max_iter=500,
                        random_state=42,
                        learning_rate_init=lr,
                        alpha=alpha,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        verbose=False,
                    )
                    mlp.fit(X_tr, y_tr)
                    acc = mlp.score(X_te, y_te)
                    if acc > best_acc:
                        best_acc = acc

        layer_scores[L] = best_acc

    _report(layer_scores, gate, label='MLP')
    _gate_check(layer_scores, gate)
    return layer_scores


# ── Combined: run both, side-by-side report, return max per layer ──────────────

def score_all_layers_both(
    H_pos: dict,
    H_neg: dict,
    gate: float = PROBE_GATE,
) -> dict[int, float]:
    """
    Run LR and MLP probes on every layer. Print side-by-side comparison.
    Returns dict[layer -> max(LR_acc, MLP_acc)] used for downstream layer selection.
    """
    n_pos = len(next(iter(H_pos.values())))
    n_neg = len(next(iter(H_neg.values())))
    d     = next(iter(H_pos.values())).shape[-1]
    print(f"\nProbe dataset:  H+={n_pos}  H-={n_neg}  "
          f"total={n_pos+n_neg}  d={d}  layers={len(H_pos)}")
    print("Preprocessing:  L2 row-normalise → StandardScaler (80/20 split)")

    print("\n── Logistic Regression ────────────────────────────────────────────")
    lr_scores = score_all_layers(H_pos, H_neg, gate=gate)

    print("\n── MLP (64 hidden, early stopping, grid lr×alpha) ─────────────────")
    mlp_scores = score_all_layers_nn(H_pos, H_neg, gate=gate)

    # Side-by-side table
    print(f"\n{'Layer':>6}  {'LR':>7}  {'MLP':>7}  {'Best':>7}  {'Winner':>6}")
    print('─' * 44)
    combined: dict[int, float] = {}
    for L in sorted(lr_scores.keys()):
        lr_a  = lr_scores.get(L, 0.0)
        mlp_a = mlp_scores.get(L, 0.0)
        best  = max(lr_a, mlp_a)
        winner = 'MLP' if mlp_a > lr_a else 'LR '
        combined[L] = best
        marker = '  ✓' if best > gate else ''
        print(f"  {L:02d}    {lr_a:.3f}    {mlp_a:.3f}    {best:.3f}    {winner}{marker}")

    best_L = max(combined, key=combined.get)
    print(f"\nBest layer overall: L={best_L}  acc={combined[best_L]:.4f}  "
          f"(LR={lr_scores[best_L]:.4f}  MLP={mlp_scores[best_L]:.4f})")
    return combined


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _report(layer_scores: dict[int, float], gate: float, label: str = '') -> None:
    lbl  = f' [{label}]' if label else ''
    best = max(layer_scores.values()) if layer_scores else 0.0
    print(f"\nPer-layer probe accuracy{lbl}  (gate={gate:.0%}  best={best:.3f}):")
    for L, acc in sorted(layer_scores.items()):
        bar    = '█' * int(acc * 40)
        marker = '  ✓' if acc > gate else ''
        print(f"  Layer {L:02d}: {acc:.3f}  {bar}{marker}")


def _gate_check(layer_scores: dict[int, float], gate: float) -> None:
    passing = [L for L, acc in layer_scores.items() if acc > gate]
    if not passing:
        best_L   = max(layer_scores, key=layer_scores.get)
        best_acc = layer_scores[best_L]
        raise RuntimeError(
            f"Probe gate failed: no layer exceeded {gate:.0%}. "
            f"Best was layer {best_L} at {best_acc:.3f}. "
            "Check hidden-state quality or increase D_steer size."
        )
    print(f"Gate PASSED: {len(passing)} layer(s) > {gate:.0%}  -> {passing}")
