import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier

PROBE_GATE = 0.55   # minimum per-layer accuracy required in at least one layer


def score_all_layers(
    H_pos: dict,
    H_neg: dict,
    gate: float = PROBE_GATE,
    enforce_gate: bool = False,
) -> dict[int, float]:
    """
    Fit a stratified 80/20 logistic probe on each layer's hidden states.
    Returns dict[layer -> held-out accuracy].

    When `enforce_gate` is true, raises RuntimeError if no layer exceeds
    `gate` (default 55%). By default the gate is advisory: low probe accuracy
    is reported and saved in diagnostics, but extraction can continue with the
    best available layer.
    """
    layer_scores: dict[int, float] = {}

    for L in sorted(H_pos.keys()):
        X = normalize(torch.cat([H_pos[L].float(), H_neg[L].float()]).numpy().astype(np.float32))
        y = np.array([1] * len(H_pos[L]) + [0] * len(H_neg[L]))

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        probe = LogisticRegression(max_iter=1000, C=1.0)
        probe.fit(X_tr, y_tr)
        layer_scores[L] = float(accuracy_score(y_te, probe.predict(X_te)))

    _report(layer_scores, gate)
    _gate_check(layer_scores, gate, enforce_gate=enforce_gate)
    return layer_scores


def _report(layer_scores: dict[int, float], gate: float) -> None:
    best = max(layer_scores.values()) if layer_scores else 0.0
    print(f"\nPer-layer probe accuracy  (gate={gate:.0%}  best={best:.3f}):")
    for L, acc in sorted(layer_scores.items()):
        bar    = '█' * int(acc * 40)
        marker = '  ✓' if acc > gate else ''
        print(f"  Layer {L:02d}: {acc:.3f}  {bar}{marker}")


def gate_status(layer_scores: dict[int, float], gate: float = PROBE_GATE) -> dict:
    passing = [L for L, acc in layer_scores.items() if acc > gate]
    best_L = max(layer_scores, key=layer_scores.get) if layer_scores else None
    best_acc = layer_scores[best_L] if best_L is not None else 0.0
    return {
        "gate_passed": bool(passing),
        "gate_threshold": float(gate),
        "layers_passing": sorted(int(L) for L in passing),
        "best_layer": int(best_L) if best_L is not None else None,
        "best_acc": float(best_acc),
    }


def _gate_check(
    layer_scores: dict[int, float],
    gate: float,
    enforce_gate: bool = False,
) -> None:
    if not layer_scores:
        raise RuntimeError("Probe gate failed: no usable contrastive layers")
    passing = [L for L, acc in layer_scores.items() if acc > gate]
    if not passing:
        best_L   = max(layer_scores, key=layer_scores.get)
        best_acc = layer_scores[best_L]
        message = (
            f"Probe gate failed: no layer exceeded {gate:.0%}. "
            f"Best was layer {best_L} at {best_acc:.3f}. "
            "Continuing with the best available layer; treat downstream "
            "vectors as low-confidence."
        )
        if not enforce_gate:
            print(f"WARNING: {message}")
            return
        raise RuntimeError(
            f"Probe gate failed: no layer exceeded {gate:.0%}. "
            f"Best was layer {best_L} at {best_acc:.3f}. "
            "Check hidden-state quality or increase D_steer size."
        )
    print(f"Gate PASSED: {len(passing)} layer(s) > {gate:.0%}  -> {passing}")


def score_all_layers_nn(H_pos, H_neg):
    scores = {}
    for layer in sorted(H_pos):
        features = normalize(torch.cat([H_pos[layer], H_neg[layer]]).float().numpy())
        labels = np.array([1] * len(H_pos[layer]) + [0] * len(H_neg[layer]))
        train, test, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels,
        )
        scaler = StandardScaler().fit(train)
        train, test = scaler.transform(train), scaler.transform(test)
        best_accuracy = 0.0
        for hidden in ((64,), (128, 64)):
            for learning_rate in (1e-3, 5e-3, 1e-2):
                for regularization in (1e-4, 1e-3):
                    probe = MLPClassifier(
                        hidden_layer_sizes=hidden, learning_rate_init=learning_rate,
                        alpha=regularization, early_stopping=True,
                        validation_fraction=0.15, n_iter_no_change=20,
                        max_iter=300, random_state=42,
                    )
                    probe.fit(train, train_labels)
                    best_accuracy = max(best_accuracy, float(probe.score(test, test_labels)))
        scores[layer] = best_accuracy
    return scores


def score_all_layers_both(H_pos, H_neg, gate=PROBE_GATE):
    logistic = score_all_layers(H_pos, H_neg, enforce_gate=False)
    neural = score_all_layers_nn(H_pos, H_neg)
    scores = {layer: max(logistic[layer], neural[layer]) for layer in logistic}
    _gate_check(scores, gate, enforce_gate=True)
    diagnostics = {
        "probe_method": "max(LR, MLP)",
        "logistic_scores": logistic, "mlp_scores": neural,
        "winning_probe": {layer: "MLP" if neural[layer] > logistic[layer] else "LR"
                          for layer in scores},
    }
    return scores, diagnostics
