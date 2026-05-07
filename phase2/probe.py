import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def score_all_layers(H_pos: dict, H_neg: dict) -> dict[int, float]:
    """Fit a logistic probe per layer; return dict[layer -> val accuracy]."""
    layer_scores: dict[int, float] = {}

    for L in sorted(H_pos.keys()):
        X = torch.cat([H_pos[L], H_neg[L]]).numpy().astype(np.float32)
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

    print("\nPer-layer probe accuracy:")
    for L, acc in sorted(layer_scores.items()):
        bar = '█' * int(acc * 40)
        print(f"  Layer {L:02d}: {acc:.3f}  {bar}")

    return layer_scores
