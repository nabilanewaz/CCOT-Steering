"""Compare methods on an outer holdout excluded from vector extraction."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


def compare_methods(H_pos, H_neg, v_truth, U_truth, selected_layers, heldout=None, best_layer=None):
    if heldout is None:
        raise ValueError("Method comparison requires an independent extraction holdout")
    test_pos, test_neg = heldout
    layer = best_layer if best_layer is not None else selected_layers[0]
    train_states = torch.cat([H_pos[layer], H_neg[layer]]).float().numpy()
    test_states = torch.cat([test_pos[layer], test_neg[layer]]).float().numpy()
    train_labels = np.array([1] * len(H_pos[layer]) + [0] * len(H_neg[layer]))
    test_labels = np.array([1] * len(test_pos[layer]) + [0] * len(test_neg[layer]))
    train_unit, test_unit = normalize(train_states), normalize(test_states)
    direction = v_truth.float().numpy()
    features = {"dom": ((train_unit @ direction)[:, None], (test_unit @ direction)[:, None])}
    if U_truth is not None:
        basis = U_truth.float().numpy()
        features["cpca"] = (np.linalg.norm(train_unit @ basis, axis=1)[:, None],
                            np.linalg.norm(test_unit @ basis, axis=1)[:, None])
    scores = {}
    for method, (train, test) in features.items():
        probe = LogisticRegression(max_iter=1000).fit(train, train_labels)
        scores[method] = float(probe.score(test, test_labels))
    winner = max(scores, key=scores.get)
    return winner, scores


def select_best_source_method(ccot_res, base_res):
    scores = {
        (source, method): accuracy
        for source, result in (("ccot", ccot_res), ("base", base_res))
        for method, accuracy in result.get("method_accs", {}).items()
    }
    source, method = max(scores, key=scores.get)
    return source, method, scores[(source, method)]
