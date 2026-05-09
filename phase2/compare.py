import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def compare_methods(
    H_pos: dict[int, torch.Tensor],
    H_neg: dict[int, torch.Tensor],
    v_truth: torch.Tensor,
    U_truth: torch.Tensor,
    selected_layers: list[int],
) -> tuple[str, dict]:
    """
    Project H+/H- through DoM and cPCA subspace; compare held-out probe accuracy.
    Returns (winner, {'dom': acc_dom, 'cpca': acc_cpca}).

    Decision rule (spec §2.6):
      cPCA wins if its probe acc exceeds DoM probe acc by more than 2 pp;
      otherwise DoM wins (simpler, more robust).
    """
    layers = [L for L in selected_layers if L in H_pos]
    if not layers:
        print("No selected layers in H_pos — defaulting to 'dom'")
        return 'dom', {'dom': 0.0, 'cpca': 0.0}

    H_pos_sel = torch.cat([H_pos[L] for L in layers])
    H_neg_sel = torch.cat([H_neg[L] for L in layers])
    H_all = torch.cat([H_pos_sel, H_neg_sel]).numpy().astype(np.float32)
    y     = np.array([1] * len(H_pos_sel) + [0] * len(H_neg_sel))

    X_tr, X_te, y_tr, y_te = train_test_split(
        H_all, y, test_size=0.2, random_state=42, stratify=y
    )

    # Method A: 1-D projection onto v_truth (DoM)
    v_np = v_truth.numpy().reshape(-1, 1)
    probe_dom = LogisticRegression(max_iter=500)
    probe_dom.fit(X_tr @ v_np, y_tr)
    acc_dom = float(accuracy_score(y_te, probe_dom.predict(X_te @ v_np)))

    # Method B: r-D projection onto U_truth subspace (cPCA)
    U_np    = U_truth.numpy()
    scaler  = StandardScaler()
    proj_tr = scaler.fit_transform(X_tr @ U_np)
    proj_te = scaler.transform(X_te @ U_np)
    probe_sub = LogisticRegression(max_iter=500)
    probe_sub.fit(proj_tr, y_tr)
    acc_cpca = float(accuracy_score(y_te, probe_sub.predict(proj_te)))

    gap    = acc_cpca - acc_dom
    winner = 'cpca' if gap > 0.02 else 'dom'
    label  = 'B (cPCA)' if winner == 'cpca' else 'A (DoM)'

    print(f"\nMethod comparison on held-out D_steer:")
    print(f"  Method A (DoM):  probe acc = {acc_dom:.3f}")
    print(f"  Method B (cPCA): probe acc = {acc_cpca:.3f}  (gap = {gap:+.3f})")
    print(f"  -> Selected: Method {label}")

    return winner, {'dom': acc_dom, 'cpca': acc_cpca}


def select_best_source_method(
    ccot_res: dict,
    base_res: dict,
) -> tuple[str, str, float]:
    """
    Compare probe accuracy across all four (source, method) combinations:
        (ccot | base) × (dom | cpca)

    Uses method_accs stored in each source's result dict.  Prints a 2×2
    table and returns (best_source, best_method, best_acc).
    """
    def _acc(res: dict, method: str) -> float:
        return res.get('method_accs', {}).get(method) or 0.0

    combos = {
        ('ccot', 'dom'):  _acc(ccot_res, 'dom'),
        ('ccot', 'cpca'): _acc(ccot_res, 'cpca'),
        ('base', 'dom'):  _acc(base_res, 'dom'),
        ('base', 'cpca'): _acc(base_res, 'cpca'),
    }

    best_key = max(combos, key=combos.get)

    print(f"\nSource × Method comparison (held-out D_steer probe accuracy):")
    print(f"  {'source':<8}  {'DoM':>8}  {'cPCA':>8}")
    print(f"  {'─' * 28}")
    for source in ('ccot', 'base'):
        row = []
        for method in ('dom', 'cpca'):
            acc  = combos[(source, method)]
            mark = ' *' if (source, method) == best_key else '  '
            row.append(f"{acc:>7.3f}{mark}")
        print(f"  {source:<8}  {'  '.join(row)}")

    best_source, best_method = best_key
    best_acc = combos[best_key]
    print(f"\n  Winner: source={best_source}  method={best_method}  "
          f"probe_acc={best_acc:.3f}")
    return best_source, best_method, best_acc
