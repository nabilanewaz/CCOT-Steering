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
) -> str:
    """
    Project H+/H- through each method and compare linear separability on a
    held-out split. Returns 'dom' or 'cpca'.

    Decision rule (spec §2.6):
      - cPCA wins if its probe acc exceeds DoM probe acc by more than 2 pp.
      - Otherwise DoM wins (simpler, more robust).
    """
    layers = [L for L in selected_layers if L in H_pos]
    if not layers:
        print("No selected layers in H_pos — defaulting to 'dom'")
        return 'dom'

    H_pos_sel = torch.cat([H_pos[L] for L in layers])
    H_neg_sel = torch.cat([H_neg[L] for L in layers])
    H_all = torch.cat([H_pos_sel, H_neg_sel]).numpy().astype(np.float32)
    y     = np.array([1] * len(H_pos_sel) + [0] * len(H_neg_sel))

    X_tr, X_te, y_tr, y_te = train_test_split(
        H_all, y, test_size=0.2, random_state=42, stratify=y
    )

    # Method A: 1-D projection onto v_truth
    v_np = v_truth.numpy().reshape(-1, 1)
    probe_dom = LogisticRegression(max_iter=500)
    probe_dom.fit(X_tr @ v_np, y_tr)
    acc_dom = accuracy_score(y_te, probe_dom.predict(X_te @ v_np))

    # Method B: r-D projection onto U_truth subspace
    U_np    = U_truth.numpy()
    scaler  = StandardScaler()
    proj_tr = scaler.fit_transform(X_tr @ U_np)
    proj_te = scaler.transform(X_te @ U_np)
    probe_sub = LogisticRegression(max_iter=500)
    probe_sub.fit(proj_tr, y_tr)
    acc_sub = accuracy_score(y_te, probe_sub.predict(proj_te))

    print(f"\nMethod comparison on held-out D_steer:")
    print(f"  Method A (DoM):   probe acc = {acc_dom:.3f}")
    print(f"  Method B (cPCA):  probe acc = {acc_sub:.3f}")

    gap = acc_sub - acc_dom
    winner = 'cpca' if gap > 0.02 else 'dom'
    label = 'B (cPCA)' if winner == 'cpca' else 'A (DoM)'
    print(f"  -> Selected: Method {label}  (gap = {gap:+.3f})")
    return winner
