# ─────────────────────────────────────────────────────────────────────────────
# CCOT Probe Test — LR vs MLP + Multi-Layer DoM (run in Google Colab)
#
# Steps:
#   1. Run Cell 1 to install dependencies
#   2. Run Cell 2 to upload  vectors/S2/qwen25_math1.5b/ccot_hstates_cache.pt
#   3. Run Cell 3 onwards
# ─────────────────────────────────────────────────────────────────────────────

# ══ CELL 1 — Install ══════════════════════════════════════════════════════════
# !pip install torch --index-url https://download.pytorch.org/whl/cpu -q
# !pip install scikit-learn matplotlib -q

# ══ CELL 2 — Upload hidden states cache ══════════════════════════════════════
from google.colab import files
uploaded = files.upload()   # select ccot_hstates_cache.pt from your machine

# ══ CELL 3 — Load hidden states ══════════════════════════════════════════════
import torch, numpy as np

CACHE_PATH = 'ccot_hstates_cache.pt'

data  = torch.load(CACHE_PATH, map_location='cpu')
H_pos = data['H_pos']   # dict[layer_idx -> Tensor (n_pos, d)]
H_neg = data['H_neg']   # dict[layer_idx -> Tensor (n_neg, d)]

L0 = sorted(H_pos.keys())[0]
n_pos, n_neg = len(H_pos[L0]), len(H_neg[L0])
d = H_pos[L0].shape[-1]
print(f"Layers  : {sorted(H_pos.keys())}")
print(f"H+      : {n_pos} samples")
print(f"H-      : {n_neg} samples")
print(f"Total   : {n_pos + n_neg}")
print(f"Hidden d: {d}")
print(f"Imbalance ratio: {n_pos/n_neg:.2f}:1")

# ══ CELL 4 — Preprocessing helper (1:1 balanced) ════════════════════════════
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split

def build_xy(H_pos, H_neg, L):
    """Undersample larger class to 1:1, L2-normalise, 80/20 split, StandardScaler."""
    H_p, H_n = H_pos[L], H_neg[L]
    n_min = min(len(H_p), len(H_n))

    # Fixed-seed undersample so all layers use the same examples
    rng   = np.random.default_rng(42)
    p_idx = rng.choice(len(H_p), size=n_min, replace=False)
    n_idx = rng.choice(len(H_n), size=n_min, replace=False)

    X_raw = np.vstack([H_p[p_idx].numpy().astype(np.float32),
                       H_n[n_idx].numpy().astype(np.float32)])
    y = np.array([1]*n_min + [0]*n_min)

    X = normalize(X_raw, norm='l2')

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), y_tr, y_te

n_min = min(n_pos, n_neg)
print(f"\nbuild_xy: undersampling H+ from {n_pos} → {n_min}  (1:1 balanced, {n_min*2} total)")

# ══ CELL 5 — Logistic Regression probe ═══════════════════════════════════════
from sklearn.linear_model import LogisticRegression

lr_scores = {}
for L in sorted(H_pos.keys()):
    X_tr, X_te, y_tr, y_te = build_xy(H_pos, H_neg, L)
    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(X_tr, y_tr)
    lr_scores[L] = probe.score(X_te, y_te)
    print(f"  Layer {L:02d}: LR acc = {lr_scores[L]:.4f}")

print(f"\nBest LR layer: L={max(lr_scores, key=lr_scores.get)}  "
      f"acc={max(lr_scores.values()):.4f}")

# ══ CELL 6 — MLP probe (64) + (128→64) grid search ═══════════════════════════
from sklearn.neural_network import MLPClassifier

LR_GRID     = [1e-3, 5e-3, 1e-2]
ALPHA_GRID  = [1e-4, 1e-3]
HIDDEN_GRID = [(64,), (128, 64)]

mlp_scores      = {}
mlp_best_params = {}

for L in sorted(H_pos.keys()):
    X_tr, X_te, y_tr, y_te = build_xy(H_pos, H_neg, L)

    best_acc, best_p = 0.0, {}
    for hidden in HIDDEN_GRID:
        for lr in LR_GRID:
            for alpha in ALPHA_GRID:
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
                    best_p   = {'hidden': hidden, 'lr': lr, 'alpha': alpha}

    mlp_scores[L]      = best_acc
    mlp_best_params[L] = best_p
    print(f"  Layer {L:02d}: MLP acc = {best_acc:.4f}  {best_p}")

print(f"\nBest MLP layer: L={max(mlp_scores, key=mlp_scores.get)}  "
      f"acc={max(mlp_scores.values()):.4f}")

# ══ CELL 7 — Side-by-side comparison table ════════════════════════════════════
GATE = 0.55

print(f"\n{'Layer':>6}  {'LR':>7}  {'MLP':>7}  {'Best':>7}  {'Winner':>6}")
print('─' * 48)
combined = {}
for L in sorted(lr_scores.keys()):
    lr_a  = lr_scores[L]
    mlp_a = mlp_scores[L]
    best  = max(lr_a, mlp_a)
    combined[L] = best
    winner = 'MLP' if mlp_a > lr_a else 'LR '
    marker = '  ✓' if best > GATE else ''
    print(f"  {L:02d}    {lr_a:.4f}  {mlp_a:.4f}  {best:.4f}    {winner}{marker}")

best_L = max(combined, key=combined.get)
print(f"\nBest layer overall : L={best_L}  combined_acc={combined[best_L]:.4f}")
print(f"  LR  : {lr_scores[best_L]:.4f}")
print(f"  MLP : {mlp_scores[best_L]:.4f}")

# ══ CELL 8 — Multi-layer DoM extraction ═════════════════════════════════════
# Compute unit-norm DoM vector at every layer using balanced H+/H-
dom_vectors = {}
rng_b = np.random.default_rng(42)
for L in H_pos:
    H_p = H_pos[L]
    H_n = H_neg[L]
    n_min_L = min(len(H_p), len(H_n))
    p_idx = rng_b.choice(len(H_p), size=n_min_L, replace=False)
    n_idx = rng_b.choice(len(H_n), size=n_min_L, replace=False)
    mu_pos = H_p[p_idx].float().mean(dim=0)
    mu_neg = H_n[n_idx].float().mean(dim=0)
    v_raw  = mu_pos - mu_neg
    dom_vectors[L] = v_raw / (v_raw.norm() + 1e-8)

# Pick top-3 layers by combined probe score
top3 = sorted(combined, key=combined.get, reverse=True)[:3]
print(f"Top-3 layers by probe score: {top3}")
for L in top3:
    print(f"  L={L:02d}  probe_acc={combined[L]:.4f}  "
          f"dom_raw_norm={((H_pos[L].float().mean(0) - H_neg[L].float().mean(0)).norm()):.4f}")

# Simulate save_multilayer_dom_vectors
multilayer_payload = {
    'top_layers':    top3,
    'layer_vectors': {L: dom_vectors[L] for L in top3},
    'layer_scores':  {L: combined[L]    for L in top3},
    'top_k':         3,
}
torch.save(multilayer_payload, 'ccot_multilayer_dom_test.pt')
loaded = torch.load('ccot_multilayer_dom_test.pt', map_location='cpu')
print(f"\nSave/load verified  top_layers={loaded['top_layers']}")
for L in loaded['top_layers']:
    v = loaded['layer_vectors'][L]
    print(f"  L={L:02d}  shape={tuple(v.shape)}  norm={v.norm().item():.6f}")

# ══ CELL 9 — Multi-layer SVM 5-fold CV (balanced) ════════════════════════════
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

def cv_multilayer(layers, label, pca_k=50):
    X_pos = torch.cat([H_pos[L] for L in layers], dim=-1).numpy().astype(np.float32)
    X_neg = torch.cat([H_neg[L] for L in layers], dim=-1).numpy().astype(np.float32)
    # Balance
    n_m   = min(len(X_pos), len(X_neg))
    rng2  = np.random.default_rng(42)
    X_pos = X_pos[rng2.choice(len(X_pos), n_m, replace=False)]
    X_neg = X_neg[rng2.choice(len(X_neg), n_m, replace=False)]
    X     = normalize(np.vstack([X_pos, X_neg]), norm='l2')
    y     = np.array([1]*n_m + [0]*n_m)

    from sklearn.preprocessing import StandardScaler
    X_sc  = StandardScaler().fit_transform(X)
    pipe  = Pipeline([('pca', PCA(n_components=min(pca_k, X_sc.shape[1]-1))),
                      ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))])
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s     = cross_val_score(pipe, X_sc, y, cv=cv)
    print(f"  {label}  layers={layers}  d={X.shape[1]}  "
          f"5-fold CV: {s.mean():.4f} ± {s.std():.4f}  folds={[f'{x:.3f}' for x in s]}")
    return s.mean()

print("Multi-layer SVM 5-fold CV (balanced):")
cv_multilayer(top3,                                           "top-3 (probe best)")
top3_dom = sorted(dom_vectors, key=lambda L: (H_pos[L].float().mean(0)-H_neg[L].float().mean(0)).norm(), reverse=True)[:3]
cv_multilayer(top3_dom,                                       "top-3 (dom norm)")
cv_multilayer(sorted(combined, key=combined.get, reverse=True)[:5], "top-5 (probe best)")

print(f"\nSingle-layer best: L={best_L}  MLP_acc={mlp_scores[best_L]:.4f}")

# ══ CELL 10 — Plot ════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

layers   = sorted(lr_scores.keys())
lr_vals  = [lr_scores[L]  for L in layers]
mlp_vals = [mlp_scores[L] for L in layers]

plt.figure(figsize=(14, 5))
plt.plot(layers, lr_vals,  'b-o', label='LR (balanced)',        linewidth=2)
plt.plot(layers, mlp_vals, 'r-s', label='MLP (balanced, grid)', linewidth=2)
plt.axhline(GATE,  color='gray',   linestyle='--', label=f'Gate ({GATE})')
plt.axvline(best_L, color='purple', linestyle=':', label=f'Best L={best_L}')
for L in top3:
    plt.axvline(L, color='orange', linestyle=':', alpha=0.5)
plt.xlabel('Layer')
plt.ylabel('Held-out Accuracy')
plt.title('LR vs MLP Probe (1:1 balanced) — per layer (qwen25_math1.5b)')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(layers)
plt.tight_layout()
plt.savefig('probe_comparison_balanced.png', dpi=150)
plt.show()
print("Saved: probe_comparison_balanced.png")
