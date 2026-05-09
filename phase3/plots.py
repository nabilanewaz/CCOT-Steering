"""Phase 3 diagnostic plots: α-tuning loss curves and α-sweep accuracy curve."""
import os


def plot_loss_curves(history: list, out_path: str) -> None:
    """
    Plot L_ans, L_align, L_mag, and total training loss per epoch.
    Saved as a PNG. Silently skips if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[plots] matplotlib not installed — skipping loss curves")
        return

    if not history:
        return

    epochs  = [h['epoch']       for h in history]
    L_ans   = [h.get('L_ans',   0.0) for h in history]
    L_align = [h.get('L_align', 0.0) for h in history]
    L_mag   = [h.get('L_mag',   0.0) for h in history]
    total   = [h.get('total_train', 0.0) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: all components on one axis
    ax = axes[0]
    ax.plot(epochs, L_ans,   label='L_ans',   marker='o', linewidth=2, color='steelblue')
    ax.plot(epochs, L_align, label='L_align', marker='s', linestyle='--', color='darkorange')
    ax.plot(epochs, L_mag,   label='L_mag',   marker='^', linestyle=':', color='green')
    ax.plot(epochs, total,   label='total',   marker='x', color='black', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training loss components')
    ax.legend(fontsize=8)
    ax.set_xticks(epochs)

    # Right: L_ans fraction to confirm it dominates
    ax2 = axes[1]
    dom_frac = [
        la / max(t, 1e-8) for la, t in zip(L_ans, total)
    ]
    ax2.plot(epochs, dom_frac, marker='o', color='steelblue', linewidth=2)
    ax2.axhline(0.9, color='red', linestyle='--', linewidth=1, label='90% threshold')
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L_ans / total')
    ax2.set_title('L_ans dominance fraction')
    ax2.legend(fontsize=8)
    ax2.set_xticks(epochs)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plots] Loss curves -> {out_path}")


def plot_alpha_diagnostic(sweep_data: dict, out_path: str) -> None:
    """
    Plot accuracy vs α from the diagnostic grid sweep.
    Marks the learned α* with a vertical dashed line.
    Saved as a PNG. Silently skips if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[plots] matplotlib not installed — skipping alpha diagnostic plot")
        return

    sweep      = sweep_data.get('sweep', [])
    alpha_star = sweep_data.get('alpha_star', None)
    model_tag  = sweep_data.get('model_tag', '')

    if not sweep:
        return

    alphas = [s['alpha']    for s in sweep]
    accs   = [s['accuracy'] for s in sweep]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, accs, marker='o', linewidth=2, label='Accuracy', color='steelblue')

    if alpha_star is not None:
        ax.axvline(alpha_star, color='red', linestyle='--', linewidth=1.5,
                   label=f'α* = {alpha_star:.2f}')

    ax.set_xlabel('α  (steering strength)')
    ax.set_ylabel('Accuracy on D_val subset')
    ax.set_xscale('symlog', linthresh=0.1)
    ax.set_title(f'Diagnostic α sweep — {model_tag}')
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plots] Alpha diagnostic -> {out_path}")


def plot_lambda_sweep_heatmap(sweep_data: dict, out_path: str) -> None:
    """
    4×4 heatmap of ES val loss over the (λ_a, λ_m) grid.
    Collapsed cells are hatched. Saved as PNG.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"[plots] matplotlib not installed — skipping λ heatmap")
        return

    grid     = sweep_data.get('grid', [])
    selected = sweep_data.get('selected', {})
    if not grid:
        return

    la_vals = sorted(set(r['lambda_a'] for r in grid))
    lm_vals = sorted(set(r['lambda_m'] for r in grid))

    Z        = np.full((len(lm_vals), len(la_vals)), float('nan'))
    collapse = np.zeros((len(lm_vals), len(la_vals)), dtype=bool)

    for r in grid:
        i = lm_vals.index(r['lambda_m'])
        j = la_vals.index(r['lambda_a'])
        Z[i, j]        = r['es_loss']
        collapse[i, j] = r['norm_collapse']

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = float(np.nanpercentile(Z, 95))
    im   = ax.imshow(Z, aspect='auto', origin='lower',
                     vmin=float(np.nanmin(Z)), vmax=vmax, cmap='viridis_r')

    # Hatch collapsed cells
    for i in range(len(lm_vals)):
        for j in range(len(la_vals)):
            if collapse[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, hatch='////', edgecolor='red'))

    # Star on selected
    if selected:
        try:
            sj = la_vals.index(selected['lambda_a'])
            si = lm_vals.index(selected['lambda_m'])
            ax.plot(sj, si, '*', markersize=16, color='white', markeredgecolor='black')
        except ValueError:
            pass

    ax.set_xticks(range(len(la_vals)))
    ax.set_xticklabels([str(v) for v in la_vals])
    ax.set_yticks(range(len(lm_vals)))
    ax.set_yticklabels([str(v) for v in lm_vals])
    ax.set_xlabel('λ_a')
    ax.set_ylabel('λ_m')
    ax.set_title(f"λ sweep ES loss — {sweep_data.get('model_tag', '')}")
    plt.colorbar(im, ax=ax, label='ES val loss')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plots] λ heatmap -> {out_path}")
