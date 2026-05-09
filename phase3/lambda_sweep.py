"""Phase 3 λ sweep: grid search over (λ_a, λ_m) to confirm L_ans dominates.

Runs tune_alpha for max_epochs=2 at each of the 16 combinations.
Selection rule: lowest ES val loss among non-collapsed combinations.
Norm-collapse criterion: λ_m · L_mag > COLLAPSE_RATIO · max(λ_a · L_align, ε).
"""
import json
import os

import torch

from phase3.alpha import tune_alpha

LAMBDA_A_GRID   = [0.0, 0.01, 0.1, 1.0]
LAMBDA_M_GRID   = [0.0, 0.01, 0.1, 1.0]
COLLAPSE_RATIO  = 0.5    # flag if weighted L_mag exceeds half of weighted L_align


def sweep_lambda_grid(
    model,
    tokenizer,
    D_sub: list,
    v_truth: torch.Tensor,
    layer_star: int,
    device: str,
    model_tag: str,
    ratio: float = 0.7,
    out_path: str = 'lambda_sweep.json',
    max_epochs: int = 2,
) -> dict:
    """
    Run tune_alpha for all 16 (λ_a, λ_m) combinations on D_sub (≤200 examples).
    Selects the combination minimising ES val loss subject to no norm collapse.
    Saves full grid to out_path.

    Returns {'lambda_a': ..., 'lambda_m': ...} for the selected combination.
    """
    rows = []

    for la in LAMBDA_A_GRID:
        for lm in LAMBDA_M_GRID:
            print(f"\n[λ-sweep] λ_a={la}  λ_m={lm}")
            _, history = tune_alpha(
                model, tokenizer, D_sub, v_truth, layer_star, device,
                model_tag=model_tag, ratio=ratio,
                lambda_a=la, lambda_m=lm,
                max_epochs=max_epochs,
                es_patience=2,
            )

            if history:
                last         = history[-1]
                L_ans_val    = last.get('L_ans',   float('inf'))
                la_L_align   = la * last.get('L_align', 0.0)
                lm_L_mag     = lm * last.get('L_mag',   0.0)
                es_loss      = last.get('es_loss', float('inf'))
                collapse     = lm_L_mag > COLLAPSE_RATIO * max(la_L_align, 1e-6)
            else:
                L_ans_val  = float('inf')
                la_L_align = 0.0
                lm_L_mag   = 0.0
                es_loss    = float('inf')
                collapse   = False

            flag = ' !! COLLAPSE' if collapse else ''
            print(f"  es={es_loss:.4f}  L_ans={L_ans_val:.4f}  "
                  f"λ_a·L_align={la_L_align:.4f}  λ_m·L_mag={lm_L_mag:.4f}{flag}")

            rows.append({
                'lambda_a':          la,
                'lambda_m':          lm,
                'es_loss':           es_loss,
                'L_ans':             L_ans_val,
                'L_align_weighted':  la_L_align,
                'L_mag_weighted':    lm_L_mag,
                'norm_collapse':     collapse,
            })

    # Select best non-collapsed combination
    valid = [r for r in rows if not r['norm_collapse']]
    if not valid:
        print("[λ-sweep] All combinations flag as collapsed — falling back to (0.1, 0.01)")
        best_la, best_lm = 0.1, 0.01
    else:
        best_row = min(valid, key=lambda r: r['es_loss'])
        best_la  = best_row['lambda_a']
        best_lm  = best_row['lambda_m']

    # Summary table
    print(f"\n[λ-sweep] Grid summary  ({model_tag})")
    print(f"  {'λ_a':>6}  {'λ_m':>6}  {'es_loss':>9}  {'L_ans':>8}  "
          f"{'collapse':>8}")
    print(f"  {'─' * 50}")
    for r in rows:
        star = ' *' if (r['lambda_a'] == best_la and r['lambda_m'] == best_lm) else '  '
        cflag = '     YES' if r['norm_collapse'] else '      no'
        print(f"  {r['lambda_a']:>6.3f}  {r['lambda_m']:>6.3f}  "
              f"{r['es_loss']:>9.4f}  {r['L_ans']:>8.4f}  {cflag}{star}")
    print(f"\n  Selected: λ_a={best_la}  λ_m={best_lm}")

    selected = {'lambda_a': best_la, 'lambda_m': best_lm}
    payload  = {
        'model_tag': model_tag,
        'selected':  selected,
        'grid':      rows,
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  -> {out_path}")

    return selected
