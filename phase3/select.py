"""Select the best steered configuration from Phase 3 results (spec §3.9)."""
import json
import os
import re
from math import sqrt


def _wilson_lower(accuracy: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p      = accuracy
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    margin = (z * sqrt(max(p * (1 - p) / n + z ** 2 / (4 * n ** 2), 0.0))) / (1 + z ** 2 / n)
    return centre - margin


def select_best_steered_config(
    results_dir: str,
    model_tag: str,
) -> dict:
    """
    Read phase3_val.json, pick the steered condition with the highest
    accuracy on D_val (flip rate is the tiebreaker).
    Writes phase3_best_config.yaml and returns the selection dict.
    """
    ph3_path = os.path.join(results_dir, 'phase3_val.json')
    if not os.path.exists(ph3_path):
        raise FileNotFoundError(f"phase3_val.json missing: {ph3_path}")

    with open(ph3_path) as f:
        records = json.load(f)

    # Steered conditions: vector_method in ('dom', 'cpca')
    steered = [r for r in records if r.get('vector_method') in ('dom', 'cpca')]
    if not steered:
        print(f"[PH3-select] No steered results found in {ph3_path}")
        return {}

    best = max(steered, key=lambda r: (r['accuracy'], r['flip_rate']))

    n = best.get('n_examples', 1) or 1
    wl = _wilson_lower(best['accuracy'], n)

    # CCoT accuracy at same latent-token budget (reference)
    latent_match = re.search(r'_L(\d+)', best.get('condition', ''))
    latent_tokens = int(latent_match.group(1)) if latent_match else None
    ccot_cond = f"ccot_L{latent_tokens}" if latent_tokens else None
    ccot_rec    = next((r for r in records if r['condition'] == ccot_cond), None)
    ccot_acc    = ccot_rec['accuracy'] if ccot_rec else None

    selection = {
        'model_tag':        model_tag,
        'best_condition':   best['condition'],
        'latent_tokens':    latent_tokens,
        'ratio':            best.get('ratio'),
        'vector_source':    best.get('vector_source'),
        'vector_method':    best.get('vector_method'),
        'alpha_star':       best.get('alpha'),
        'steered_accuracy': best['accuracy'],
        'ccot_accuracy':    ccot_acc,
        'flip_rate':        best['flip_rate'],
        'reasoning_tokens': best.get('reasoning_tokens'),
        'actual_ratio':     best.get('actual_ratio'),
        'wilson_lower_95':  wl,
        'selection_metric': 'accuracy_then_flip_rate',
    }

    print(f"\n[PH3-select] {model_tag}: {best['condition']}")
    print(f"  acc={best['accuracy']:.4f}  Wilson95lo={wl:.4f}  "
          f"flip={best['flip_rate']:.4f}  α*={best.get('alpha'):.4f}")
    if ccot_acc is not None:
        print(f"  vs CCoT acc={ccot_acc:.4f}  "
              f"gain={best['accuracy'] - ccot_acc:+.4f}")

    # Write YAML
    out_path = os.path.join(results_dir, 'phase3_best_config.yaml')
    try:
        import yaml
        with open(out_path, 'w') as f:
            yaml.dump(selection, f, default_flow_style=False)
    except ImportError:
        with open(out_path, 'w') as f:
            for k, v in selection.items():
                f.write(f"{k}: {v}\n")
    print(f"  -> {out_path}")

    return selection
