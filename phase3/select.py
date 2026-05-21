"""Select the best steered configuration from Phase 3 results (spec §3.9)."""
import json
import os
import re
from math import sqrt
from pathlib import Path


def _wilson_lower(accuracy: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p      = accuracy
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    margin = (z * sqrt(max(p * (1 - p) / n + z ** 2 / (4 * n ** 2), 0.0))) / (1 + z ** 2 / n)
    return centre - margin




def _load_phase2_meta(results_dir: str) -> dict:
    candidates = [Path(results_dir) / 'phase2_meta.json']
    parts = Path(results_dir).parts
    if 'results' in parts:
        idx = parts.index('results')
        candidates.append(Path(*parts[:idx], 'vectors', *parts[idx + 1:], 'phase2_meta.json'))
    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return {}


def _source_gate_passed(meta: dict, source: str, default_gate: float = 0.55) -> bool:
    gate_info = meta.get(f'{source}_probe_gate') or {}
    if 'gate_passed' in gate_info:
        return bool(gate_info.get('gate_passed'))
    score = meta.get(f'{source}_max_probe_score')
    threshold = gate_info.get('gate_threshold', default_gate)
    return bool(score is not None and float(score) > float(threshold))


def _write_selection_yaml(selection: dict, out_path: str) -> None:
    try:
        import yaml
        with open(out_path, 'w') as f:
            yaml.dump(selection, f, default_flow_style=False)
    except ImportError:
        with open(out_path, 'w') as f:
            for k, v in selection.items():
                f.write(f"{k}: {v}\n")

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

    meta = _load_phase2_meta(results_dir)

    # Steered conditions: vector_method in ('dom', 'cpca'), but only from
    # sources whose Phase 2 probe gate passed. Failed-gate vectors can remain in
    # phase3_val.json for analysis; they are not eligible for locked selection.
    steered_all = [r for r in records if r.get('vector_method') in ('dom', 'cpca')]
    steered = [
        r for r in steered_all
        if _source_gate_passed(meta, str(r.get('vector_source') or ''))
    ]
    out_path = os.path.join(results_dir, 'phase3_best_config.yaml')
    if not steered:
        ccot_records = [r for r in records if str(r.get('condition', '')).startswith('ccot_L')]
        if not ccot_records:
            print(f"[PH3-select] No eligible steered or CCoT results found in {ph3_path}")
            return {}
        best_ccot = max(ccot_records, key=lambda r: r['accuracy'])
        latent_match = re.search(r'_L(\d+)', best_ccot.get('condition', ''))
        latent_tokens = int(latent_match.group(1)) if latent_match else None
        n = best_ccot.get('n_examples', 1) or 1
        wl = _wilson_lower(best_ccot['accuracy'], n)
        selection = {
            'model_tag':        model_tag,
            'best_condition':   best_ccot['condition'],
            'latent_tokens':    latent_tokens,
            'ratio':            best_ccot.get('ratio'),
            'vector_source':    None,
            'vector_method':    'none',
            'alpha_star':       0.0,
            'steered_accuracy': best_ccot['accuracy'],
            'ccot_accuracy':    best_ccot['accuracy'],
            'flip_rate':        0.0,
            'reasoning_tokens': best_ccot.get('reasoning_tokens'),
            'actual_ratio':     best_ccot.get('actual_ratio'),
            'wilson_lower_95':  wl,
            'selection_metric': 'fallback_ccot_probe_gate',
            'selection_reason': 'all_vectors_failed_probe_gate',
            'excluded_vector_sources': sorted({
                str(r.get('vector_source')) for r in steered_all if r.get('vector_source')
            }),
        }
        print(f"\n[PH3-select] {model_tag}: {best_ccot['condition']} (fallback)")
        print(
            f"  acc={best_ccot['accuracy']:.4f}  Wilson95lo={wl:.4f}  "
            "reason=all_vectors_failed_probe_gate"
        )
        _write_selection_yaml(selection, out_path)
        print(f"  -> {out_path}")
        return selection

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

    selection['selection_reason'] = 'best_probe_passing_steered_config'
    selection['excluded_vector_sources'] = sorted({
        str(r.get('vector_source'))
        for r in steered_all
        if r.get('vector_source') and not _source_gate_passed(meta, str(r.get('vector_source')))
    })

    _write_selection_yaml(selection, out_path)
    print(f"  -> {out_path}")

    return selection
