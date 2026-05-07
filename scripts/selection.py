import json
from math import sqrt
import numpy as np
import os


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple:
    p_hat  = correct / n
    centre = (p_hat + z**2 / (2*n)) / (1 + z**2 / n)
    margin = (z * sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / (1 + z**2/n)
    return centre - margin, centre + margin


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _n_items(value) -> int:
    if isinstance(value, int):
        return value
    return len(value)


def compute_config_score(results_dir: str, model_tags: list,
                          cfg_id: str) -> dict:
    scores = {}
    for model_tag in model_tags:
        path = os.path.join(results_dir, cfg_id, model_tag, 'steered_val.json')
        m   = load_json(path)
        n   = m['n_examples']
        cor = round(m['steered_accuracy'] * n)
        lo, hi = wilson_ci(cor, n)
        scores[model_tag] = {
            'accuracy':  m['steered_accuracy'],
            'ci_lower':  lo,
            'ci_upper':  hi,
            'n':         n,
            'flip_rate': m.get('flip_rate', 0.0),
            'probe_acc': m.get('probe_accuracy', 0.0),
        }

    mean_acc   = float(np.mean([s['accuracy']  for s in scores.values()]))
    mean_lower = float(np.mean([s['ci_lower']  for s in scores.values()]))
    mean_flip  = float(np.mean([s['flip_rate'] for s in scores.values()]))
    mean_probe = float(np.mean([s['probe_acc'] for s in scores.values()]))

    return {
        'cfg_id':     cfg_id,
        'per_model':  scores,
        'mean_acc':   mean_acc,
        'mean_lower': mean_lower,
        'mean_flip':  mean_flip,
        'mean_probe': mean_probe,
    }


def print_selection_table(scores: dict):
    print(f"\n{'Config':<8} {'n_val':>7} {'Acc':>8} {'95% CI':>16} {'Lower':>8} {'Flip':>8} {'Probe':>8}")
    print('─' * 70)
    for cfg_id, s in scores.items():
        n  = list(s['per_model'].values())[0]['n']
        lo = s['mean_lower']
        hi = float(np.mean([v['ci_upper'] for v in s['per_model'].values()]))
        print(f"{cfg_id:<8} {n:>7} {s['mean_acc']:>8.3f}   [{lo:.3f}, {hi:.3f}]  {lo:>8.3f} {s['mean_flip']:>8.3f} {s['mean_probe']:>8.3f}")


def select_best_config(splits: dict, results_dir: str, model_tags: list) -> tuple:
    scores = {
        cfg: compute_config_score(results_dir, model_tags, cfg)
        for cfg in ['S1', 'S2', 'S3', 'S4']
    }

    print_selection_table(scores)

    winner = max(scores, key=lambda c: scores[c]['mean_lower'])
    print(f"\nWinning config: {winner}  (mean lower CI = {scores[winner]['mean_lower']:.4f})")

    # Write selected.yaml (finalise step)
    try:
        import yaml
    except Exception:
        yaml = None

    if yaml is not None:
        record = {
            'winning_config':   winner,
            'seed':             42,
            'n_train':          _n_items(splits[winner]['D_train']),
            'n_steer':          _n_items(splits[winner]['D_steer']),
            'n_val':            _n_items(splits[winner]['D_val']),
            'n_test':           1319,
            'selection_metric': 'mean_wilson_lower_steered_val_accuracy',
            'selection_value':  round(scores[winner]['mean_lower'], 4),
            'flip_rate':        round(scores[winner]['mean_flip'],  4),
            'probe_accuracy':   round(scores[winner]['mean_probe'], 4),
        }
        os.makedirs('configs', exist_ok=True)
        with open('configs/selected.yaml', 'w') as f:
            yaml.dump(record, f)
        print('Winner locked -> configs/selected.yaml')

    return winner, scores


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results')
    parser.add_argument('--splits', default=None, help='Path to splits meta json (optional)')
    args = parser.parse_args()

    if args.splits:
        with open(args.splits) as f:
            splits = json.load(f)
    else:
        # minimal placeholder: fill counts with expected values
        splits = {
            'S1': {'D_train': 5231, 'D_steer': 747, 'D_val': 1495},
            'S2': {'D_train': 4484, 'D_steer': 1495, 'D_val': 1495},
            'S3': {'D_train': 4484, 'D_steer': 747, 'D_val': 2242},
            'S4': {'D_train': 3737, 'D_steer': 1495, 'D_val': 2242},
        }

    MODEL_TAGS = ['llama32_3b', 'phi2', 'qwen25_3b', 'qwen25_math1.5b']
    select_best_config(splits, args.results, MODEL_TAGS)
