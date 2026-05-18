import json
import random
import os
from typing import Dict


def build_all_splits(pool_path: str, seed: int = 42, out_dir: str = None) -> Dict:
    with open(pool_path) as f:
        pool = [json.loads(l) for l in f]

    random.seed(seed)
    random.shuffle(pool)
    n = len(pool)

    # S1: 70% train / 10% steer / 20% val
    # S2: 60% train / 10% steer / 30% val
    # S3: 60% train / 20% steer / 20% val
    # S4: 50% train / 20% steer / 30% val
    configs = {
        'S1': (0.70, 0.10),
        'S2': (0.60, 0.10),
        'S3': (0.60, 0.20),
        'S4': (0.50, 0.20),
    }
    splits = {}
    for cfg_id, (tr_frac, st_frac) in configs.items():
        n_tr = round(n * tr_frac)
        n_st = round(n * st_frac)
        splits[cfg_id] = {
            'D_train': pool[:n_tr],
            'D_steer': pool[n_tr : n_tr + n_st],
            'D_val':   pool[n_tr + n_st:],
        }
        print(f"{cfg_id}: train={n_tr}  steer={n_st}  val={len(splits[cfg_id]['D_val'])}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        meta = {}
        for cfg_id, parts in splits.items():
            meta[cfg_id] = {k: len(v) for k, v in parts.items()}
            for part_name, examples in parts.items():
                path = os.path.join(out_dir, f"{cfg_id}_{part_name}.jsonl")
                with open(path, 'w', encoding='utf-8') as f:
                    for ex in examples:
                        f.write(json.dumps(ex) + "\n")
        with open(os.path.join(out_dir, 'splits_meta.json'), 'w') as mf:
            json.dump(meta, mf, indent=2)

    return splits


if __name__ == '__main__':
    import argparse
    import sys

    from utils.dataset_paths import get_train_pool_path, init_project_dataset

    p = argparse.ArgumentParser()
    p.add_argument('--pool', default=None, help='Train pool JSONL (default: active dataset train.jsonl)')
    p.add_argument('--dataset', default=None, choices=('gsm8k', 'svamp', 'prontoqa'),
                   help='Active dataset id (used when --pool is omitted)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='configs/splits')
    args = p.parse_args()
    init_project_dataset(args.dataset, interactive=sys.stdin.isatty())
    pool = args.pool or get_train_pool_path()
    build_all_splits(pool, seed=args.seed, out_dir=args.out)
