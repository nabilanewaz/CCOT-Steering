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

    # Fixed split: 60% train / 20% steer / 20% val
    n_tr = round(n * 0.60)
    n_st = round(n * 0.20)
    splits = {
        'S2': {
            'D_train': pool[:n_tr],
            'D_steer': pool[n_tr : n_tr + n_st],
            'D_val':   pool[n_tr + n_st:],
        }
    }
    print(f"S2: train={n_tr}  steer={n_st}  val={len(splits['S2']['D_val'])}")

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
    p = argparse.ArgumentParser()
    p.add_argument('--pool', default='gsm8k/train.jsonl')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='configs/splits')
    args = p.parse_args()
    build_all_splits(args.pool, seed=args.seed, out_dir=args.out)
