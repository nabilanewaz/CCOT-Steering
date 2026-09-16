"""Build the full S3 allocation: 60% train, 10% steer, 30% validation."""
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.artifacts import EXPERIMENT_VERSION, atomic_json, dataset_fingerprint
from utils.experiment_config import split_counts


def build_all_splits(pool_path: str, seed: int = 42, out_dir: str = None) -> dict:
    with open(pool_path, encoding="utf-8") as stream:
        pool = [json.loads(line) for line in stream if line.strip()]
    random.Random(seed).shuffle(pool)
    counts = split_counts(len(pool))
    train_end = counts["D_train"]
    steer_end = train_end + counts["D_steer"]
    parts = {
        "D_train": pool[:train_end],
        "D_steer": pool[train_end:steer_end],
        "D_val": pool[steer_end:],
    }
    question_sets = {
        role: {" ".join(item["question"].split()).casefold() for item in examples}
        for role, examples in parts.items()
    }
    roles = list(parts)
    for index, role in enumerate(roles):
        for other in roles[index + 1:]:
            overlap = question_sets[role] & question_sets[other]
            if overlap:
                raise ValueError(f"Question leakage between {role} and {other}: {len(overlap)}")
    print(f"S3: train={counts['D_train']} steer={counts['D_steer']} val={counts['D_val']}")
    if out_dir:
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        for role, examples in parts.items():
            with (root / f"S3_{role}.jsonl").open("w", encoding="utf-8") as stream:
                for example in examples:
                    stream.write(json.dumps(example, ensure_ascii=False) + "\n")
        atomic_json(root / "splits_meta.json", {"S3": counts})
        atomic_json(root / "split_manifest.json", {
            "experiment_version": EXPERIMENT_VERSION,
            "config": "S3", "seed": seed, "ratios": [0.6, 0.1, 0.3],
            "pool_path": str(Path(pool_path).resolve()), "pool_size": len(pool),
            "counts": counts,
            "fingerprints": {role: dataset_fingerprint(rows) for role, rows in parts.items()},
        })
    return {"S3": parts}


if __name__ == "__main__":
    import argparse
    from utils.dataset_paths import get_train_pool_path, init_project_dataset, split_output_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool")
    parser.add_argument("--dataset", default="gsm8k", choices=("gsm8k", "svamp", "prontoqa"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    init_project_dataset(args.dataset, interactive=False)
    build_all_splits(args.pool or get_train_pool_path(), args.seed, args.out or split_output_dir())
