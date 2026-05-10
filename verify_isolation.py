"""Data isolation invariant checks.

Verifies that:
  1. No D_train / D_steer / D_val split overlaps with D_test (by item id)
  2. D_train and D_steer are disjoint within each split config

Run this before every Phase 2 and Phase 4 execution.

Usage:
    python verify_isolation.py
"""
import json
import sys


def main():
    from scripts.build_splits import build_all_splits
    from utils.dataset_paths import get_test_path, get_train_pool_path, init_project_dataset

    init_project_dataset(None, interactive=sys.stdin.isatty())
    train_pool = get_train_pool_path()
    test_path = get_test_path()

    splits = build_all_splits(train_pool, seed=42)

    # Load test IDs
    try:
        with open(test_path, encoding="utf-8") as f:
            test_ids = {json.loads(l)["id"] for l in f}
    except FileNotFoundError:
        print(f"[warn] Test file not found: {test_path}  (skipping D_test checks)")
        test_ids = set()

    failures = []

    for cfg_id, split in splits.items():
        for subset in ("D_train", "D_steer", "D_val"):
            ids = {item["id"] for item in split[subset]}
            if test_ids:
                overlap = ids & test_ids
                if overlap:
                    failures.append(
                        f"LEAKAGE: {cfg_id}/{subset} overlaps D_test on {len(overlap)} id(s)"
                    )

        train_ids = {item["id"] for item in split["D_train"]}
        steer_ids = {item["id"] for item in split["D_steer"]}
        overlap   = train_ids & steer_ids
        if overlap:
            failures.append(
                f"LEAKAGE: {cfg_id} D_train overlaps D_steer on {len(overlap)} id(s)"
            )

    if failures:
        print("ISOLATION CHECK FAILED:")
        for msg in failures:
            print(f"  ✗ {msg}")
        sys.exit(1)
    else:
        total = sum(
            len(split[s])
            for split in splits.values()
            for s in ("D_train", "D_steer", "D_val")
        )
        print(f"All isolation checks passed. "
              f"({len(splits)} configs, {total} total split examples, "
              f"{len(test_ids)} test ids checked)")


if __name__ == "__main__":
    main()
