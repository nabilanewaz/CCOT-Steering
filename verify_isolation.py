"""Audit question-only identities across full train/steer/val/test allocations."""
import argparse
import itertools
import json
import sys

from scripts.build_splits import build_all_splits
from utils.data import select_test_examples
from utils.dataset_paths import get_test_path, get_train_pool_path, init_project_dataset


def question_key(item):
    return " ".join(item["question"].split()).casefold()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="gsm8k", choices=("gsm8k", "svamp", "prontoqa"))
    args = parser.parse_args()
    init_project_dataset(args.dataset, interactive=False)
    parts = build_all_splits(get_train_pool_path())["S3"]
    with open(get_test_path(), encoding="utf-8") as stream:
        test = select_test_examples([json.loads(line) for line in stream if line.strip()])
    parts["D_test"] = test
    keys = {role: {question_key(item) for item in rows} for role, rows in parts.items()}
    failures = []
    for first, second in itertools.combinations(parts, 2):
        overlap = keys[first] & keys[second]
        if overlap:
            failures.append(f"{first}/{second}: {len(overlap)} overlapping normalized questions")
    if failures:
        raise SystemExit("ISOLATION FAILED:\n" + "\n".join(failures))
    print("Full-data isolation passed:", {role: len(rows) for role, rows in parts.items()})


if __name__ == "__main__":
    main()
