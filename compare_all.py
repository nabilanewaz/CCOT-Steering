"""Split selection: read all Phase 3 val results and pick the winning config.

Run after all Phase 3 sweeps complete:
    python compare_all.py
    python compare_all.py --results results --models llama32_3b,phi2
"""
import argparse
import json
import sys

from scripts.build_splits import build_all_splits
from scripts.selection import select_best_config

MODEL_TAGS = ['llama32_3b', 'phi2', 'qwen25_3b', 'qwen25_math1.5b']
TRAIN_POOL = 'gsm8k/train.jsonl'


def main():
    parser = argparse.ArgumentParser(description='Select winning split config.')
    parser.add_argument('--results', default='results',
                        help='Base results directory (default: results)')
    parser.add_argument('--models', default=None,
                        help='Comma-separated model tags (default: all four)')
    parser.add_argument('--pool', default=TRAIN_POOL,
                        help='Path to GSM8K train pool (for split size counts)')
    args = parser.parse_args()

    model_tags = args.models.split(',') if args.models else MODEL_TAGS

    try:
        splits = build_all_splits(args.pool, seed=42)
    except FileNotFoundError:
        print(f"[warn] Train pool not found at {args.pool} — using placeholder split counts.")
        splits = {
            'S1': {'D_train': [None] * 5231, 'D_steer': [None] * 747,  'D_val': [None] * 1495},
            'S2': {'D_train': [None] * 4484, 'D_steer': [None] * 1495, 'D_val': [None] * 1495},
            'S3': {'D_train': [None] * 4484, 'D_steer': [None] * 747,  'D_val': [None] * 2242},
            'S4': {'D_train': [None] * 3737, 'D_steer': [None] * 1495, 'D_val': [None] * 2242},
        }

    winner, scores = select_best_config(splits, args.results, model_tags)
    return winner, scores


if __name__ == '__main__':
    winner, _ = main()
    print(f"\nRun next:  python evaluate_final.py")
    sys.exit(0)
