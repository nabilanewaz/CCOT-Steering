"""CLI entry point: python -m phase3 --model-tag ... --base-model-id ... etc."""
import argparse
import json

import torch

from phase3.evaluate import run_phase3_evaluation
from phase3.select import select_best_steered_config
from utils.experiment_config import require_exact_count, samples_per_phase


def _load_jsonl(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    if len(examples) < samples_per_phase():
        raise ValueError(f"{path} has fewer than {samples_per_phase()} examples")
    examples = examples[:samples_per_phase()]
    require_exact_count(examples, "D_val")
    return examples


def main():
    parser = argparse.ArgumentParser(description='Run Phase 3 steered evaluation.')
    parser.add_argument('--model-tag',       required=True)
    parser.add_argument('--base-model-id',   required=True)
    parser.add_argument('--checkpoints-dir', required=True)
    parser.add_argument('--val-jsonl',       required=True)
    parser.add_argument('--vectors-dir',     required=True)
    parser.add_argument('--results-dir',     required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    D_val = _load_jsonl(args.val_jsonl)

    run_phase3_evaluation(
        model_tag=args.model_tag,
        base_model_id=args.base_model_id,
        checkpoints_dir=args.checkpoints_dir,
        D_val=D_val,
        vectors_dir=args.vectors_dir,
        results_dir=args.results_dir,
        device=args.device,
    )
    select_best_steered_config(args.results_dir, args.model_tag)


if __name__ == '__main__':
    main()
