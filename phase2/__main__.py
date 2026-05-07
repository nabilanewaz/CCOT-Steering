import argparse
import json

import torch

from phase2.run import run_phase2_all_sources


def _load_jsonl(path: str) -> list:
	with open(path, encoding='utf-8') as f:
		return [json.loads(line) for line in f]


def main():
	parser = argparse.ArgumentParser(description='Run Phase 2 truth-vector extraction.')
	parser.add_argument('--model-tag', required=True)
	parser.add_argument('--base-model-id', required=True)
	parser.add_argument('--checkpoints-dir', required=True)
	parser.add_argument('--steer-jsonl', required=True)
	parser.add_argument('--vectors-dir', required=True)
	parser.add_argument('--results-dir', required=True)
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	args = parser.parse_args()

	D_steer = _load_jsonl(args.steer_jsonl)
	run_phase2_all_sources(
		model_tag=args.model_tag,
		base_model_id=args.base_model_id,
		checkpoints_dir=args.checkpoints_dir,
		D_steer=D_steer,
		device=args.device,
		vectors_dir=args.vectors_dir,
		results_dir=args.results_dir,
	)


if __name__ == '__main__':
	main()
