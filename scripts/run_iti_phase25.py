"""
Standalone Phase 2.5: ITI head collection.

Run this AFTER Phase 2 is complete to collect per-head attention activations,
probe each (layer, head) pair, and save top-K ITI vectors.

Usage:
  python scripts/run_iti_phase25.py --model qwen25_math1.5b --config S2 --dataset gsm8k
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from scripts.build_splits import build_all_splits
from utils.dataset_paths import get_train_pool_path, init_project_dataset
from phase2.loaders import load_ccot_frozen, find_boundary_idx_ccot
from phase2.run import run_iti_phase2

MODEL_ID_MAP = {
    'llama32_3b':      'meta-llama/Llama-3.2-3B',
    'phi2':            'microsoft/phi-2',
    'qwen25_3b':       'Qwen/Qwen2.5-3B',
    'qwen25_math1.5b': 'Qwen/Qwen2.5-Math-1.5B',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   required=True, choices=list(MODEL_ID_MAP))
    parser.add_argument('--config',  default='S2')
    parser.add_argument('--dataset', default=None, choices=('gsm8k', 'svamp', 'prontoqa'))
    parser.add_argument('--device',  default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--top-k',   type=int, default=48,
                        help='Number of attention heads to select (ITI)')
    parser.add_argument('--n-rollouts', type=int, default=10,
                        help='Rollouts per question for head activation collection')
    args = parser.parse_args()

    init_project_dataset(args.dataset, interactive=sys.stdin.isatty())

    splits      = build_all_splits(get_train_pool_path(), seed=42)
    D_steer     = splits[args.config]['D_steer']
    model_tag   = args.model
    base_id     = MODEL_ID_MAP[model_tag]
    ckpt_dir    = f"checkpoints/{args.config}/{model_tag}"
    vectors_dir = f"vectors/{args.config}/{model_tag}"

    meta_path = os.path.join(vectors_dir, 'phase2_meta.json')
    if not os.path.exists(meta_path):
        print(f"ERROR: phase2_meta.json not found at {meta_path}")
        print("Run Phase 2 first:  python pipeline.py --phase 2 ...")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    ratio_int = meta.get('best_ccot_ratio', 6)
    ccot_ckpt = os.path.join(ckpt_dir, f'ccot_R{ratio_int}')

    iti_out = os.path.join(vectors_dir, 'ccot_iti_heads.pt')
    if os.path.exists(iti_out):
        print(f"ITI vectors already exist: {iti_out}")
        print("Delete the file to re-run ITI collection.")
        return

    print(f"\nLoading CCoT R={ratio_int} model: {ccot_ckpt}")
    model, tok = load_ccot_frozen(base_id, ccot_ckpt, args.device)

    prompt_fn = lambda item, ri=ratio_int: (
        f"Question: {item['question']}\n\n[compress:0.{ri}]\n"
    )

    run_iti_phase2(
        model=model,
        tokenizer=tok,
        D_steer=D_steer,
        model_tag=model_tag,
        source_tag='ccot',
        boundary_idx_fn=find_boundary_idx_ccot,
        device=args.device,
        vectors_dir=vectors_dir,
        prompt_fn=prompt_fn,
        N_rollouts=args.n_rollouts,
        top_k=args.top_k,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nITI Phase 2.5 complete.  Output: {iti_out}")
    print(f"Now run Phase 3 to evaluate ITI:")
    print(f"  python pipeline.py --phase 3 --model {args.model} "
          f"--config {args.config} --dataset gsm8k")


if __name__ == '__main__':
    main()
