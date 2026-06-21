"""
Quick test of LR vs MLP probe on existing hidden states cache.
Run from repo root:
    python scripts/test_probe.py
    python scripts/test_probe.py --model qwen25_math1.5b --config S2
"""
import argparse, torch
from phase2.probe import score_all_layers_both

parser = argparse.ArgumentParser()
parser.add_argument('--model',  default='qwen25_math1.5b')
parser.add_argument('--config', default='S2')
parser.add_argument('--source', default='ccot')
args = parser.parse_args()

cache_path = f'vectors/{args.config}/{args.model}/{args.source}_hstates_cache.pt'
print(f'Loading {cache_path} ...')
data = torch.load(cache_path, map_location='cpu')

H_pos = data['H_pos']   # dict[layer -> Tensor]
H_neg = data['H_neg']
print(f'Layers: {sorted(H_pos.keys())}')
print(f'Samples per class: {len(next(iter(H_pos.values())))}')
print(f'Hidden dim: {next(iter(H_pos.values())).shape[-1]}')
print()

scores = score_all_layers_both(H_pos, H_neg)
best_L = max(scores, key=scores.get)
print(f'\nBest layer: L={best_L}  combined_acc={scores[best_L]:.4f}')
