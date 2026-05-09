"""Master pipeline runner for the CCOT-Steering experiment.

Usage
-----
  python pipeline.py --phase 0                       # all phases, all configs/models
  python pipeline.py --phase 1                       # Phase 1 training + eval only
  python pipeline.py --phase 1 --config S2 --model llama32_3b
  python pipeline.py --phase 2 --config S2
  python pipeline.py --phase 3
  python pipeline.py --phase 4                       # delegates to evaluate_final.py

Checkpoint resume: every step checks whether its output file already exists and
skips if so — safe to re-run after interruption.
"""
import argparse
import os
import subprocess
import sys

import torch
import yaml

from scripts.build_splits import build_all_splits
from scripts.selection import select_best_config
from phase1.compress import build_ccot_cache, load_cache
from phase1.train import train_cot, train_ccot
from phase1.evaluate import run_phase1_evaluation, print_comparison_table
from phase2.run import run_phase2_all_sources
from phase3.evaluate import run_phase3_evaluation
from phase3.select import select_best_steered_config


CONFIGS = ['S1', 'S2', 'S3', 'S4']
MODEL_TAGS = ['llama32_3b', 'phi2', 'qwen25_3b', 'qwen25_math1.5b']
RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9]

MODEL_ID_MAP = {
    'llama32_3b':      'meta-llama/Llama-3.2-3B',
    'phi2':            'microsoft/phi-2',
    'qwen25_3b':       'Qwen/Qwen2.5-3B',
    'qwen25_math1.5b': 'Qwen/Qwen2.5-Math-1.5B',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _done(path: str) -> bool:
    return os.path.exists(path)


def _update_selected_phase3_best(model_tag: str, selection: dict) -> None:
    """Merge one model's Phase 3 best config into configs/selected.yaml."""
    path = 'configs/selected.yaml'
    cfg = {}
    if os.path.exists(path):
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    phase3_best = dict(cfg.get('phase3_best') or {})
    phase3_best[model_tag] = selection
    cfg['phase3_best'] = phase3_best
    os.makedirs('configs', exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ── Phase runners ─────────────────────────────────────────────────────────────

def _run_phase1(configs_to_run, models_to_run, splits, device):
    print("\n" + "=" * 70)
    print("PHASE 1: Training + Evaluation")
    print("=" * 70)

    for cfg_id in configs_to_run:
        D_train   = splits[cfg_id]['D_train']
        D_val     = splits[cfg_id]['D_val']
        cache_dir = f"cache/{cfg_id}"

        # Build compression cache once per split config
        try:
            from llmlingua import PromptCompressor
            compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            build_ccot_cache(D_train, RATIOS, cache_dir, compressor)
        except ImportError:
            print("[warn] llmlingua not installed — skipping cache build. "
                  "Ensure cache/*.jsonl files exist before CCoT training.")

        for model_tag in models_to_run:
            base_id  = MODEL_ID_MAP[model_tag]
            ckpt_dir = f"checkpoints/{cfg_id}/{model_tag}"
            res_dir  = f"results/{cfg_id}/{model_tag}"

            # Stage 1: CoT
            cot_out = os.path.join(ckpt_dir, 'cot')
            if not _done(os.path.join(cot_out, 'adapter_config.json')):
                print(f"\n[{cfg_id}][{model_tag}] Stage 1: CoT fine-tuning")
                train_cot(base_id, D_train, cot_out, model_tag)
            else:
                print(f"[{cfg_id}][{model_tag}] CoT checkpoint exists — skipping")

            # Stage 2: CCoT per ratio
            for ratio in RATIOS:
                rtag     = f"R{int(ratio * 10)}"
                ccot_out = os.path.join(ckpt_dir, f"ccot_{rtag}")
                if _done(os.path.join(ccot_out, 'adapter_config.json')):
                    print(f"[{cfg_id}][{model_tag}] CCoT {rtag} exists — skipping")
                    continue
                cache_path = os.path.join(cache_dir, f"compressed_{rtag}.jsonl")
                if not _done(cache_path):
                    raise FileNotFoundError(
                        f"Compression cache missing: {cache_path}\n"
                        "Run:  python preprocess_compress.py"
                    )
                compressed_cache = load_cache(cache_path)
                print(f"\n[{cfg_id}][{model_tag}] Stage 2: CCoT R={ratio}")
                train_ccot(base_id, D_train, compressed_cache,
                           ratio, ccot_out, model_tag)

            # Phase 1 evaluation
            out_path = os.path.join(res_dir, 'phase1_val.json')
            if not _done(out_path):
                print(f"\n[{cfg_id}][{model_tag}] Phase 1 evaluation on D_val")
                results = run_phase1_evaluation(
                    model_tag=model_tag,
                    base_model_id=base_id,
                    D_val=D_val,
                    device=device,
                    checkpoints_dir=ckpt_dir,
                    results_dir=res_dir,
                )
                print_comparison_table(results)
            else:
                print(f"[{cfg_id}][{model_tag}] Phase 1 val results exist — skipping")


def _run_phase2(configs_to_run, models_to_run, splits, device):
    print("\n" + "=" * 70)
    print("PHASE 2: Truth Vector Extraction")
    print("=" * 70)

    for cfg_id in configs_to_run:
        D_steer = splits[cfg_id]['D_steer']

        for model_tag in models_to_run:
            base_id     = MODEL_ID_MAP[model_tag]
            ckpt_dir    = f"checkpoints/{cfg_id}/{model_tag}"
            vectors_dir = f"vectors/{cfg_id}/{model_tag}"
            res_dir     = f"results/{cfg_id}/{model_tag}"

            if not _done(os.path.join(vectors_dir, 'ccot_dom.pt')):
                print(f"\n[{cfg_id}][{model_tag}] Phase 2: vector extraction")
                run_phase2_all_sources(
                    model_tag=model_tag,
                    base_model_id=base_id,
                    checkpoints_dir=ckpt_dir,
                    D_steer=D_steer,
                    device=device,
                    vectors_dir=vectors_dir,
                    results_dir=res_dir,
                )
            else:
                print(f"[{cfg_id}][{model_tag}] Vectors exist — skipping")


def _run_phase3(configs_to_run, models_to_run, splits, device):
    print("\n" + "=" * 70)
    print("PHASE 3: α-Tuning + Steered Evaluation")
    print("=" * 70)

    for cfg_id in configs_to_run:
        D_val = splits[cfg_id]['D_val']

        for model_tag in models_to_run:
            base_id     = MODEL_ID_MAP[model_tag]
            ckpt_dir    = f"checkpoints/{cfg_id}/{model_tag}"
            vectors_dir = f"vectors/{cfg_id}/{model_tag}"
            res_dir     = f"results/{cfg_id}/{model_tag}"

            out_path = os.path.join(res_dir, 'phase3_val.json')
            if not _done(out_path):
                print(f"\n[{cfg_id}][{model_tag}] Phase 3")
                run_phase3_evaluation(
                    model_tag=model_tag,
                    base_model_id=base_id,
                    checkpoints_dir=ckpt_dir,
                    D_val=D_val,
                    vectors_dir=vectors_dir,
                    results_dir=res_dir,
                    device=device,
                )
                selection = select_best_steered_config(res_dir, model_tag)
                if selection:
                    _update_selected_phase3_best(model_tag, selection)
            else:
                print(f"[{cfg_id}][{model_tag}] Phase 3 val results exist — skipping")


def _run_selection(models_to_run, splits):
    print("\n" + "=" * 70)
    print("SPLIT SELECTION")
    print("=" * 70)
    winner, _ = select_best_config(splits, 'results', models_to_run)
    print(f"\nWinner: {winner}  (configs/selected.yaml updated)")


def _run_phase4():
    print("\n" + "=" * 70)
    print("PHASE 4: Final Evaluation on D_test")
    print("=" * 70)
    if not _done('configs/selected.yaml'):
        print("[PH4] configs/selected.yaml missing — run split selection first.")
        sys.exit(1)
    print("Delegating to evaluate_final.py (the only script that may open test.jsonl).")
    result = subprocess.run([sys.executable, 'evaluate_final.py'], check=False)
    if result.returncode != 0:
        print(f"[PH4] evaluate_final.py exited with code {result.returncode}")
        sys.exit(result.returncode)
    print("[PH4] Final evaluation complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='CCOT-Steering pipeline runner.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--phase', type=int, default=0,
        help='0=all  1=train+eval  2=vectors  3=steer  4=test',
    )
    parser.add_argument(
        '--config', type=str, default='all',
        help='Split config(s): all | S1 | S1,S2',
    )
    parser.add_argument(
        '--model', type=str, default='all',
        help='Model tag(s): all | llama32_3b | llama32_3b,phi2',
    )
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    args = parser.parse_args()

    configs_to_run = CONFIGS if args.config == 'all' else args.config.split(',')
    models_to_run  = MODEL_TAGS if args.model == 'all' else args.model.split(',')

    bad_cfg   = [c for c in configs_to_run if c not in CONFIGS]
    bad_model = [m for m in models_to_run  if m not in MODEL_TAGS]
    if bad_cfg:
        parser.error(f"Unknown config(s): {bad_cfg}. Valid: {CONFIGS}")
    if bad_model:
        parser.error(f"Unknown model(s): {bad_model}. Valid: {MODEL_TAGS}")

    print(f"Device:  {args.device}")
    print(f"Configs: {configs_to_run}")
    print(f"Models:  {models_to_run}")

    splits = build_all_splits('gsm8k/train.jsonl', seed=42)

    if args.phase in (0, 1):
        _run_phase1(configs_to_run, models_to_run, splits, args.device)

    if args.phase in (0, 2):
        _run_phase2(configs_to_run, models_to_run, splits, args.device)

    if args.phase in (0, 3):
        _run_phase3(configs_to_run, models_to_run, splits, args.device)
        _run_selection(models_to_run, splits)

    if args.phase in (0, 4):
        _run_phase4()


if __name__ == '__main__':
    main()
