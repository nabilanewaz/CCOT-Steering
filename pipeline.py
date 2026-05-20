"""Master pipeline runner for the CCOT-Steering experiment.

Usage
-----
  python pipeline.py --phase 0                       # all phases, all configs/models
  python pipeline.py --phase 1                       # Phase 1 training + eval only
  python pipeline.py --phase 1 --config S2 --model llama32_3b
  python pipeline.py --phase 2 --config S2
  python pipeline.py --phase 3
  python pipeline.py --phase 4                       # delegates to evaluate_final.py
  python pipeline.py --phase 5                       # SVAMP transfer eval (frozen GSM8K steering)

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
from utils.dataset_paths import (
    get_active_dataset_id,
    get_test_path,
    get_train_pool_path,
    init_project_dataset,
    phase4_subprocess_env,
)
from phase1.train import train_coconut_phase1
from phase1.evaluate import run_phase1_evaluation, print_comparison_table
from phase2.run import run_phase2_all_sources
from phase3.evaluate import run_phase3_evaluation
from phase3.select import select_best_steered_config


CONFIGS = ['S2']
MODEL_TAGS = ['llama32_3b', 'phi2', 'qwen25_0.5b', 'qwen25_3b', 'qwen25_math1.5b']
LATENT_TOKEN_COUNTS = [3, 4, 6]

MODEL_ID_MAP = {
    'llama32_3b':      'meta-llama/Llama-3.2-3B',
    'phi2':            'microsoft/phi-2',
    'qwen25_0.5b':     'Qwen/Qwen2.5-0.5B',
    'qwen25_3b':       'Qwen/Qwen2.5-3B',
    'qwen25_math1.5b': 'Qwen/Qwen2.5-Math-1.5B',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _done(path: str) -> bool:
    return os.path.exists(path)

def _checkpoint_ready(path: str) -> bool:
    return _done(os.path.join(path, "adapter_config.json")) or _done(os.path.join(path, "config.json"))


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
    cfg['winning_config'] = 'S2'
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
        for model_tag in models_to_run:
            base_id  = MODEL_ID_MAP[model_tag]
            ckpt_dir = f"checkpoints/{cfg_id}/{model_tag}"
            res_dir  = f"results/{cfg_id}/{model_tag}"

            cot_out = os.path.join(ckpt_dir, 'cot')
            all_latent_ready = all(
                _checkpoint_ready(os.path.join(ckpt_dir, f"ccot_L{int(n)}"))
                for n in LATENT_TOKEN_COUNTS
            )
            if not (_checkpoint_ready(cot_out) and all_latent_ready):
                print(f"\n[{cfg_id}][{model_tag}] Coconut phase1 single-run (20 epochs) + latent export")
                train_coconut_phase1(
                    base_model_id=base_id,
                    D_train=D_train,
                    checkpoints_dir=ckpt_dir,
                    results_dir=res_dir,
                    model_tag=model_tag,
                    latent_token_counts=LATENT_TOKEN_COUNTS,
                )
            else:
                print(f"[{cfg_id}][{model_tag}] Coconut latent checkpoints exist — skipping")

            # Phase 1 evaluation
            phase1_eval_paths = [
                os.path.join(res_dir, 'phase1_latent_sweep.json'),
                os.path.join(res_dir, 'phase1_best_latent.json'),
                os.path.join(res_dir, 'phase1_val.json'),
            ]
            if not all(_done(path) for path in phase1_eval_paths):
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
                print(f"[{cfg_id}][{model_tag}] Phase 1 latent results exist — skipping")


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


def _run_phase4(models_to_run=None):
    print("\n" + "=" * 70)
    print("PHASE 4: Final Evaluation on D_test")
    print("=" * 70)
    if not _done('configs/selected.yaml'):
        print("[PH4] configs/selected.yaml missing — run split selection first.")
        sys.exit(1)
    print("Delegating to evaluate_final.py (the only script that may open test.jsonl).")
    cmd = [sys.executable, 'evaluate_final.py']
    if models_to_run:
        cmd.extend(['--model', ','.join(models_to_run)])
    result = subprocess.run(
        cmd,
        check=False,
        env=phase4_subprocess_env(),
    )
    if result.returncode != 0:
        print(f"[PH4] evaluate_final.py exited with code {result.returncode}")
        sys.exit(result.returncode)
    print("[PH4] Final evaluation complete.")


def _run_phase5():
    print("\n" + "=" * 70)
    print("PHASE 5: SVAMP transfer evaluation (GSM8K-frozen v_truth, alpha_star)")
    print("=" * 70)
    if not _done("configs/selected.yaml"):
        print("[PH5] configs/selected.yaml missing — run split selection / Phase 3 first.")
        sys.exit(1)
    print(
        "Delegating to evaluate_final.py: D_test=SVAMP, same vectors/checkpoints as Phase 4."
    )
    env = os.environ.copy()
    env["CCOT_DATASET"] = "svamp"
    result = subprocess.run(
        [
            sys.executable,
            "evaluate_final.py",
            "--dataset",
            "svamp",
            "--results-dir",
            "results/final_svamp_transfer",
        ],
        check=False,
        env=env,
    )
    if result.returncode != 0:
        print(f"[PH5] evaluate_final.py exited with code {result.returncode}")
        sys.exit(result.returncode)
    print("[PH5] Transfer evaluation complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='CCOT-Steering pipeline runner.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--phase', type=int, default=0,
        help='0=all  1=train+eval  2=vectors  3=steer  4=test  5=svamp_transfer',
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
    parser.add_argument(
        '--dataset', type=str, default=None,
        choices=('gsm8k', 'svamp', 'prontoqa'),
        help='Dataset id (omit to prompt interactively on a TTY, or set CCOT_DATASET)',
    )
    args = parser.parse_args()

    if args.phase == 5:
        _run_phase5()
        return

    init_project_dataset(args.dataset, interactive=sys.stdin.isatty())
    print(
        f"Dataset: {get_active_dataset_id()}  "
        f"train_pool={get_train_pool_path()}  D_test={get_test_path()}"
    )

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

    splits = build_all_splits(get_train_pool_path(), seed=42)

    if args.phase in (0, 1):
        _run_phase1(configs_to_run, models_to_run, splits, args.device)

    if args.phase in (0, 2):
        _run_phase2(configs_to_run, models_to_run, splits, args.device)

    if args.phase in (0, 3):
        _run_phase3(configs_to_run, models_to_run, splits, args.device)
        _run_selection(models_to_run, splits)

    if args.phase in (0, 4):
        _run_phase4(models_to_run)


if __name__ == '__main__':
    main()
