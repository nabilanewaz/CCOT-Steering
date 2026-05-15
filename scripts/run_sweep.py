"""Phase 1 + Phase 2 + Phase 3 sweep, then Phase 4 final evaluation.

Flow:
  Phases 1–3: run for each split config (S1–S4) × each backbone.
  Selection:  scripts/selection.py picks the winning split by mean Wilson CI.
  Phase 4:    evaluate_final.py runs once on D_test with locked configs.
"""
import argparse
import os
import subprocess
import sys
import torch
import yaml

from scripts.build_splits import build_all_splits
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

CFG_ID     = 'S2'   # fixed 60/20/20 split
MODEL_TAGS = ['llama32_3b', 'phi2', 'qwen25_3b', 'qwen25_math1.5b']
LATENT_TOKEN_COUNTS = [3, 4, 6]

MODEL_ID_MAP = {
    'llama32_3b':      'meta-llama/Llama-3.2-3B',
    'phi2':            'microsoft/phi-2',
    'qwen25_3b':       'Qwen/Qwen2.5-3B',
    'qwen25_math1.5b': 'Qwen/Qwen2.5-Math-1.5B',
}

def _checkpoint_ready(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json")) or os.path.exists(os.path.join(path, "config.json"))


def _update_selected_phase3_best(model_tag: str, selection: dict) -> None:
    """Append the locked Phase 3 choice for one backbone into configs/selected.yaml."""
    selected_path = 'configs/selected.yaml'
    if os.path.exists(selected_path):
        with open(selected_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    phase3_best = dict(cfg.get('phase3_best') or {})
    phase3_best[model_tag] = selection
    cfg['phase3_best'] = phase3_best
    cfg['winning_config'] = 'S2'

    os.makedirs('configs', exist_ok=True)
    with open(selected_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        '--dataset', default=None, choices=('gsm8k', 'svamp', 'prontoqa'),
        help='Dataset id (omit to prompt on a TTY or use CCOT_DATASET / configs/active_dataset.txt)',
    )
    args, _rest = ap.parse_known_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    init_project_dataset(args.dataset, interactive=sys.stdin.isatty())
    print(
        f"Dataset: {get_active_dataset_id()}  "
        f"train_pool={get_train_pool_path()}  D_test={get_test_path()}"
    )
    splits    = build_all_splits(get_train_pool_path(), seed=42)
    D_train   = splits[CFG_ID]['D_train']
    D_steer   = splits[CFG_ID]['D_steer']
    D_val     = splits[CFG_ID]['D_val']
    for model_tag in MODEL_TAGS:
            base_model_id = MODEL_ID_MAP[model_tag]
            ckpt_dir      = f"checkpoints/{CFG_ID}/{model_tag}"
            results_dir   = f"results/{CFG_ID}/{model_tag}"

            # ── Phase 1 training (single Coconut run + compat export) ────────
            cot_out = os.path.join(ckpt_dir, 'cot')
            all_latent_ready = all(
                _checkpoint_ready(os.path.join(ckpt_dir, f"ccot_L{int(n)}"))
                for n in LATENT_TOKEN_COUNTS
            )
            if not (_checkpoint_ready(cot_out) and all_latent_ready):
                train_coconut_phase1(
                    base_model_id=base_model_id,
                    D_train=D_train,
                    checkpoints_dir=ckpt_dir,
                    results_dir=results_dir,
                    model_tag=model_tag,
                    latent_token_counts=LATENT_TOKEN_COUNTS,
                )
            else:
                print(f"[PH1] Coconut latent checkpoints exist, skipping: {ckpt_dir}")

            # ── Phase 1 evaluation ────────────────────────────────────────────
            phase1_results = run_phase1_evaluation(
                model_tag=model_tag,
                base_model_id=base_model_id,
                D_val=D_val,
                device=device,
                checkpoints_dir=ckpt_dir,
                results_dir=results_dir,
            )
            print_comparison_table(phase1_results)

            # ── Phase 2: truth vector extraction ─────────────────────────────
            vectors_dir = f"vectors/{CFG_ID}/{model_tag}"
            run_phase2_all_sources(
                model_tag=model_tag,
                base_model_id=base_model_id,
                checkpoints_dir=ckpt_dir,
                D_steer=D_steer,
                device=device,
                vectors_dir=vectors_dir,
                results_dir=results_dir,
            )

            # ── Phase 3: steered evaluation ───────────────────────────────────
            run_phase3_evaluation(
                model_tag=model_tag,
                base_model_id=base_model_id,
                checkpoints_dir=ckpt_dir,
                D_val=D_val,
                vectors_dir=vectors_dir,
                results_dir=results_dir,
                device=device,
            )
            selection = select_best_steered_config(results_dir, model_tag)
            if selection:
                _update_selected_phase3_best(model_tag, selection)


def run_phase4():
    """
    Run the single-pass D_test evaluation using the locked configs in selected.yaml.
    Phase 4 is invoked as a subprocess so that the load_test_set() guard
    in utils/data.py sees 'evaluate_final.py' as the permitted caller.
    """
    if not os.path.exists('configs/selected.yaml'):
        print("[PH4] configs/selected.yaml missing — Phase 3 selection incomplete, skipping Phase 4.")
        return

    print(f"\n[PH4] Launching evaluate_final.py (split={CFG_ID})...")
    result = subprocess.run(
        [sys.executable, 'evaluate_final.py'],
        check=False,
        env=phase4_subprocess_env(),
    )
    if result.returncode != 0:
        print(f"[PH4] evaluate_final.py exited with code {result.returncode}")
    else:
        print("[PH4] Final evaluation complete.")


if __name__ == '__main__':
    main()
    run_phase4()
