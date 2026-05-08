"""Phase 1 + Phase 2 + Phase 3 sweep, then Phase 4 final evaluation.

Flow:
  Phases 1–3: run for each split config (S1–S4) × each backbone.
  Selection:  scripts/selection.py picks the winning split by mean Wilson CI.
  Phase 4:    evaluate_final.py runs once on D_test with locked configs.
"""
import os
import subprocess
import sys
import torch
import yaml

from scripts.build_splits import build_all_splits
from phase1.compress import build_ccot_cache, load_cache
from phase1.train import train_cot, train_ccot
from phase1.evaluate import run_phase1_evaluation, print_comparison_table
from phase2.run import run_phase2_all_sources
from phase3.evaluate import run_phase3_evaluation
from phase3.select import select_best_steered_config

CONFIGS    = ['S1', 'S2', 'S3', 'S4']
MODEL_TAGS = ['llama32_3b', 'phi2', 'qwen25_3b', 'qwen25_math1.5b']
RATIOS     = [0.5, 0.6, 0.7, 0.8, 0.9]

MODEL_ID_MAP = {
    'llama32_3b':      'meta-llama/Llama-3.2-3B',
    'phi2':            'microsoft/phi-2',
    'qwen25_3b':       'Qwen/Qwen2.5-3B',
    'qwen25_math1.5b': 'Qwen/Qwen2.5-Math-1.5B',
}


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

    os.makedirs('configs', exist_ok=True)
    with open(selected_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    splits = build_all_splits('gsm8k/train.jsonl', seed=42)

    for cfg_id in CONFIGS:
        D_train = splits[cfg_id]['D_train']
        D_steer = splits[cfg_id]['D_steer']
        D_val   = splits[cfg_id]['D_val']
        cache_dir = f"cache/{cfg_id}"

        # ── Offline compression (once per config, shared across backbones) ────
        try:
            from llmlingua import PromptCompressor
            compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            build_ccot_cache(D_train, RATIOS, cache_dir, compressor)
        except ImportError:
            print("llmlingua not installed — skipping compression cache build. "
                  "Ensure cache files exist before running CCoT training.")

        for model_tag in MODEL_TAGS:
            base_model_id = MODEL_ID_MAP[model_tag]
            ckpt_dir      = f"checkpoints/{cfg_id}/{model_tag}"
            results_dir   = f"results/{cfg_id}/{model_tag}"

            # ── Phase 1 training ──────────────────────────────────────────────
            cot_out = os.path.join(ckpt_dir, 'cot')
            if not os.path.exists(os.path.join(cot_out, 'adapter_config.json')):
                train_cot(base_model_id, D_train, cot_out, model_tag)
            else:
                print(f"[PH1] CoT checkpoint exists, skipping: {cot_out}")

            for ratio in RATIOS:
                ccot_out = os.path.join(ckpt_dir, f"ccot_R{int(ratio * 10)}")
                if os.path.exists(os.path.join(ccot_out, 'adapter_config.json')):
                    print(f"[PH1] CCoT R={ratio} checkpoint exists, skipping: {ccot_out}")
                    continue
                cache_path = os.path.join(cache_dir, f"compressed_R{int(ratio * 10)}.jsonl")
                if not os.path.exists(cache_path):
                    raise FileNotFoundError(
                        f"Compression cache missing: {cache_path}\n"
                        "Run build_ccot_cache() first."
                    )
                compressed_cache = load_cache(cache_path)
                train_ccot(base_model_id, D_train, compressed_cache,
                           ratio, ccot_out, model_tag)

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
            vectors_dir = f"vectors/{cfg_id}/{model_tag}"
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
    Select the winning split config and run the single-pass D_test evaluation.
    Phase 4 is invoked as a subprocess so that the load_test_set() guard
    in utils/data.py sees 'evaluate_final.py' as the permitted caller.
    """
    from scripts.selection import select_best_config
    from scripts.build_splits import build_all_splits

    splits = build_all_splits('gsm8k/train.jsonl', seed=42)
    winner, _ = select_best_config(splits, 'results', MODEL_TAGS)

    if not os.path.exists('configs/selected.yaml'):
        print("[PH4] configs/selected.yaml missing — selection step failed, skipping Phase 4.")
        return

    print(f"\n[PH4] Launching evaluate_final.py (winning config = {winner})...")
    result = subprocess.run(
        [sys.executable, 'evaluate_final.py'],
        check=False,
    )
    if result.returncode != 0:
        print(f"[PH4] evaluate_final.py exited with code {result.returncode}")
    else:
        print("[PH4] Final evaluation complete.")


if __name__ == '__main__':
    main()
    run_phase4()
