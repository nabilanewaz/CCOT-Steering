"""Orchestration scaffold for the Phase 1 -> Phase 2 -> alpha-tuning sweep.

This file provides the high-level loop and placeholders for the heavy lifting
functions. Replace the placeholder implementations with your actual training
and evaluation code.
"""
import os
from scripts.build_splits import build_all_splits

CONFIGS    = ['S1', 'S2', 'S3', 'S4']
MODEL_TAGS = ['llama32_3b', 'phi2', 'qwen25_3b', 'qwen25_math1.5b']


def run_phase1_training(model_tag, D_train, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # TODO: implement training (CoT & CCoT). Save checkpoints into output_dir.
    print(f"[PH1] (placeholder) train {model_tag} on {len(D_train)} examples -> {output_dir}")


def run_phase1_evaluation(model_tag, D_val, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    # TODO: evaluate the Phase 1 models on D_val and write phase1_val.json
    print(f"[PH1] (placeholder) eval {model_tag} on {len(D_val)} examples -> {results_dir}")


def run_phase2(model_tag, D_steer, vectors_dir):
    os.makedirs(vectors_dir, exist_ok=True)
    # TODO: extract truth vectors and save into vectors_dir
    print(f"[PH2] (placeholder) extract vectors for {model_tag} from {len(D_steer)} examples -> {vectors_dir}")


def run_alpha_tuning(model_tag, D_val, vectors_dir):
    # TODO: run alpha tuning and return alpha_star
    print(f"[ALPHA] (placeholder) tune alpha for {model_tag} on {len(D_val)} examples using {vectors_dir}")
    return 0.1


def run_steered_evaluation(model_tag, D_val, alpha_star, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    # TODO: run steered evaluation and write steered_val.json with keys:
    # {n_examples, steered_accuracy, flip_rate, probe_accuracy}
    print(f"[STEER] (placeholder) steered eval {model_tag} with alpha={alpha_star} -> {results_dir}")


def main():
    splits = build_all_splits('gsm8k/train.jsonl', seed=42)

    for cfg_id in CONFIGS:
        D_train = splits[cfg_id]['D_train']
        D_steer = splits[cfg_id]['D_steer']
        D_val   = splits[cfg_id]['D_val']

        for model_tag in MODEL_TAGS:

            run_phase1_training(
                model_tag, D_train,
                output_dir=f"checkpoints/{cfg_id}/{model_tag}"
            )
            run_phase1_evaluation(
                model_tag, D_val,
                results_dir=f"results/{cfg_id}/{model_tag}"
            )

            run_phase2(
                model_tag, D_steer,
                vectors_dir=f"vectors/{cfg_id}/{model_tag}"
            )

            alpha_star = run_alpha_tuning(
                model_tag, D_val,
                vectors_dir=f"vectors/{cfg_id}/{model_tag}"
            )
            run_steered_evaluation(
                model_tag, D_val, alpha_star,
                results_dir=f"results/{cfg_id}/{model_tag}"
            )


if __name__ == '__main__':
    main()
