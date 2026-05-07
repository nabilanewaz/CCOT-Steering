import os
import json
from utils.data import load_test_set


def run_final_evaluation():
    D_test = load_test_set('gsm8k/test.jsonl')
    print(f"Loaded D_test with {len(D_test)} examples")

    # Expect configs/selected.yaml to exist and point to the winning config.
    import yaml
    with open('configs/selected.yaml') as f:
        sel = yaml.safe_load(f)

    winner = sel['winning_config']
    MODEL_TAGS = ['llama32_3b', 'phi2', 'qwen25_3b', 'qwen25_math1.5b']
    out_dir = 'results/final'
    os.makedirs(out_dir, exist_ok=True)

    # Placeholder: copy or compute final test results per model and write JSONs.
    for model_tag in MODEL_TAGS:
        # In a real run you'd load the locked checkpoints/vectors and run evaluation.
        out_path = os.path.join(out_dir, f"{model_tag}_test.json")
        with open(out_path, 'w') as f:
            json.dump({'n_test': len(D_test), 'note': f'Placeholder for {model_tag} (winner={winner})'}, f)
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    run_final_evaluation()
