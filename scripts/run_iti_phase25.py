"""Optional full-D_steer head extraction after Phase 2."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from phase2.run import run_iti_phase2
from scripts.build_splits import build_all_splits
from utils.dataset_paths import get_train_pool_path, init_project_dataset, artifact_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="S3", choices=("S3",))
    parser.add_argument("--dataset", default="gsm8k", choices=("gsm8k", "svamp", "prontoqa"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    init_project_dataset(args.dataset, interactive=False)
    split = build_all_splits(get_train_pool_path())[args.config]
    suffix = f"{args.config}/{args.model}"
    run_iti_phase2(args.model, f"{artifact_root('checkpoints')}/{suffix}", split["D_steer"], args.device,
                   f"{artifact_root('vectors')}/{suffix}", f"{artifact_root('results')}/{suffix}")


if __name__ == "__main__":
    main()
