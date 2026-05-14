"""Build the phase1 compatibility cache.

Run once before Phase 1 training. Writes cache/S2/compressed_R*.jsonl files
needed by downstream orchestration contracts.

Usage:
    python preprocess_compress.py              # uses S2 (default)
    python preprocess_compress.py --all        # all split configs (also just S2)
"""
import argparse
import os

RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    parser = argparse.ArgumentParser(description="Build phase1 compatibility cache.")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--config", type=str, default="S2",
                     help="Single split config to build cache for (default: S2)")
    grp.add_argument("--all", action="store_true",
                     help="Build cache for all four split configs")
    parser.add_argument("--pool", default=None,
                        help="Path to train pool JSONL (default: active dataset)")
    parser.add_argument("--dataset", default=None, choices=("gsm8k", "svamp", "prontoqa"),
                        help="Active dataset id when --pool is omitted")
    args = parser.parse_args()

    import sys

    from phase1.compress import build_ccot_cache
    from scripts.build_splits import build_all_splits
    from utils.dataset_paths import get_train_pool_path, init_project_dataset

    init_project_dataset(args.dataset, interactive=sys.stdin.isatty())
    pool = args.pool or get_train_pool_path()
    splits = build_all_splits(pool, seed=42)

    configs = list(splits.keys()) if args.all else [args.config]
    for cfg_id in configs:
        print(f"\n--- {cfg_id} ---")
        D_train   = splits[cfg_id]["D_train"]
        cache_dir = f"cache/{cfg_id}"
        build_ccot_cache(D_train, RATIOS, cache_dir, compressor=None)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
