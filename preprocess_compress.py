"""Build the offline compression cache for CCoT training.

Run once before Phase 1 training.  Compresses every reasoning trace in D_train
(largest split S1 — all smaller splits are subsets) at each ratio and writes
JSONL files to cache/{cfg_id}/compressed_R*.jsonl.

Usage:
    python preprocess_compress.py              # uses S1 (largest D_train)
    python preprocess_compress.py --config S2  # single split config
    python preprocess_compress.py --all        # all four split configs
"""
import argparse
import json
import os

RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9]
TRAIN_POOL = "gsm8k/train.jsonl"


def _build_cache(D_train: list, cache_dir: str, compressor):
    os.makedirs(cache_dir, exist_ok=True)
    for ratio in RATIOS:
        rtag      = f"R{int(ratio * 10)}"
        out_path  = os.path.join(cache_dir, f"compressed_{rtag}.jsonl")
        if os.path.exists(out_path):
            print(f"  Cache {rtag} exists ({out_path}) — skipping")
            continue

        print(f"  Compressing R={ratio}  ({len(D_train)} examples)...")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in D_train:
                reasoning = item["answer"].split("####")[0].strip()
                result = compressor.compress_prompt(
                    reasoning,
                    rate=ratio,
                    force_tokens=["\n", "."],
                    drop_consecutive=True,
                )
                compressed = result["compressed_prompt"]
                n_orig = len(reasoning.split())
                n_comp = len(compressed.split())
                f.write(json.dumps({
                    "id":             item.get("id", ""),
                    "compressed":     compressed,
                    "actual_ratio":   n_comp / max(n_orig, 1),
                    "target_ratio":   ratio,
                    "original_len":   n_orig,
                    "compressed_len": n_comp,
                }) + "\n")
        print(f"    -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build CCoT compression cache.")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--config", type=str, default="S1",
                     help="Single split config to build cache for (default: S1)")
    grp.add_argument("--all", action="store_true",
                     help="Build cache for all four split configs")
    parser.add_argument("--pool", default=TRAIN_POOL,
                        help="Path to GSM8K train pool JSONL")
    args = parser.parse_args()

    try:
        from llmlingua import PromptCompressor
    except ImportError:
        raise SystemExit(
            "llmlingua not installed. Install with:\n"
            "  pip install llmlingua"
        )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LLMLingua-2 compressor on {device}...")
    compressor = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,
        device_map=device,
    )

    from scripts.build_splits import build_all_splits
    splits = build_all_splits(args.pool, seed=42)

    configs = list(splits.keys()) if args.all else [args.config]
    for cfg_id in configs:
        print(f"\n--- {cfg_id} ---")
        D_train   = splits[cfg_id]["D_train"]
        cache_dir = f"cache/{cfg_id}"
        _build_cache(D_train, cache_dir, compressor)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
