import json
import os


def compress_reasoning(reasoning: str, ratio: float, compressor) -> str:
    result = compressor.compress_prompt(
        reasoning,
        rate=ratio,
        force_tokens=['\n', '.'],
        drop_consecutive=True,
    )
    return result['compressed_prompt']


def build_ccot_cache(D_train: list, ratios: list, cache_dir: str, compressor) -> None:
    """Offline: compress every D_train reasoning trace at each ratio and save to JSONL."""
    os.makedirs(cache_dir, exist_ok=True)
    for ratio in ratios:
        cache_path = os.path.join(cache_dir, f"compressed_R{int(ratio * 10)}.jsonl")
        if os.path.exists(cache_path):
            print(f"Cache exists for R={ratio}, skipping.")
            continue
        print(f"Compressing at R={ratio}...")
        with open(cache_path, 'w', encoding='utf-8') as f:
            for item in D_train:
                full_reasoning = item['answer'].split('####')[0].strip()
                compressed = compress_reasoning(full_reasoning, ratio, compressor)
                n_orig = len(full_reasoning.split())
                n_comp = len(compressed.split())
                f.write(json.dumps({
                    'id':             item.get('id', ''),
                    'compressed':     compressed,
                    'actual_ratio':   n_comp / max(n_orig, 1),
                    'target_ratio':   ratio,
                    'original_len':   n_orig,
                    'compressed_len': n_comp,
                }) + '\n')
        print(f"  Done — {cache_path}")


def load_cache(cache_path: str) -> list:
    with open(cache_path, encoding='utf-8') as f:
        return [json.loads(l) for l in f]
