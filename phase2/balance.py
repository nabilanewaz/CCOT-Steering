"""Stratified balanced undersampling for H+/H− class imbalance (spec §2.8)."""
import numpy as np

IMBALANCE_THRESHOLD = 1.0   # always balance — equal H+ and H-
N_BUCKETS = 4               # difficulty quartiles (0 = hard, 3 = easy)


def difficulty_bucket(frac_correct: float) -> int:
    """Map per-question correct fraction to a difficulty bucket index 0..3."""
    return min(int(frac_correct * N_BUCKETS), N_BUCKETS - 1)


def check_balance(H_plus_raw: dict, H_minus_raw: dict) -> tuple[float, int, int]:
    """
    Return (ratio, n_pos, n_neg).
    ratio = max(|H+|,|H-|) / min(|H+|,|H-|)  — symmetric, always >= 1.
    """
    if not H_plus_raw:
        return 0.0, 0, 0
    ref_L = next(iter(H_plus_raw))
    n_pos = len(H_plus_raw[ref_L])
    n_neg = len(H_minus_raw.get(ref_L, []))
    ratio = max(n_pos, n_neg) / max(min(n_pos, n_neg), 1)
    return ratio, n_pos, n_neg


def _stratified_downsample(
    H_small: dict,
    H_large: dict,
    bucket_small: dict,
    bucket_large: dict,
    rng: np.random.Generator,
    label_large: str,
) -> dict:
    """
    Downsample H_large (per layer) to match the count in H_small, stratified by
    difficulty bucket. H_small is not touched.
    """
    H_large_bal: dict = {}

    for L in H_small:
        n_small = len(H_small[L])
        h_large_L = H_large.get(L, [])
        n_large = len(h_large_L)

        if n_large <= n_small:
            H_large_bal[L] = h_large_L
            continue

        small_b = np.asarray(bucket_small.get(L, [0] * n_small), dtype=np.int32)
        large_b = np.asarray(bucket_large.get(L, [0] * n_large), dtype=np.int32)
        keep_idx = []

        for b in range(N_BUCKETS):
            idx_small_b = np.where(small_b == b)[0]
            idx_large_b = np.where(large_b == b)[0]
            if len(idx_small_b) == 0 or len(idx_large_b) == 0:
                continue
            n_sample = min(len(idx_small_b), len(idx_large_b))
            chosen = rng.choice(idx_large_b, size=n_sample, replace=False)
            keep_idx.extend(chosen.tolist())

        keep_idx.sort()
        H_large_bal[L] = [h_large_L[i] for i in keep_idx]

        n_bal = len(H_large_bal[L])
        print(f"  Layer {L:02d}: {label_large} {n_large} -> {n_bal}  "
              f"(balanced to match {n_small})  [stratified, {N_BUCKETS} buckets]")

    return H_large_bal


def stratified_balance(
    H_plus_raw: dict,
    H_minus_raw: dict,
    bucket_pos: dict,
    bucket_neg: dict,
    threshold: float = IMBALANCE_THRESHOLD,
    seed: int = 42,
) -> tuple[dict, dict]:
    """
    Per-layer stratified undersampling to equalise H+ and H−.

    Whichever class is larger is downsampled to match the smaller, stratified
    by difficulty bucket. Both directions are handled:
      - H− > H+: downsample H−, keep all H+
      - H+ > H−: downsample H+, keep all H−
    """
    rng = np.random.default_rng(seed)

    if not H_plus_raw:
        return H_plus_raw, H_minus_raw

    ref_L = next(iter(H_plus_raw))
    n_pos = len(H_plus_raw[ref_L])
    n_neg = len(H_minus_raw.get(ref_L, []))

    if n_neg >= n_pos:
        print(f"  H- ({n_neg}) >= H+ ({n_pos}): downsampling H-")
        H_minus_bal = _stratified_downsample(
            H_plus_raw, H_minus_raw, bucket_pos, bucket_neg, rng, "H-"
        )
        return H_plus_raw, H_minus_bal
    else:
        print(f"  H+ ({n_pos}) > H- ({n_neg}): downsampling H+")
        H_plus_bal = _stratified_downsample(
            H_minus_raw, H_plus_raw, bucket_neg, bucket_pos, rng, "H+"
        )
        return H_plus_bal, H_minus_raw
