"""Stratified balanced undersampling for H+/H− class imbalance (spec §2.8)."""
import numpy as np
from collections import defaultdict

IMBALANCE_THRESHOLD = 3.0   # trigger rebalancing if |H−| / |H+| > this
N_BUCKETS = 4               # difficulty quartiles (0 = hard, 3 = easy)


def difficulty_bucket(frac_correct: float) -> int:
    """Map per-question correct fraction to a difficulty bucket index 0..3."""
    return min(int(frac_correct * N_BUCKETS), N_BUCKETS - 1)


def check_balance(H_plus_raw: dict, H_minus_raw: dict) -> tuple[float, int, int]:
    """
    Return (ratio, n_pos, n_neg) using the first available layer as reference.
    ratio = |H−| / |H+|.
    """
    if not H_plus_raw:
        return 0.0, 0, 0
    ref_L = next(iter(H_plus_raw))
    n_pos = len(H_plus_raw[ref_L])
    n_neg = len(H_minus_raw.get(ref_L, []))
    return n_neg / max(n_pos, 1), n_pos, n_neg


def stratified_balance(
    H_plus_raw: dict,
    H_minus_raw: dict,
    bucket_pos: dict,
    bucket_neg: dict,
    threshold: float = IMBALANCE_THRESHOLD,
    seed: int = 42,
) -> tuple[dict, dict]:
    """
    Per-layer stratified undersampling of H− when |H−|/|H+| > threshold.

    Within each difficulty bucket, H− is downsampled to match the H+ count
    in that bucket. H+ is kept intact. This preserves the difficulty
    distribution while correcting the class imbalance.

    Args:
        H_plus_raw:  dict[L -> list[Tensor [d]]]
        H_minus_raw: dict[L -> list[Tensor [d]]]
        bucket_pos:  dict[L -> list[int]]  — difficulty bucket per H+ sample
        bucket_neg:  dict[L -> list[int]]  — difficulty bucket per H- sample
        threshold:   ratio above which rebalancing is triggered
        seed:        RNG seed for reproducibility

    Returns:
        (H_plus_raw, H_minus_bal)  — H+ unchanged, H- undersampled
    """
    rng = np.random.default_rng(seed)
    H_minus_bal: dict = {}

    for L in H_plus_raw:
        n_pos = len(H_plus_raw[L])
        h_neg_L = H_minus_raw.get(L, [])
        n_neg = len(h_neg_L)
        ratio = n_neg / max(n_pos, 1)

        if ratio <= threshold:
            H_minus_bal[L] = h_neg_L
            continue

        # Collect H- indices to keep, bucket by bucket
        pos_b = np.asarray(bucket_pos.get(L, [0] * n_pos), dtype=np.int32)
        neg_b = np.asarray(bucket_neg.get(L, [0] * n_neg), dtype=np.int32)
        keep_idx = []

        for b in range(N_BUCKETS):
            idx_pos_b = np.where(pos_b == b)[0]
            idx_neg_b = np.where(neg_b == b)[0]
            if len(idx_pos_b) == 0 or len(idx_neg_b) == 0:
                continue
            n_sample = min(len(idx_pos_b), len(idx_neg_b))
            chosen = rng.choice(idx_neg_b, size=n_sample, replace=False)
            keep_idx.extend(chosen.tolist())

        keep_idx.sort()
        H_minus_bal[L] = [h_neg_L[i] for i in keep_idx]

        n_bal = len(H_minus_bal[L])
        print(f"  Layer {L:02d}: H- {n_neg} -> {n_bal}  "
              f"(ratio {ratio:.1f}x -> {n_bal/max(n_pos,1):.2f}x)  "
              f"[stratified, {N_BUCKETS} buckets]")

    return H_plus_raw, H_minus_bal
