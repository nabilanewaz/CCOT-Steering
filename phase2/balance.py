"""Balance both classes within question-difficulty quartiles."""
import numpy as np

IMBALANCE_THRESHOLD = 1.0
N_BUCKETS = 4


def difficulty_bucket(frac_correct: float) -> int:
    return min(int(frac_correct * N_BUCKETS), N_BUCKETS - 1)


def check_balance(H_plus_raw: dict, H_minus_raw: dict) -> tuple[float, int, int]:
    if not H_plus_raw:
        return 0.0, 0, 0
    layer = next(iter(H_plus_raw))
    n_pos, n_neg = len(H_plus_raw[layer]), len(H_minus_raw.get(layer, []))
    return max(n_pos, n_neg) / max(min(n_pos, n_neg), 1), n_pos, n_neg


def stratified_balance(
    H_plus_raw, H_minus_raw, bucket_pos, bucket_neg,
    threshold=IMBALANCE_THRESHOLD, seed=42,
):
    balanced_pos, balanced_neg = {}, {}
    for layer in H_plus_raw:
        rng = np.random.default_rng(seed)
        positives = H_plus_raw[layer]
        negatives = H_minus_raw.get(layer, [])
        pos_buckets = np.asarray(bucket_pos.get(layer, [0] * len(positives)))
        neg_buckets = np.asarray(bucket_neg.get(layer, [0] * len(negatives)))
        pos_keep, neg_keep = [], []
        for bucket in range(N_BUCKETS):
            pos_indices = np.flatnonzero(pos_buckets == bucket)
            neg_indices = np.flatnonzero(neg_buckets == bucket)
            size = min(len(pos_indices), len(neg_indices))
            pos_keep.extend(rng.choice(pos_indices, size, replace=False).tolist())
            neg_keep.extend(rng.choice(neg_indices, size, replace=False).tolist())
        balanced_pos[layer] = [positives[index] for index in sorted(pos_keep)]
        balanced_neg[layer] = [negatives[index] for index in sorted(neg_keep)]
    return balanced_pos, balanced_neg
