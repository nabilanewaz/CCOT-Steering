"""Balanced per-head probing, directions, and projected-activation scales."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def probe_all_heads(H_pos, H_neg, top_k=48, seed=42):
    scores, directions, sigmas = {}, {}, {}
    generator = torch.Generator().manual_seed(seed)
    for key in sorted(H_pos):
        count = min(len(H_pos[key]), len(H_neg[key]))
        positive = H_pos[key][torch.randperm(len(H_pos[key]), generator=generator)[:count]].float()
        negative = H_neg[key][torch.randperm(len(H_neg[key]), generator=generator)[:count]].float()
        states = torch.cat([positive, negative])
        features = normalize(states.numpy())
        labels = np.array([1] * count + [0] * count)
        train, test, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=seed, stratify=labels,
        )
        probe = LogisticRegression(max_iter=1000, random_state=seed).fit(train, train_labels)
        scores[key] = float(probe.score(test, test_labels))
        direction = positive.mean(0) - negative.mean(0)
        directions[key] = direction / direction.norm().clamp_min(1e-8)
        all_states = torch.cat([H_pos[key], H_neg[key]]).float()
        sigmas[key] = float((all_states @ directions[key]).std(unbiased=False))
    if not scores:
        raise RuntimeError("ITI extraction produced no heads with sufficient examples in both classes")
    top_heads = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return {
        "top_k": len(top_heads), "top_heads": top_heads,
        "head_scores": {key: scores[key] for key in top_heads},
        "head_directions": {key: directions[key] for key in top_heads},
        "head_sigmas": {key: sigmas[key] for key in top_heads},
    }
