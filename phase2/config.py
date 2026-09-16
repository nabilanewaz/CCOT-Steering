"""Phase 2 settings from PHASE2.md, adapted to Coconut checkpoints."""

_DEFAULT_CONFIG = {
    "N": 10,
    "cpca_variant": "full",
    "r_per_layer": 3,
    "r_final": 10,
    "beta": 0.5,
    "threshold_multiplier": 0.5,
    "min_samples": 200,
    "extraction": "mean_gen",
    "gen_window": 20,
}

MODEL_PHASE2_CONFIG = {
    "qwen25_3b": dict(_DEFAULT_CONFIG),
    "qwen25_0.5b": dict(_DEFAULT_CONFIG),
    "qwen25_math1.5b": {
        **_DEFAULT_CONFIG, "cpca_variant": "shrunk",
        "r_final": 8, "threshold_multiplier": 0.4,
    },
    "llama32_3b": {**_DEFAULT_CONFIG, "cpca_variant": "randomized"},
    "phi2": dict(_DEFAULT_CONFIG),
}


def get_model_config(model_tag: str) -> dict:
    return MODEL_PHASE2_CONFIG.get(model_tag, _DEFAULT_CONFIG).copy()
