"""Model-specific Phase 2 hyperparameters (spec §2.8)."""

MODEL_PHASE2_CONFIG = {
    'llama32_3b': {
        'N': 20,
        'cpca_variant':         'randomized',
        'r_per_layer':          3,
        'r_final':              10,
        'beta':                 0.5,
        'threshold_multiplier': 0.5,
        'min_samples':          200,
    },
    'phi2': {
        'N': 20,
        'cpca_variant':         'full',
        'r_per_layer':          3,
        'r_final':              10,
        'beta':                 0.5,
        'threshold_multiplier': 0.3,
        'min_samples':          200,
    },
    'qwen25_0.5b': {
        'N': 20,
        'cpca_variant':         'full',
        'r_per_layer':          2,
        'r_final':              6,
        'beta':                 0.5,
        'threshold_multiplier': 0.5,
        'min_samples':          200,
    },
    'qwen25_3b': {
        'N': 20,
        'cpca_variant':         'full',
        'r_per_layer':          3,
        'r_final':              10,
        'beta':                 0.5,
        'threshold_multiplier': 0.5,
        'min_samples':          200,
    },
    'qwen25_math1.5b': {
        'N': 20,
        'cpca_variant':         'shrunk',
        'r_per_layer':          2,
        'r_final':              6,
        'beta':                 0.5,
        'threshold_multiplier': 0.6,
        'min_samples':          300,
    },
}

_DEFAULT_CONFIG = {
    'N': 20,
    'cpca_variant':         'full',
    'r_per_layer':          3,
    'r_final':              10,
    'beta':                 0.5,
    'threshold_multiplier': 0.5,
    'min_samples':          200,
}


def get_model_config(model_tag: str) -> dict:
    return MODEL_PHASE2_CONFIG.get(model_tag, _DEFAULT_CONFIG).copy()
