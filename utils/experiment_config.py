"""Validated access to experiment-wide data-count settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs" / "protocol.yaml"
DATA_ROLES = ("D_train", "D_steer", "D_val", "D_test")


@lru_cache(maxsize=1)
def load_protocol() -> dict:
    with PROTOCOL_PATH.open(encoding="utf-8") as f:
        protocol = yaml.safe_load(f) or {}
    return protocol


def samples_per_phase() -> int:
    value = load_protocol().get("samples_per_phase")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"{PROTOCOL_PATH}: samples_per_phase must be a positive integer, got {value!r}"
        )
    return value


def protocol_seed() -> int:
    value = load_protocol().get("seed")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{PROTOCOL_PATH}: seed must be an integer, got {value!r}")
    return value


def require_exact_count(examples: list, role: str) -> None:
    if role not in DATA_ROLES:
        raise ValueError(f"Unknown data role {role!r}; expected one of {DATA_ROLES}")
    expected = samples_per_phase()
    actual = len(examples)
    if actual != expected:
        raise ValueError(f"{role} must contain exactly {expected} examples; got {actual}")
