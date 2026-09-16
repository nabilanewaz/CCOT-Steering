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


def split_counts(pool_size: int) -> dict[str, int]:
    if pool_size < 10:
        raise ValueError("A 60/10/30 split requires at least ten examples")
    train = round(pool_size * 0.60)
    steer = round(pool_size * 0.10)
    return {"D_train": train, "D_steer": steer, "D_val": pool_size - train - steer}


def expected_count(role: str) -> int:
    from utils.dataset_paths import get_active_dataset_id

    sizes = load_protocol()["dataset_sizes"][get_active_dataset_id()]
    return sizes["test"] if role == "D_test" else split_counts(sizes["train"])[role]


def protocol_seed() -> int:
    value = load_protocol().get("seed")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{PROTOCOL_PATH}: seed must be an integer, got {value!r}")
    return value


def require_exact_count(examples: list, role: str) -> None:
    if role not in DATA_ROLES:
        raise ValueError(f"Unknown data role {role!r}; expected one of {DATA_ROLES}")
    expected = expected_count(role)
    actual = len(examples)
    if actual != expected:
        raise ValueError(f"{role} must contain exactly {expected} examples; got {actual}")
