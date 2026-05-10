"""Active dataset selection (GSM8K, SVAMP, ProntoQA) and canonical JSONL paths.

Non-interactive modes (no prompt):
  - Environment variable ``CCOT_DATASET`` = ``gsm8k`` | ``svamp`` | ``prontoqa``
  - CLI ``--dataset <id>`` on supported entry points
  - File ``configs/active_dataset.txt`` (written after an interactive choice, or by pipeline)

Train pool and test paths are always ``<dataset_id>/train.jsonl`` and ``<dataset_id>/test.jsonl``,
matching the GSM8K layout: official HF **train** split only in ``train.jsonl``, official **test**
split only in ``test.jsonl``, each line ``{"id", "question", "answer"}`` with per-file ids starting at 0.
"""
from __future__ import annotations

import os
import sys

DATASET_IDS: tuple[str, ...] = ("gsm8k", "svamp", "prontoqa")

_ACTIVE: str | None = None
_PERSIST_REL = os.path.join("configs", "active_dataset.txt")


def _persist_path() -> str:
    return os.path.abspath(_PERSIST_REL)


def _read_persisted() -> str | None:
    path = _persist_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip().lower()
        return raw if raw in DATASET_IDS else None
    except OSError:
        return None


def _write_persisted(dataset_id: str) -> None:
    os.makedirs(os.path.dirname(_persist_path()) or ".", exist_ok=True)
    with open(_persist_path(), "w", encoding="utf-8") as f:
        f.write(dataset_id + "\n")


def _prompt_choice() -> str:
    print("\nSelect dataset (all phases use the same JSONL layout):")
    print("  1) gsm8k   — grade-school math (default)")
    print("  2) svamp   — challenge math word problems")
    print("  3) prontoqa — synthetic logical reasoning (True/False)")
    default = _read_persisted() or "gsm8k"
    tip = f" [default: {default}]" if default else ""
    raw = input(f"Enter 1–3 or name{tip}: ").strip().lower()
    if not raw:
        return default
    if raw in ("1", "gsm8k", "g"):
        return "gsm8k"
    if raw in ("2", "svamp", "s"):
        return "svamp"
    if raw in ("3", "prontoqa", "p", "pronto"):
        return "prontoqa"
    if raw in DATASET_IDS:
        return raw
    print(f"Unrecognized choice {raw!r} — using {default}.", file=sys.stderr)
    return default


def set_active_dataset(dataset_id: str, *, persist: bool = True) -> None:
    if dataset_id not in DATASET_IDS:
        raise ValueError(f"Unknown dataset {dataset_id!r}; expected one of {DATASET_IDS}")
    global _ACTIVE
    _ACTIVE = dataset_id
    if persist:
        _write_persisted(dataset_id)


def get_active_dataset_id() -> str:
    global _ACTIVE
    if _ACTIVE in DATASET_IDS:
        return _ACTIVE
    env = os.environ.get("CCOT_DATASET", "").strip().lower()
    if env in DATASET_IDS:
        _ACTIVE = env
        return env
    disk = _read_persisted()
    if disk in DATASET_IDS:
        _ACTIVE = disk
        return disk
    _ACTIVE = "gsm8k"
    return "gsm8k"


def get_train_pool_path() -> str:
    d = get_active_dataset_id()
    return os.path.join(d, "train.jsonl")


def get_test_path() -> str:
    d = get_active_dataset_id()
    return os.path.join(d, "test.jsonl")


def init_project_dataset(
    cli_dataset: str | None = None,
    *,
    interactive: bool | None = None,
    persist: bool = True,
) -> str:
    """Resolve and set the active dataset for this process.

    * ``cli_dataset`` — explicit id from argparse (highest priority after env).
    * ``interactive`` — if ``None``, prompt only when a TTY is attached and no explicit source.
    """
    global _ACTIVE

    env = os.environ.get("CCOT_DATASET", "").strip().lower()
    if env in DATASET_IDS:
        _ACTIVE = env
        return env

    if cli_dataset:
        cid = cli_dataset.strip().lower()
        if cid not in DATASET_IDS:
            raise ValueError(f"--dataset must be one of {list(DATASET_IDS)}, got {cli_dataset!r}")
        set_active_dataset(cid, persist=persist)
        return cid

    if _ACTIVE in DATASET_IDS:
        return _ACTIVE

    if interactive is None:
        interactive = sys.stdin.isatty()

    if interactive:
        choice = _prompt_choice()
        set_active_dataset(choice, persist=persist)
        return choice

    disk = _read_persisted()
    if disk in DATASET_IDS:
        _ACTIVE = disk
        return disk

    _ACTIVE = "gsm8k"
    return "gsm8k"


def phase4_subprocess_env() -> dict:
    """Environment dict for spawning ``evaluate_final.py`` with the same dataset."""
    env = os.environ.copy()
    env["CCOT_DATASET"] = get_active_dataset_id()
    return env
