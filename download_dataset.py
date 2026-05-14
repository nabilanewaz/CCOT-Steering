"""Download GSM8K, SVAMP, or ProntoQA and write JSONL in the **same contract as GSM8K**.

For every dataset:

* **Splits:** ``train.jsonl`` ← Hugging Face official ``train`` split only;
  ``test.jsonl`` ← official ``test`` split only (same isolation story as GSM8K).
* **Records:** each line is ``{"id": str, "question": str, "answer": str}``.
* **Ids:** split-qualified unique ids (``train_<i>`` / ``test_<i>``) to keep
  train/val/test leakage checks strict by id.
* **Files:** ``<out_dir>/train.jsonl`` and ``<out_dir>/test.jsonl``, written with the same
  streaming pattern as GSM8K (UTF-8, one ``json.dumps`` per line).

``answer`` must contain ``####`` with the final label after it (GSM8K-native for GSM8K;
normalized for SVAMP / ProntoQA so Phase~1 compression sees a reasoning span).

Usage:
  python download_dataset.py              # interactive dataset menu
  python download_dataset.py --dataset svamp
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import re
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def download_gsm8k(out_dir: str = "gsm8k") -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install `datasets` (pip install datasets)") from e

    os.makedirs(out_dir, exist_ok=True)
    # Use the fully-qualified Hub id to avoid local-folder shadowing
    # when a `./gsm8k` directory already exists in the project.
    ds = load_dataset("openai/gsm8k", "main")
    train_path = os.path.join(out_dir, "train.jsonl")
    test_path = os.path.join(out_dir, "test.jsonl")
    n_train = 0
    with open(train_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds["train"]):
            rec = {"id": f"train_{i}", "question": item["question"], "answer": item["answer"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_train += 1
    n_test = 0
    with open(test_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds["test"]):
            rec = {"id": f"test_{i}", "question": item["question"], "answer": item["answer"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_test += 1
    print(f"GSM8K: {n_train} train -> {train_path}")
    print(f"GSM8K: {n_test} test  -> {test_path}")


def download_svamp(out_dir: str = "svamp") -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install `datasets` (pip install datasets)") from e

    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("ChilleD/SVAMP")
    train_path = os.path.join(out_dir, "train.jsonl")
    test_path = os.path.join(out_dir, "test.jsonl")

    def _svamp_record(i: int, row: dict, split: str) -> dict:
        # Mirror hub field names (ChilleD/SVAMP); fall back for lowercase variants.
        body = (row.get("Body") or row.get("body") or "").strip()
        q = (row.get("Question") or row.get("question") or "").strip()
        q_concat = (row.get("question_concat") or row.get("Question_concat") or "").strip()
        question = q_concat if q_concat else f"{body} {q}".strip()
        eq = (row.get("Equation") or row.get("equation") or "").strip()
        ans = str(row.get("Answer", row.get("answer", ""))).strip()
        # One reasoning block then #### final number (same structural pattern as GSM8K answers).
        reasoning = (
            f"{body}\n\n{q}\n\n"
            f"To solve this, we use the equation {eq}.\n\n"
            f"Therefore, the numerical answer is {ans}."
        ).strip()
        return {"id": f"{split}_{i}", "question": question, "answer": f"{reasoning}\n\n####\n{ans}"}

    n_train = 0
    with open(train_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds["train"]):
            f.write(json.dumps(_svamp_record(i, row, "train"), ensure_ascii=False) + "\n")
            n_train += 1
    n_test = 0
    with open(test_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds["test"]):
            f.write(json.dumps(_svamp_record(i, row, "test"), ensure_ascii=False) + "\n")
            n_test += 1
    print(f"SVAMP: {n_train} train -> {train_path}")
    print(f"SVAMP: {n_test} test  -> {test_path}")


def _parse_options(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            if isinstance(v, list):
                return [str(x) for x in v]
        except (SyntaxError, ValueError):
            pass
        return [raw]
    return [str(raw)]


def _gold_from_mc(options: list[str], letter: str) -> str:
    letter = (letter or "").strip().upper()[:1]
    if not letter:
        return ""
    pat = re.compile(rf"^{re.escape(letter)}\s*[\).:]\s*(.+)", re.I)
    for opt in options:
        m = pat.match(str(opt).strip())
        if m:
            return m.group(1).strip().lower()
    return letter.lower()


def _prontoqa_record(i: int, row: dict, split: str) -> dict:
    ctx = (row.get("context") or "").strip()
    q = (row.get("question") or "").strip()
    opts = _parse_options(row.get("options"))
    cot = (row.get("chain_of_thought") or "").strip()
    letter = str(row.get("answer", "")).strip()
    gold = _gold_from_mc(opts, letter)
    opt_line = " | ".join(opts) if opts else ""
    question = f"{ctx}\n\n{q}\n\nAnswer choices: {opt_line}".strip()
    return {"id": f"{split}_{i}", "question": question, "answer": f"{cot}\n\n####\n{gold}"}


def download_prontoqa(out_dir: str = "prontoqa") -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install `datasets` (pip install datasets)") from e

    # Same split policy as GSM8K: official ``train`` → train pool, official ``test`` → D_test.
    ds = load_dataset("smoorsmith/prontoqa")
    if "train" not in ds or "test" not in ds:
        raise SystemExit("smoorsmith/prontoqa must expose 'train' and 'test' splits.")

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    test_path = os.path.join(out_dir, "test.jsonl")

    n_train = 0
    with open(train_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds["train"]):
            f.write(json.dumps(_prontoqa_record(i, row, "train"), ensure_ascii=False) + "\n")
            n_train += 1
    n_test = 0
    with open(test_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds["test"]):
            f.write(json.dumps(_prontoqa_record(i, row, "test"), ensure_ascii=False) + "\n")
            n_test += 1
    print(f"ProntoQA: {n_train} train -> {train_path}")
    print(f"ProntoQA: {n_test} test  -> {test_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GSM8K / SVAMP / ProntoQA as JSONL.")
    parser.add_argument(
        "--dataset",
        choices=("gsm8k", "svamp", "prontoqa"),
        default=None,
        help="Dataset id (omit for interactive menu)",
    )
    args = parser.parse_args()

    from utils.dataset_paths import init_project_dataset

    if args.dataset:
        ds_id = args.dataset
    else:
        if sys.stdin.isatty():
            print("Which dataset to download?")
            print("  1) gsm8k")
            print("  2) svamp")
            print("  3) prontoqa")
            choice = input("Enter 1–3 [1]: ").strip() or "1"
            ds_id = {"1": "gsm8k", "2": "svamp", "3": "prontoqa"}.get(choice, "gsm8k")
        else:
            ds_id = "gsm8k"

    init_project_dataset(ds_id, interactive=False, persist=True)

    if ds_id == "gsm8k":
        download_gsm8k()
    elif ds_id == "svamp":
        download_svamp()
    else:
        download_prontoqa()


if __name__ == "__main__":
    main()
