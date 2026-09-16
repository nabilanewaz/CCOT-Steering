"""Stable identities and atomic persistence for full-experiment artifacts."""
import hashlib
import json
import os
from pathlib import Path


EXPERIMENT_VERSION = "full_60_10_30_v1"


def fingerprint(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_fingerprint(examples: list) -> str:
    return fingerprint(examples)


def file_fingerprint(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(path) -> dict:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint missing: {root}")
    files = []
    for entry in sorted(root.iterdir()):
        if entry.is_file():
            stat = entry.stat()
            files.append([entry.name, stat.st_size, stat.st_mtime_ns])
    return {"path": str(root), "files": files}


def atomic_json(path, payload) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, target)
