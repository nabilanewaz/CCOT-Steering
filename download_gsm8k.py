"""Backward-compatible entry point: downloads GSM8K JSONL via ``download_dataset``."""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from download_dataset import download_gsm8k  # noqa: E402


def main():
    download_gsm8k()


if __name__ == "__main__":
    main()
