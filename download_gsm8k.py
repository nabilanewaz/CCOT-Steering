"""Download GSM8K from HuggingFace and write to gsm8k/train.jsonl and gsm8k/test.jsonl."""
import json
import os


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "datasets package not found. Install with:\n"
            "  pip install datasets"
        )

    os.makedirs("gsm8k", exist_ok=True)
    print("Downloading GSM8K from HuggingFace...")
    ds = load_dataset("gsm8k", "main")

    train_path = "gsm8k/train.jsonl"
    test_path  = "gsm8k/test.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds["train"]):
            f.write(json.dumps({
                "id":       i,
                "question": item["question"],
                "answer":   item["answer"],
            }) + "\n")
    print(f"  Train: {len(ds['train'])} examples -> {train_path}")

    with open(test_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds["test"]):
            f.write(json.dumps({
                "id":       i,
                "question": item["question"],
                "answer":   item["answer"],
            }) + "\n")
    print(f"  Test:  {len(ds['test'])} examples -> {test_path}")
    print("Done.")


if __name__ == "__main__":
    main()
