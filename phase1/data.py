class SimpleDataset:
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)

    @property
    def features(self):
        if self._data:
            return {k: None for k in self._data[0].keys()}
        return {}

    def map(self, func, remove_columns=None, num_proc=None):
        del num_proc
        mapped = []
        for item in self._data:
            result = func(item)
            if remove_columns:
                result = {k: v for k, v in result.items() if k not in remove_columns}
            mapped.append(result)
        return SimpleDataset(mapped)

    def shuffle(self, seed=42):
        import random

        rng = random.Random(seed)
        shuffled = list(self._data)
        rng.shuffle(shuffled)
        return SimpleDataset(shuffled)


def get_hf_dataset(raw_data, tokenizer):
    tokenized = []
    for sample in raw_data:
        q_tok = tokenizer.encode(sample["question"] + "\n", add_special_tokens=True)
        s_tok = [tokenizer.encode(s + "\n", add_special_tokens=False) for s in sample["steps"]]
        a_tok = tokenizer.encode("#### " + sample["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
        tokenized.append(
            {
                "qid": sample.get("qid"),
                "question": sample["question"],
                "question_tokenized": q_tok,
                "steps_tokenized": s_tok,
                "answer_tokenized": a_tok,
                "ground_truth": sample["answer"],
            }
        )
    return SimpleDataset(tokenized)
