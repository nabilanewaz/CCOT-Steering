def format_cot(item: dict) -> str:
    reasoning = item['answer'].split('####')[0].strip()
    answer    = item['answer'].split('####')[1].strip()
    return (
        f"Question: {item['question']}\n\n"
        f"Reasoning: {reasoning}\n\n"
        f"Answer: {answer}"
    )


def format_ccot(item: dict, compressed_reasoning: str, ratio: float) -> str:
    answer = item['answer'].split('####')[1].strip()
    return (
        f"Question: {item['question']}\n\n"
        f"[compress:{ratio}]\n"
        f"{compressed_reasoning}\n\n"
        f"Answer: {answer}"
    )
