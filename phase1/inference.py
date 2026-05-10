import json
import os
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> str | None:
    """Return the final answer string from model output (number, bool word, or #### suffix), or None."""
    text = text.strip()
    if '####' in text:
        return text.split('####')[1].strip()
    m_bool = re.search(r'\b(true|false)\b', text, flags=re.I)
    if m_bool:
        return m_bool.group(1).lower()
    # Match integers and decimals, including comma-separated thousands
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def normalize_answer(text: str) -> str:
    """Normalize to a plain integer string for comparison."""
    if text is None:
        return ''
    text = re.sub(r'[,$%]', '', str(text).strip()).replace(',', '')
    try:
        return str(int(float(text)))
    except (ValueError, OverflowError):
        return text.lower()


def extract_reasoning_span(text: str) -> str:
    """Strip [compress:R] tag and everything after 'Answer:' marker."""
    parts = text.split('\n\nAnswer:')
    reasoning_block = parts[0]
    lines = reasoning_block.strip().split('\n')
    return '\n'.join(l for l in lines if not l.startswith('[compress:'))


# ── Model loaders ─────────────────────────────────────────────────────────────

def _trust(model_id: str) -> bool:
    return 'qwen' in model_id.lower()


def load_base_frozen(base_model_id: str, device: str):
    """Load the untuned base model (no LoRA)."""
    trust = _trust(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype='auto', trust_remote_code=trust
    ).to(device)
    model.eval()
    return model, tokenizer


def load_finetuned(checkpoint_dir: str, device: str):
    """Load a LoRA-fine-tuned model from a directory saved by PeftModel.save_pretrained."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_cfg_path = os.path.join(checkpoint_dir, 'adapter_config.json')
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_id = adapter_cfg.get('base_model_name_or_path', checkpoint_dir)
        trust = _trust(base_id)
        base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype='auto', trust_remote_code=trust
        )
        model = PeftModel.from_pretrained(base, checkpoint_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, torch_dtype='auto'
        )

    model = model.to(device)
    model.eval()
    return model, tokenizer


# ── Inference functions ───────────────────────────────────────────────────────

def run_no_cot(model, tokenizer, item: dict, device: str) -> tuple[str | None, str]:
    """Direct answer from frozen base model — no reasoning generated."""
    enc = tokenizer(
        f"Question: {item['question']}\n\nAnswer:",
        return_tensors='pt',
    ).to(device)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=32)
    text = tokenizer.decode(
        out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()
    return extract_answer(text), ''


def run_cot(model, tokenizer, item: dict, device: str) -> tuple[str | None, str]:
    enc = tokenizer(
        f"Question: {item['question']}\n\nReasoning:",
        return_tensors='pt',
    ).to(device)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=512)
    generated = tokenizer.decode(
        out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
    )
    parts = generated.split('\n\nAnswer:')
    reasoning = parts[0].strip()
    raw_answer = parts[1].strip() if len(parts) > 1 else ''
    return extract_answer(raw_answer), reasoning


def run_ccot(model, tokenizer, item: dict, ratio: float,
             device: str) -> tuple[str | None, str]:
    enc = tokenizer(
        f"Question: {item['question']}\n\n[compress:{ratio}]\n",
        return_tensors='pt',
    ).to(device)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=256)
    generated = tokenizer.decode(
        out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
    )
    reasoning = extract_reasoning_span(generated)
    parts = generated.split('\n\nAnswer:')
    raw_answer = parts[1].strip() if len(parts) > 1 else ''
    return extract_answer(raw_answer), reasoning


def run_trimmed_cot(cot_model, tokenizer, item: dict, token_budget: int,
                    device: str) -> tuple[str | None, str]:
    """
    Two-stage: generate reasoning capped at token_budget tokens, then
    force answer decoding from the truncated context. Keeps the answer prompt
    well-formed regardless of where truncation falls in the reasoning.
    """
    cot_model.eval()

    # Stage 1 — truncated reasoning
    reasoning_enc = tokenizer(
        f"Question: {item['question']}\n\nReasoning:",
        return_tensors='pt',
    ).to(device)
    with torch.no_grad():
        reasoning_ids = cot_model.generate(
            **reasoning_enc,
            do_sample=False,
            max_new_tokens=token_budget,
        )
    reasoning_text = tokenizer.decode(
        reasoning_ids[0][reasoning_enc['input_ids'].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # Stage 2 — answer from truncated context
    answer_enc = tokenizer(
        f"Question: {item['question']}\n\n"
        f"Reasoning: {reasoning_text}\n\n"
        f"Answer:",
        return_tensors='pt',
    ).to(device)
    with torch.no_grad():
        answer_ids = cot_model.generate(
            **answer_enc,
            do_sample=False,
            max_new_tokens=32,
        )
    answer_text = tokenizer.decode(
        answer_ids[0][answer_enc['input_ids'].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return extract_answer(answer_text), reasoning_text


# ── Budget computation ────────────────────────────────────────────────────────

def compute_per_example_budgets(cot_model, tokenizer, dataset: list,
                                 device: str, ratio: float) -> list[int]:
    """
    For each example, run CoT at greedy decoding to get T_full^(i), then
    return budget = round(ratio * T_full^(i)), floored at 10 tokens.
    """
    budgets = []
    cot_model.eval()
    for item in dataset:
        enc = tokenizer(
            f"Question: {item['question']}\n\nReasoning:",
            return_tensors='pt',
        ).to(device)
        with torch.no_grad():
            out_ids = cot_model.generate(**enc, do_sample=False, max_new_tokens=512)
        generated = tokenizer.decode(
            out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
        )
        reasoning = generated.split('\n\nAnswer:')[0].strip()
        t_full = len(tokenizer.encode(reasoning, add_special_tokens=False))
        budgets.append(max(10, round(ratio * t_full)))
    return budgets


def measure_ccot_token_counts(ccot_model, tokenizer, dataset: list,
                               device: str, ratio: float) -> dict:
    """Run CCoT model on dataset and return reasoning token count statistics."""
    lengths = []
    ccot_model.eval()
    for item in dataset:
        enc = tokenizer(
            f"Question: {item['question']}\n\n[compress:{ratio}]\n",
            return_tensors='pt',
        ).to(device)
        with torch.no_grad():
            out_ids = ccot_model.generate(**enc, do_sample=False, max_new_tokens=256)
        full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        reasoning = extract_reasoning_span(full_text)
        lengths.append(len(tokenizer.encode(reasoning, add_special_tokens=False)))
    return {
        'mean':        float(np.mean(lengths)),
        'median':      float(np.median(lengths)),
        'std':         float(np.std(lengths)),
        'per_example': lengths,
        'ratio':       ratio,
    }
