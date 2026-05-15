import json
import os
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.torch_compat import patch_transformers_custom_op_registration


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
    """Strip control tags and everything after 'Answer:' marker."""
    parts = text.split('\n\nAnswer:')
    reasoning_block = parts[0]
    lines = reasoning_block.strip().split('\n')
    return '\n'.join(
        l for l in lines
        if not l.startswith('[compress:') and not l.startswith('[latents:')
    )


# ── Model loaders ─────────────────────────────────────────────────────────────

def _trust(model_id: str) -> bool:
    return 'qwen' in model_id.lower()


def _fallback_token_id(tokenizer, model, *names: str) -> int | None:
    for name in names:
        token_id = getattr(tokenizer, f"{name}_token_id", None)
        if token_id is not None:
            return token_id
        token_id = getattr(getattr(model, "config", None), f"{name}_token_id", None)
        if token_id is not None:
            return token_id
        token_id = getattr(getattr(model, "generation_config", None), f"{name}_token_id", None)
        if token_id is not None:
            return token_id
    return None


def _single_token_id(token_id) -> int | None:
    if isinstance(token_id, (list, tuple)):
        return token_id[0] if token_id else None
    return token_id


def _ensure_generation_token_ids(model, tokenizer) -> dict:
    eos_id = _fallback_token_id(tokenizer, model, "eos", "sep")
    pad_id = _single_token_id(_fallback_token_id(tokenizer, model, "pad", "eos", "unk"))
    if pad_id is None:
        pad_id = 0

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = pad_id
        if eos_id is not None:
            model.config.eos_token_id = eos_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = pad_id
        if eos_id is not None:
            model.generation_config.eos_token_id = eos_id

    kwargs = {"pad_token_id": pad_id}
    if eos_id is not None:
        kwargs["eos_token_id"] = eos_id
    return kwargs


def _generate(model, tokenizer, **kwargs):
    generation_kwargs = _ensure_generation_token_ids(model, tokenizer)
    generation_kwargs.update(kwargs)
    return model.generate(**generation_kwargs)


def load_base_frozen(base_model_id: str, device: str):
    """Load the untuned base model (no LoRA)."""
    patch_transformers_custom_op_registration()
    trust = _trust(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype='auto', trust_remote_code=trust
    ).to(device)
    _ensure_generation_token_ids(model, tokenizer)
    model.eval()
    return model, tokenizer


def load_finetuned(checkpoint_dir: str, device: str):
    """Load a LoRA-fine-tuned model from a directory saved by PeftModel.save_pretrained."""
    patch_transformers_custom_op_registration()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_cfg_path = os.path.join(checkpoint_dir, 'adapter_config.json')
    if os.path.exists(adapter_cfg_path):
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError(
                f"Checkpoint at {checkpoint_dir} is a LoRA adapter but `peft` failed to import. "
                "Install compatible `peft`/`transformers` versions or use a full-model checkpoint."
            ) from e
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
    _ensure_generation_token_ids(model, tokenizer)
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
        out = _generate(model, tokenizer, **enc, do_sample=False, max_new_tokens=32)
    text = tokenizer.decode(
        out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()
    return extract_answer(text), ''


def run_cot(model, tokenizer, item: dict, device: str) -> tuple[str | None, str]:
    enc = tokenizer(
        f"{item['question']}\n<|start-latent|><|latent|><|latent|><|end-latent|>\n",
        return_tensors='pt',
    ).to(device)
    with torch.no_grad():
        out = _generate(model, tokenizer, **enc, do_sample=False, max_new_tokens=512)
    generated = tokenizer.decode(
        out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
    )
    parts = generated.split('####')
    reasoning = parts[0].strip()
    raw_answer = parts[1].strip() if len(parts) > 1 else generated
    return extract_answer(raw_answer), reasoning


def latent_prompt(question: str, n_latents: int) -> str:
    n_latents = max(1, int(n_latents))
    latent_span = "<|latent|>" * n_latents
    return f"{question}\n<|start-latent|>{latent_span}<|end-latent|>\n"


def run_ccot(model, tokenizer, item: dict, n_latents: int,
             device: str) -> tuple[str | None, str]:
    enc = tokenizer(latent_prompt(item['question'], n_latents), return_tensors='pt').to(device)
    with torch.no_grad():
        out = _generate(model, tokenizer, **enc, do_sample=False, max_new_tokens=256)
    generated = tokenizer.decode(
        out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
    )
    reasoning = extract_reasoning_span(generated)
    parts = generated.split('####')
    raw_answer = parts[1].strip() if len(parts) > 1 else generated
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
        f"{item['question']}\n<|start-latent|><|latent|><|latent|><|end-latent|>\n",
        return_tensors='pt',
    ).to(device)
    with torch.no_grad():
        reasoning_ids = _generate(
            cot_model,
            tokenizer,
            **reasoning_enc,
            do_sample=False,
            max_new_tokens=token_budget,
        )
    reasoning_text = tokenizer.decode(
        reasoning_ids[0][reasoning_enc['input_ids'].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # Stage 2 — answer from truncated context
    answer_enc = tokenizer(reasoning_text + "\n#### ", return_tensors='pt').to(device)
    with torch.no_grad():
        answer_ids = _generate(
            cot_model,
            tokenizer,
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
            f"{item['question']}\n<|start-latent|><|latent|><|latent|><|end-latent|>\n",
            return_tensors='pt',
        ).to(device)
        with torch.no_grad():
            out_ids = _generate(cot_model, tokenizer, **enc, do_sample=False, max_new_tokens=512)
        generated = tokenizer.decode(
            out_ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
        )
        reasoning = generated.split('\n\nAnswer:')[0].strip()
        t_full = len(tokenizer.encode(reasoning, add_special_tokens=False))
        budgets.append(max(10, round(ratio * t_full)))
    return budgets


def measure_ccot_token_counts(ccot_model, tokenizer, dataset: list,
                               device: str, n_latents: int) -> dict:
    """Run CCoT model on dataset and return reasoning token count statistics."""
    lengths = []
    ccot_model.eval()
    for item in dataset:
        enc = tokenizer(latent_prompt(item['question'], n_latents), return_tensors='pt').to(device)
        with torch.no_grad():
            out_ids = _generate(ccot_model, tokenizer, **enc, do_sample=False, max_new_tokens=256)
        full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        reasoning = extract_reasoning_span(full_text)
        lengths.append(len(tokenizer.encode(reasoning, add_special_tokens=False)))
    return {
        'mean':        float(np.mean(lengths)),
        'median':      float(np.median(lengths)),
        'std':         float(np.std(lengths)),
        'per_example': lengths,
        'latent_tokens': n_latents,
    }
