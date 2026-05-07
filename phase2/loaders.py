import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def _needs_trust(model_id: str) -> bool:
    return 'qwen' in model_id.lower()


def get_transformer_layers(model):
    """Return the transformer layer list regardless of PEFT wrapping."""
    try:
        if isinstance(model, PeftModel):
            # PeftModel → LoraModel (.base_model) → base CausalLM (.model) → inner model (.model)
            return model.base_model.model.model.layers
    except AttributeError:
        pass
    return model.model.layers


def load_ccot_frozen(base_model_id: str, lora_adapter_path: str, device: str):
    """Load Source A: CCoT LoRA checkpoint, all parameters frozen."""
    trust = _needs_trust(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float32,
        device_map=device, trust_remote_code=trust,
    )
    model = PeftModel.from_pretrained(base, lora_adapter_path)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, tokenizer


def load_base_frozen(base_model_id: str, device: str):
    """Load the raw pre-trained base model with no LoRA, all parameters frozen."""
    trust = _needs_trust(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float32,
        device_map=device, trust_remote_code=trust,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, tokenizer


# ── Boundary token finders ────────────────────────────────────────────────────

def find_boundary_idx_ccot(input_ids: torch.Tensor, tokenizer) -> int:
    """
    Return the sequence index of the last compressed-reasoning token for Source A.
    Tries the '</think>' delimiter first; falls back to the token just before
    '\n\nAnswer:' since our CCoT format uses that separator rather than </think>.
    """
    end_id = tokenizer.convert_tokens_to_ids("</think>")
    if end_id is not None and end_id != tokenizer.unk_token_id:
        positions = (input_ids[0] == end_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            return positions[-1].item()
    return find_boundary_idx_base(input_ids, tokenizer)


def find_boundary_idx_base(input_ids: torch.Tensor, tokenizer) -> int:
    """Return the index of the last reasoning token before '\n\nAnswer:'."""
    sep = tokenizer.encode("\n\nAnswer:", add_special_tokens=False)
    ids = input_ids[0].tolist()
    for i in range(len(ids) - len(sep), -1, -1):
        if ids[i:i + len(sep)] == sep:
            return max(0, i - 1)
    raise ValueError("Answer separator '\\n\\nAnswer:' not found in output.")
