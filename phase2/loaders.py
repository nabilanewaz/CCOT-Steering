import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.torch_compat import patch_transformers_custom_op_registration


def _needs_trust(model_id: str) -> bool:
    return 'qwen' in model_id.lower()


def _sync_pad_token_id(model, tokenizer) -> None:
    """Align model config with tokenizer so generate() does not log pad_token_id warnings."""
    pid = getattr(tokenizer, 'pad_token_id', None)
    if pid is None:
        return
    if getattr(model.config, 'pad_token_id', None) != pid:
        model.config.pad_token_id = pid
    gen = getattr(model, 'generation_config', None)
    if gen is not None and getattr(gen, 'pad_token_id', None) != pid:
        gen.pad_token_id = pid


def get_transformer_layers(model):
    """Return the transformer layer list regardless of PEFT wrapping."""
    try:
        from peft import PeftModel as _PeftModel
    except ImportError:
        _PeftModel = None
    if _PeftModel is not None:
        try:
            if isinstance(model, _PeftModel):
                # PeftModel → LoraModel (.base_model) → base CausalLM (.model) → inner model (.model)
                return model.base_model.model.model.layers
        except AttributeError:
            pass
    return model.model.layers


def load_ccot_frozen(base_model_id: str, lora_adapter_path: str, device: str):
    """Load Source A checkpoint (LoRA adapter or full Coconut checkpoint), frozen."""
    patch_transformers_custom_op_registration()
    trust = _needs_trust(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_cfg = f"{lora_adapter_path}/adapter_config.json"
    if os.path.exists(adapter_cfg):
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError(
                f"Checkpoint at {lora_adapter_path} is a LoRA adapter but `peft` failed to import. "
                "Install compatible `peft`/`transformers` versions or use a full-model checkpoint."
            ) from e
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float32,
            device_map=device, trust_remote_code=trust,
        )
        model = PeftModel.from_pretrained(base, lora_adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            lora_adapter_path,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=trust,
        )

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    _sync_pad_token_id(model, tokenizer)
    return model, tokenizer


def load_base_frozen(base_model_id: str, device: str):
    """Load the raw pre-trained base model with no LoRA, all parameters frozen."""
    patch_transformers_custom_op_registration()
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
    _sync_pad_token_id(model, tokenizer)
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
    """Return the index of the last reasoning token before answer delimiter."""
    sep = tokenizer.encode("\n\nAnswer:", add_special_tokens=False)
    ids = input_ids[0].tolist()
    for i in range(len(ids) - len(sep), -1, -1):
        if ids[i:i + len(sep)] == sep:
            return max(0, i - 1)
    sep_hash = tokenizer.encode("####", add_special_tokens=False)
    for i in range(len(ids) - len(sep_hash), -1, -1):
        if ids[i:i + len(sep_hash)] == sep_hash:
            return max(0, i - 1)
    end_latent = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    if end_latent is not None and end_latent != tokenizer.unk_token_id:
        positions = (input_ids[0] == end_latent).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            return positions[-1].item()
    return max(0, input_ids.shape[1] - 2)
