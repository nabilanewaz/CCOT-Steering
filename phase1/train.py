import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from phase1.format import format_cot, format_ccot

# Per-backbone hyperparameters (spec §1.10)
MODEL_HPARAMS = {
    'llama32_3b':      {'lr': 2e-4, 'batch': 4, 'grad_accum': 4, 'epochs': 3},
    'phi2':            {'lr': 1e-4, 'batch': 4, 'grad_accum': 4, 'epochs': 3},
    'qwen25_3b':       {'lr': 2e-4, 'batch': 4, 'grad_accum': 4, 'epochs': 3},
    # Smaller model; Math pretraining means CoT converges faster — 1 epoch for CoT
    'qwen25_math1.5b': {'lr': 2e-4, 'batch': 8, 'grad_accum': 2, 'epochs': 1},
}

_DEFAULT_HP = {'lr': 2e-4, 'batch': 4, 'grad_accum': 4, 'epochs': 3}


def get_target_modules(model_id: str) -> list:
    targets = {
        'llama': ["q_proj", "v_proj", "k_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"],
        'phi':   ["q_proj", "v_proj", "k_proj", "dense", "fc1", "fc2"],
        'qwen':  ["q_proj", "v_proj", "k_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"],
    }
    m = model_id.lower()
    for key, modules in targets.items():
        if key in m:
            return modules
    return ["q_proj", "v_proj"]


def _needs_trust_remote_code(model_id: str) -> bool:
    return 'qwen' in model_id.lower()


def _load_base(model_id: str):
    trust = _needs_trust_remote_code(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype='auto', trust_remote_code=trust
    )
    return model, tokenizer


def _make_lora_config(model_id: str, lora_r: int, lora_alpha: int) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=get_target_modules(model_id),
        bias='none',
    )


class CoTDataset(Dataset):
    def __init__(self, items: list, tokenizer, max_length: int = 512):
        self.data = [format_cot(item) for item in items]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.data[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = enc['input_ids'].squeeze()
        return {
            'input_ids':      input_ids,
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels':         input_ids.clone(),
        }


class CCoTDataset(Dataset):
    def __init__(self, items: list, compressed_cache: list, ratio: float,
                 tokenizer, max_length: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        cache_map = {entry['id']: entry for entry in compressed_cache}
        for item in items:
            entry = cache_map.get(item.get('id', ''))
            if entry is None:
                continue
            self.samples.append(format_ccot(item, entry['compressed'], ratio))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = enc['input_ids'].squeeze()
        return {
            'input_ids':      input_ids,
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels':         input_ids.clone(),
        }


def _training_args(output_dir: str, hp: dict) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hp['epochs'],
        per_device_train_batch_size=hp['batch'],
        gradient_accumulation_steps=hp['grad_accum'],
        learning_rate=hp['lr'],
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy='epoch',
        report_to='none',
    )


def train_cot(base_model_id: str, D_train: list, output_dir: str,
              model_tag: str, lora_r: int = 16, lora_alpha: int = 32):
    hp = MODEL_HPARAMS.get(model_tag, _DEFAULT_HP)
    model, tokenizer = _load_base(base_model_id)
    model = get_peft_model(model, _make_lora_config(base_model_id, lora_r, lora_alpha))
    model.print_trainable_parameters()

    Trainer(
        model=model,
        args=_training_args(output_dir, hp),
        train_dataset=CoTDataset(D_train, tokenizer),
    ).train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"CoT model saved → {output_dir}")


def train_ccot(base_model_id: str, D_train: list, compressed_cache: list,
               ratio: float, output_dir: str, model_tag: str,
               lora_r: int = 16, lora_alpha: int = 32):
    hp = MODEL_HPARAMS.get(model_tag, _DEFAULT_HP)
    model, tokenizer = _load_base(base_model_id)
    model = get_peft_model(model, _make_lora_config(base_model_id, lora_r, lora_alpha))

    dataset = CCoTDataset(D_train, compressed_cache, ratio, tokenizer)
    if len(dataset) == 0:
        raise ValueError(
            f"CCoTDataset is empty for R={ratio}. "
            "Check that compressed_cache ids match D_train item ids."
        )

    Trainer(
        model=model,
        args=_training_args(output_dir, {**hp, 'epochs': 3}),
        train_dataset=dataset,
    ).train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"CCoT model (R={ratio}) saved → {output_dir}")
