import json
import os
import time

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from peft import LoraConfig, get_peft_model, TaskType

from phase1.format import format_cot, format_ccot

# Per-backbone hyperparameters (spec §1.10)
# CCoT always uses 3 epochs regardless of model — it is a harder generation task
# than CoT, even for math-pretrained models.
MODEL_HPARAMS = {
    'llama32_3b':      {'lr': 2e-4, 'batch': 4, 'grad_accum': 4, 'epochs': 3},
    'phi2':            {'lr': 1e-4, 'batch': 4, 'grad_accum': 4, 'epochs': 3},
    'qwen25_3b':       {'lr': 2e-4, 'batch': 4, 'grad_accum': 4, 'epochs': 3},
    # Math pretraining means CoT converges in 1 epoch; CCoT still uses 3.
    'qwen25_math1.5b': {'lr': 2e-4, 'batch': 8, 'grad_accum': 2, 'epochs': 1},
}
# CCoT epoch override — always 3 regardless of per-model CoT epochs.
CCOT_EPOCHS = 3

_DEFAULT_HP = {'lr': 2e-4, 'batch': 4, 'grad_accum': 4, 'epochs': 3}


class _EpochLogger(TrainerCallback):
    """Logs per-step loss live and collects per-epoch summaries."""

    def __init__(self, n_epochs: int, n_steps_per_epoch: int):
        self.n_epochs = n_epochs
        self.n_steps_per_epoch = n_steps_per_epoch
        self.epoch_log: list[dict] = []
        self._last_loss: float = float('nan')

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **_):
        if logs is None:
            return
        loss = logs.get('loss')
        lr   = logs.get('learning_rate')
        if loss is not None:
            self._last_loss = loss
            step  = state.global_step
            epoch = state.epoch or 0
            print(f"    step {step:>5}  epoch {epoch:5.2f}/{self.n_epochs}  "
                  f"loss={loss:.4f}" + (f"  lr={lr:.2e}" if lr else ""))

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **_):
        epoch_num = round(state.epoch or 0)
        # Pull the last logged train loss for this epoch
        train_loss = float('nan')
        for entry in reversed(state.log_history):
            if 'loss' in entry:
                train_loss = entry['loss']
                break
        record = {
            'epoch':      epoch_num,
            'train_loss': round(train_loss, 6) if train_loss == train_loss else None,
            'step':       state.global_step,
        }
        self.epoch_log.append(record)
        print(f"\n  ── Epoch {epoch_num}/{self.n_epochs} complete  "
              f"train_loss={train_loss:.4f}  global_step={state.global_step} ──\n")


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
        logging_strategy='steps',
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=None,      # keep every epoch checkpoint
        report_to='none',
    )


def _print_train_config(label: str, model_tag: str, base_model_id: str,
                        hp: dict, n_examples: int, output_dir: str,
                        lora_r: int, lora_alpha: int) -> None:
    bar = '═' * 64
    print(f"\n{bar}")
    print(f"  {label}  [{model_tag}]")
    print(f"  Base model    : {base_model_id}")
    print(f"  Epochs        : {hp['epochs']}")
    print(f"  LR            : {hp['lr']:.1e}  (cosine, warmup=5%)")
    print(f"  Batch / accum : {hp['batch']} × {hp['grad_accum']}  "
          f"(eff. batch = {hp['batch'] * hp['grad_accum']})")
    print(f"  LoRA r/alpha  : {lora_r} / {lora_alpha}  (dropout=0.05)")
    print(f"  Train examples: {n_examples}")
    steps_per_epoch = max(1, n_examples // (hp['batch'] * hp['grad_accum']))
    total_steps     = steps_per_epoch * hp['epochs']
    print(f"  Steps/epoch   : ~{steps_per_epoch}  total ~{total_steps}")
    print(f"  fp16          : {torch.cuda.is_available()}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Checkpoints   : one per epoch  (save_total_limit=None → all kept)")
    print(bar)


def _save_training_info(
    output_dir: str,
    label: str,
    model_tag: str,
    base_model_id: str,
    hp: dict,
    n_examples: int,
    lora_r: int,
    lora_alpha: int,
    epoch_log: list,
    elapsed_s: float,
) -> None:
    info = {
        'label':         label,
        'model_tag':     model_tag,
        'base_model_id': base_model_id,
        'hyperparams': {
            'epochs':           hp['epochs'],
            'lr':               hp['lr'],
            'batch_size':       hp['batch'],
            'grad_accum_steps': hp['grad_accum'],
            'eff_batch_size':   hp['batch'] * hp['grad_accum'],
            'lora_r':           lora_r,
            'lora_alpha':       lora_alpha,
            'lora_dropout':     0.05,
            'lr_scheduler':     'cosine',
            'warmup_ratio':     0.05,
        },
        'dataset': {
            'n_train': n_examples,
        },
        'epoch_log':    epoch_log,
        'elapsed_s':    round(elapsed_s, 2),
        'output_dir':   output_dir,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'training_info.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  Training info -> {path}")


def train_cot(base_model_id: str, D_train: list, output_dir: str,
              model_tag: str, lora_r: int = 16, lora_alpha: int = 32):
    hp = MODEL_HPARAMS.get(model_tag, _DEFAULT_HP)
    _print_train_config('CoT fine-tuning', model_tag, base_model_id,
                        hp, len(D_train), output_dir, lora_r, lora_alpha)

    model, tokenizer = _load_base(base_model_id)
    model = get_peft_model(model, _make_lora_config(base_model_id, lora_r, lora_alpha))
    model.print_trainable_parameters()

    dataset = CoTDataset(D_train, tokenizer)
    steps_per_epoch = max(1, len(dataset) // (hp['batch'] * hp['grad_accum']))
    callback = _EpochLogger(n_epochs=hp['epochs'], n_steps_per_epoch=steps_per_epoch)

    t0 = time.time()
    Trainer(
        model=model,
        args=_training_args(output_dir, hp),
        train_dataset=dataset,
        callbacks=[callback],
    ).train()
    elapsed = time.time() - t0

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _save_training_info(output_dir, 'cot', model_tag, base_model_id,
                        hp, len(dataset), lora_r, lora_alpha,
                        callback.epoch_log, elapsed)
    print(f"\n  CoT model saved → {output_dir}  ({elapsed:.0f}s)")
    _print_checkpoint_tree(output_dir, hp['epochs'])


def train_ccot(base_model_id: str, D_train: list, compressed_cache: list,
               ratio: float, output_dir: str, model_tag: str,
               lora_r: int = 16, lora_alpha: int = 32):
    hp_base = MODEL_HPARAMS.get(model_tag, _DEFAULT_HP)
    hp = {**hp_base, 'epochs': CCOT_EPOCHS}
    _print_train_config(f'CCoT fine-tuning  R={ratio}', model_tag, base_model_id,
                        hp, len(D_train), output_dir, lora_r, lora_alpha)

    model, tokenizer = _load_base(base_model_id)
    model = get_peft_model(model, _make_lora_config(base_model_id, lora_r, lora_alpha))

    dataset = CCoTDataset(D_train, compressed_cache, ratio, tokenizer)
    if len(dataset) == 0:
        raise ValueError(
            f"CCoTDataset is empty for R={ratio}. "
            "Check that compressed_cache ids match D_train item ids."
        )

    steps_per_epoch = max(1, len(dataset) // (hp['batch'] * hp['grad_accum']))
    callback = _EpochLogger(n_epochs=hp['epochs'], n_steps_per_epoch=steps_per_epoch)

    t0 = time.time()
    Trainer(
        model=model,
        args=_training_args(output_dir, hp),
        train_dataset=dataset,
        callbacks=[callback],
    ).train()
    elapsed = time.time() - t0

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _save_training_info(output_dir, f'ccot_R{int(ratio * 10)}', model_tag,
                        base_model_id, hp, len(dataset), lora_r, lora_alpha,
                        callback.epoch_log, elapsed)
    print(f"\n  CCoT model (R={ratio}) saved → {output_dir}  ({elapsed:.0f}s)")
    _print_checkpoint_tree(output_dir, hp['epochs'])


def _print_checkpoint_tree(output_dir: str, n_epochs: int) -> None:
    """Print which epoch checkpoints were saved inside output_dir."""
    try:
        entries = sorted(os.listdir(output_dir))
    except OSError:
        return
    ckpts = [e for e in entries if e.startswith('checkpoint-')]
    if not ckpts:
        return
    print(f"\n  Saved checkpoints in {output_dir}:")
    for ckpt in ckpts:
        p = os.path.join(output_dir, ckpt)
        size_mb = sum(
            os.path.getsize(os.path.join(p, f))
            for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))
        ) / 1e6
        print(f"    {ckpt}/  ({size_mb:.1f} MB)")
    print(f"    adapter_config.json + adapter_model.*  (final, epoch {n_epochs})")
