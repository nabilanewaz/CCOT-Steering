import itertools
import json
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from tqdm.auto import tqdm

from phase1.data import get_hf_dataset
from phase1.modeling import Coconut
from utils.torch_compat import patch_transformers_custom_op_registration

MODEL_HPARAMS = {
    "llama32_3b": {"lr": 1e-5, "batch": 1, "grad_accum": 64, "epochs": 50},
    "phi2": {"lr": 1e-5, "batch": 1, "grad_accum": 64, "epochs": 50},
    "qwen25_3b": {"lr": 1e-5, "batch": 1, "grad_accum": 64, "epochs": 50},
    "qwen25_math1.5b": {"lr": 1e-5, "batch": 1, "grad_accum": 128, "epochs": 50},
}
_DEFAULT_HP = {"lr": 1e-5, "batch": 1, "grad_accum": 64, "epochs": 50}
MAX_SEQ_LEN = 512
MAX_LATENT_TOKENS = 6
C_THOUGHT = 2
HYBRID_MODE = True
RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9]
CANONICAL_DIRNAME = "_coconut_phase1"


def _parse_steps(answer: str) -> tuple[list[str], str]:
    if "####" in answer:
        reasoning, final = answer.split("####", 1)
        final = final.strip()
    else:
        reasoning, final = answer, ""
    steps = [s.strip() for s in reasoning.split("\n") if s.strip()]
    return steps, final


def _to_coconut_examples(items: list[dict]) -> list[dict]:
    converted = []
    for idx, item in enumerate(items):
        steps, final = _parse_steps(item["answer"])
        converted.append({
            "qid": item.get("id", f"train_{idx:05d}"),
            "question": item["question"],
            "steps": steps,
            "answer": final,
            "ground_truth": final,
        })
    return converted


def _init_model(base_model_id: str, device: str):
    patch_transformers_custom_op_registration()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=("qwen" in base_model_id.lower()),
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=("qwen" in base_model_id.lower()),
    )
    added_pad = False
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            added_pad = True

    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    latent_id, start_id, end_id = tokenizer.convert_tokens_to_ids(
        ["<|latent|>", "<|start-latent|>", "<|end-latent|>"]
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        input_embeds = model.get_input_embeddings()
        init_id = tokenizer.encode("The", add_special_tokens=False)[0]
        init_ids = [latent_id, start_id, end_id]
        if added_pad:
            init_ids.append(tokenizer.pad_token_id)
        for token_id in init_ids:
            input_embeds.weight.data[token_id] = input_embeds.weight.data[init_id].clone()
            if hasattr(model, "lm_head") and model.lm_head is not None:
                model.lm_head.weight.data[token_id] = model.lm_head.weight.data[init_id].clone()
        input_embeds.weight.requires_grad = True

    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id).to(device)
    return coconut_model, tokenizer, latent_id, start_id, end_id


class MyCollator:
    def __init__(self, tokenizer, latent_id: int, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.latent_id = latent_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        earliest_latent = [f["input_ids"].index(self.latent_id) for f in features if self.latent_id in f["input_ids"]]
        if earliest_latent:
            latest_earliest = max(earliest_latent)
            for feature in features:
                pad = latest_earliest - feature["input_ids"].index(self.latent_id) if self.latent_id in feature["input_ids"] else 0
                feature["input_ids"] = [self.tokenizer.pad_token_id] * pad + feature["input_ids"]
                feature["attention_mask"] = [0] * pad + feature["attention_mask"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * pad + feature["labels"]

        labels = [f.pop("labels") for f in features] if "labels" in features[0] else None
        batch = pad_without_fast_tokenizer_warning(self.tokenizer, features, padding=True, return_tensors="pt")
        if labels:
            max_len = batch["input_ids"].shape[1]
            batch["labels"] = torch.tensor([l + [self.label_pad_token_id] * (max_len - len(l)) for l in labels])
        return batch


def _get_stage_info(epoch: int) -> tuple[int, bool, bool]:
    if epoch < 6:
        return 0, False, epoch == 0
    if epoch < 9:
        return 1, False, epoch == 6
    if epoch < 12:
        return 2, False, epoch == 9
    if epoch < 15:
        return 3, False, epoch == 12
    return 4, True, epoch == 15


def _build_stage_dataset(base_dataset, stage: int, drop_remaining: bool, start_id: int, latent_id: int, end_id: int):
    def _process(sample):
        if len(sample["steps_tokenized"]) > 0 and HYBRID_MODE:
            skeleton = sample["steps_tokenized"][0]
            remaining = sample["steps_tokenized"][1:]
        else:
            skeleton = []
            remaining = sample["steps_tokenized"]

        steps_to_drop = min(stage, len(remaining))
        if drop_remaining:
            kept_remaining_steps = []
            n_latent_tokens = MAX_LATENT_TOKENS
        else:
            kept_remaining_steps = remaining[steps_to_drop:]
            n_latent_tokens = steps_to_drop * C_THOUGHT

        kept_remaining_text = list(itertools.chain.from_iterable(kept_remaining_steps))
        tokens = (
            sample["question_tokenized"]
            + skeleton
            + [start_id]
            + [latent_id] * n_latent_tokens
            + [end_id]
            + kept_remaining_text
            + sample["answer_tokenized"]
        )
        mask_len = len(sample["question_tokenized"]) + len(skeleton) + n_latent_tokens + 2
        labels = [-100] * mask_len + tokens[mask_len:]
        tokens = tokens[:MAX_SEQ_LEN]
        labels = labels[:MAX_SEQ_LEN]
        return {"input_ids": tokens, "labels": labels, "attention_mask": [1] * len(tokens)}

    return base_dataset.map(_process, remove_columns=list(base_dataset.features)).shuffle(seed=42)


def _run_coconut_training(base_model_id: str, D_train: list, output_dir: str, model_tag: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hp = MODEL_HPARAMS.get(model_tag, _DEFAULT_HP)
    coconut_model, tokenizer, latent_id, start_id, end_id = _init_model(base_model_id, device)

    raw_ds = _to_coconut_examples(D_train)
    n_val = max(1, int(round(len(raw_ds) * 0.1))) if len(raw_ds) > 1 else 0
    train_raw = raw_ds[:-n_val] if n_val else raw_ds
    val_raw = raw_ds[-n_val:] if n_val else []
    ds = get_hf_dataset(train_raw, tokenizer)
    collator = MyCollator(tokenizer, latent_id=latent_id)

    optimizer = None
    scheduler = None
    loss_history = []
    losses_per_stage = {i: [] for i in range(5)}
    stage_transition_epochs = []
    val_acc_history = []
    drift_by_token = {"<|start-latent|>": [], "<|latent|>": [], "<|end-latent|>": []}
    input_embeds_ref = coconut_model.base_causallm.get_input_embeddings().weight.detach().clone()

    def _phase1_val_accuracy() -> float:
        if not val_raw:
            return 0.0
        coconut_model.eval()
        correct = 0
        tqdm.write(f"[phase1][{model_tag}] running validation on {len(val_raw)} examples...")
        val_iter = tqdm(
            val_raw,
            desc=f"Val {model_tag}",
            leave=False,
            dynamic_ncols=True,
        )
        for sample in val_iter:
            prompt = (
                sample["question"] + "\n<|start-latent|>"
                + "<|latent|>" * MAX_LATENT_TOKENS + "<|end-latent|>"
            )
            inp = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids, _, _ = coconut_model.generate_with_latents(
                    inp,
                    max_new_tokens=64,
                    temperature=0.0,
                )
            decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            pred = sample["answer"].replace(",", "").strip()
            if pred and pred in decoded.replace(",", ""):
                correct += 1
            val_iter.set_postfix({"acc": f"{correct / max(1, val_iter.n):.4f}"})
        return correct / max(len(val_raw), 1)

    for epoch in range(hp["epochs"]):
        stage, drop_remaining, reset_opt = _get_stage_info(epoch)
        train_ds = _build_stage_dataset(ds, stage, drop_remaining, start_id, latent_id, end_id)
        train_loader = DataLoader(train_ds, batch_size=hp["batch"], collate_fn=collator, shuffle=True)
        if optimizer is None or reset_opt:
            stage_transition_epochs.append(epoch)
            optimizer = torch.optim.AdamW(coconut_model.parameters(), lr=hp["lr"], weight_decay=0.01, eps=1e-8)
            epochs_in_stage = 3 if stage < 4 else max(1, hp["epochs"] - 12)
            total_steps = max(1, (len(train_loader) * epochs_in_stage) // hp["grad_accum"])
            warmup_steps = max(10, int(total_steps * 0.1))
            def _lr_lambda(step: int):
                if step < warmup_steps:
                    return float(step) / max(1, warmup_steps)
                return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

        coconut_model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{hp['epochs']} | Stage {stage}")
        total_loss = 0.0
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = coconut_model(input_ids, attention_mask, labels)
                loss = outputs.loss.to(torch.float32) / hp["grad_accum"]
            loss.backward()
            total_loss += float(loss.item() * hp["grad_accum"])
            if (step + 1) % hp["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(coconut_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            pbar.set_postfix({"loss": total_loss / max(step + 1, 1)})

        epoch_avg_loss = total_loss / max(len(train_loader), 1)
        loss_history.append(epoch_avg_loss)
        losses_per_stage[stage].append(epoch_avg_loss)
        with torch.no_grad():
            w = coconut_model.base_causallm.get_input_embeddings().weight.detach()
            tok_map = {
                "<|start-latent|>": start_id,
                "<|latent|>": latent_id,
                "<|end-latent|>": end_id,
            }
            for name, tid in tok_map.items():
                drift = torch.nn.functional.cosine_similarity(
                    w[tid].float().unsqueeze(0),
                    input_embeds_ref[tid].float().unsqueeze(0),
                    dim=-1,
                ).item()
                drift_by_token[name].append(drift)
        val_acc_history.append(_phase1_val_accuracy())
        tqdm.write(
            f"[phase1][{model_tag}] epoch={epoch + 1}/{hp['epochs']} "
            f"stage={stage} train_loss={epoch_avg_loss:.4f} "
            f"val_acc={val_acc_history[-1]:.4f} "
            f"(train_n={len(train_raw)} val_n={len(val_raw)})"
        )

    os.makedirs(output_dir, exist_ok=True)
    coconut_model.base_causallm.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "coconut_meta.json"), "w", encoding="utf-8") as f:
        f.write(
            "{\n"
            f'  "architecture": "coconut",\n'
            f'  "base_model_id": "{base_model_id}",\n'
            f'  "latent_start_token": "<|start-latent|>",\n'
            f'  "latent_token": "<|latent|>",\n'
            f'  "latent_end_token": "<|end-latent|>"\n'
            "}\n"
        )
    print(f"Coconut model saved -> {output_dir}")
    return {
        "loss_history": loss_history,
        "losses_per_stage": losses_per_stage,
        "stage_transition_epochs": stage_transition_epochs,
        "val_accuracy": val_acc_history,
        "embedding_drift": drift_by_token,
        "n_train": len(train_raw),
        "n_val": len(val_raw),
        "epochs": hp["epochs"],
    }


def _plot_curves(metrics: dict, phase1_plot_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    os.makedirs(phase1_plot_dir, exist_ok=True)

    # Stage losses
    plt.figure(figsize=(8, 4))
    for stage, vals in metrics["losses_per_stage"].items():
        if vals:
            plt.plot(range(len(vals)), vals, label=f"stage{stage}")
    plt.xlabel("Epoch-in-stage")
    plt.ylabel("Loss")
    plt.title("Phase1 Stage Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(phase1_plot_dir, "stage_loss_curve.png"))
    plt.close()

    # Validation accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(metrics["val_accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Phase1 Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(phase1_plot_dir, "val_accuracy.png"))
    plt.close()

    # Embedding drift
    plt.figure(figsize=(8, 4))
    for token_name, values in metrics["embedding_drift"].items():
        plt.plot(values, label=token_name)
    plt.xlabel("Epoch")
    plt.ylabel("Cosine similarity vs init")
    plt.title("Latent Token Embedding Drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(phase1_plot_dir, "embedding_drift.png"))
    plt.close()


def _materialize_alias_dir(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(os.path.join(dst_dir, "config.json")):
        return
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst) or os.path.islink(dst):
            continue
        rel_src = os.path.relpath(src, os.path.dirname(dst))
        try:
            os.symlink(rel_src, dst)
        except OSError:
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)


def export_compat_checkpoints(checkpoints_dir: str, ratios: list[float] = RATIOS) -> None:
    canonical_dir = os.path.join(checkpoints_dir, CANONICAL_DIRNAME)
    if not os.path.exists(os.path.join(canonical_dir, "config.json")):
        raise FileNotFoundError(f"Canonical Coconut checkpoint not found: {canonical_dir}")

    cot_dir = os.path.join(checkpoints_dir, "cot")
    _materialize_alias_dir(canonical_dir, cot_dir)
    for ratio in ratios:
        rtag = f"R{int(ratio * 10)}"
        ccot_dir = os.path.join(checkpoints_dir, f"ccot_{rtag}")
        _materialize_alias_dir(canonical_dir, ccot_dir)

    meta_path = os.path.join(checkpoints_dir, "compat_export_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": CANONICAL_DIRNAME,
                "ratios": ratios,
                "exported_dirs": ["cot"] + [f"ccot_R{int(r * 10)}" for r in ratios],
            },
            f,
            indent=2,
        )


def train_coconut_phase1(
    base_model_id: str,
    D_train: list,
    checkpoints_dir: str,
    results_dir: str,
    model_tag: str,
    ratios: list[float] = RATIOS,
):
    canonical_dir = os.path.join(checkpoints_dir, CANONICAL_DIRNAME)
    marker_path = os.path.join(canonical_dir, "config.json")
    if not os.path.exists(marker_path):
        metrics = _run_coconut_training(base_model_id, D_train, canonical_dir, model_tag)
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "phase1_training_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        cfg_id = os.path.basename(os.path.dirname(checkpoints_dir))
        model_id = os.path.basename(checkpoints_dir)
        phase1_plot_dir = os.path.join("plots", cfg_id, model_id, "phase1")
        _plot_curves(metrics, phase1_plot_dir)
    else:
        print(f"Canonical Coconut checkpoint exists -> {canonical_dir} (skipping retrain)")

    export_compat_checkpoints(checkpoints_dir, ratios=ratios)


def train_cot(base_model_id: str, D_train: list, output_dir: str, model_tag: str, lora_r: int = 16, lora_alpha: int = 32):
    del lora_r, lora_alpha
    checkpoints_dir = os.path.dirname(output_dir)
    results_dir = os.path.join("results", checkpoints_dir.split("/")[-2], checkpoints_dir.split("/")[-1])
    train_coconut_phase1(base_model_id, D_train, checkpoints_dir, results_dir, model_tag)


def train_ccot(
    base_model_id: str,
    D_train: list,
    compressed_cache: list,
    ratio: float,
    output_dir: str,
    model_tag: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    del compressed_cache, ratio, lora_r, lora_alpha
    checkpoints_dir = os.path.dirname(output_dir)
    results_dir = os.path.join("results", checkpoints_dir.split("/")[-2], checkpoints_dir.split("/")[-1])
    train_coconut_phase1(base_model_id, D_train, checkpoints_dir, results_dir, model_tag)
