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
from phase1.inference import extract_answer, normalize_answer
from phase1.modeling import Coconut
from utils.experiment_config import require_exact_count
from utils.torch_compat import patch_transformers_custom_op_registration

MODEL_HPARAMS = {
    "llama32_3b": {"lr": 1e-4, "batch": 1, "grad_accum": 128, "epochs": 30},
    "phi2": {"lr": 1e-4, "batch": 1, "grad_accum": 128, "epochs": 30},
    "qwen25_0.5b": {"lr": 1e-4, "batch": 1, "grad_accum": 128, "epochs": 30},
    "qwen25_3b": {"lr": 1e-4, "batch": 1, "grad_accum": 128, "epochs": 30},
    "qwen25_math1.5b": {"lr": 1e-4, "batch": 1, "grad_accum": 128, "epochs": 30},
}
_DEFAULT_HP = {"lr": 1e-4, "batch": 1, "grad_accum": 128, "epochs": 30}
MAX_SEQ_LEN = 512
MAX_LATENT_TOKENS = 6
C_THOUGHT = 2
MAX_LATENT_STAGE = 3
LATENT_TOKEN_COUNTS = [3, 4, 6]
CANONICAL_DIRNAME = "_coconut_phase1"
BEST_DIRNAME = "_coconut_phase1_best"
LATENT_ONLY_BEST_DIRNAME = "_coconut_phase1_best_latent_only"
COT_BEST_DIRNAME = "_coconut_phase1_cot_best"
CURRICULUM_VERSION = "paper_gsm8k_c2_stage0_6_stage123_3_final_to_30_v1"
BEST_CHECKPOINT_MIN_STAGE = 4
LATENT_ONLY_CHECKPOINT_MIN_STAGE = 4
TRAIN_USE_KV_CACHE = False
TRAIN_DETACH_LATENTS = False


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
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        # cuDNN SDPA backward has produced non-finite gradients in the latent
        # curriculum on some torch/CUDA stacks. Prefer the other SDPA kernels.
        torch.backends.cuda.enable_cudnn_sdp(False)
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
        # Match the reference implementation's initialization for new tokens.
        init_id = tokenizer.encode("<<", add_special_tokens=False)[0]
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
    if stage < 0 or stage > MAX_LATENT_STAGE + 1:
        raise ValueError(f"Invalid Coconut curriculum stage: {stage}")

    def _process(sample):
        question = sample["question_tokenized"]
        if stage == 0:
            reasoning = list(itertools.chain.from_iterable(sample["steps_tokenized"]))
            tokens = question + reasoning + sample["answer_tokenized"]
            mask_len = len(question)
        else:
            fully_latent = drop_remaining or stage > MAX_LATENT_STAGE
            n_latent_tokens = MAX_LATENT_TOKENS if fully_latent else stage * C_THOUGHT
            kept_steps = [] if fully_latent else sample["steps_tokenized"][stage:]
            reasoning = list(itertools.chain.from_iterable(kept_steps))
            prefix = question + [start_id] + [latent_id] * n_latent_tokens + [end_id]
            tokens = prefix + reasoning + sample["answer_tokenized"]
            mask_len = len(prefix)

        labels = [-100] * mask_len + tokens[mask_len:]
        tokens = tokens[:MAX_SEQ_LEN]
        labels = labels[:MAX_SEQ_LEN]
        return {"input_ids": tokens, "labels": labels, "attention_mask": [1] * len(tokens)}

    return base_dataset.map(_process, remove_columns=list(base_dataset.features)).shuffle(seed=42)


def _save_coconut_checkpoint(
    coconut_model,
    tokenizer,
    output_dir: str,
    base_model_id: str,
    role: str,
    epoch: int | None = None,
    val_accuracy: float | None = None,
    uses_coconut_wrapper: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    coconut_model.base_causallm.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    meta = {
        "architecture": "coconut" if uses_coconut_wrapper else "causal_lm_cot",
        "base_model_id": base_model_id,
        "latent_start_token": "<|start-latent|>",
        "latent_token": "<|latent|>",
        "latent_end_token": "<|end-latent|>",
        "checkpoint_role": role,
        "uses_coconut_wrapper": uses_coconut_wrapper,
        "curriculum_version": CURRICULUM_VERSION,
        "c_thought": C_THOUGHT,
        "max_latent_stage": MAX_LATENT_STAGE,
        "final_latent_tokens": MAX_LATENT_TOKENS,
    }
    if epoch is not None:
        meta["epoch"] = epoch
    if val_accuracy is not None:
        meta["val_accuracy"] = val_accuracy
    with open(os.path.join(output_dir, "coconut_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def _run_coconut_training(base_model_id: str, D_train: list, D_val: list, output_dir: str, model_tag: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hp = dict(MODEL_HPARAMS.get(model_tag, _DEFAULT_HP))
    coconut_model, tokenizer, latent_id, start_id, end_id = _init_model(base_model_id, device)

    train_raw = _to_coconut_examples(D_train)
    val_raw = _to_coconut_examples(D_val)
    ds = get_hf_dataset(train_raw, tokenizer)
    collator = MyCollator(tokenizer, latent_id=latent_id)

    optimizer = None
    loss_history = []
    losses_per_stage = {i: [] for i in range(5)}
    stage_transition_epochs = []
    val_acc_history = []
    drift_by_token = {"<|start-latent|>": [], "<|latent|>": [], "<|end-latent|>": []}
    input_embeds_ref = coconut_model.base_causallm.get_input_embeddings().weight.detach().clone()
    checkpoint_root = os.path.dirname(output_dir)
    best_dir = os.path.join(checkpoint_root, BEST_DIRNAME)
    latent_only_best_dir = os.path.join(checkpoint_root, LATENT_ONLY_BEST_DIRNAME)
    cot_best_dir = os.path.join(checkpoint_root, COT_BEST_DIRNAME)
    best_val_acc = -float("inf")
    best_epoch = None
    best_stage = None
    latent_only_best_val_acc = -float("inf")
    latent_only_best_epoch = None
    latent_only_best_stage = None
    cot_best_val_acc = -float("inf")
    cot_best_epoch = None
    skipped_nonfinite_losses = 0
    skipped_nonfinite_steps = 0

    def _phase1_val_accuracy(stage: int, drop_remaining: bool) -> float:
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
            if stage == 0:
                prompt = sample["question"] + "\n"
            else:
                fully_latent = drop_remaining or stage > MAX_LATENT_STAGE
                n_latents = MAX_LATENT_TOKENS if fully_latent else stage * C_THOUGHT
                prompt = (
                    sample["question"] + "\n<|start-latent|>"
                    + "<|latent|>" * n_latents + "<|end-latent|>"
                )
            inp = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids, _, _ = coconut_model.generate_with_latents(
                    inp,
                    max_new_tokens=64,
                    temperature=0.0,
                )
            generated = tokenizer.decode(
                out_ids[0][inp.shape[1]:], skip_special_tokens=True
            )
            pred = normalize_answer(extract_answer(generated))
            gold = normalize_answer(sample["answer"])
            if pred and pred == gold:
                correct += 1
            val_iter.set_postfix({"acc": f"{correct / max(1, val_iter.n):.4f}"})
        return correct / max(len(val_raw), 1)

    for epoch in range(hp["epochs"]):
        stage, drop_remaining, reset_opt = _get_stage_info(epoch)
        train_ds = _build_stage_dataset(ds, stage, drop_remaining, start_id, latent_id, end_id)
        train_loader = DataLoader(train_ds, batch_size=hp["batch"], collate_fn=collator, shuffle=True)
        if optimizer is None or reset_opt:
            stage_transition_epochs.append(epoch)
            optimizer = torch.optim.AdamW(
                coconut_model.parameters(), lr=hp["lr"], weight_decay=0.01, eps=1e-8
            )

        coconut_model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{hp['epochs']} | Stage {stage}")
        total_loss = 0.0
        finite_loss_steps = 0
        epoch_skipped_losses = 0
        epoch_skipped_steps = 0
        accumulated_steps = 0
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = coconut_model(
                    input_ids,
                    attention_mask,
                    labels,
                    detach_latents=TRAIN_DETACH_LATENTS,
                    use_kv_cache=TRAIN_USE_KV_CACHE,
                )
                loss = outputs.loss.to(torch.float32) / hp["grad_accum"]
            if not torch.isfinite(loss):
                skipped_nonfinite_losses += 1
                epoch_skipped_losses += 1
                optimizer.zero_grad(set_to_none=True)
                accumulated_steps = 0
                pbar.set_postfix({
                    "loss": total_loss / max(finite_loss_steps, 1),
                    "skipped": epoch_skipped_losses,
                })
                continue
            loss.backward()
            total_loss += float(loss.item() * hp["grad_accum"])
            finite_loss_steps += 1
            accumulated_steps += 1
            accumulation_boundary = accumulated_steps == hp["grad_accum"]
            last_batch = step + 1 == len(train_loader)
            if accumulation_boundary or (last_batch and accumulated_steps > 0):
                if accumulated_steps < hp["grad_accum"]:
                    scale = hp["grad_accum"] / accumulated_steps
                    for param in coconut_model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(scale)
                grad_norm = torch.nn.utils.clip_grad_norm_(coconut_model.parameters(), max_norm=1.0)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                else:
                    skipped_nonfinite_steps += 1
                    epoch_skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                accumulated_steps = 0
            pbar.set_postfix({
                "loss": total_loss / max(finite_loss_steps, 1),
                "skipped": epoch_skipped_losses + epoch_skipped_steps,
            })

        epoch_avg_loss = total_loss / max(finite_loss_steps, 1)
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
        val_acc = _phase1_val_accuracy(stage, drop_remaining)
        val_acc_history.append(val_acc)
        epoch_number = epoch + 1
        if stage == 0 and val_acc > cot_best_val_acc:
            cot_best_val_acc = val_acc
            cot_best_epoch = epoch_number
            _save_coconut_checkpoint(
                coconut_model,
                tokenizer,
                cot_best_dir,
                base_model_id,
                role="cot_stage0_best",
                epoch=epoch_number,
                val_accuracy=val_acc,
                uses_coconut_wrapper=False,
            )
            tqdm.write(
                f"[phase1][{model_tag}] best CoT checkpoint saved -> {cot_best_dir} "
                f"(epoch={epoch_number} val_acc={val_acc:.4f})"
            )
        eligible_for_best = stage >= BEST_CHECKPOINT_MIN_STAGE
        if eligible_for_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch_number
            best_stage = stage
            _save_coconut_checkpoint(
                coconut_model,
                tokenizer,
                best_dir,
                base_model_id,
                role="best",
                epoch=epoch_number,
                val_accuracy=val_acc,
            )
            tqdm.write(
                f"[phase1][{model_tag}] best checkpoint saved -> {best_dir} "
                f"(epoch={epoch_number} stage={stage} val_acc={val_acc:.4f})"
            )
        eligible_for_latent_only_best = stage >= LATENT_ONLY_CHECKPOINT_MIN_STAGE
        if eligible_for_latent_only_best and val_acc > latent_only_best_val_acc:
            latent_only_best_val_acc = val_acc
            latent_only_best_epoch = epoch_number
            latent_only_best_stage = stage
            _save_coconut_checkpoint(
                coconut_model,
                tokenizer,
                latent_only_best_dir,
                base_model_id,
                role="best_latent_only",
                epoch=epoch_number,
                val_accuracy=val_acc,
            )
            tqdm.write(
                f"[phase1][{model_tag}] best latent-only checkpoint saved -> {latent_only_best_dir} "
                f"(epoch={epoch_number} stage={stage} val_acc={val_acc:.4f})"
            )
        tqdm.write(
            f"[phase1][{model_tag}] epoch={epoch_number}/{hp['epochs']} "
            f"stage={stage} train_loss={epoch_avg_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"skipped_nonfinite={epoch_skipped_losses + epoch_skipped_steps} "
            f"(train_n={len(train_raw)} val_n={len(val_raw)})"
        )

    _save_coconut_checkpoint(
        coconut_model,
        tokenizer,
        output_dir,
        base_model_id,
        role="final",
        epoch=len(loss_history),
        val_accuracy=val_acc_history[-1] if val_acc_history else None,
    )
    print(f"Coconut final model saved -> {output_dir}")
    if best_epoch is None:
        tqdm.write(
            f"[phase1][{model_tag}] WARNING: no finite eligible latent-stage best checkpoint "
            f"was found; falling back to final checkpoint for compatibility export."
        )
        _save_coconut_checkpoint(
            coconut_model,
            tokenizer,
            best_dir,
            base_model_id,
            role="best",
            epoch=len(loss_history),
            val_accuracy=val_acc_history[-1] if val_acc_history else None,
        )
        print(f"Coconut best model saved -> {best_dir}")
    if latent_only_best_epoch is None:
        tqdm.write(
            f"[phase1][{model_tag}] WARNING: no finite latent-only checkpoint "
            f"was found; falling back to final checkpoint for latent-only export."
        )
        _save_coconut_checkpoint(
            coconut_model,
            tokenizer,
            latent_only_best_dir,
            base_model_id,
            role="best_latent_only",
            epoch=len(loss_history),
            val_accuracy=val_acc_history[-1] if val_acc_history else None,
        )
        print(f"Coconut latent-only best model saved -> {latent_only_best_dir}")
    return {
        "loss_history": loss_history,
        "losses_per_stage": losses_per_stage,
        "stage_transition_epochs": stage_transition_epochs,
        "val_accuracy": val_acc_history,
        "embedding_drift": drift_by_token,
        "n_train": len(train_raw),
        "n_val": len(val_raw),
        "n_phase_examples": len(train_raw),
        "n_validation_examples": len(val_raw),
        "validation_source": "D_val",
        "epochs": hp["epochs"],
        "completed_epochs": len(loss_history),
        "fixed_epoch_schedule": True,
        "learning_rate": hp["lr"],
        "effective_batch_size": hp["batch"] * hp["grad_accum"],
        "curriculum_version": CURRICULUM_VERSION,
        "c_thought": C_THOUGHT,
        "max_latent_stage": MAX_LATENT_STAGE,
        "final_latent_tokens": MAX_LATENT_TOKENS,
        "best_val_accuracy": best_val_acc if best_epoch is not None else None,
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "best_checkpoint_min_stage": BEST_CHECKPOINT_MIN_STAGE,
        "best_checkpoint_policy": "fully_latent_stage4_only",
        "latent_only_best_val_accuracy": latent_only_best_val_acc if latent_only_best_epoch is not None else None,
        "latent_only_best_epoch": latent_only_best_epoch,
        "latent_only_best_stage": latent_only_best_stage,
        "latent_only_checkpoint_min_stage": LATENT_ONLY_CHECKPOINT_MIN_STAGE,
        "latent_only_best_checkpoint_dir": latent_only_best_dir,
        "cot_best_val_accuracy": cot_best_val_acc if cot_best_epoch is not None else None,
        "cot_best_epoch": cot_best_epoch,
        "cot_best_checkpoint_dir": cot_best_dir,
        "best_checkpoint_dir": best_dir,
        "train_use_kv_cache": TRAIN_USE_KV_CACHE,
        "train_detach_latents": TRAIN_DETACH_LATENTS,
        "skipped_nonfinite_losses": skipped_nonfinite_losses,
        "skipped_nonfinite_steps": skipped_nonfinite_steps,
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
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst) or os.path.islink(dst):
            if os.path.islink(dst) or os.path.isfile(dst):
                os.unlink(dst)
            elif os.path.isdir(dst):
                shutil.rmtree(dst)
        rel_src = os.path.relpath(src, os.path.dirname(dst))
        try:
            os.symlink(rel_src, dst)
        except OSError:
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)


def export_compat_checkpoints(
    checkpoints_dir: str,
    latent_token_counts: list[int] = LATENT_TOKEN_COUNTS,
) -> None:
    coconut_source = os.path.join(checkpoints_dir, BEST_DIRNAME)
    cot_source = os.path.join(checkpoints_dir, COT_BEST_DIRNAME)
    source_specs = ((coconut_source, True), (cot_source, False))
    for source, expected_wrapper in source_specs:
        config_path = os.path.join(source, "config.json")
        meta_path = os.path.join(source, "coconut_meta.json")
        if not os.path.exists(config_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Required Phase 1 checkpoint not found: {source}")
        with open(meta_path, encoding="utf-8") as f:
            checkpoint_meta = json.load(f)
        if (
            checkpoint_meta.get("curriculum_version") != CURRICULUM_VERSION
            or checkpoint_meta.get("uses_coconut_wrapper") is not expected_wrapper
        ):
            raise RuntimeError(f"Stale or mislabeled Phase 1 checkpoint: {source}")

    _materialize_alias_dir(cot_source, os.path.join(checkpoints_dir, "cot"))
    for n_latents in latent_token_counts:
        ccot_dir = os.path.join(checkpoints_dir, f"ccot_L{int(n_latents)}")
        _materialize_alias_dir(coconut_source, ccot_dir)

    meta_path = os.path.join(checkpoints_dir, "compat_export_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cot_source": COT_BEST_DIRNAME,
                "coconut_source": BEST_DIRNAME,
                "latent_token_counts": latent_token_counts,
                "exported_dirs": ["cot"] + [f"ccot_L{int(n)}" for n in latent_token_counts],
            },
            f,
            indent=2,
        )
        f.write("\n")


def _phase1_training_current(
    results_dir: str, n_phase_examples: int, n_validation_examples: int
) -> bool:
    metrics_path = os.path.join(results_dir, "phase1_training_metrics.json")
    if not os.path.exists(metrics_path):
        return False
    try:
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metrics.get("n_phase_examples") == n_phase_examples
        and metrics.get("n_train") == n_phase_examples
        and metrics.get("n_validation_examples") == n_validation_examples
        and metrics.get("n_val") == n_validation_examples
        and metrics.get("validation_source") == "D_val"
        and metrics.get("curriculum_version") == CURRICULUM_VERSION
        and metrics.get("epochs") == 30
        and metrics.get("completed_epochs") == 30
        and metrics.get("fixed_epoch_schedule") is True
        and metrics.get("learning_rate") == 1e-4
        and metrics.get("effective_batch_size") == 128
        and metrics.get("c_thought") == C_THOUGHT
        and metrics.get("max_latent_stage") == MAX_LATENT_STAGE
        and metrics.get("final_latent_tokens") == MAX_LATENT_TOKENS
        and metrics.get("train_use_kv_cache") == TRAIN_USE_KV_CACHE
        and metrics.get("train_detach_latents") == TRAIN_DETACH_LATENTS
        and metrics.get("best_checkpoint_min_stage") == BEST_CHECKPOINT_MIN_STAGE
        and metrics.get("best_checkpoint_policy") == "fully_latent_stage4_only"
        and metrics.get("best_stage") == MAX_LATENT_STAGE + 1
        and metrics.get("latent_only_best_stage") == MAX_LATENT_STAGE + 1
        and metrics.get("cot_best_epoch") is not None
    )


def _remove_phase1_eval_artifacts(results_dir: str) -> None:
    for filename in (
        "phase1_latent_sweep.json",
        "phase1_best_latent.json",
        "phase1_val.json",
        "phase1_latent_predictions.jsonl",
        "phase1_comparison_predictions.jsonl",
    ):
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            os.remove(path)


def train_coconut_phase1(
    base_model_id: str,
    D_train: list,
    D_val: list,
    checkpoints_dir: str,
    results_dir: str,
    model_tag: str,
    latent_token_counts: list[int] = LATENT_TOKEN_COUNTS,
):
    require_exact_count(D_train, "D_train")
    require_exact_count(D_val, "D_val")
    canonical_dir = os.path.join(checkpoints_dir, CANONICAL_DIRNAME)
    best_dir = os.path.join(checkpoints_dir, BEST_DIRNAME)
    latent_only_best_dir = os.path.join(checkpoints_dir, LATENT_ONLY_BEST_DIRNAME)
    cot_best_dir = os.path.join(checkpoints_dir, COT_BEST_DIRNAME)
    required_markers = [
        os.path.join(canonical_dir, "config.json"),
        os.path.join(best_dir, "config.json"),
        os.path.join(latent_only_best_dir, "config.json"),
        os.path.join(cot_best_dir, "config.json"),
        os.path.join(canonical_dir, "coconut_meta.json"),
        os.path.join(best_dir, "coconut_meta.json"),
        os.path.join(latent_only_best_dir, "coconut_meta.json"),
        os.path.join(cot_best_dir, "coconut_meta.json"),
    ]
    checkpoints_ready = all(os.path.exists(path) for path in required_markers)
    training_current = _phase1_training_current(results_dir, len(D_train), len(D_val))
    if not checkpoints_ready or not training_current:
        if checkpoints_ready and not training_current:
            print(
                "Phase 1 checkpoints exist but were produced with stale stability/curriculum "
                "settings; retraining."
            )
        metrics = _run_coconut_training(base_model_id, D_train, D_val, canonical_dir, model_tag)
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "phase1_training_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        _remove_phase1_eval_artifacts(results_dir)
        cfg_id = os.path.basename(os.path.dirname(checkpoints_dir))
        model_id = os.path.basename(checkpoints_dir)
        phase1_plot_dir = os.path.join("plots", cfg_id, model_id, "phase1")
        _plot_curves(metrics, phase1_plot_dir)
    else:
        print(
            "Phase 1 checkpoints exist -> "
            f"{canonical_dir}, {best_dir}, {latent_only_best_dir}, {cot_best_dir} (skipping retrain)"
        )

    export_compat_checkpoints(checkpoints_dir, latent_token_counts=latent_token_counts)


def train_cot(base_model_id: str, D_train: list, D_val: list, output_dir: str, model_tag: str, lora_r: int = 16, lora_alpha: int = 32):
    del lora_r, lora_alpha
    checkpoints_dir = os.path.dirname(output_dir)
    results_dir = os.path.join("results", checkpoints_dir.split("/")[-2], checkpoints_dir.split("/")[-1])
    train_coconut_phase1(base_model_id, D_train, D_val, checkpoints_dir, results_dir, model_tag)


def train_ccot(
    base_model_id: str,
    D_train: list,
    D_val: list,
    compressed_cache: list,
    latent_tokens: int,
    output_dir: str,
    model_tag: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    del compressed_cache, latent_tokens, lora_r, lora_alpha
    checkpoints_dir = os.path.dirname(output_dir)
    results_dir = os.path.join("results", checkpoints_dir.split("/")[-2], checkpoints_dir.split("/")[-1])
    train_coconut_phase1(base_model_id, D_train, D_val, checkpoints_dir, results_dir, model_tag)
