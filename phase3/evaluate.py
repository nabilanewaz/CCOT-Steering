"""Full-validation condition grid with provenance-checked, per-condition resume."""
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm

from phase1.inference import (
    cot_prompt, extract_answer, extract_reasoning_span, latent_prompt,
    load_base_frozen, load_finetuned, normalize_answer, run_trimmed_cot,
)
from phase3.alpha import tune_alpha
from phase3.hooks import condition_hooks, generation_scope
from utils.artifacts import (
    EXPERIMENT_VERSION, atomic_json, checkpoint_identity, dataset_fingerprint,
    file_fingerprint, fingerprint,
)
from utils.experiment_config import load_protocol, protocol_seed, require_exact_count

STEERED_METHODS = ("dom", "cpca", "multilayer_dom", "multilayer_dom_mlp", "iti")
SKIP_CONDITIONS = frozenset()


@dataclass
class ConditionResult:
    condition: str
    model_tag: str
    ratio: Optional[float]
    vector_source: Optional[str]
    vector_method: Optional[str]
    alpha: Optional[float]
    accuracy: float
    flip_rate: float
    reasoning_tokens: float
    actual_ratio: float
    latency_sec: float
    answer_found_rate: float
    n_examples: int = 0


def _load_meta(vectors_dir):
    with open(os.path.join(vectors_dir, "phase2_meta.json")) as stream:
        return json.load(stream)


def _require_phase2_inputs(vectors_dir):
    required = ("phase2_meta.json", "ccot_dom.pt", "ccot_multilayer_dom.pt", "ccot_shuffled_dom.pt")
    missing = [name for name in required if not os.path.isfile(os.path.join(vectors_dir, name))
               or os.path.getsize(os.path.join(vectors_dir, name)) == 0]
    if missing:
        raise RuntimeError(f"Phase 3 cannot start: Phase 2 artifacts missing or empty: {missing}")
    meta = _load_meta(vectors_dir)
    if meta.get("experiment_version") != EXPERIMENT_VERSION:
        raise RuntimeError("Phase 2 artifacts predate the full experiment; rerun Phase 2")
    for name, expected in meta["artifacts"].items():
        path = os.path.join(vectors_dir, name)
        if not os.path.exists(path) or file_fingerprint(path) != expected:
            raise RuntimeError(f"Phase 2 artifact missing or changed: {path}")


def load_source_artifacts(vectors_dir, source, meta):
    def read(suffix):
        path = os.path.join(vectors_dir, f"{source}_{suffix}.pt")
        return torch.load(path, map_location="cpu", weights_only=False)

    artifacts = {key: read(key) for key in ("dom", "multilayer_dom", "shuffled_dom")}
    rank = meta["ccot_r_final"]
    if meta.get(f"{source}_has_cpca"):
        artifacts["cpca"] = read(f"cpca_r{rank}")
    if meta.get(f"{source}_has_shuffled_cpca"):
        artifacts["shuffled_cpca"] = read(f"shuffled_cpca_r{rank}")
    iti_path = os.path.join(vectors_dir, f"{source}_iti_heads.pt")
    if os.path.exists(iti_path):
        iti = read("iti_heads")
        identity = iti.get("identity", {})
        if (identity.get("dataset") != meta["dataset_fingerprint"]
                or identity.get("checkpoint") != meta["checkpoints"][source]):
            raise RuntimeError(f"Stale ITI artifact: {iti_path}; rerun Phase 2.5")
        artifacts["iti"] = iti
    return artifacts


def available_methods(artifacts):
    methods = ["noise", "dom", "multilayer_dom", "multilayer_dom_mlp", "shuf_dom", "neg_dom"]
    if "cpca" in artifacts:
        methods.extend(["cpca", "neg_cpca"])
    if "shuffled_cpca" in artifacts:
        methods.append("shuf_cpca")
    if "iti" in artifacts:
        methods.append("iti")
    return methods


def generate_condition(model, tokenizer, prompt, device, method=None, artifacts=None, alpha=0.0, max_new_tokens=256):
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    length = encoded["input_ids"].shape[1]
    context = condition_hooks(model, method, artifacts or {}, alpha, device) if method else nullcontext()
    with generation_scope(model, length), context:
        with torch.no_grad():
            output = model.generate(**encoded, max_new_tokens=max_new_tokens,
                                    do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0, length:], skip_special_tokens=True)


def example_record(text, item, tokenizer, elapsed):
    prediction = extract_answer(text)
    gold = item["answer"].split("####", 1)[1].strip()
    reasoning = extract_reasoning_span(text)
    return {
        "id": item.get("id"), "question_hash": fingerprint(item["question"]),
        "prediction": prediction, "generated_text": text,
        "correct": prediction is not None and normalize_answer(prediction) == normalize_answer(gold),
        "answer_found": prediction is not None,
        "reasoning_tokens": len(tokenizer.encode(reasoning, add_special_tokens=False)),
        "latency_sec": elapsed,
    }


def _result(condition, model_tag, source, method, alpha, rows, baseline, full_mean):
    count = len(rows)
    wrong = [index for index, row in enumerate(baseline or []) if not row["correct"]]
    mean_tokens = sum(row["reasoning_tokens"] for row in rows) / count
    return ConditionResult(
        condition, model_tag, None, source, method, alpha,
        sum(row["correct"] for row in rows) / count,
        sum(rows[index]["correct"] for index in wrong) / len(wrong) if wrong else 0.0,
        mean_tokens, mean_tokens / full_mean if full_mean else 0.0,
        sum(row["latency_sec"] for row in rows) / count,
        sum(row["answer_found"] for row in rows) / count, count,
    )


def phase3_identity(model_tag, checkpoints_dir, D_val, vectors_dir, max_new_tokens=256):
    meta = _load_meta(vectors_dir)
    budget = meta["best_ccot_latent_tokens"]
    checkpoint_paths = {"cot": os.path.join(checkpoints_dir, "cot"),
                        "ccot": os.path.join(checkpoints_dir, f"ccot_L{budget}")}
    current_checkpoints = {key: checkpoint_identity(path) for key, path in checkpoint_paths.items()}
    for source in meta["sources"]:
        key = "ccot" if source == "ccot" else "cot"
        if meta["checkpoints"][source] != current_checkpoints[key]:
            raise RuntimeError(f"Checkpoint changed since Phase 2: {source}; rerun extraction")
    vector_files = dict(meta["artifacts"])
    for source in meta["sources"]:
        path = os.path.join(vectors_dir, f"{source}_iti_heads.pt")
        if os.path.exists(path):
            vector_files[f"{source}_iti_heads.pt"] = file_fingerprint(path)
    return {
        "experiment_version": EXPERIMENT_VERSION, "model_tag": model_tag,
        "validation": dataset_fingerprint(D_val), "n_val": len(D_val),
        "phase2": file_fingerprint(os.path.join(vectors_dir, "phase2_meta.json")),
        "artifacts": vector_files, "protocol": load_protocol(),
        "checkpoints": current_checkpoints,
        "injection": "generated_positions_only", "max_new_tokens": max_new_tokens,
    }


def _load_alpha(vectors_dir, source):
    return float(torch.load(os.path.join(vectors_dir, f"{source}_alpha_star.pt"), map_location="cpu", weights_only=False))


def _tune_and_save_alpha(model_tag, checkpoints_dir, D_val, vectors_dir, device, meta, results_dir, signature):
    from phase3.lambda_sweep import sweep_lambda_grid

    protocol = load_protocol()
    budget = meta["best_ccot_latent_tokens"]
    for source in meta["sources"]:
        alpha_path = os.path.join(vectors_dir, f"{source}_alpha_star.pt")
        metadata_path = os.path.join(results_dir, f"{source}_alpha_meta.json")
        if os.path.exists(alpha_path) and os.path.exists(metadata_path):
            with open(metadata_path) as stream:
                saved = json.load(stream)
            if saved.get("signature") == signature and saved.get("alpha_sha256") == file_fingerprint(alpha_path):
                continue
        checkpoint = os.path.join(checkpoints_dir, f"ccot_L{budget}" if source == "ccot" else "cot")
        model, tokenizer = load_finetuned(checkpoint, device)
        artifacts = load_source_artifacts(vectors_dir, source, meta)
        vector = artifacts["dom"]["v_truth"]
        layer = artifacts["dom"]["best_layer"]
        prompt_fn = (lambda item: latent_prompt(item["question"], budget)) if source == "ccot" else (lambda item: cot_prompt(item["question"]))
        sweep_path = os.path.join(results_dir, f"{source}_lambda_sweep.json")
        try:
            sweep = None
            if os.path.exists(sweep_path):
                with open(sweep_path) as stream:
                    cached = json.load(stream)
                if cached.get("signature") == signature:
                    sweep = cached["selected"]
            if sweep is None:
                sweep = sweep_lambda_grid(
                    model, tokenizer, D_val[:protocol["lambda_sweep_examples"]], vector, layer,
                    device, model_tag, latent_tokens=budget, out_path=sweep_path, max_epochs=2,
                    prompt_fn=prompt_fn, prompt_mode=source,
                )
                with open(sweep_path) as stream:
                    cached = json.load(stream)
                cached["signature"] = signature
                atomic_json(sweep_path, cached)
            alpha, history = tune_alpha(
                model, tokenizer, D_val[:protocol["alpha_tune_examples"]], vector, layer, device,
                model_tag=model_tag, latent_tokens=budget, lambda_a=sweep["lambda_a"],
                lambda_m=sweep["lambda_m"], prompt_fn=prompt_fn, es_patience=protocol["es_patience"],
                max_epochs=protocol["max_epochs"], lr=protocol["alpha_lr"],
            )
            torch.save(alpha.cpu(), alpha_path)
            atomic_json(metadata_path, {
                "signature": signature, "alpha": float(alpha),
                "alpha_sha256": file_fingerprint(alpha_path),
                "n_tune": min(len(D_val), protocol["alpha_tune_examples"]),
                "n_lambda": min(len(D_val), protocol["lambda_sweep_examples"]),
                "dataset_role": "D_val", "injection": "generated_positions_only",
            })
            atomic_json(os.path.join(results_dir, f"{source}_alpha_history.json"), {
                "source": source, "model_tag": model_tag, "history": history, **sweep,
            })
            from phase3.plots import plot_loss_curves, plot_lambda_sweep_heatmap
            plot_loss_curves(history, os.path.join(results_dir, f"{source}_loss_curves.png"))
            plot_lambda_sweep_heatmap(cached, os.path.join(results_dir, f"{source}_lambda_heatmap.png"))
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def alpha_diagnostic(model, tokenizer, data, prompt_fn, artifacts, method, device, candidates):
    rows = []
    for alpha in candidates:
        correct = 0
        for item in data:
            text = generate_condition(model, tokenizer, prompt_fn(item), device,
                                      method, artifacts, alpha)
            prediction = extract_answer(text)
            gold = item["answer"].split("####", 1)[1].strip()
            correct += prediction is not None and normalize_answer(prediction) == normalize_answer(gold)
        rows.append({"alpha": alpha, "accuracy": correct / len(data),
                     "n_examples": len(data), "n_correct": correct})
    return rows


def run_phase3_evaluation(
    model_tag, base_model_id, checkpoints_dir, D_val, vectors_dir, results_dir, device,
    max_new_tokens=256,
):
    require_exact_count(D_val, "D_val")
    _require_phase2_inputs(vectors_dir)
    meta = _load_meta(vectors_dir)
    identity = phase3_identity(model_tag, checkpoints_dir, D_val, vectors_dir, max_new_tokens)
    signature = fingerprint(identity)
    os.makedirs(results_dir, exist_ok=True)
    interim_path = os.path.join(results_dir, "phase3_val_interim.json")
    example_path = os.path.join(results_dir, "phase3_examples.json")
    run_path = os.path.join(results_dir, "phase3_run_meta.json")
    results, saved_examples = [], {}
    if os.path.exists(run_path):
        with open(run_path) as stream:
            previous = json.load(stream)
        if previous.get("signature") == signature and os.path.exists(interim_path) and os.path.exists(example_path):
            with open(interim_path) as stream:
                results = [ConditionResult(**row) for row in json.load(stream)]
            with open(example_path) as stream:
                saved_examples = json.load(stream)
            results = [row for row in results if row.n_examples == len(D_val)
                       and len(saved_examples.get(row.condition, [])) == len(D_val)]
    atomic_json(run_path, {**identity, "signature": signature, "complete": False, "phase3_eval_version": 3})
    _tune_and_save_alpha(model_tag, checkpoints_dir, D_val, vectors_dir, device, meta, results_dir, signature)
    alphas = {source: _load_alpha(vectors_dir, source) for source in meta["sources"]}
    full_mean = next((row.reasoning_tokens for row in results if row.condition == "full_cot"), 0.0)

    def evaluate(name, model, tokenizer, prompt_fn, source=None, method=None, alpha=None,
                 artifacts=None, baseline=None, budgets=None, token_limit=max_new_tokens):
        nonlocal results
        if any(row.condition == name for row in results):
            print(f"[RESUME] {name}: {len(D_val)} examples")
            return saved_examples[name]
        torch.manual_seed(int(fingerprint([protocol_seed(), name])[:8], 16))
        rows = []
        for index, item in enumerate(tqdm(D_val, desc=name, unit="example")):
            started = time.time()
            if budgets is not None:
                with generation_scope(model, len(tokenizer.encode(cot_prompt(item["question"])))):
                    if method:
                        with condition_hooks(model, method, artifacts, alpha, device):
                            prediction, reasoning = run_trimmed_cot(model, tokenizer, item, budgets[index], device)
                    else:
                        prediction, reasoning = run_trimmed_cot(model, tokenizer, item, budgets[index], device)
                text = reasoning + ("\n#### " + prediction if prediction is not None else "")
            else:
                text = generate_condition(model, tokenizer, prompt_fn(item), device,
                                          method, artifacts, alpha or 0.0, token_limit)
            rows.append(example_record(text, item, tokenizer, time.time() - started))
        if name == "no_cot":
            for row in rows:
                row["reasoning_tokens"] = 0
        result = _result(name, model_tag, source, method, alpha, rows, baseline, full_mean)
        results.append(result)
        saved_examples[name] = rows
        atomic_json(example_path, saved_examples)
        atomic_json(interim_path, [asdict(row) for row in results])
        print(f"{name}: accuracy={result.accuracy:.4f}, n={result.n_examples}")
        return rows

    if not any(row.condition == "no_cot" for row in results):
        model, tokenizer = load_base_frozen(base_model_id, device)
        evaluate("no_cot", model, tokenizer, lambda item: f"Question: {item['question']}\n\nAnswer:", token_limit=32)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not any(row.condition == "full_cot" for row in results):
        model, tokenizer = load_finetuned(os.path.join(checkpoints_dir, "cot"), device)
        evaluate("full_cot", model, tokenizer, lambda item: cot_prompt(item["question"]), token_limit=512)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    full_mean = sum(row["reasoning_tokens"] for row in saved_examples["full_cot"]) / len(D_val)
    for result in results:
        result.actual_ratio = result.reasoning_tokens / full_mean if full_mean else 0.0
    budget = meta["best_ccot_latent_tokens"]
    tag = f"L{budget}"
    model, tokenizer = load_finetuned(os.path.join(checkpoints_dir, f"ccot_{tag}"), device)
    prompt_fn = lambda item: latent_prompt(item["question"], budget)
    baseline = evaluate(f"ccot_{tag}", model, tokenizer, prompt_fn)
    diagnostics = {}
    try:
        for source in meta["sources"]:
            artifacts = load_source_artifacts(vectors_dir, source, meta)
            iti_alpha = None
            if "iti" in artifacts:
                path = os.path.join(results_dir, f"iti_alpha_diagnostic_{source}.json")
                cached = {}
                if os.path.exists(path):
                    with open(path) as stream:
                        cached = json.load(stream)
                if cached.get("signature") != signature:
                    subset = D_val[:load_protocol()["diagnostic_examples"]]
                    sweep = alpha_diagnostic(model, tokenizer, subset, prompt_fn, artifacts, "iti", device,
                                             [0.5, 1, 2, 5, 10, 15, 20])
                    best = max(sweep, key=lambda row: (row["accuracy"], -row["alpha"]))
                    cached = {"signature": signature, "best_alpha": best["alpha"], "sweep": sweep,
                              "n_examples": len(subset), "dataset_role": "D_val"}
                    atomic_json(path, cached)
                iti_alpha = cached["best_alpha"]
            for method in available_methods(artifacts):
                alpha = iti_alpha if method == "iti" else alphas[source]
                evaluate(f"{method}_{tag}_{source}", model, tokenizer, prompt_fn,
                         source, method, alpha, artifacts, baseline)
            diagnostics[source] = {"methods": available_methods(artifacts), "alpha": alphas[source],
                                   "iti_alpha": iti_alpha}
        diagnostic_path = os.path.join(results_dir, "alpha_diagnostic.json")
        cached = {}
        if os.path.exists(diagnostic_path):
            with open(diagnostic_path) as stream:
                cached = json.load(stream)
        if cached.get("signature") != signature:
            subset = D_val[:load_protocol()["diagnostic_examples"]]
            sweep = alpha_diagnostic(model, tokenizer, subset, prompt_fn,
                                     load_source_artifacts(vectors_dir, "ccot", meta), "dom", device,
                                     load_protocol()["diagnostic_alphas"])
            cached = {"signature": signature, "model_tag": model_tag, "source": "ccot",
                      "alpha_star": alphas["ccot"], "sweep": sweep, "n_examples": len(subset),
                      "dataset_role": "D_val"}
            atomic_json(diagnostic_path, cached)
        from phase3.plots import plot_alpha_diagnostic
        plot_alpha_diagnostic(cached, os.path.join(results_dir, "alpha_diagnostic.png"))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    budgets = [max(1, budget + row["reasoning_tokens"]) for row in baseline]
    atomic_json(os.path.join(results_dir, f"phase3_budgets_{tag}.json"), {"signature": signature, "budgets": budgets})
    atomic_json(os.path.join(results_dir, f"phase3_ccot_correct_{tag}.json"),
                {"signature": signature, "correct": [row["correct"] for row in baseline]})
    model, tokenizer = load_finetuned(os.path.join(checkpoints_dir, "cot"), device)
    try:
        evaluate(f"trimmed_{tag}", model, tokenizer, lambda item: cot_prompt(item["question"]),
                 baseline=baseline, budgets=budgets)
        if "base" in meta["sources"]:
            artifacts = load_source_artifacts(vectors_dir, "base", meta)
            evaluate(f"trimmed_dom_{tag}", model, tokenizer, lambda item: cot_prompt(item["question"]),
                     "base", "dom", alphas["base"], artifacts, baseline, budgets)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(os.path.join(results_dir, "phase3_val.json"), [asdict(row) for row in results])
    atomic_json(os.path.join(results_dir, "phase3_diagnostics.json"), diagnostics)
    atomic_json(run_path, {**identity, "signature": signature, "complete": True, "phase3_eval_version": 3})
    from phase3.select import select_best_steered_config
    selection = select_best_steered_config(results_dir, model_tag)
    atomic_json(os.path.join(results_dir, "steered_val.json"), {
        **selection, "n_examples": len(D_val),
        "probe_accuracy": meta.get(f"{selection.get('vector_source', 'ccot')}_max_probe_score", 0.0),
    })
    return results
