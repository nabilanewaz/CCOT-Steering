"""Full Phase 2 extraction, provenance-aware caching, and optional ITI."""
import json
import os
import time

import torch
from sklearn.model_selection import train_test_split

from phase1.inference import cot_prompt, latent_prompt, load_finetuned
from phase2.collect import collect_hidden_states
from phase2.compare import compare_methods, select_best_source_method
from phase2.config import get_model_config
from phase2.cpca import (
    select_layers, cpca_full, cpca_shrunk, cpca_randomized, run_cpca_sweep,
    weighted_subspace_merge, save_subspace, compute_shuffled_cpca, save_shuffled_subspace,
)
from phase2.dom import (
    compute_per_layer_dom, compute_best_layer_dom, compute_shuffled_dom,
    save_dom_vector, save_shuffled_vector, save_multilayer_dom_vectors,
)
from phase2.loaders import find_boundary_idx_ccot, find_boundary_idx_base
from phase2.probe import gate_status, score_all_layers_both
from utils.artifacts import (
    EXPERIMENT_VERSION, atomic_json, checkpoint_identity, dataset_fingerprint, file_fingerprint,
)
from utils.experiment_config import load_protocol, protocol_seed, require_exact_count

_CPCA_FN_MAP = {"full": cpca_full, "shrunk": cpca_shrunk, "randomized": cpca_randomized}


def _hidden_state_cache_usable(cache):
    positive, negative = cache.get("H_pos") or {}, cache.get("H_neg") or {}
    return any(positive[layer].numel() > 0 and negative[layer].numel() > 0
               for layer in set(positive) & set(negative))


def _split_extraction_holdout(H_pos, H_neg):
    fit_pos, fit_neg, test_pos, test_neg = {}, {}, {}, {}
    for layer in H_pos:
        fit_pos[layer], test_pos[layer] = train_test_split(H_pos[layer], test_size=0.2, random_state=42)
        fit_neg[layer], test_neg[layer] = train_test_split(H_neg[layer], test_size=0.2, random_state=42)
    return fit_pos, fit_neg, test_pos, test_neg


def run_phase2_source(
    model, tokenizer, D_steer, model_tag, source_tag, boundary_idx_fn, device,
    vectors_dir, prompt_fn=None, prompt_mode="unknown", N=10, beta=0.5,
    r_per_layer=3, r_final=10, threshold_multiplier=0.5, min_samples=200,
    cpca_variant="full", extraction="mean_gen", gen_window=20, checkpoint=None,
):
    os.makedirs(vectors_dir, exist_ok=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    torch.manual_seed(protocol_seed())
    started = time.time()
    timings = {}
    cache_meta = {
        "experiment_version": EXPERIMENT_VERSION,
        "data": dataset_fingerprint(D_steer), "n_steer": len(D_steer),
        "model_tag": model_tag, "source": source_tag, "prompt_mode": prompt_mode,
        "N": N, "min_samples": min_samples, "extraction": extraction,
        "gen_window": gen_window, "checkpoint": checkpoint, "seed": protocol_seed(),
    }
    cache_path = os.path.join(vectors_dir, f"{source_tag}_hstates_cache.pt")
    cache = None
    if os.path.exists(cache_path):
        candidate = torch.load(cache_path, map_location="cpu", weights_only=False)
        if candidate.get("cache_meta") == cache_meta and _hidden_state_cache_usable(candidate):
            cache = candidate
    if cache is None:
        positive, negative, collection = collect_hidden_states(
            model, tokenizer, D_steer, N, device, boundary_idx_fn, source_tag,
            prompt_fn=prompt_fn, min_samples=min_samples, extraction=extraction, gen_window=gen_window,
        )
        if not positive or not negative:
            rows = collection.get("per_layer", {}).values()
            max_pos = max((row.get("h_pos", 0) for row in rows), default=0)
            max_neg = max((row.get("h_neg", 0) for row in collection.get("per_layer", {}).values()), default=0)
            atomic_json(os.path.join(vectors_dir, f"{source_tag}_diagnostics.json"), {"collection": collection})
            raise RuntimeError(
                f"Phase 2 has no contrastive layer: largest observed H+={max_pos}, H-={max_neg}; "
                f"min_samples={min_samples}. No incomplete cache was saved."
            )
        cache = {"H_pos": positive, "H_neg": negative, "collection_diag": collection, "cache_meta": cache_meta}
        temporary = cache_path + ".tmp"
        torch.save(cache, temporary)
        os.replace(temporary, cache_path)
    positive, negative = cache["H_pos"], cache["H_neg"]
    timings["collection"] = time.time() - started
    fit_pos, fit_neg, test_pos, test_neg = _split_extraction_holdout(positive, negative)
    probe_start = time.time()
    try:
        scores, probe_diagnostics = score_all_layers_both(fit_pos, fit_neg)
    except RuntimeError as error:
        atomic_json(os.path.join(vectors_dir, f"{source_tag}_diagnostics.json"), {
            "collection": cache["collection_diag"], "probe_error": str(error),
        })
        raise
    timings["probe"] = time.time() - probe_start
    directions = compute_per_layer_dom(fit_pos, fit_neg)
    truth, best_layer = compute_best_layer_dom(directions, scores)
    save_dom_vector(truth, model_tag, source_tag, vectors_dir, best_layer=best_layer)
    save_multilayer_dom_vectors(directions, scores, model_tag, source_tag, vectors_dir)
    shuffled, shuffled_stats = compute_shuffled_dom(fit_pos, fit_neg, best_layer, truth)
    save_shuffled_vector(shuffled, model_tag, source_tag, vectors_dir, best_layer=best_layer)
    selected = select_layers(scores, multiplier=threshold_multiplier)
    cpca_start = time.time()
    cpca_function = _CPCA_FN_MAP[cpca_variant]
    sweep = run_cpca_sweep(fit_pos, fit_neg, selected, cpca_function)
    basis, weights, shuffled_basis = None, {}, None
    if sweep:
        subspaces = {layer: (row[0], row[1]) for layer, row in sweep.items()}
        basis, weights = weighted_subspace_merge(subspaces, scores, directions, truth, r_final)
        sweep_meta = {layer: {"k": row[2], "beta": row[3], "acc": row[4]} for layer, row in sweep.items()}
        save_subspace(basis, selected, model_tag, source_tag, r_final, beta, vectors_dir,
                      layer_scores=scores, sweep_meta=sweep_meta)
        shuffled_basis = compute_shuffled_cpca(
            fit_pos, fit_neg, selected, cpca_function, directions, scores, truth, r_final,
        )
        if shuffled_basis is not None:
            save_shuffled_subspace(shuffled_basis, model_tag, source_tag, r_final, vectors_dir)
    timings["cpca"] = time.time() - cpca_start
    winner, accuracies = compare_methods(
        fit_pos, fit_neg, truth, basis, selected, heldout=(test_pos, test_neg), best_layer=best_layer,
    )
    diagnostics = {
        "collection": cache["collection_diag"],
        "probe": {**probe_diagnostics, **gate_status(scores), "layer_scores": scores},
        "dom": {"best_layer": best_layer}, "shuffled_dom": shuffled_stats,
        "layer_selection": {"selected_layers": selected},
        "cpca_sweep": {layer: {"k": row[2], "beta": row[3], "accuracy": row[4]} for layer, row in sweep.items()},
        "subspace_merge": {"requested_rank": r_final, "actual_rank": basis.shape[1] if basis is not None else 0,
                           "layer_weights": weights},
        "method_comparison": {"winner": winner, **accuracies, "holdout_fraction": 0.2},
        "step_times_s": {**timings, "total": time.time() - started},
        "freeze": {"trainable_params": 0, "model_training": False},
    }
    atomic_json(os.path.join(vectors_dir, f"{source_tag}_diagnostics.json"), diagnostics)
    return {
        "v_truth": truth, "U_truth": basis, "best_layer": best_layer, "layer_scores": scores,
        "selected_layers": selected, "method_accs": accuracies, "winner": winner,
        "diagnostics": diagnostics, "has_cpca": basis is not None,
        "has_shuffled_cpca": shuffled_basis is not None,
    }


def pick_best_ccot_latent_tokens(results_dir, model_tag):
    best_path = os.path.join(results_dir, "phase1_best_latent.json")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Phase 1 selection required; no fallback latent budget: {best_path}")
    with open(best_path) as stream:
        payload = json.load(stream)
    budget = int(payload["latent_tokens"])
    if budget not in (3, 4, 6):
        raise ValueError(f"Unsupported Phase 1 latent budget: {budget}")
    return budget


def run_iti_phase2(model_tag, checkpoints_dir, D_steer, device, vectors_dir, results_dir, sources=None):
    from phase2.collect_heads import collect_head_activations
    from phase2.probe_heads import probe_all_heads

    with open(os.path.join(vectors_dir, "phase2_meta.json")) as stream:
        meta = json.load(stream)
    budget = meta["best_ccot_latent_tokens"]
    for source in sources or meta["sources"]:
        checkpoint = os.path.join(checkpoints_dir, f"ccot_L{budget}" if source == "ccot" else "cot")
        identity = {"version": EXPERIMENT_VERSION, "dataset": dataset_fingerprint(D_steer),
                    "checkpoint": checkpoint_identity(checkpoint), "source": source,
                    "N": 10, "temperature": 0.8, "min_samples": 30, "max_new_tokens": 128}
        cache_path = os.path.join(vectors_dir, f"{source}_head_hstates_cache.pt")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False) if os.path.exists(cache_path) else {}
        if cache.get("identity") != identity:
            model, tokenizer = load_finetuned(checkpoint, device)
            prompt = (lambda item: latent_prompt(item["question"], budget)) if source == "ccot" else (lambda item: cot_prompt(item["question"]))
            boundary = find_boundary_idx_ccot if source == "ccot" else find_boundary_idx_base
            try:
                positive, negative = collect_head_activations(model, tokenizer, D_steer, device, prompt, boundary)
                num_heads = int(model.config.num_attention_heads)
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            cache = {"identity": identity, "H_pos": positive, "H_neg": negative, "num_heads": num_heads}
            torch.save(cache, cache_path)
        payload = probe_all_heads(cache["H_pos"], cache["H_neg"], top_k=load_protocol().get("iti_top_k", 48))
        payload.update(model_tag=model_tag, source_tag=source, identity=identity, num_heads=cache["num_heads"])
        torch.save(payload, os.path.join(vectors_dir, f"{source}_iti_heads.pt"))


def run_phase2_all_sources(
    model_tag, base_model_id, checkpoints_dir, D_steer, device, vectors_dir, results_dir,
    run_source_b=None, run_iti=None,
):
    require_exact_count(D_steer, "D_steer")
    protocol = load_protocol()
    run_source_b = protocol.get("run_source_b", False) if run_source_b is None else run_source_b
    run_iti = protocol.get("run_iti", False) if run_iti is None else run_iti
    sources = ["ccot", "base"] if run_source_b else ["ccot"]
    budget = pick_best_ccot_latent_tokens(results_dir, model_tag)
    config = get_model_config(model_tag)
    config.update(extraction=protocol["extraction"], gen_window=protocol["gen_window"])
    results, checkpoints = {}, {}
    os.makedirs(vectors_dir, exist_ok=True)
    for source in sources:
        path = os.path.join(checkpoints_dir, f"ccot_L{budget}" if source == "ccot" else "cot")
        checkpoints[source] = checkpoint_identity(path)
        model, tokenizer = load_finetuned(path, device)
        prompt = (lambda item: latent_prompt(item["question"], budget)) if source == "ccot" else (lambda item: cot_prompt(item["question"]))
        boundary = find_boundary_idx_ccot if source == "ccot" else find_boundary_idx_base
        try:
            results[source] = run_phase2_source(
                model, tokenizer, D_steer, model_tag, source, boundary, device, vectors_dir,
                prompt_fn=prompt, prompt_mode="latent_prompt" if source == "ccot" else "cot_prompt",
                checkpoint=checkpoints[source], **config,
            )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    best_source, best_method, accuracy = select_best_source_method(results["ccot"], results.get("base", {}))
    meta = {
        "experiment_version": EXPERIMENT_VERSION, "phase2_prompt_version": 3,
        "model_tag": model_tag, "n_steer": len(D_steer), "sources": sources,
        "dataset_fingerprint": dataset_fingerprint(D_steer), "checkpoints": checkpoints,
        "config": config, "best_ccot_latent_tokens": budget, "best_source": best_source,
        "best_method": best_method, "best_probe_acc": accuracy,
        "ccot_r_final": config["r_final"], "extraction": config["extraction"],
    }
    for source, result in results.items():
        meta.update({
            f"{source}_best_layer": result["best_layer"],
            f"{source}_selected_layers": result["selected_layers"],
            f"{source}_max_probe_score": max(result["layer_scores"].values()),
            f"{source}_probe_gate": result["diagnostics"]["probe"],
            f"{source}_method_accs": result["method_accs"],
            f"{source}_has_cpca": result["has_cpca"],
            f"{source}_has_shuffled_cpca": result["has_shuffled_cpca"],
        })
        atomic_json(os.path.join(results_dir, f"phase2_{source}_diagnostics.json"), result["diagnostics"])
    artifact_names = []
    for source, result in results.items():
        artifact_names.extend(f"{source}_{suffix}.pt" for suffix in ("dom", "multilayer_dom", "shuffled_dom"))
        if result["has_cpca"]:
            artifact_names.append(f"{source}_cpca_r{config['r_final']}.pt")
        if result["has_shuffled_cpca"]:
            artifact_names.append(f"{source}_shuffled_cpca_r{config['r_final']}.pt")
    meta["artifacts"] = {name: file_fingerprint(os.path.join(vectors_dir, name)) for name in artifact_names}
    atomic_json(os.path.join(vectors_dir, "phase2_meta.json"), meta)
    atomic_json(os.path.join(results_dir, "phase2_meta.json"), meta)
    if run_iti:
        run_iti_phase2(model_tag, checkpoints_dir, D_steer, device, vectors_dir, results_dir, sources)
    return results
