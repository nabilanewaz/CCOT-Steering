"""Full Phase 2 runner: collect hidden states → probe → DoM → cPCA → compare."""
import json
import os
import time

import torch

from phase2.config import get_model_config
from phase2.loaders import (
    load_ccot_frozen,
    find_boundary_idx_ccot,
    find_boundary_idx_base,
)
from phase2.collect import collect_hidden_states
from phase2.probe import PROBE_GATE, score_all_layers
from phase2.dom import (
    compute_per_layer_dom,
    compute_best_layer_dom,
    compute_shuffled_dom,
    report_cross_source_alignment,
    save_dom_vector,
    save_shuffled_vector,
)
from phase2.cpca import (
    select_layers,
    cpca_full,
    cpca_shrunk,
    cpca_randomized,
    run_cpca_sweep,
    weighted_subspace_merge,
    save_subspace,
    compute_shuffled_cpca,
    save_shuffled_subspace,
)
from phase2.compare import compare_methods, select_best_source_method

_CPCA_FN_MAP = {
    'full':       cpca_full,
    'shrunk':     cpca_shrunk,
    'randomized': cpca_randomized,
}


def _tensor_shape(value) -> list[int] | None:
    return list(value.shape) if value is not None and hasattr(value, "shape") else None


def _layer_selection_diagnostics(
    layer_scores: dict[int, float],
    threshold_multiplier: float,
    selected_layers: list[int],
) -> dict:
    if not layer_scores:
        return {
            "threshold": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "layers_passing": [],
            "selected_layers": selected_layers,
        }
    vals = torch.tensor(list(layer_scores.values()), dtype=torch.float32)
    mean = vals.mean().item()
    std = vals.std(unbiased=False).item()
    threshold = mean + threshold_multiplier * std
    return {
        "threshold": threshold,
        "mean": mean,
        "std": std,
        "layers_passing": sorted(int(L) for L, s in layer_scores.items() if s >= threshold),
        "selected_layers": sorted(int(L) for L in selected_layers),
    }


def _save_diagnostics(vectors_dir: str, source: str, diagnostics: dict) -> str:
    os.makedirs(vectors_dir, exist_ok=True)
    path = os.path.join(vectors_dir, f"{source}_diagnostics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved {source} diagnostics -> {path}")
    return path


def run_phase2_source(
    model,
    tokenizer,
    D_steer: list,
    model_tag: str,
    source_tag: str,
    boundary_idx_fn,
    device: str,
    vectors_dir: str,
    prompt_fn=None,
    N: int = 20,
    beta: float = 0.5,
    r_per_layer: int = 3,
    r_final: int = 10,
    threshold_multiplier: float = 0.5,
    min_samples: int = 200,
    cpca_variant: str = 'full',
) -> dict:
    """
    Phase 2 extraction for a single (model, source) pair.
    Saves {source}_dom.pt and {source}_cpca_r{r_final}.pt to vectors_dir.
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    header = f"Phase 2: {model_tag} | source={source_tag}"
    print(f"\n{'='*len(header)}\n{header}\n{'='*len(header)}")
    print(
        f"Phase 2 freeze check [{source_tag}]: "
        f"trainable_params={trainable_params} frozen_params={frozen_params}"
    )
    source_start = time.time()
    step_times: dict[str, float] = {}

    def _step(n: int, label: str) -> float:
        elapsed = time.time() - source_start
        print(f"\n{'-' * 72}")
        print(f"STEP {n}/9 [{source_tag}] {label} | elapsed={elapsed:.1f}s")
        print(f"{'-' * 72}")
        return time.time()

    def _finish_step(key: str, started: float) -> None:
        step_times[key] = time.time() - started

    t = _step(1, "collect hidden states")
    os.makedirs(vectors_dir, exist_ok=True)
    hstates_cache = os.path.join(vectors_dir, f"{source_tag}_hstates_cache.pt")
    if os.path.exists(hstates_cache):
        print(f"  Loading cached hidden states from {hstates_cache}")
        cache = torch.load(hstates_cache, map_location="cpu")
        H_pos, H_neg = cache["H_pos"], cache["H_neg"]
        collection_diag = dict(cache.get("collection_diag") or {})
        collection_diag["cached"] = True
        collection_diag["cache_path"] = hstates_cache
    else:
        H_pos, H_neg, collection_diag = collect_hidden_states(
            model, tokenizer, D_steer, N, device,
            boundary_idx_fn, source_tag,
            prompt_fn=prompt_fn,
            min_samples=min_samples,
        )
        collection_diag = dict(collection_diag or {})
        collection_diag["cached"] = False
        collection_diag["cache_path"] = hstates_cache
        torch.save(
            {
                "H_pos": H_pos,
                "H_neg": H_neg,
                "collection_diag": collection_diag,
            },
            hstates_cache,
        )
        print(f"  Hidden states cached -> {hstates_cache}")
    _finish_step("1_collection", t)
    if not H_pos:
        print("No layers passed min_samples threshold — aborting this source.")
        diagnostics = {
            "collection": collection_diag,
            "freeze": {
                "trainable_params": trainable_params,
                "frozen_params": frozen_params,
                "model_training": bool(model.training),
            },
            "step_times_s": {**step_times, "total": time.time() - source_start},
        }
        _save_diagnostics(vectors_dir, source_tag, diagnostics)
        return {"diagnostics": diagnostics}

    # ── Layer probe scores ────────────────────────────────────────────────────
    t = _step(2, "score layer probes")
    layer_scores = score_all_layers(H_pos, H_neg)
    _finish_step("2_probe", t)

    # ── Method A: best-layer DoM ──────────────────────────────────────────────
    t = _step(3, "compute best-layer DoM")
    dom_vectors         = compute_per_layer_dom(H_pos, H_neg)
    v_truth, best_layer = compute_best_layer_dom(dom_vectors, layer_scores)
    dom_cosines = {
        str(L): float(torch.dot(v.float(), v_truth.float()).item())
        for L, v in dom_vectors.items()
    }
    _finish_step("3_dom", t)

    # ── Control: shuffled-label DoM ───────────────────────────────────────────
    t = _step(4, "compute shuffled-label DoM control")
    v_shuffled, shuffled_dom_stats = compute_shuffled_dom(H_pos, H_neg, best_layer, v_truth)
    save_shuffled_vector(v_shuffled, model_tag, source_tag, vectors_dir,
                         best_layer=best_layer)
    _finish_step("4_shuffled_dom", t)

    # ── Layer selection ───────────────────────────────────────────────────────
    t = _step(5, "select cPCA layers")
    selected_layers = select_layers(layer_scores, multiplier=threshold_multiplier)
    layer_selection_diag = _layer_selection_diagnostics(
        layer_scores, threshold_multiplier, selected_layers
    )
    _finish_step("5_layer_selection", t)

    # ── Method B: cPCA sweep (k ∈ {1,2,5,10}, β ∈ {0.3,0.5,0.7}) ────────────
    t = _step(6, "run cPCA sweep")
    cpca_fn = _CPCA_FN_MAP.get(cpca_variant, cpca_full)

    cpca_results = run_cpca_sweep(H_pos, H_neg, selected_layers, cpca_fn)
    subspaces  = {L: (U, lam) for L, (U, lam, _, _, _) in cpca_results.items()}
    sweep_meta = {L: {'k': k, 'beta': b, 'acc': acc}
                  for L, (_, _, k, b, acc) in cpca_results.items()}
    cpca_sweep_diag = {
        str(L): {
            "best_k": int(k),
            "best_beta": float(b),
            "best_acc": float(acc),
            "h_pos": int(H_pos[L].shape[0]),
            "h_neg": int(H_neg[L].shape[0]),
        }
        for L, (_, _, k, b, acc) in cpca_results.items()
    }
    _finish_step("6_cpca_sweep", t)

    if not subspaces:
        print("No subspaces computed — saving DoM only, skipping Method B.")
        save_dom_vector(v_truth, model_tag, source_tag, vectors_dir,
                        best_layer=best_layer)
        method_accs = {'dom': layer_scores.get(best_layer, 0.0), 'cpca': 0.0}
        diagnostics = {
            "collection": collection_diag,
            "freeze": {
                "trainable_params": trainable_params,
                "frozen_params": frozen_params,
                "model_training": bool(model.training),
            },
            "probe": {
                "layer_scores": {str(L): float(s) for L, s in layer_scores.items()},
                "gate_threshold": PROBE_GATE,
                "layers_passing": sorted(int(L) for L, s in layer_scores.items() if s > PROBE_GATE),
            },
            "dom": {
                "best_layer": int(best_layer),
                "v_truth_norm": float(v_truth.norm().item()),
                "per_layer_cosines_to_best": dom_cosines,
            },
            "shuffled_dom": shuffled_dom_stats,
            "layer_selection": layer_selection_diag,
            "cpca_sweep": cpca_sweep_diag,
            "subspace_merge": {"r_final": r_final, "U_shape": None, "layer_weights": {}},
            "method_comparison": {
                "dom_acc": float(method_accs["dom"]),
                "cpca_acc": 0.0,
                "gap": -float(method_accs["dom"]),
                "winner": "dom",
            },
            "shuffled_cpca": {"U_shape": None},
            "step_times_s": {**step_times, "total": time.time() - source_start},
        }
        _save_diagnostics(vectors_dir, source_tag, diagnostics)
        return {
            'v_truth':    v_truth,    'U_truth':       None,
            'v_shuffled': v_shuffled, 'best_layer':    best_layer,
            'layer_scores':    layer_scores,
            'selected_layers': selected_layers, 'sweep_meta': {},
            'winner':     'dom', 'dom_vectors': dom_vectors, 'subspaces': {},
            'method_accs': method_accs,
            'diagnostics': diagnostics,
        }

    t = _step(7, "merge weighted cPCA subspaces")
    U_truth, layer_weights = weighted_subspace_merge(
        subspaces, layer_scores, dom_vectors, v_truth, r_final
    )
    _finish_step("7_subspace_merge", t)

    # ── Method comparison ─────────────────────────────────────────────────────
    t = _step(8, "compare methods and save vectors")
    winner, method_accs = compare_methods(
        H_pos, H_neg, v_truth, U_truth, selected_layers
    )

    save_dom_vector(v_truth, model_tag, source_tag, vectors_dir,
                    best_layer=best_layer)
    save_subspace(U_truth, selected_layers, model_tag, source_tag,
                  r_final, beta, vectors_dir,
                  layer_scores=layer_scores, sweep_meta=sweep_meta)
    _finish_step("8_method_compare_save", t)

    # ── Control: Shuffled-Label cPCA ──────────────────────────────────────────
    t = _step(9, "compute shuffled-label cPCA control")
    U_shuffled_cpca = compute_shuffled_cpca(
        H_pos, H_neg, selected_layers, cpca_fn,
        dom_vectors, layer_scores, v_truth, r_final,
    )
    if U_shuffled_cpca is not None:
        save_shuffled_subspace(U_shuffled_cpca, model_tag, source_tag,
                               r_final, vectors_dir)
    _finish_step("9_shuffled_cpca", t)

    total_elapsed = time.time() - source_start
    diagnostics = {
        "collection": collection_diag,
        "freeze": {
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "model_training": bool(model.training),
        },
        "probe": {
            "layer_scores": {str(L): float(s) for L, s in layer_scores.items()},
            "gate_threshold": PROBE_GATE,
            "layers_passing": sorted(int(L) for L, s in layer_scores.items() if s > PROBE_GATE),
        },
        "dom": {
            "best_layer": int(best_layer),
            "v_truth_norm": float(v_truth.norm().item()),
            "per_layer_cosines_to_best": dom_cosines,
        },
        "shuffled_dom": shuffled_dom_stats,
        "layer_selection": layer_selection_diag,
        "cpca_sweep": cpca_sweep_diag,
        "subspace_merge": {
            "r_final": int(r_final),
            "U_shape": _tensor_shape(U_truth),
            "layer_weights": {str(L): float(w) for L, w in layer_weights.items()},
        },
        "method_comparison": {
            "dom_acc": float(method_accs.get("dom", 0.0)),
            "cpca_acc": float(method_accs.get("cpca", 0.0)),
            "gap": float(method_accs.get("cpca", 0.0) - method_accs.get("dom", 0.0)),
            "winner": winner,
        },
        "shuffled_cpca": {"U_shape": _tensor_shape(U_shuffled_cpca)},
        "step_times_s": {**step_times, "total": total_elapsed},
    }
    _save_diagnostics(vectors_dir, source_tag, diagnostics)

    print(f"\n{'=' * 72}")
    print(f"Phase 2 source complete: {model_tag} | source={source_tag}")
    print(f"  best_layer={best_layer}  probe_acc={layer_scores.get(best_layer, 0.0):.3f}")
    print(f"  winner={winner}  method_accs={method_accs}")
    print(f"  v_truth_shape={_tensor_shape(v_truth)}  U_truth_shape={_tensor_shape(U_truth)}")
    print(f"  shuffled_dom_shape={_tensor_shape(v_shuffled)}  shuffled_cpca_shape={_tensor_shape(U_shuffled_cpca)}")
    print(f"  total_time={total_elapsed:.1f}s")
    for key, seconds in step_times.items():
        print(f"    {key}: {seconds:.1f}s")
    print(f"{'=' * 72}")

    return {
        'v_truth':         v_truth,
        'U_truth':         U_truth,
        'v_shuffled':      v_shuffled,
        'U_shuffled_cpca': U_shuffled_cpca,
        'best_layer':      best_layer,
        'layer_scores':    layer_scores,
        'selected_layers': selected_layers,
        'sweep_meta':      sweep_meta,
        'winner':          winner,
        'method_accs':     method_accs,
        'dom_vectors':     dom_vectors,
        'subspaces':       subspaces,
        'diagnostics':     diagnostics,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_best_ccot_latent_tokens(results_dir: str, model_tag: str) -> int:
    """
    Read Phase 1 latent-token results and return the CCoT latent count that
    maximizes validation accuracy. Defaults to 4 if no Phase 1 result exists.
    """
    best_path = os.path.join(results_dir, 'phase1_best_latent.json')
    if os.path.exists(best_path):
        with open(best_path) as f:
            best = json.load(f)
        n_latents = int(best.get('latent_tokens') or 4)
        print(
            f"Best CCoT latent budget for {model_tag}: L={n_latents} "
            f"(val acc = {best.get('accuracy', 0.0):.3f})"
        )
        return n_latents

    sweep_path = os.path.join(results_dir, 'phase1_latent_sweep.json')
    if not os.path.exists(sweep_path):
        print(f"Phase 1 latent results missing in {results_dir} — defaulting to L=4")
        return 4

    with open(sweep_path) as f:
        records = json.load(f)

    candidates = [
        r for r in records
        if str(r.get('condition', '')).startswith('ccot_L') and r.get('latent_tokens')
    ]
    if not candidates:
        print(f"No latent-token CCoT records found in {sweep_path} — defaulting to L=4")
        return 4

    best = max(candidates, key=lambda r: (r.get('accuracy', 0.0), int(r['latent_tokens'])))
    n_latents = int(best['latent_tokens'])
    print(
        f"Best CCoT latent budget for {model_tag}: L={n_latents} "
        f"(val acc = {best.get('accuracy', 0.0):.3f})"
    )
    return n_latents


def run_phase2_all_sources(
    model_tag: str,
    base_model_id: str,
    checkpoints_dir: str,
    D_steer: list,
    device: str,
    vectors_dir: str,
    results_dir: str,
) -> dict:
    """
    Load Source A (best CCoT checkpoint) and Source B (CoT checkpoint),
    run Phase 2 extraction for both, save vectors, and write phase2_meta.json.
    """
    phase_start = time.time()
    cfg = get_model_config(model_tag)
    best_latent_tokens = pick_best_ccot_latent_tokens(results_dir, model_tag)
    ccot_ckpt = os.path.join(checkpoints_dir, f'ccot_L{best_latent_tokens}')

    results: dict = {}

    # ── Source A: CCoT fine-tuned model ──────────────────────────────────────
    print(f"\n{'#' * 72}")
    print(f"SOURCE A/2: CCoT latent checkpoint | model={model_tag} | L={best_latent_tokens}")
    print(f"{'#' * 72}")
    print(f"\nLoading Source A  CCoT L={best_latent_tokens}: {ccot_ckpt}")
    ccot_model, tok_a = load_ccot_frozen(base_model_id, ccot_ckpt, device)

    ccot_prompt_fn = (
        lambda item, n=best_latent_tokens:
        f"{item['question']}\n<|start-latent|>{'<|latent|>' * n}<|end-latent|>\n"
    )
    results['ccot'] = run_phase2_source(
        ccot_model, tok_a, D_steer, model_tag,
        source_tag='ccot',
        boundary_idx_fn=find_boundary_idx_ccot,
        device=device,
        vectors_dir=vectors_dir,
        prompt_fn=ccot_prompt_fn,
        **cfg,
    )
    del ccot_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Source B: CoT fine-tuned model (Phase 1, Stage 1 checkpoint) ─────────
    # Using load_finetuned from phase1 (loads LoRA adapter), then freeze
    print(f"\n{'#' * 72}")
    print(f"SOURCE B/2: CoT checkpoint | model={model_tag}")
    print(f"{'#' * 72}")
    from phase1.inference import load_finetuned
    cot_ckpt = os.path.join(checkpoints_dir, 'cot')
    print(f"\nLoading Source B  CoT checkpoint: {cot_ckpt}")
    cot_model, tok_b = load_finetuned(cot_ckpt, device)
    for param in cot_model.parameters():
        param.requires_grad = False
    cot_model.eval()

    cot_prompt_fn = (
        lambda item: f"{item['question']}\n<|start-latent|><|latent|><|latent|><|end-latent|>\n"
    )
    results['base'] = run_phase2_source(
        cot_model, tok_b, D_steer, model_tag,
        source_tag='base',
        boundary_idx_fn=find_boundary_idx_base,
        device=device,
        vectors_dir=vectors_dir,
        prompt_fn=cot_prompt_fn,
        **cfg,
    )
    del cot_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Cross-source DoM alignment ────────────────────────────────────────────
    ccot_res = results.get('ccot', {})
    base_res = results.get('base', {})
    v_ccot   = ccot_res.get('v_truth')
    v_base   = base_res.get('v_truth')
    cross_cos = None
    if v_ccot is not None and v_base is not None:
        cross_cos = report_cross_source_alignment(
            v_ccot, v_base,
            best_L_a=ccot_res.get('best_layer', -1),
            best_L_b=base_res.get('best_layer', -1),
        )

    # ── Select best source × method ───────────────────────────────────────────
    print("\n--- Source × Method Selection ---")
    best_source, best_method, best_acc = select_best_source_method(
        ccot_res, base_res
    )

    # ── Save metadata ─────────────────────────────────────────────────────────
    def _pick_layer_star(result: dict) -> int:
        ls  = result.get('layer_scores', {})
        sel = result.get('selected_layers', [])
        candidates = [L for L in sel if L in ls]
        pool = candidates if candidates else list(ls.keys())
        return max(pool, key=ls.get) if pool else 0

    ccot_ls = ccot_res.get('layer_scores', {})
    base_ls = base_res.get('layer_scores', {})

    meta = {
        'model_tag':              model_tag,
        'best_ccot_latent_tokens': best_latent_tokens,
        'best_ccot_condition':    f'ccot_L{best_latent_tokens}',
        # Per-source winner (dom vs cpca)
        'ccot_winner_method':     ccot_res.get('winner'),
        'base_winner_method':     base_res.get('winner'),
        'ccot_method_accs':       ccot_res.get('method_accs', {}),
        'base_method_accs':       base_res.get('method_accs', {}),
        # Overall winner across sources and methods
        'best_source':            best_source,
        'best_method':            best_method,
        'best_probe_acc':         best_acc,
        # Layer info
        'ccot_best_layer':        ccot_res.get('best_layer', _pick_layer_star(ccot_res)),
        'base_best_layer':        base_res.get('best_layer', _pick_layer_star(base_res)),
        'ccot_selected_layers':   ccot_res.get('selected_layers', []),
        'base_selected_layers':   base_res.get('selected_layers', []),
        'ccot_layer_scores':      {str(L): s for L, s in ccot_ls.items()},
        'base_layer_scores':      {str(L): s for L, s in base_ls.items()},
        'ccot_max_probe_score':   max(ccot_ls.values()) if ccot_ls else 0.0,
        'base_max_probe_score':   max(base_ls.values()) if base_ls else 0.0,
        'cross_source_cos':       cross_cos,
        'ccot_r_final':           cfg.get('r_final', 10),
        'phase2_models_frozen':   True,
        'ccot_trainable_params':  int(ccot_res.get('diagnostics', {}).get('freeze', {}).get('trainable_params', 0)),
        'base_trainable_params':  int(base_res.get('diagnostics', {}).get('freeze', {}).get('trainable_params', 0)),
    }
    os.makedirs(vectors_dir, exist_ok=True)
    meta_path = os.path.join(vectors_dir, 'phase2_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nPhase 2 metadata -> {meta_path}")

    total_elapsed = time.time() - phase_start
    print(f"\n{'=' * 72}")
    print(f"PHASE 2 COMPLETE | model={model_tag}")
    print(f"  total_time={total_elapsed:.1f}s")
    for source, res in (("ccot", ccot_res), ("base", base_res)):
        diag = res.get("diagnostics", {})
        method = res.get("winner")
        best_layer = res.get("best_layer")
        layer_scores = res.get("layer_scores", {})
        probe_acc = layer_scores.get(best_layer, 0.0) if best_layer is not None else 0.0
        print(f"  source={source}: best_layer={best_layer} probe_acc={probe_acc:.3f} winner={method}")
        print(f"    v_truth_shape={_tensor_shape(res.get('v_truth'))} U_truth_shape={_tensor_shape(res.get('U_truth'))}")
        step_times = diag.get("step_times_s", {})
        if step_times:
            timing = ", ".join(
                f"{k}={v:.1f}s" for k, v in step_times.items()
                if isinstance(v, (int, float))
            )
            print(f"    step_times: {timing}")
    print(f"  overall winner: source={best_source} method={best_method} probe_acc={best_acc:.3f}")
    print(f"{'=' * 72}")

    return results
