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
from phase2.probe import score_all_layers
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


def _step(n: int, total: int, label: str, t0: float) -> float:
    """Print a numbered step banner and return a new step-start timestamp."""
    elapsed = time.time() - t0
    bar = '═' * 60
    print(f"\n{bar}")
    print(f"  STEP {n}/{total}  {label}  (+{elapsed:.1f}s total)")
    print(bar)
    return time.time()


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
    Saves {source}_dom.pt, {source}_cpca_r{r_final}.pt, and
    {source}_diagnostics.json to vectors_dir.
    """
    N_STEPS = 9
    phase_start = time.time()
    step_times: dict = {}
    diag: dict = {
        'model_tag': model_tag,
        'source':    source_tag,
        'n_steer_questions': len(D_steer),
        'n_rollouts':        N,
    }

    header = f"Phase 2  [{model_tag}]  source={source_tag}"
    print(f"\n{'═' * max(len(header), 60)}")
    print(f"  {header}")
    print(f"  D_steer={len(D_steer)}  N_rollouts={N}  "
          f"r_final={r_final}  min_samples={min_samples}  "
          f"cpca_variant={cpca_variant}")
    print(f"{'═' * max(len(header), 60)}")

    # ── STEP 1: Collect hidden states ─────────────────────────────────────────
    t_step = _step(1, N_STEPS, 'Collecting hidden states', phase_start)
    H_pos, H_neg = collect_hidden_states(
        model, tokenizer, D_steer, N, device,
        boundary_idx_fn, source_tag,
        prompt_fn=prompt_fn,
        min_samples=min_samples,
    )
    step_times['collection'] = round(time.time() - t_step, 2)

    if not H_pos:
        print("No layers passed min_samples threshold — aborting this source.")
        return {}

    diag['collection'] = {
        'layers_included':    sorted(H_pos.keys()),
        'n_layers_included':  len(H_pos),
        'samples_per_layer':  {
            str(L): {'H_pos': H_pos[L].shape[0], 'H_neg': H_neg[L].shape[0],
                     'hidden_dim': H_pos[L].shape[1]}
            for L in sorted(H_pos.keys())
        },
    }

    # ── STEP 2: Layer probe scores ────────────────────────────────────────────
    t_step = _step(2, N_STEPS, 'Logistic probe — scoring all layers', phase_start)
    layer_scores = score_all_layers(H_pos, H_neg)
    step_times['probe'] = round(time.time() - t_step, 2)

    passing_gate = [L for L, s in layer_scores.items() if s > 0.65]
    diag['probe'] = {
        'gate_threshold':     0.65,
        'layer_scores':       {str(L): round(s, 4) for L, s in sorted(layer_scores.items())},
        'best_layer':         max(layer_scores, key=layer_scores.get),
        'best_score':         round(max(layer_scores.values()), 4),
        'layers_passing_gate': passing_gate,
        'n_passing':           len(passing_gate),
    }

    # ── STEP 3: Method A — best-layer DoM ────────────────────────────────────
    t_step = _step(3, N_STEPS, 'Method A — Difference of Means (DoM)', phase_start)
    dom_vectors         = compute_per_layer_dom(H_pos, H_neg)
    v_truth, best_layer = compute_best_layer_dom(dom_vectors, layer_scores)
    step_times['dom'] = round(time.time() - t_step, 2)

    per_layer_cos = {
        str(L): round(torch.dot(dom_vectors[L], v_truth).item(), 4)
        for L in sorted(dom_vectors.keys())
    }
    diag['dom'] = {
        'best_layer':          best_layer,
        'best_probe_acc':      round(layer_scores.get(best_layer, float('nan')), 4),
        'v_truth_norm':        round(v_truth.norm().item(), 6),
        'v_truth_shape':       list(v_truth.shape),
        'per_layer_cos_to_best': per_layer_cos,
    }

    # ── STEP 4: Control — shuffled-label DoM ─────────────────────────────────
    t_step = _step(4, N_STEPS, 'Control — Shuffled-Label DoM', phase_start)
    v_shuffled, shuf_dom_stats = compute_shuffled_dom(
        H_pos, H_neg, best_layer, v_truth
    )
    save_shuffled_vector(v_shuffled, model_tag, source_tag, vectors_dir,
                         best_layer=best_layer)
    step_times['shuffled_dom'] = round(time.time() - t_step, 2)
    diag['shuffled_dom'] = shuf_dom_stats

    # ── STEP 5: cPCA layer selection ──────────────────────────────────────────
    t_step = _step(5, N_STEPS,
                   f'cPCA layer selection (threshold_multiplier={threshold_multiplier})',
                   phase_start)
    cpca_fn = _CPCA_FN_MAP.get(cpca_variant, cpca_full)
    selected_layers = select_layers(layer_scores, multiplier=threshold_multiplier)
    step_times['layer_selection'] = round(time.time() - t_step, 2)

    scores_arr = [layer_scores[L] for L in sorted(layer_scores.keys())]
    import statistics as _stats
    diag['layer_selection'] = {
        'cpca_variant':     cpca_variant,
        'mean_probe_score': round(_stats.mean(scores_arr), 4),
        'std_probe_score':  round(_stats.stdev(scores_arr) if len(scores_arr) > 1 else 0.0, 4),
        'selected_layers':  selected_layers,
        'n_selected':       len(selected_layers),
    }

    # ── STEP 6: Method B — cPCA sweep ────────────────────────────────────────
    t_step = _step(6, N_STEPS,
                   f'Method B — cPCA sweep  ({len(selected_layers)} layers × 4k × 3β)',
                   phase_start)
    cpca_results = run_cpca_sweep(H_pos, H_neg, selected_layers, cpca_fn)
    subspaces  = {L: (U, lam) for L, (U, lam, _, _, _) in cpca_results.items()}
    sweep_meta = {L: {'k': k, 'beta': b}
                  for L, (_, _, k, b, _) in cpca_results.items()}
    step_times['cpca_sweep'] = round(time.time() - t_step, 2)

    diag['cpca_sweep'] = {
        str(L): {
            'best_k':    k,
            'best_beta': b,
            'best_acc':  round(acc, 4),
            'H_pos':     H_pos[L].shape[0],
            'H_neg':     H_neg[L].shape[0],
        }
        for L, (_, _, k, b, acc) in cpca_results.items()
    }

    if not subspaces:
        print("No subspaces computed — saving DoM only, skipping Method B.")
        save_dom_vector(v_truth, model_tag, source_tag, vectors_dir,
                        best_layer=best_layer)
        diag['winner'] = 'dom'
        diag['step_times_s'] = step_times
        _save_diagnostics(diag, vectors_dir, source_tag)
        return {
            'v_truth':    v_truth,    'U_truth':       None,
            'v_shuffled': v_shuffled, 'best_layer':    best_layer,
            'layer_scores':    layer_scores,
            'selected_layers': selected_layers, 'sweep_meta': {},
            'winner':     'dom', 'dom_vectors': dom_vectors, 'subspaces': {},
            'method_accs': {'dom': layer_scores.get(best_layer, 0.0), 'cpca': 0.0},
        }

    # ── STEP 7: Weighted subspace merge ───────────────────────────────────────
    t_step = _step(7, N_STEPS,
                   f'Weighted subspace merge  r_final={r_final}',
                   phase_start)
    U_truth, layer_weights = weighted_subspace_merge(
        subspaces, layer_scores, dom_vectors, v_truth, r_final
    )
    save_dom_vector(v_truth, model_tag, source_tag, vectors_dir,
                    best_layer=best_layer)
    save_subspace(U_truth, selected_layers, model_tag, source_tag,
                  r_final, beta, vectors_dir,
                  layer_scores=layer_scores, sweep_meta=sweep_meta)
    step_times['subspace_merge'] = round(time.time() - t_step, 2)

    diag['subspace_merge'] = {
        'r_final':       r_final,
        'U_shape':       list(U_truth.shape),
        'layer_weights': {str(L): round(w, 6) for L, w in layer_weights.items()},
    }

    # ── STEP 8: Method comparison (DoM vs cPCA) ───────────────────────────────
    t_step = _step(8, N_STEPS, 'Method comparison — DoM vs cPCA', phase_start)
    winner, method_accs = compare_methods(
        H_pos, H_neg, v_truth, U_truth, selected_layers
    )
    step_times['method_comparison'] = round(time.time() - t_step, 2)

    diag['method_comparison'] = {
        'dom_acc':  round(method_accs.get('dom',  0.0), 4),
        'cpca_acc': round(method_accs.get('cpca', 0.0), 4),
        'gap':      round(method_accs.get('cpca', 0.0) - method_accs.get('dom', 0.0), 4),
        'winner':   winner,
    }

    # ── STEP 9: Control — shuffled-label cPCA ────────────────────────────────
    t_step = _step(9, N_STEPS, 'Control — Shuffled-Label cPCA', phase_start)
    U_shuffled_cpca = compute_shuffled_cpca(
        H_pos, H_neg, selected_layers, cpca_fn,
        dom_vectors, layer_scores, v_truth, r_final,
    )
    if U_shuffled_cpca is not None:
        save_shuffled_subspace(U_shuffled_cpca, model_tag, source_tag,
                               r_final, vectors_dir)
        diag['shuffled_cpca'] = {'U_shape': list(U_shuffled_cpca.shape)}
    else:
        diag['shuffled_cpca'] = None
    step_times['shuffled_cpca'] = round(time.time() - t_step, 2)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - phase_start, 2)
    step_times['total'] = total_elapsed
    diag['step_times_s'] = step_times
    diag['winner'] = winner

    print(f"\n{'═' * 60}")
    print(f"  Phase 2 complete  [{model_tag}]  source={source_tag}")
    print(f"  Best layer : L={best_layer}  probe_acc={layer_scores.get(best_layer, 0):.3f}")
    print(f"  DoM acc    : {method_accs.get('dom', 0):.3f}")
    print(f"  cPCA acc   : {method_accs.get('cpca', 0):.3f}")
    print(f"  Winner     : {winner}")
    print(f"  v_truth    : shape={list(v_truth.shape)}  norm={v_truth.norm().item():.4f}")
    print(f"  U_truth    : shape={list(U_truth.shape)}")
    print(f"  Total time : {total_elapsed:.1f}s")
    print(f"  Step times : " +
          "  ".join(f"{k}={v}s" for k, v in step_times.items() if k != 'total'))
    print(f"{'═' * 60}")

    _save_diagnostics(diag, vectors_dir, source_tag)

    return {
        'v_truth':         v_truth,
        'U_truth':         U_truth,
        'v_shuffled':      v_shuffled,
        'best_layer':      best_layer,
        'layer_scores':    layer_scores,
        'selected_layers': selected_layers,
        'sweep_meta':      sweep_meta,
        'winner':          winner,
        'method_accs':     method_accs,
        'dom_vectors':     dom_vectors,
        'subspaces':       subspaces,
    }


def _save_diagnostics(diag: dict, vectors_dir: str, source_tag: str) -> None:
    os.makedirs(vectors_dir, exist_ok=True)
    path = os.path.join(vectors_dir, f'{source_tag}_diagnostics.json')
    with open(path, 'w') as f:
        json.dump(diag, f, indent=2)
    print(f"\n  Diagnostics -> {path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_best_ccot_ratio(results_dir: str, model_tag: str) -> int:
    """
    Read phase1_val.json and return the CCoT ratio (integer, e.g. 7 for R=0.7)
    that maximises mechanism gain (CCoT acc − Trimmed CoT acc) on D_val.
    Defaults to 7 if the file is missing or no match is found.
    """
    path = os.path.join(results_dir, 'phase1_val.json')
    if not os.path.exists(path):
        print(f"phase1_val.json not found at {path} — defaulting to R=7")
        return 7

    with open(path) as f:
        records = json.load(f)

    res_map = {r['condition']: r for r in records}
    best_ratio_int, best_gain = 7, float('-inf')

    for ri in [5, 6, 7, 8, 9]:
        k_ccot    = f'ccot_R{ri}'
        k_trimmed = f'trimmed_cot_R{ri}'
        if k_ccot in res_map and k_trimmed in res_map:
            gain = res_map[k_ccot]['accuracy'] - res_map[k_trimmed]['accuracy']
            if gain > best_gain:
                best_gain, best_ratio_int = gain, ri

    print(f"Best CCoT ratio for {model_tag}: R=0.{best_ratio_int}  "
          f"(mechanism gain = {best_gain:.3f})")
    return best_ratio_int


def run_phase2_all_sources(
    model_tag: str,
    base_model_id: str,
    checkpoints_dir: str,
    D_steer: list,
    device: str,
    vectors_dir: str,
    results_dir: str,
    run_source_b: bool = False,
) -> dict:
    """
    Load Source A (best CCoT checkpoint) and Source B (CoT checkpoint),
    run Phase 2 extraction for both, save vectors, and write phase2_meta.json.
    """
    run_start = time.time()
    cfg = get_model_config(model_tag)
    ratio_int = pick_best_ccot_ratio(results_dir, model_tag)
    ccot_ckpt = os.path.join(checkpoints_dir, f'ccot_R{ratio_int}')

    print(f"\n{'█' * 64}")
    print(f"  PHASE 2 START  [{model_tag}]")
    print(f"  D_steer={len(D_steer)}  device={device}")
    print(f"  Source A: CCoT R=0.{ratio_int}  ckpt={ccot_ckpt}")
    print(f"  Source B: CoT           ckpt={os.path.join(checkpoints_dir, 'cot')}")
    print(f"  Vectors -> {vectors_dir}")
    print(f"{'█' * 64}")

    results: dict = {}

    # ── Source A: CCoT fine-tuned model ──────────────────────────────────────
    print(f"\n{'▓' * 64}")
    print(f"  SOURCE A  CCoT R=0.{ratio_int}")
    print(f"{'▓' * 64}")
    print(f"\nLoading Source A  CCoT R=0.{ratio_int}: {ccot_ckpt}")
    ccot_model, tok_a = load_ccot_frozen(base_model_id, ccot_ckpt, device)

    ccot_prompt_fn = (
        lambda item, ri=ratio_int:
        f"Question: {item['question']}\n\n[compress:0.{ri}]\n"
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
    if run_source_b:
        from phase1.inference import load_finetuned
        cot_ckpt = os.path.join(checkpoints_dir, 'cot')

        print(f"\n{'▓' * 64}")
        print(f"  SOURCE B  CoT")
        print(f"{'▓' * 64}")
        print(f"\nLoading Source B  CoT checkpoint: {cot_ckpt}")
        cot_model, tok_b = load_finetuned(cot_ckpt, device)
        for param in cot_model.parameters():
            param.requires_grad = False
        cot_model.eval()

        cot_prompt_fn = (
            lambda item: f"Question: {item['question']}\n\nReasoning:"
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
    else:
        print(f"\n  Source B (CoT) skipped  [run_source_b=False]")

    # ── Cross-source DoM alignment ────────────────────────────────────────────
    ccot_res = results.get('ccot', {})
    base_res = results.get('base', {})
    v_ccot   = ccot_res.get('v_truth')
    v_base   = base_res.get('v_truth')
    cross_cos = None
    if v_ccot is not None and v_base is not None:
        print(f"\n{'═' * 60}")
        print(f"  Cross-Source Alignment")
        print(f"{'═' * 60}")
        cross_cos = report_cross_source_alignment(
            v_ccot, v_base,
            best_L_a=ccot_res.get('best_layer', -1),
            best_L_b=base_res.get('best_layer', -1),
        )

    # ── Select best source × method ───────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Source × Method Selection")
    print(f"{'═' * 60}")
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
        'best_ccot_ratio':        ratio_int,
        'ccot_winner_method':     ccot_res.get('winner'),
        'base_winner_method':     base_res.get('winner'),
        'ccot_method_accs':       ccot_res.get('method_accs', {}),
        'base_method_accs':       base_res.get('method_accs', {}),
        'best_source':            best_source,
        'best_method':            best_method,
        'best_probe_acc':         best_acc,
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
    }
    os.makedirs(vectors_dir, exist_ok=True)
    meta_path = os.path.join(vectors_dir, 'phase2_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    total_elapsed = round(time.time() - run_start, 2)
    print(f"\n{'█' * 64}")
    print(f"  PHASE 2 COMPLETE  [{model_tag}]")
    print(f"  Best source    : {best_source}")
    print(f"  Best method    : {best_method}")
    print(f"  Best probe acc : {best_acc:.3f}")
    print(f"  CCoT best L    : {meta['ccot_best_layer']}  "
          f"(max probe={meta['ccot_max_probe_score']:.3f})")
    print(f"  Base best L    : {meta['base_best_layer']}  "
          f"(max probe={meta['base_max_probe_score']:.3f})")
    if cross_cos is not None:
        print(f"  Cross-source cos: {cross_cos:+.4f}")
    print(f"  Total time     : {total_elapsed:.1f}s")
    print(f"  Saved files:")
    for fname in [
        'ccot_dom.pt', 'base_dom.pt',
        f'ccot_cpca_r{cfg.get("r_final", 10)}.pt',
        f'base_cpca_r{cfg.get("r_final", 10)}.pt',
        'ccot_shuffled_dom.pt', 'base_shuffled_dom.pt',
        f'ccot_shuffled_cpca_r{cfg.get("r_final", 10)}.pt',
        f'base_shuffled_cpca_r{cfg.get("r_final", 10)}.pt',
        'ccot_diagnostics.json', 'base_diagnostics.json',
        'phase2_meta.json',
    ]:
        p = os.path.join(vectors_dir, fname)
        mark = '✓' if os.path.exists(p) else '—'
        print(f"    {mark}  {p}")
    print(f"{'█' * 64}")

    return results
