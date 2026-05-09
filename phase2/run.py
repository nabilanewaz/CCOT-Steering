"""Full Phase 2 runner: collect hidden states → probe → DoM → cPCA → compare."""
import json
import os

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
)
from phase2.compare import compare_methods, select_best_source_method

_CPCA_FN_MAP = {
    'full':       cpca_full,
    'shrunk':     cpca_shrunk,
    'randomized': cpca_randomized,
}


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
    header = f"Phase 2: {model_tag} | source={source_tag}"
    print(f"\n{'='*len(header)}\n{header}\n{'='*len(header)}")

    H_pos, H_neg = collect_hidden_states(
        model, tokenizer, D_steer, N, device,
        boundary_idx_fn, source_tag,
        prompt_fn=prompt_fn,
        min_samples=min_samples,
    )
    if not H_pos:
        print("No layers passed min_samples threshold — aborting this source.")
        return {}

    # ── Layer probe scores ────────────────────────────────────────────────────
    print("\n--- Layer Probe Scores ---")
    layer_scores = score_all_layers(H_pos, H_neg)

    # ── Method A: best-layer DoM ──────────────────────────────────────────────
    print("\n--- Method A: Best-Layer DoM ---")
    dom_vectors         = compute_per_layer_dom(H_pos, H_neg)
    v_truth, best_layer = compute_best_layer_dom(dom_vectors, layer_scores)

    # ── Control: shuffled-label DoM ───────────────────────────────────────────
    print("\n--- Control: Shuffled-Label DoM ---")
    v_shuffled = compute_shuffled_dom(H_pos, H_neg, best_layer, v_truth)
    save_shuffled_vector(v_shuffled, model_tag, source_tag, vectors_dir,
                         best_layer=best_layer)

    # ── Method B: cPCA sweep (k ∈ {1,2,5,10}, β ∈ {0.3,0.5,0.7}) ────────────
    print("\n--- Method B: cPCA Sweep ---")
    selected_layers = select_layers(layer_scores, multiplier=threshold_multiplier)
    cpca_fn = _CPCA_FN_MAP.get(cpca_variant, cpca_full)

    cpca_results = run_cpca_sweep(H_pos, H_neg, selected_layers, cpca_fn)
    subspaces  = {L: (U, lam) for L, (U, lam, _, _) in cpca_results.items()}
    sweep_meta = {L: {'k': k, 'beta': b}
                  for L, (_, _, k, b) in cpca_results.items()}

    if not subspaces:
        print("No subspaces computed — saving DoM only, skipping Method B.")
        save_dom_vector(v_truth, model_tag, source_tag, vectors_dir,
                        best_layer=best_layer)
        return {
            'v_truth':    v_truth,    'U_truth':       None,
            'v_shuffled': v_shuffled, 'best_layer':    best_layer,
            'layer_scores':    layer_scores,
            'selected_layers': selected_layers, 'sweep_meta': {},
            'winner':     'dom', 'dom_vectors': dom_vectors, 'subspaces': {},
            'method_accs': {'dom': layer_scores.get(best_layer, 0.0), 'cpca': 0.0},
        }

    print("\n--- Weighted Subspace Merge ---")
    U_truth = weighted_subspace_merge(
        subspaces, layer_scores, dom_vectors, v_truth, r_final
    )

    # ── Method comparison ─────────────────────────────────────────────────────
    print("\n--- Method Comparison ---")
    winner, method_accs = compare_methods(
        H_pos, H_neg, v_truth, U_truth, selected_layers
    )

    save_dom_vector(v_truth, model_tag, source_tag, vectors_dir,
                    best_layer=best_layer)
    save_subspace(U_truth, selected_layers, model_tag, source_tag,
                  r_final, beta, vectors_dir,
                  layer_scores=layer_scores, sweep_meta=sweep_meta)

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
) -> dict:
    """
    Load Source A (best CCoT checkpoint) and Source B (CoT checkpoint),
    run Phase 2 extraction for both, save vectors, and write phase2_meta.json.
    """
    cfg = get_model_config(model_tag)
    ratio_int = pick_best_ccot_ratio(results_dir, model_tag)
    ccot_ckpt = os.path.join(checkpoints_dir, f'ccot_R{ratio_int}')

    results: dict = {}

    # ── Source A: CCoT fine-tuned model ──────────────────────────────────────
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
    # Using load_finetuned from phase1 (loads LoRA adapter), then freeze
    from phase1.inference import load_finetuned
    cot_ckpt = os.path.join(checkpoints_dir, 'cot')
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
        'best_ccot_ratio':        ratio_int,
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
    }
    os.makedirs(vectors_dir, exist_ok=True)
    meta_path = os.path.join(vectors_dir, 'phase2_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nPhase 2 metadata -> {meta_path}")

    return results
