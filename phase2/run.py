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
    compute_global_dom,
    save_dom_vector,
)
from phase2.cpca import (
    select_layers,
    cpca_full,
    cpca_shrunk,
    cpca_randomized,
    analyze_eigenspectrum,
    weighted_subspace_merge,
    save_subspace,
)
from phase2.compare import compare_methods

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
    N: int = 15,
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

    # ── Method A: multi-layer DoM ─────────────────────────────────────────────
    print("\n--- Method A: Multi-Layer DoM ---")
    dom_vectors = compute_per_layer_dom(H_pos, H_neg)
    v_truth     = compute_global_dom(dom_vectors, layer_scores)

    # ── Method B: threshold cPCA ──────────────────────────────────────────────
    print("\n--- Method B: Threshold cPCA ---")
    selected_layers = select_layers(layer_scores, multiplier=threshold_multiplier)
    cpca_fn = _CPCA_FN_MAP.get(cpca_variant, cpca_full)

    subspaces: dict = {}
    for L in selected_layers:
        if L not in H_pos:
            continue
        print(f"\n  cPCA at layer {L}...")
        try:
            U_L, lam_L = cpca_fn(H_pos[L], H_neg[L], r=r_per_layer, beta=beta)
            analyze_eigenspectrum(lam_L, L)
            subspaces[L] = (U_L, lam_L)
        except Exception as exc:
            print(f"  Layer {L}: cPCA failed ({exc}), skipping.")

    if not subspaces:
        print("No subspaces computed — saving DoM only, skipping Method B.")
        save_dom_vector(v_truth, model_tag, source_tag, vectors_dir)
        return {
            'v_truth': v_truth, 'U_truth': None,
            'layer_scores': layer_scores, 'selected_layers': selected_layers,
            'winner': 'dom', 'dom_vectors': dom_vectors, 'subspaces': {},
        }

    print("\n--- Weighted Subspace Merge ---")
    U_truth = weighted_subspace_merge(
        subspaces, layer_scores, dom_vectors, v_truth, r_final
    )

    # ── Method comparison ─────────────────────────────────────────────────────
    print("\n--- Method Comparison ---")
    winner = compare_methods(H_pos, H_neg, v_truth, U_truth, selected_layers)

    save_dom_vector(v_truth, model_tag, source_tag, vectors_dir)
    save_subspace(U_truth, selected_layers, model_tag, source_tag,
                  r_final, beta, vectors_dir)

    return {
        'v_truth':         v_truth,
        'U_truth':         U_truth,
        'layer_scores':    layer_scores,
        'selected_layers': selected_layers,
        'winner':          winner,
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

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        'model_tag':          model_tag,
        'best_ccot_ratio':    ratio_int,
        'ccot_winner_method': results.get('ccot', {}).get('winner'),
        'base_winner_method': results.get('base', {}).get('winner'),
    }
    os.makedirs(vectors_dir, exist_ok=True)
    meta_path = os.path.join(vectors_dir, 'phase2_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nPhase 2 metadata -> {meta_path}")

    return results
