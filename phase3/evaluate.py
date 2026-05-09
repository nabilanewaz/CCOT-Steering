"""Phase 3 evaluation: full condition grid, alpha tuning, flip-rate computation.

Conditions evaluated per backbone (spec §3.2):
  Fixed (×1):      no_cot, full_cot
  Per ratio (×5):  ccot_R*, trimmed_R*, noise_{src}, dom_{src}, cpca_{src}, trimmed_dom
  → ~52 evaluations per backbone on D_val.
"""
import glob as _glob
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from phase1.inference import (
    compute_per_example_budgets,
    extract_answer,
    load_finetuned,
    normalize_answer,
    run_cot,
    run_no_cot,
    run_trimmed_cot,
)
from phase2.loaders import find_boundary_idx_base, find_boundary_idx_ccot
from phase3.alpha import tune_alpha
from phase3.hooks import (
    get_injection_layer,
    make_cpca_hook,
    make_dom_hook,
    make_noise_hook,
    run_with_hook,
)

RATIOS   = [0.5, 0.6, 0.7, 0.8, 0.9]
SOURCES  = ('ccot', 'base')


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class ConditionResult:
    condition:         str
    model_tag:         str
    ratio:             Optional[float]
    vector_source:     Optional[str]    # 'ccot' | 'base' | None
    vector_method:     Optional[str]    # 'dom' | 'cpca' | 'noise' | None
    alpha:             Optional[float]
    accuracy:          float
    flip_rate:         float            # wrong→right vs CCoT baseline at same R
    reasoning_tokens:  float
    actual_ratio:      float            # mean_reasoning_tokens / full_cot_mean_tokens
    latency_sec:       float
    answer_found_rate: float
    n_examples:        int = 0


# ── Small helpers ──────────────────────────────────────────────────────────────

def _score(text: str, gold: str) -> bool:
    pred = extract_answer(text)
    return normalize_answer(pred) == normalize_answer(gold) if pred else False


def _boundary_from_prompt(tokenizer, prompt: str) -> int:
    return max(0, len(tokenizer.encode(prompt, add_special_tokens=False)) - 1)


def _load_meta(vectors_dir: str) -> dict:
    path = os.path.join(vectors_dir, 'phase2_meta.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"phase2_meta.json not found: {path}")
    with open(path) as f:
        return json.load(f)


def _load_vector(vectors_dir: str, source: str, method: str,
                 r_final: Optional[int] = None) -> torch.Tensor:
    if method == 'dom':
        return torch.load(
            os.path.join(vectors_dir, f'{source}_dom.pt'), map_location='cpu'
        )['v_truth']
    files = (
        [os.path.join(vectors_dir, f'{source}_cpca_r{r_final}.pt')]
        if r_final else
        sorted(_glob.glob(os.path.join(vectors_dir, f'{source}_cpca_r*.pt')))
    )
    if not files or not os.path.exists(files[-1]):
        raise FileNotFoundError(f"No cPCA vector for source={source} in {vectors_dir}")
    return torch.load(files[-1], map_location='cpu')['U_truth']


def _load_shuffled_vector(
    vectors_dir: str,
    source: str,
    method: str,
    r_final: Optional[int] = None,
) -> torch.Tensor:
    """Load shuffled-label control vector (v_shuffled or U_shuffled)."""
    if method == 'dom':
        path = os.path.join(vectors_dir, f'{source}_shuffled_dom.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Shuffled DoM missing: {path}")
        return torch.load(path, map_location='cpu')['v_shuffled']
    # cPCA
    if r_final:
        candidates = [os.path.join(vectors_dir, f'{source}_shuffled_cpca_r{r_final}.pt')]
    else:
        candidates = sorted(
            _glob.glob(os.path.join(vectors_dir, f'{source}_shuffled_cpca_r*.pt'))
        )
    if not candidates or not os.path.exists(candidates[-1]):
        raise FileNotFoundError(
            f"Shuffled cPCA missing for source={source} in {vectors_dir}"
        )
    return torch.load(candidates[-1], map_location='cpu')['U_shuffled']


def _alpha_path(vectors_dir: str, source: str) -> str:
    return os.path.join(vectors_dir, f'{source}_alpha_star.pt')


def _load_alpha(vectors_dir: str, source: str) -> float:
    return torch.load(_alpha_path(vectors_dir, source)).item()


# ── Per-example evaluation helpers ────────────────────────────────────────────

def _eval_one(
    model, tokenizer, item: dict, prompt: str,
    layer_star: Optional[int], hook_factory,
    device: str, max_new_tokens: int,
    boundary_fn=None,   # find_boundary_idx_ccot | find_boundary_idx_base | None
) -> tuple[bool, bool, int, float]:
    """
    Run one example (optionally with a hook).
    When hook_factory is given and boundary_fn is provided, probe-generates first
    to find the reasoning/answer boundary (spec §3.8), then steers the real pass.
    Returns (correct, answer_found, n_tok, latency_sec).
    """
    gold = item['answer'].split('####')[1].strip()
    t0   = time.time()

    if hook_factory is not None:
        if boundary_fn is not None:
            # Spec §3.8: probe-generate (no_grad) to locate the semantic boundary
            enc = tokenizer(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                probe_ids = model.generate(
                    **enc, do_sample=False, max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id,
                )
            try:
                b_idx = boundary_fn(probe_ids, tokenizer)
            except (ValueError, Exception):
                b_idx = max(0, enc['input_ids'].shape[1] - 1)
        else:
            b_idx = _boundary_from_prompt(tokenizer, prompt)

        hook_fn = hook_factory(b_idx)
        text    = run_with_hook(model, tokenizer, prompt, layer_star,
                                hook_fn, device, max_new_tokens)
    else:
        enc = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][enc['input_ids'].shape[1]:],
                                skip_special_tokens=True)

    lat   = time.time() - t0
    found = extract_answer(text) is not None
    ok    = _score(text, gold)
    n_tok = len(tokenizer.encode(text, add_special_tokens=False))
    return ok, found, n_tok, lat


def _build_result(
    condition, model_tag, ratio, vector_source, vector_method, alpha,
    correct_list, found_list, tokens_list, latencies,
    ccot_correct, full_cot_mean_tokens,
) -> ConditionResult:
    n   = len(correct_list)
    acc = sum(correct_list) / n if n else 0.0

    wrong_idx = [i for i, c in enumerate(ccot_correct) if not c] if ccot_correct else []
    flip_rate = (
        sum(correct_list[i] for i in wrong_idx) / len(wrong_idx)
        if wrong_idx else 0.0
    )

    mean_tok = sum(tokens_list) / len(tokens_list) if tokens_list else 0.0
    act_r    = mean_tok / full_cot_mean_tokens if full_cot_mean_tokens else 0.0

    return ConditionResult(
        condition=condition, model_tag=model_tag,
        ratio=ratio, vector_source=vector_source,
        vector_method=vector_method, alpha=alpha,
        accuracy=acc, flip_rate=flip_rate,
        reasoning_tokens=mean_tok,
        actual_ratio=act_r,
        latency_sec=sum(latencies) / len(latencies) if latencies else 0.0,
        answer_found_rate=sum(found_list) / n if n else 0.0,
        n_examples=n,
    )


# ── Alpha tuning (pre-run, per source) ────────────────────────────────────────

def _tune_and_save_alpha(
    model_tag: str,
    checkpoints_dir: str,
    D_val: list,
    vectors_dir: str,
    device: str,
    meta: dict,
    results_dir: str = None,
) -> None:
    """
    For each source:
      1. Run λ sweep on a 200-example D_val subset to select (λ_a, λ_m).
      2. Run full gradient-based α* tuning with the selected lambdas.
      3. Save alpha_star, training history JSON, and loss-curve PNG.
    Results are cached per source — rerun is skipped if alpha_star file exists.
    """
    best_ratio_int = meta['best_ccot_ratio']
    best_ratio_flt = best_ratio_int / 10.0

    source_ckpt = {
        'ccot': os.path.join(checkpoints_dir, f'ccot_R{best_ratio_int}'),
        'base': os.path.join(checkpoints_dir, 'cot'),
    }
    source_L_star = {
        'ccot': meta.get('ccot_layer_star'),
        'base': meta.get('base_layer_star'),
    }

    for source in SOURCES:
        out_path = _alpha_path(vectors_dir, source)
        if os.path.exists(out_path):
            print(f"[PH3] alpha_star for source={source} cached: {out_path}")
            continue

        v_dom  = _load_vector(vectors_dir, source, 'dom')
        L_star = source_L_star[source]
        if L_star is None:
            try:
                L_star = get_injection_layer(vectors_dir, source)
            except FileNotFoundError:
                L_star = meta.get('ccot_layer_star', 14)

        print(f"\n[PH3] Tuning alpha  source={source}  L*={L_star}")
        model, tok = load_finetuned(source_ckpt[source], device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        # ── Step 1: λ sweep on 200-example subset ────────────────────────────
        sweep_path = os.path.join(vectors_dir, f'{source}_lambda_sweep.json')
        if os.path.exists(sweep_path):
            with open(sweep_path) as fp:
                sweep_sel = json.load(fp)['selected']
            lambda_a = sweep_sel['lambda_a']
            lambda_m = sweep_sel['lambda_m']
            print(f"[PH3] λ sweep cached: λ_a={lambda_a}  λ_m={lambda_m}")
        else:
            from phase3.lambda_sweep import sweep_lambda_grid
            D_sub    = D_val[:min(200, len(D_val))]
            sweep_sel = sweep_lambda_grid(
                model, tok, D_sub, v_dom, L_star, device, model_tag,
                ratio=best_ratio_flt,
                out_path=sweep_path,
                max_epochs=2,
            )
            lambda_a = sweep_sel['lambda_a']
            lambda_m = sweep_sel['lambda_m']
            # Plot heatmap if results_dir provided
            if results_dir:
                try:
                    from phase3.plots import plot_lambda_sweep_heatmap
                    with open(sweep_path) as fp:
                        sweep_data = json.load(fp)
                    plot_lambda_sweep_heatmap(
                        sweep_data,
                        os.path.join(results_dir, f'{source}_lambda_heatmap.png'),
                    )
                except Exception as e:
                    print(f"  [plot] {e}")

        # ── Step 2: Full α* tuning with selected lambdas ──────────────────────
        alpha_star, history = tune_alpha(
            model, tok, D_val, v_dom, L_star, device,
            model_tag=model_tag, ratio=best_ratio_flt,
            lambda_a=lambda_a, lambda_m=lambda_m,
        )
        torch.save(alpha_star, out_path)
        print(f"  alpha_star={alpha_star.item():.4f}  -> {out_path}")

        # ── Step 3: Persist history and plots ─────────────────────────────────
        if results_dir:
            hist_path = os.path.join(results_dir, f'{source}_alpha_history.json')
            with open(hist_path, 'w') as fp:
                json.dump({
                    'model_tag': model_tag,
                    'source':    source,
                    'lambda_a':  lambda_a,
                    'lambda_m':  lambda_m,
                    'history':   history,
                }, fp, indent=2)
            try:
                from phase3.plots import plot_loss_curves
                plot_loss_curves(
                    history,
                    os.path.join(results_dir, f'{source}_loss_curves.png'),
                )
            except Exception as e:
                print(f"  [plot] {e}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_phase3_evaluation(
    model_tag: str,
    base_model_id: str,
    checkpoints_dir: str,
    D_val: list,
    vectors_dir: str,
    results_dir: str,
    device: str,
    max_new_tokens: int = 256,
) -> list[ConditionResult]:
    """
    Evaluate all Phase 3 conditions on D_val.
    Writes phase3_val.json, steered_val.json, alpha_diagnostic.json.
    """
    os.makedirs(results_dir, exist_ok=True)
    meta    = _load_meta(vectors_dir)
    r_final = meta.get('ccot_r_final', 10)

    # ── Pre-compute alpha per source ──────────────────────────────────────────
    _tune_and_save_alpha(model_tag, checkpoints_dir, D_val,
                         vectors_dir, device, meta, results_dir=results_dir)
    alphas = {s: _load_alpha(vectors_dir, s) for s in SOURCES}
    print(f"\n[PH3] Alpha stars: {alphas}")

    results: list[ConditionResult] = []

    # ── No CoT baseline (base model, no LoRA) ─────────────────────────────────
    print("\n[PH3] Evaluating: No CoT")
    from phase1.inference import load_base_frozen
    base_model, tok_base = load_base_frozen(base_model_id, device)
    c_list, f_list, tok_list, lat_list = [], [], [], []
    for item in D_val:
        prompt = f"Question: {item['question']}\n\nAnswer:"
        ok, fd, nt, lt = _eval_one(base_model, tok_base, item, prompt,
                                   None, None, device, 32)
        c_list.append(ok); f_list.append(fd)
        tok_list.append(0); lat_list.append(lt)
    results.append(_build_result(
        'no_cot', model_tag, None, None, None, None,
        c_list, f_list, tok_list, lat_list, None, 1.0,
    ))
    print(f"  no_cot: acc={results[-1].accuracy:.3f}")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── CoT model: load once, use for Full CoT + Trimmed CoT conditions ───────
    cot_ckpt  = os.path.join(checkpoints_dir, 'cot')
    cot_model, tok_cot = load_finetuned(cot_ckpt, device)
    for p in cot_model.parameters():
        p.requires_grad = False
    cot_model.eval()

    # Full CoT
    print("\n[PH3] Evaluating: Full CoT")
    c_list, f_list, tok_list, lat_list = [], [], [], []
    for item in D_val:
        t0 = time.time()
        pred, reasoning = run_cot(cot_model, tok_cot, item, device)
        lt = time.time() - t0
        gold = item['answer'].split('####')[1].strip()
        ok   = normalize_answer(pred) == normalize_answer(gold) if pred else False
        fd   = pred is not None
        nt   = len(tok_cot.encode(reasoning, add_special_tokens=False)) if reasoning else 0
        c_list.append(ok); f_list.append(fd)
        tok_list.append(nt); lat_list.append(lt)
    full_cot_mean_tokens = sum(tok_list) / max(len(tok_list), 1)
    results.append(_build_result(
        'full_cot', model_tag, None, None, None, None,
        c_list, f_list, tok_list, lat_list, None, full_cot_mean_tokens,
    ))
    print(f"  full_cot: acc={results[-1].accuracy:.3f}  "
          f"mean_tok={full_cot_mean_tokens:.1f}")

    # Pre-compute per-example budgets for all ratios (done while CoT is loaded)
    budgets_by_ratio: dict[float, list[int]] = {}
    for ratio in RATIOS:
        print(f"  [budget] computing per-example budgets for R={ratio}…")
        budgets_by_ratio[ratio] = compute_per_example_budgets(
            cot_model, tok_cot, D_val, device, ratio
        )

    # ── Per-ratio loop ─────────────────────────────────────────────────────────
    for ratio in RATIOS:
        ratio_int = int(ratio * 10)
        rtag      = f"R{ratio_int}"
        budgets   = budgets_by_ratio[ratio]
        print(f"\n{'='*55}\n[PH3] Ratio = {ratio}  ({rtag})\n{'='*55}")

        # Trimmed CoT (CoT model, same token budget as CCoT)
        print(f"  Evaluating: Trimmed CoT (R={ratio})")
        c_list, f_list, tok_list, lat_list = [], [], [], []
        for i, item in enumerate(D_val):
            t0   = time.time()
            pred, reasoning = run_trimmed_cot(cot_model, tok_cot, item,
                                              budgets[i], device)
            lt   = time.time() - t0
            gold = item['answer'].split('####')[1].strip()
            ok   = normalize_answer(pred) == normalize_answer(gold) if pred else False
            nt   = len(tok_cot.encode(reasoning or '', add_special_tokens=False))
            c_list.append(ok); f_list.append(pred is not None)
            tok_list.append(nt); lat_list.append(lt)
        results.append(_build_result(
            f'trimmed_R{ratio_int}', model_tag, ratio, None, None, None,
            c_list, f_list, tok_list, lat_list, None, full_cot_mean_tokens,
        ))
        print(f"    trimmed_R{ratio_int}: acc={results[-1].accuracy:.3f}")

        # Load CCoT model for this ratio
        ccot_ckpt  = os.path.join(checkpoints_dir, f'ccot_{rtag}')
        if not os.path.exists(os.path.join(ccot_ckpt, 'adapter_config.json')):
            print(f"  [SKIP] CCoT checkpoint missing: {ccot_ckpt}")
            continue
        ccot_model, tok_ccot = load_finetuned(ccot_ckpt, device)
        for p in ccot_model.parameters():
            p.requires_grad = False
        ccot_model.eval()
        ccot_prompt_fn = lambda item: (
            f"Question: {item['question']}\n\n[compress:{ratio}]\n"
        )

        # CCoT baseline (no steering) — collect per-example correct flags
        print(f"  Evaluating: CCoT (R={ratio})")
        c_list, f_list, tok_list, lat_list = [], [], [], []
        for item in D_val:
            ok, fd, nt, lt = _eval_one(
                ccot_model, tok_ccot, item, ccot_prompt_fn(item),
                None, None, device, max_new_tokens,
            )
            c_list.append(ok); f_list.append(fd)
            tok_list.append(nt); lat_list.append(lt)
        ccot_correct = list(c_list)    # used for flip rate below
        results.append(_build_result(
            f'ccot_{rtag}', model_tag, ratio, None, None, None,
            c_list, f_list, tok_list, lat_list, None, full_cot_mean_tokens,
        ))
        print(f"    ccot_{rtag}: acc={results[-1].accuracy:.3f}")

        # Steered conditions (both sources)
        for source in SOURCES:
            alpha = alphas[source]
            try:
                L_star = get_injection_layer(vectors_dir, source)
            except FileNotFoundError:
                L_star = meta.get(f'{source}_layer_star',
                                  meta.get('ccot_layer_star', 14))

            try:
                v_dom = _load_vector(vectors_dir, source, 'dom')
            except FileNotFoundError:
                print(f"  [SKIP] DoM vector missing for source={source}")
                continue

            try:
                U_cpca = _load_vector(vectors_dir, source, 'cpca', r_final)
                has_cpca = True
            except FileNotFoundError:
                U_cpca  = None
                has_cpca = False

            # Condition 5: Random Noise
            print(f"  Evaluating: Random Noise (R={ratio}, src={source})")
            c_list, f_list, tok_list, lat_list = [], [], [], []
            noise_fac = lambda b, a=alpha: make_noise_hook(b, a, device)
            for item in D_val:
                ok, fd, nt, lt = _eval_one(
                    ccot_model, tok_ccot, item, ccot_prompt_fn(item),
                    L_star, noise_fac, device, max_new_tokens,
                    boundary_fn=find_boundary_idx_ccot,
                )
                c_list.append(ok); f_list.append(fd)
                tok_list.append(nt); lat_list.append(lt)
            results.append(_build_result(
                f'noise_{rtag}_{source}', model_tag, ratio, source, 'noise', alpha,
                c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens,
            ))
            print(f"    noise_{rtag}_{source}: acc={results[-1].accuracy:.3f}  "
                  f"flip={results[-1].flip_rate:.3f}")

            # Condition 6: CCoT + DoM
            print(f"  Evaluating: CCoT + DoM (R={ratio}, src={source})")
            c_list, f_list, tok_list, lat_list = [], [], [], []
            dom_fac = lambda b, v=v_dom, a=alpha: make_dom_hook(b, v, a, device)
            for item in D_val:
                ok, fd, nt, lt = _eval_one(
                    ccot_model, tok_ccot, item, ccot_prompt_fn(item),
                    L_star, dom_fac, device, max_new_tokens,
                    boundary_fn=find_boundary_idx_ccot,
                )
                c_list.append(ok); f_list.append(fd)
                tok_list.append(nt); lat_list.append(lt)
            results.append(_build_result(
                f'dom_{rtag}_{source}', model_tag, ratio, source, 'dom', alpha,
                c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens,
            ))
            print(f"    dom_{rtag}_{source}: acc={results[-1].accuracy:.3f}  "
                  f"flip={results[-1].flip_rate:.3f}")

            # Condition 7: CCoT + cPCA
            if has_cpca:
                print(f"  Evaluating: CCoT + cPCA (R={ratio}, src={source})")
                c_list, f_list, tok_list, lat_list = [], [], [], []
                cpca_fac = lambda b, U=U_cpca, a=alpha: make_cpca_hook(b, U, a, device)
                for item in D_val:
                    ok, fd, nt, lt = _eval_one(
                        ccot_model, tok_ccot, item, ccot_prompt_fn(item),
                        L_star, cpca_fac, device, max_new_tokens,
                        boundary_fn=find_boundary_idx_ccot,
                    )
                    c_list.append(ok); f_list.append(fd)
                    tok_list.append(nt); lat_list.append(lt)
                results.append(_build_result(
                    f'cpca_{rtag}_{source}', model_tag, ratio, source, 'cpca', alpha,
                    c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens,
                ))
                print(f"    cpca_{rtag}_{source}: acc={results[-1].accuracy:.3f}  "
                      f"flip={results[-1].flip_rate:.3f}")

            # ── Controls ──────────────────────────────────────────────────────

            # Control A: Shuffled-label DoM vector at α*
            # Rules out: "the extraction procedure itself creates a useful artifact"
            try:
                v_shuf = _load_shuffled_vector(vectors_dir, source, 'dom')
                print(f"  Evaluating: Shuffled-label DoM (R={ratio}, src={source})")
                c_list, f_list, tok_list, lat_list = [], [], [], []
                shuf_dom_fac = lambda b, v=v_shuf, a=alpha: make_dom_hook(b, v, a, device)
                for item in D_val:
                    ok, fd, nt, lt = _eval_one(
                        ccot_model, tok_ccot, item, ccot_prompt_fn(item),
                        L_star, shuf_dom_fac, device, max_new_tokens,
                        boundary_fn=find_boundary_idx_ccot,
                    )
                    c_list.append(ok); f_list.append(fd)
                    tok_list.append(nt); lat_list.append(lt)
                results.append(_build_result(
                    f'shuf_dom_{rtag}_{source}', model_tag, ratio,
                    source, 'shuf_dom', alpha,
                    c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens,
                ))
                print(f"    shuf_dom_{rtag}_{source}: acc={results[-1].accuracy:.3f}")
            except FileNotFoundError:
                print(f"  [SKIP] Shuffled DoM vector missing for source={source}")

            # Control B: Negative DoM direction at α*
            # Injects −v_truth; should degrade accuracy if the direction is meaningful
            print(f"  Evaluating: Negative DoM (R={ratio}, src={source})")
            c_list, f_list, tok_list, lat_list = [], [], [], []
            neg_dom_fac = lambda b, v=v_dom, a=alpha: make_dom_hook(b, v, -a, device)
            for item in D_val:
                ok, fd, nt, lt = _eval_one(
                    ccot_model, tok_ccot, item, ccot_prompt_fn(item),
                    L_star, neg_dom_fac, device, max_new_tokens,
                    boundary_fn=find_boundary_idx_ccot,
                )
                c_list.append(ok); f_list.append(fd)
                tok_list.append(nt); lat_list.append(lt)
            results.append(_build_result(
                f'neg_dom_{rtag}_{source}', model_tag, ratio,
                source, 'neg_dom', alpha,
                c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens,
            ))
            print(f"    neg_dom_{rtag}_{source}: acc={results[-1].accuracy:.3f}  "
                  f"flip={results[-1].flip_rate:.3f}")

            if has_cpca:
                # Control C: Negative cPCA — subtract subspace projection at α*
                # h' = h − α·σ·U·Uᵀ·ĥ; should degrade if subspace is meaningful
                print(f"  Evaluating: Negative cPCA (R={ratio}, src={source})")
                c_list, f_list, tok_list, lat_list = [], [], [], []
                neg_cpca_fac = lambda b, U=U_cpca, a=alpha: make_cpca_hook(b, U, -a, device)
                for item in D_val:
                    ok, fd, nt, lt = _eval_one(
                        ccot_model, tok_ccot, item, ccot_prompt_fn(item),
                        L_star, neg_cpca_fac, device, max_new_tokens,
                        boundary_fn=find_boundary_idx_ccot,
                    )
                    c_list.append(ok); f_list.append(fd)
                    tok_list.append(nt); lat_list.append(lt)
                results.append(_build_result(
                    f'neg_cpca_{rtag}_{source}', model_tag, ratio,
                    source, 'neg_cpca', alpha,
                    c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens,
                ))
                print(f"    neg_cpca_{rtag}_{source}: acc={results[-1].accuracy:.3f}  "
                      f"flip={results[-1].flip_rate:.3f}")

                # Control D: Shuffled cPCA subspace at α*
                # Subspace-level analogue of shuffled-label DoM
                try:
                    U_shuf_cpca = _load_shuffled_vector(vectors_dir, source, 'cpca', r_final)
                    print(f"  Evaluating: Shuffled cPCA (R={ratio}, src={source})")
                    c_list, f_list, tok_list, lat_list = [], [], [], []
                    shuf_cpca_fac = lambda b, U=U_shuf_cpca, a=alpha: (
                        make_cpca_hook(b, U, a, device)
                    )
                    for item in D_val:
                        ok, fd, nt, lt = _eval_one(
                            ccot_model, tok_ccot, item, ccot_prompt_fn(item),
                            L_star, shuf_cpca_fac, device, max_new_tokens,
                            boundary_fn=find_boundary_idx_ccot,
                        )
                        c_list.append(ok); f_list.append(fd)
                        tok_list.append(nt); lat_list.append(lt)
                    results.append(_build_result(
                        f'shuf_cpca_{rtag}_{source}', model_tag, ratio,
                        source, 'shuf_cpca', alpha,
                        c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens,
                    ))
                    print(f"    shuf_cpca_{rtag}_{source}: acc={results[-1].accuracy:.3f}")
                except FileNotFoundError:
                    print(f"  [SKIP] Shuffled cPCA missing for source={source}")

        # Condition 8: Trimmed + DoM (best source = 'base' by convention)
        try:
            v_base_dom = _load_vector(vectors_dir, 'base', 'dom')
            alpha_base = alphas.get('base', alphas.get('ccot'))
            try:
                L_star_base = get_injection_layer(vectors_dir, 'base')
            except FileNotFoundError:
                L_star_base = meta.get('base_layer_star',
                                       meta.get('ccot_layer_star', 14))

            print(f"  Evaluating: Trimmed + DoM (R={ratio})")
            c_list, f_list, tok_list, lat_list = [], [], [], []
            trim_dom_fac = lambda b, v=v_base_dom, a=alpha_base: (
                make_dom_hook(b, v, a, device)
            )
            cot_prompt_fn = lambda item: (
                f"Question: {item['question']}\n\nReasoning:"
            )
            for i, item in enumerate(D_val):
                # Use budget as max_new_tokens (trimmed decoding)
                ok, fd, nt, lt = _eval_one(
                    cot_model, tok_cot, item, cot_prompt_fn(item),
                    L_star_base, trim_dom_fac, device, budgets[i],
                    boundary_fn=find_boundary_idx_base,
                )
                c_list.append(ok); f_list.append(fd)
                tok_list.append(nt); lat_list.append(lt)
            results.append(_build_result(
                f'trimmed_dom_{rtag}', model_tag, ratio, 'base', 'dom', alpha_base,
                c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens,
            ))
            print(f"    trimmed_dom_{rtag}: acc={results[-1].accuracy:.3f}  "
                  f"flip={results[-1].flip_rate:.3f}")
        except FileNotFoundError:
            print(f"  [SKIP] Base DoM vector missing — skipping Trimmed+DoM at R={ratio}")

        del ccot_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del cot_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Diagnostic alpha sweep (DoM, first source, subset) ────────────────────
    print(f"\n[PH3] Diagnostic alpha sweep…")
    _run_diagnostic_sweep(
        model_tag, checkpoints_dir, D_val, vectors_dir, meta, results_dir, device
    )

    # ── Save full results ──────────────────────────────────────────────────────
    ph3_path = os.path.join(results_dir, 'phase3_val.json')
    with open(ph3_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nPhase 3 results -> {ph3_path}")

    # steered_val.json for scripts/selection.py
    steered = [r for r in results if r.vector_method in ('dom', 'cpca')]
    max_probe = meta.get('ccot_max_probe_score', 0.0)
    best_s    = max(steered, key=lambda r: (r.accuracy, r.flip_rate)) if steered else None
    sv = {
        'model_tag':        model_tag,
        'best_condition':   best_s.condition if best_s else 'ccot',
        'steered_accuracy': best_s.accuracy  if best_s else 0.0,
        'flip_rate':        best_s.flip_rate if best_s else 0.0,
        'probe_accuracy':   max_probe,
        'n_examples':       len(D_val),
    }
    with open(os.path.join(results_dir, 'steered_val.json'), 'w') as f:
        json.dump(sv, f, indent=2)

    _print_phase3_table(results)
    return results


# ── Diagnostic alpha sweep ─────────────────────────────────────────────────────

def _run_diagnostic_sweep(
    model_tag, checkpoints_dir, D_val, vectors_dir, meta, results_dir, device,
    n_sub: int = 50,
) -> None:
    """Sweep alpha ∈ {0,.1,.5,1,2,5,10,20,50} on a D_val subset and save JSON."""
    alphas  = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    D_sub   = D_val[:min(n_sub, len(D_val))]
    source  = 'ccot'
    ratio_int = meta['best_ccot_ratio']

    try:
        v_dom  = _load_vector(vectors_dir, source, 'dom')
        L_star = get_injection_layer(vectors_dir, source)
    except FileNotFoundError:
        print("  [sweep skip] Missing vector or cPCA file.")
        return

    ckpt   = os.path.join(checkpoints_dir, f'ccot_R{ratio_int}')
    if not os.path.exists(os.path.join(ckpt, 'adapter_config.json')):
        print("  [sweep skip] CCoT checkpoint missing.")
        return

    model, tok = load_finetuned(ckpt, device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    ratio      = ratio_int / 10.0
    prompt_fn  = lambda item: f"Question: {item['question']}\n\n[compress:{ratio}]\n"
    alpha_star = _load_alpha(vectors_dir, source)

    sweep = []
    for a in alphas:
        c_list = []
        if a == 0.0:
            fac = None
        else:
            fac = lambda b, av=a: make_dom_hook(b, v_dom, av, device)
        for item in D_sub:
            ok, _, _, _ = _eval_one(
                model, tok, item, prompt_fn(item),
                L_star if fac else None, fac, device, 256,
                boundary_fn=find_boundary_idx_ccot if fac else None,
            )
            c_list.append(ok)
        acc    = sum(c_list) / len(c_list)
        marker = " ← α*" if abs(a - alpha_star) < 0.5 else ""
        sweep.append({'alpha': a, 'accuracy': acc})
        print(f"  [α-sweep] α={a:>5.1f}  acc={acc:.3f}{marker}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    payload = {'model_tag': model_tag, 'source': source,
               'alpha_star': alpha_star, 'sweep': sweep}
    out = os.path.join(results_dir, 'alpha_diagnostic.json')
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  Sweep -> {out}")

    try:
        from phase3.plots import plot_alpha_diagnostic
        plot_alpha_diagnostic(
            payload,
            os.path.join(results_dir, 'alpha_diagnostic.png'),
        )
    except Exception as e:
        print(f"  [plot] {e}")


# ── Printing ───────────────────────────────────────────────────────────────────

def _print_phase3_table(results: list[ConditionResult]):
    print("\n" + "=" * 72)
    print(f"{'Condition':<32} {'Acc':>6} {'Flip':>6} "
          f"{'Tok':>6} {'ActR':>6} {'Lat':>6}")
    print("-" * 72)
    for r in results:
        print(
            f"{r.condition:<32} {r.accuracy:>6.3f} {r.flip_rate:>6.3f} "
            f"{r.reasoning_tokens:>6.1f} {r.actual_ratio:>6.3f} "
            f"{r.latency_sec:>6.2f}s"
        )
    print("=" * 72)
