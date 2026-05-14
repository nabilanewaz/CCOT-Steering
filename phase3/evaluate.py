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


# ── Evaluation loop helpers ───────────────────────────────────────────────────

def _eval_loop(
    dataset: list,
    eval_fn,        # (item) -> (ok: bool, found: bool, n_tok: int, lat: float)
    label: str,
    log_every: int = None,
) -> tuple[list, list, list, list]:
    """Run eval_fn over dataset with progress prints. Returns (c, f, tok, lat) lists."""
    c_list, f_list, tok_list, lat_list = [], [], [], []
    n = len(dataset)
    every = log_every if log_every else max(1, n // 5)
    for i, item in enumerate(dataset):
        ok, fd, nt, lt = eval_fn(item)
        c_list.append(ok); f_list.append(fd)
        tok_list.append(nt); lat_list.append(lt)
        if (i + 1) % every == 0 or (i + 1) == n:
            running_acc = sum(c_list) / (i + 1)
            mean_lat    = sum(lat_list) / len(lat_list)
            mean_tok    = sum(tok_list) / len(tok_list)
            print(f"    [{i+1:>4}/{n}]  acc={running_acc:.3f}  "
                  f"tok={mean_tok:.0f}  lat={mean_lat:.2f}s")
    return c_list, f_list, tok_list, lat_list


def _cond_banner(idx: int, label: str, t_phase: float) -> float:
    """Print a condition banner and return a new step-start timestamp."""
    elapsed = time.time() - t_phase
    print(f"\n  ╔══ [{idx:>2}] {label}  (+{elapsed:.0f}s elapsed) ══")
    return time.time()


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
    bar = '─' * 60
    print(f"\n{bar}")
    print(f"  [PH3] Alpha Tuning  [{model_tag}]")
    print(f"  Sources      : {list(SOURCES)}")
    print(f"  D_val size   : {len(D_val)}")
    print(f"  λ subset     : min(200, {len(D_val)}) = {min(200, len(D_val))}")
    print(f"  vectors_dir  : {vectors_dir}")
    print(bar)

    best_ratio_int = meta['best_ccot_ratio']
    best_ratio_flt = best_ratio_int / 10.0
    t_tune_start   = time.time()

    source_ckpt = {
        'ccot': os.path.join(checkpoints_dir, f'ccot_R{best_ratio_int}'),
        'base': os.path.join(checkpoints_dir, 'cot'),
    }
    source_L_star = {
        'ccot': meta.get('ccot_best_layer'),
        'base': meta.get('base_best_layer'),
    }

    for s_idx, source in enumerate(SOURCES, 1):
        print(f"\n  ── [{s_idx}/{len(SOURCES)}] Source = {source} ──")
        out_path = _alpha_path(vectors_dir, source)
        if os.path.exists(out_path):
            cached_val = torch.load(out_path).item()
            print(f"  [CACHED] alpha_star={cached_val:.4f}  path={out_path}")
            continue

        v_dom  = _load_vector(vectors_dir, source, 'dom')
        L_star = source_L_star[source]
        if L_star is None:
            try:
                L_star = get_injection_layer(vectors_dir, source)
            except FileNotFoundError:
                L_star = meta.get('ccot_best_layer', 14)

        print(f"  Checkpoint   : {source_ckpt[source]}")
        print(f"  L* = {L_star}   v_dom.shape = {tuple(v_dom.shape)}")
        print(f"  Loading model…")
        model, tok = load_finetuned(source_ckpt[source], device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        # ── Step 1: λ sweep on 200-example subset ────────────────────────────
        sweep_path = os.path.join(vectors_dir, f'{source}_lambda_sweep.json')
        t_step = time.time()
        if os.path.exists(sweep_path):
            with open(sweep_path) as fp:
                sweep_sel = json.load(fp)['selected']
            lambda_a = sweep_sel['lambda_a']
            lambda_m = sweep_sel['lambda_m']
            print(f"  [1/3] λ sweep  CACHED  λ_a={lambda_a}  λ_m={lambda_m}")
        else:
            D_sub = D_val[:min(200, len(D_val))]
            print(f"  [1/3] λ sweep: grid on {len(D_sub)} examples  max_epochs=2…")
            from phase3.lambda_sweep import sweep_lambda_grid
            sweep_sel = sweep_lambda_grid(
                model, tok, D_sub, v_dom, L_star, device, model_tag,
                ratio=best_ratio_flt,
                out_path=sweep_path,
                max_epochs=2,
            )
            lambda_a = sweep_sel['lambda_a']
            lambda_m = sweep_sel['lambda_m']
            print(f"  [1/3] λ sweep done  λ_a={lambda_a}  λ_m={lambda_m}  "
                  f"({time.time()-t_step:.0f}s)  → {sweep_path}")
            if results_dir:
                try:
                    from phase3.plots import plot_lambda_sweep_heatmap
                    with open(sweep_path) as fp:
                        sweep_data = json.load(fp)
                    hm_path = os.path.join(results_dir, f'{source}_lambda_heatmap.png')
                    plot_lambda_sweep_heatmap(sweep_data, hm_path)
                    print(f"  [plot] heatmap → {hm_path}")
                except Exception as e:
                    print(f"  [plot] {e}")

        # ── Step 2: Full α* tuning with selected lambdas ──────────────────────
        t_step = time.time()
        print(f"\n  [2/3] α* tuning  D_val={len(D_val)}  λ_a={lambda_a}  λ_m={lambda_m}…")
        alpha_star, history = tune_alpha(
            model, tok, D_val, v_dom, L_star, device,
            model_tag=model_tag, ratio=best_ratio_flt,
            lambda_a=lambda_a, lambda_m=lambda_m,
        )
        torch.save(alpha_star, out_path)
        n_epochs   = len(history)
        final_loss = history[-1].get('loss', '?') if history else 'N/A'
        print(f"  [2/3] α* = {alpha_star.item():.4f}  epochs={n_epochs}  "
              f"final_loss={final_loss}  ({time.time()-t_step:.0f}s)")
        print(f"  Saved → {out_path}")

        # ── Step 3: Persist history and plots ─────────────────────────────────
        print(f"\n  [3/3] Saving history and plots…")
        if results_dir:
            hist_path = os.path.join(results_dir, f'{source}_alpha_history.json')
            with open(hist_path, 'w') as fp:
                json.dump({
                    'model_tag': model_tag,
                    'source':    source,
                    'lambda_a':  lambda_a,
                    'lambda_m':  lambda_m,
                    'alpha_star': alpha_star.item(),
                    'L_star':    L_star,
                    'history':   history,
                }, fp, indent=2)
            print(f"  [3/3] History → {hist_path}")
            try:
                from phase3.plots import plot_loss_curves
                lc_path = os.path.join(results_dir, f'{source}_loss_curves.png')
                plot_loss_curves(history, lc_path)
                print(f"  [plot] loss curves → {lc_path}")
            except Exception as e:
                print(f"  [plot] {e}")
        else:
            print(f"  [3/3] results_dir=None — skipping history/plot save")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  Alpha tuning complete  ({time.time()-t_tune_start:.0f}s total)")
    print(bar)


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
    Writes phase3_val.json, steered_val.json, phase3_diagnostics.json,
    alpha_diagnostic.json.  Interim saves after each ratio.
    """
    os.makedirs(results_dir, exist_ok=True)
    t_phase = time.time()
    meta    = _load_meta(vectors_dir)
    r_final = meta.get('ccot_r_final', 10)

    # ── Phase start banner ────────────────────────────────────────────────────
    bar = '═' * 64
    print(f"\n{bar}")
    print(f"  Phase 3 Evaluation  [{model_tag}]")
    print(f"  D_val = {len(D_val)} examples")
    print(f"  Ratios       : {RATIOS}")
    print(f"  Sources      : {SOURCES}")
    print(f"  max_new_tok  : {max_new_tokens}")
    print(f"  vectors_dir  : {vectors_dir}")
    print(f"  results_dir  : {results_dir}")
    print(f"  Phase-2 meta : best_ccot_ratio={meta.get('best_ccot_ratio')}  "
          f"r_final={r_final}")
    print(f"  Layer stars  : ccot={meta.get('ccot_best_layer','?')}  "
          f"base={meta.get('base_best_layer','?')}")
    print(bar)

    # ── Pre-tune alpha per source ─────────────────────────────────────────────
    _tune_and_save_alpha(model_tag, checkpoints_dir, D_val,
                         vectors_dir, device, meta, results_dir=results_dir)
    alphas = {s: _load_alpha(vectors_dir, s) for s in SOURCES}
    print(f"\n  Alpha stars loaded: { {s: round(v, 4) for s, v in alphas.items()} }")

    results:    list[ConditionResult] = []
    cond_times: dict[str, float]      = {}
    cond_idx = 0

    # ── [1] No CoT ────────────────────────────────────────────────────────────
    cond_idx += 1
    cond_start = _cond_banner(cond_idx, f'No CoT (frozen base)  base={base_model_id}', t_phase)
    from phase1.inference import load_base_frozen
    print(f"    Loading frozen base model…")
    base_model, tok_base = load_base_frozen(base_model_id, device)

    def _ev_no_cot(item):
        prompt = f"Question: {item['question']}\n\nAnswer:"
        ok, fd, _, lt = _eval_one(base_model, tok_base, item, prompt,
                                  None, None, device, 32)
        return ok, fd, 0, lt  # 0 reasoning tokens for no_cot

    c_list, f_list, tok_list, lat_list = _eval_loop(D_val, _ev_no_cot, 'no_cot')
    r = _build_result('no_cot', model_tag, None, None, None, None,
                      c_list, f_list, tok_list, lat_list, None, 1.0)
    results.append(r)
    cond_elapsed = time.time() - cond_start
    cond_times['no_cot'] = round(cond_elapsed, 2)
    print(f"  ╚══ no_cot  acc={r.accuracy:.3f}  found={r.answer_found_rate:.3f}  "
          f"({cond_elapsed:.0f}s) ══")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Load CoT model (kept alive for Full CoT + Trimmed CoT + Trimmed+DoM) ──
    cot_ckpt = os.path.join(checkpoints_dir, 'cot')
    print(f"\n  Loading CoT model: {cot_ckpt}")
    cot_model, tok_cot = load_finetuned(cot_ckpt, device)
    for p in cot_model.parameters():
        p.requires_grad = False
    cot_model.eval()

    # ── [2] Full CoT ──────────────────────────────────────────────────────────
    cond_idx += 1
    cond_start = _cond_banner(cond_idx, 'Full CoT', t_phase)

    def _ev_full_cot(item):
        t0 = time.time()
        pred, reasoning = run_cot(cot_model, tok_cot, item, device)
        lt   = time.time() - t0
        gold = item['answer'].split('####')[1].strip()
        ok   = normalize_answer(pred) == normalize_answer(gold) if pred else False
        nt   = len(tok_cot.encode(reasoning, add_special_tokens=False)) if reasoning else 0
        return ok, pred is not None, nt, lt

    c_list, f_list, tok_list, lat_list = _eval_loop(D_val, _ev_full_cot, 'full_cot')
    full_cot_mean_tokens = sum(tok_list) / max(len(tok_list), 1)
    r = _build_result('full_cot', model_tag, None, None, None, None,
                      c_list, f_list, tok_list, lat_list, None, full_cot_mean_tokens)
    results.append(r)
    cond_elapsed = time.time() - cond_start
    cond_times['full_cot'] = round(cond_elapsed, 2)
    print(f"  ╚══ full_cot  acc={r.accuracy:.3f}  mean_tok={full_cot_mean_tokens:.1f}  "
          f"({cond_elapsed:.0f}s) ══")

    # ── Pre-compute per-example budgets for all ratios ────────────────────────
    print(f"\n  ── Pre-computing per-example budgets for ratios={RATIOS} ──")
    budgets_by_ratio: dict[float, list[int]] = {}
    for ratio in RATIOS:
        t_b = time.time()
        print(f"    R={ratio}…", end='', flush=True)
        budgets_by_ratio[ratio] = compute_per_example_budgets(
            cot_model, tok_cot, D_val, device, ratio
        )
        mean_b = sum(budgets_by_ratio[ratio]) / len(budgets_by_ratio[ratio])
        print(f" done  mean_budget={mean_b:.0f} tok  ({time.time()-t_b:.0f}s)")

    # ── Per-ratio loop ─────────────────────────────────────────────────────────
    for ratio in RATIOS:
        ratio_int = int(ratio * 10)
        rtag      = f"R{ratio_int}"
        budgets   = budgets_by_ratio[ratio]
        ratio_results_start = len(results)

        print(f"\n{'▓'*64}")
        print(f"  Ratio = {ratio}  ({rtag})   "
              f"mean_budget={sum(budgets)/len(budgets):.0f} tok")
        print(f"{'▓'*64}")

        # ── Trimmed CoT ───────────────────────────────────────────────────────
        cond_idx += 1
        cond_start = _cond_banner(cond_idx, f'Trimmed CoT  R={ratio}', t_phase)
        _budgets_iter = iter(budgets)

        def _ev_trimmed(item, _bi=_budgets_iter):
            b  = next(_bi)
            t0 = time.time()
            pred, reasoning = run_trimmed_cot(cot_model, tok_cot, item, b, device)
            lt   = time.time() - t0
            gold = item['answer'].split('####')[1].strip()
            ok   = normalize_answer(pred) == normalize_answer(gold) if pred else False
            nt   = len(tok_cot.encode(reasoning or '', add_special_tokens=False))
            return ok, pred is not None, nt, lt

        c_list, f_list, tok_list, lat_list = _eval_loop(
            D_val, _ev_trimmed, f'trimmed_{rtag}')
        r = _build_result(f'trimmed_{rtag}', model_tag, ratio, None, None, None,
                          c_list, f_list, tok_list, lat_list, None, full_cot_mean_tokens)
        results.append(r)
        cond_elapsed = time.time() - cond_start
        cond_times[f'trimmed_{rtag}'] = round(cond_elapsed, 2)
        print(f"  ╚══ trimmed_{rtag}  acc={r.accuracy:.3f}  "
              f"mean_tok={r.reasoning_tokens:.1f}  ({cond_elapsed:.0f}s) ══")

        # ── Load CCoT model ───────────────────────────────────────────────────
        ccot_ckpt = os.path.join(checkpoints_dir, f'ccot_{rtag}')
        if not os.path.exists(os.path.join(ccot_ckpt, 'adapter_config.json')):
            print(f"  [SKIP] CCoT checkpoint missing: {ccot_ckpt}")
            continue
        print(f"\n  Loading CCoT model: {ccot_ckpt}")
        ccot_model, tok_ccot = load_finetuned(ccot_ckpt, device)
        for p in ccot_model.parameters():
            p.requires_grad = False
        ccot_model.eval()

        def _ccot_prompt(item, _r=ratio):
            return f"Question: {item['question']}\n\n[compress:{_r}]\n"

        # ── CCoT baseline ─────────────────────────────────────────────────────
        cond_idx += 1
        cond_start = _cond_banner(cond_idx, f'CCoT baseline  R={ratio}', t_phase)

        def _ev_ccot(item):
            return _eval_one(ccot_model, tok_ccot, item, _ccot_prompt(item),
                             None, None, device, max_new_tokens)

        c_list, f_list, tok_list, lat_list = _eval_loop(
            D_val, _ev_ccot, f'ccot_{rtag}')
        ccot_correct = list(c_list)
        r = _build_result(f'ccot_{rtag}', model_tag, ratio, None, None, None,
                          c_list, f_list, tok_list, lat_list, None, full_cot_mean_tokens)
        results.append(r)
        cond_elapsed = time.time() - cond_start
        cond_times[f'ccot_{rtag}'] = round(cond_elapsed, 2)
        ccot_acc = r.accuracy
        print(f"  ╚══ ccot_{rtag}  acc={ccot_acc:.3f}  "
              f"mean_tok={r.reasoning_tokens:.1f}  ({cond_elapsed:.0f}s) ══")

        # ── Steered conditions per source ─────────────────────────────────────
        for source in SOURCES:
            alpha = alphas[source]
            try:
                L_star = get_injection_layer(vectors_dir, source)
            except FileNotFoundError:
                L_star = meta.get(f'{source}_best_layer',
                                  meta.get('ccot_best_layer', 14))

            print(f"\n  ┌── Source={source}  L*={L_star}  α={alpha:.4f} ──")

            try:
                v_dom = _load_vector(vectors_dir, source, 'dom')
                print(f"  │   v_dom loaded  shape={tuple(v_dom.shape)}")
            except FileNotFoundError:
                print(f"  [SKIP] DoM vector missing for source={source}")
                continue

            try:
                U_cpca   = _load_vector(vectors_dir, source, 'cpca', r_final)
                has_cpca = True
                print(f"  │   U_cpca loaded  shape={tuple(U_cpca.shape)}")
            except FileNotFoundError:
                U_cpca   = None
                has_cpca = False
                print(f"  │   cPCA vector not found — cpca conditions skipped")
            print(f"  └──────────────────────────────────────────────────────")

            # ── Random Noise ──────────────────────────────────────────────────
            cond_idx += 1
            cond_start = _cond_banner(
                cond_idx, f'Random Noise  R={ratio} src={source}', t_phase)
            _noise_fac = lambda b, _a=alpha: make_noise_hook(b, _a, device)

            def _ev_noise(item, _nf=_noise_fac):
                return _eval_one(ccot_model, tok_ccot, item, _ccot_prompt(item),
                                 L_star, _nf, device, max_new_tokens,
                                 boundary_fn=find_boundary_idx_ccot)

            c_list, f_list, tok_list, lat_list = _eval_loop(
                D_val, _ev_noise, f'noise_{rtag}_{source}')
            r = _build_result(
                f'noise_{rtag}_{source}', model_tag, ratio, source, 'noise', alpha,
                c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens)
            results.append(r)
            cond_elapsed = time.time() - cond_start
            cond_times[f'noise_{rtag}_{source}'] = round(cond_elapsed, 2)
            print(f"  ╚══ noise_{rtag}_{source}  acc={r.accuracy:.3f}  "
                  f"flip={r.flip_rate:.3f}  Δ={r.accuracy-ccot_acc:+.3f}  "
                  f"({cond_elapsed:.0f}s) ══")

            # ── CCoT + DoM ────────────────────────────────────────────────────
            cond_idx += 1
            cond_start = _cond_banner(
                cond_idx, f'CCoT+DoM  R={ratio} src={source}', t_phase)
            _dom_fac = lambda b, _v=v_dom, _a=alpha: make_dom_hook(b, _v, _a, device)

            def _ev_dom(item, _df=_dom_fac):
                return _eval_one(ccot_model, tok_ccot, item, _ccot_prompt(item),
                                 L_star, _df, device, max_new_tokens,
                                 boundary_fn=find_boundary_idx_ccot)

            c_list, f_list, tok_list, lat_list = _eval_loop(
                D_val, _ev_dom, f'dom_{rtag}_{source}')
            r = _build_result(
                f'dom_{rtag}_{source}', model_tag, ratio, source, 'dom', alpha,
                c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens)
            results.append(r)
            cond_elapsed = time.time() - cond_start
            cond_times[f'dom_{rtag}_{source}'] = round(cond_elapsed, 2)
            print(f"  ╚══ dom_{rtag}_{source}  acc={r.accuracy:.3f}  "
                  f"flip={r.flip_rate:.3f}  Δ={r.accuracy-ccot_acc:+.3f}  "
                  f"({cond_elapsed:.0f}s) ══")

            # ── CCoT + cPCA ───────────────────────────────────────────────────
            if has_cpca:
                cond_idx += 1
                cond_start = _cond_banner(
                    cond_idx, f'CCoT+cPCA  R={ratio} src={source}', t_phase)
                _cpca_fac = lambda b, _U=U_cpca, _a=alpha: make_cpca_hook(b, _U, _a, device)

                def _ev_cpca(item, _cf=_cpca_fac):
                    return _eval_one(ccot_model, tok_ccot, item, _ccot_prompt(item),
                                     L_star, _cf, device, max_new_tokens,
                                     boundary_fn=find_boundary_idx_ccot)

                c_list, f_list, tok_list, lat_list = _eval_loop(
                    D_val, _ev_cpca, f'cpca_{rtag}_{source}')
                r = _build_result(
                    f'cpca_{rtag}_{source}', model_tag, ratio, source, 'cpca', alpha,
                    c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens)
                results.append(r)
                cond_elapsed = time.time() - cond_start
                cond_times[f'cpca_{rtag}_{source}'] = round(cond_elapsed, 2)
                print(f"  ╚══ cpca_{rtag}_{source}  acc={r.accuracy:.3f}  "
                      f"flip={r.flip_rate:.3f}  Δ={r.accuracy-ccot_acc:+.3f}  "
                      f"({cond_elapsed:.0f}s) ══")

            # ── Control A: Shuffled DoM ───────────────────────────────────────
            try:
                v_shuf = _load_shuffled_vector(vectors_dir, source, 'dom')
                cond_idx += 1
                cond_start = _cond_banner(
                    cond_idx,
                    f'Shuffled DoM [ctrl-A]  R={ratio} src={source}', t_phase)
                _sdom_fac = lambda b, _v=v_shuf, _a=alpha: make_dom_hook(b, _v, _a, device)

                def _ev_shuf_dom(item, _sf=_sdom_fac):
                    return _eval_one(ccot_model, tok_ccot, item, _ccot_prompt(item),
                                     L_star, _sf, device, max_new_tokens,
                                     boundary_fn=find_boundary_idx_ccot)

                c_list, f_list, tok_list, lat_list = _eval_loop(
                    D_val, _ev_shuf_dom, f'shuf_dom_{rtag}_{source}')
                r = _build_result(
                    f'shuf_dom_{rtag}_{source}', model_tag, ratio,
                    source, 'shuf_dom', alpha,
                    c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens)
                results.append(r)
                cond_elapsed = time.time() - cond_start
                cond_times[f'shuf_dom_{rtag}_{source}'] = round(cond_elapsed, 2)
                print(f"  ╚══ shuf_dom_{rtag}_{source}  acc={r.accuracy:.3f}  "
                      f"Δ={r.accuracy-ccot_acc:+.3f}  ({cond_elapsed:.0f}s) ══")
            except FileNotFoundError:
                print(f"  [SKIP] Shuffled DoM vector missing for source={source}")

            # ── Control B: Negative DoM ───────────────────────────────────────
            cond_idx += 1
            cond_start = _cond_banner(
                cond_idx,
                f'Negative DoM [ctrl-B]  R={ratio} src={source}', t_phase)
            _neg_dom_fac = lambda b, _v=v_dom, _a=alpha: make_dom_hook(b, _v, -_a, device)

            def _ev_neg_dom(item, _nf=_neg_dom_fac):
                return _eval_one(ccot_model, tok_ccot, item, _ccot_prompt(item),
                                 L_star, _nf, device, max_new_tokens,
                                 boundary_fn=find_boundary_idx_ccot)

            c_list, f_list, tok_list, lat_list = _eval_loop(
                D_val, _ev_neg_dom, f'neg_dom_{rtag}_{source}')
            r = _build_result(
                f'neg_dom_{rtag}_{source}', model_tag, ratio,
                source, 'neg_dom', alpha,
                c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens)
            results.append(r)
            cond_elapsed = time.time() - cond_start
            cond_times[f'neg_dom_{rtag}_{source}'] = round(cond_elapsed, 2)
            print(f"  ╚══ neg_dom_{rtag}_{source}  acc={r.accuracy:.3f}  "
                  f"Δ={r.accuracy-ccot_acc:+.3f}  ({cond_elapsed:.0f}s) ══")

            if has_cpca:
                # ── Control C: Negative cPCA ──────────────────────────────────
                cond_idx += 1
                cond_start = _cond_banner(
                    cond_idx,
                    f'Negative cPCA [ctrl-C]  R={ratio} src={source}', t_phase)
                _neg_cpca_fac = lambda b, _U=U_cpca, _a=alpha: make_cpca_hook(b, _U, -_a, device)

                def _ev_neg_cpca(item, _nf=_neg_cpca_fac):
                    return _eval_one(ccot_model, tok_ccot, item, _ccot_prompt(item),
                                     L_star, _nf, device, max_new_tokens,
                                     boundary_fn=find_boundary_idx_ccot)

                c_list, f_list, tok_list, lat_list = _eval_loop(
                    D_val, _ev_neg_cpca, f'neg_cpca_{rtag}_{source}')
                r = _build_result(
                    f'neg_cpca_{rtag}_{source}', model_tag, ratio,
                    source, 'neg_cpca', alpha,
                    c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens)
                results.append(r)
                cond_elapsed = time.time() - cond_start
                cond_times[f'neg_cpca_{rtag}_{source}'] = round(cond_elapsed, 2)
                print(f"  ╚══ neg_cpca_{rtag}_{source}  acc={r.accuracy:.3f}  "
                      f"Δ={r.accuracy-ccot_acc:+.3f}  ({cond_elapsed:.0f}s) ══")

                # ── Control D: Shuffled cPCA ──────────────────────────────────
                try:
                    U_shuf_cpca = _load_shuffled_vector(
                        vectors_dir, source, 'cpca', r_final)
                    cond_idx += 1
                    cond_start = _cond_banner(
                        cond_idx,
                        f'Shuffled cPCA [ctrl-D]  R={ratio} src={source}', t_phase)
                    _scpca_fac = lambda b, _U=U_shuf_cpca, _a=alpha: (
                        make_cpca_hook(b, _U, _a, device))

                    def _ev_shuf_cpca(item, _sf=_scpca_fac):
                        return _eval_one(ccot_model, tok_ccot, item, _ccot_prompt(item),
                                         L_star, _sf, device, max_new_tokens,
                                         boundary_fn=find_boundary_idx_ccot)

                    c_list, f_list, tok_list, lat_list = _eval_loop(
                        D_val, _ev_shuf_cpca, f'shuf_cpca_{rtag}_{source}')
                    r = _build_result(
                        f'shuf_cpca_{rtag}_{source}', model_tag, ratio,
                        source, 'shuf_cpca', alpha,
                        c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens)
                    results.append(r)
                    cond_elapsed = time.time() - cond_start
                    cond_times[f'shuf_cpca_{rtag}_{source}'] = round(cond_elapsed, 2)
                    print(f"  ╚══ shuf_cpca_{rtag}_{source}  acc={r.accuracy:.3f}  "
                          f"Δ={r.accuracy-ccot_acc:+.3f}  ({cond_elapsed:.0f}s) ══")
                except FileNotFoundError:
                    print(f"  [SKIP] Shuffled cPCA missing for source={source}")

        # ── Trimmed + DoM ─────────────────────────────────────────────────────
        try:
            v_base_dom = _load_vector(vectors_dir, 'base', 'dom')
            alpha_base = alphas.get('base', alphas.get('ccot'))
            try:
                L_star_base = get_injection_layer(vectors_dir, 'base')
            except FileNotFoundError:
                L_star_base = meta.get('base_best_layer',
                                       meta.get('ccot_best_layer', 14))

            cond_idx += 1
            cond_start = _cond_banner(
                cond_idx, f'Trimmed+DoM  R={ratio}  L*={L_star_base}  α={alpha_base:.4f}',
                t_phase)
            print(f"    mean_budget={sum(budgets)/len(budgets):.0f} tok")
            _trim_dom_fac = lambda b, _v=v_base_dom, _a=alpha_base: (
                make_dom_hook(b, _v, _a, device))
            _cot_prompt_fn = lambda item: f"Question: {item['question']}\n\nReasoning:"
            _budgets_iter2 = iter(budgets)

            def _ev_trim_dom(item, _tf=_trim_dom_fac, _bi=_budgets_iter2):
                b = next(_bi)
                return _eval_one(cot_model, tok_cot, item, _cot_prompt_fn(item),
                                 L_star_base, _tf, device, b,
                                 boundary_fn=find_boundary_idx_base)

            c_list, f_list, tok_list, lat_list = _eval_loop(
                D_val, _ev_trim_dom, f'trimmed_dom_{rtag}')
            r = _build_result(
                f'trimmed_dom_{rtag}', model_tag, ratio, 'base', 'dom', alpha_base,
                c_list, f_list, tok_list, lat_list, ccot_correct, full_cot_mean_tokens)
            results.append(r)
            cond_elapsed = time.time() - cond_start
            cond_times[f'trimmed_dom_{rtag}'] = round(cond_elapsed, 2)
            print(f"  ╚══ trimmed_dom_{rtag}  acc={r.accuracy:.3f}  "
                  f"flip={r.flip_rate:.3f}  Δ={r.accuracy-ccot_acc:+.3f}  "
                  f"({cond_elapsed:.0f}s) ══")
        except FileNotFoundError:
            print(f"  [SKIP] Base DoM vector missing — skipping Trimmed+DoM at R={ratio}")

        # ── Interim save after this ratio ─────────────────────────────────────
        interim_path = os.path.join(results_dir, 'phase3_val_interim.json')
        with open(interim_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        n_this_ratio = len(results) - ratio_results_start
        print(f"\n  [interim] {n_this_ratio} new conditions  "
              f"{len(results)} total → {interim_path}")

        # ── Ratio summary line ────────────────────────────────────────────────
        ratio_results = results[ratio_results_start:]
        best_r = max(ratio_results, key=lambda r: r.accuracy)
        print(f"  [ratio={ratio}] best={best_r.condition}  "
              f"acc={best_r.accuracy:.3f}  "
              f"elapsed={time.time()-t_phase:.0f}s total")

        del ccot_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del cot_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Diagnostic alpha sweep ─────────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  [PH3] Diagnostic alpha sweep (source=ccot, {50} examples)…")
    _run_diagnostic_sweep(
        model_tag, checkpoints_dir, D_val, vectors_dir, meta, results_dir, device
    )

    # ── Save full results ──────────────────────────────────────────────────────
    ph3_path = os.path.join(results_dir, 'phase3_val.json')
    with open(ph3_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n  Phase 3 results → {ph3_path}  ({len(results)} conditions)")

    # ── steered_val.json for scripts/selection.py ─────────────────────────────
    steered   = [r for r in results if r.vector_method in ('dom', 'cpca')]
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
    sv_path = os.path.join(results_dir, 'steered_val.json')
    with open(sv_path, 'w') as f:
        json.dump(sv, f, indent=2)
    print(f"  steered_val.json  → {sv_path}")
    print(f"  Best steered: {sv['best_condition']}  "
          f"acc={sv['steered_accuracy']:.3f}  flip={sv['flip_rate']:.3f}")

    # ── Phase 3 diagnostics JSON ───────────────────────────────────────────────
    total_elapsed = time.time() - t_phase
    diag = {
        'model_tag':    model_tag,
        'phase':        3,
        'n_val':        len(D_val),
        'vectors_dir':  vectors_dir,
        'results_dir':  results_dir,
        'meta_summary': {
            'best_ccot_ratio':       meta.get('best_ccot_ratio'),
            'ccot_r_final':          meta.get('ccot_r_final'),
            'ccot_best_layer':       meta.get('ccot_best_layer'),
            'base_best_layer':       meta.get('base_best_layer'),
            'ccot_max_probe_score':  meta.get('ccot_max_probe_score'),
        },
        'alpha_stars':              {s: round(v, 4) for s, v in alphas.items()},
        'full_cot_mean_tokens':     round(full_cot_mean_tokens, 2),
        'n_conditions':             len(results),
        'best_steered':             sv,
        'condition_times_s':        cond_times,
        'total_elapsed_s':          round(total_elapsed, 2),
        'results_summary': [
            {
                'condition':       r.condition,
                'accuracy':        round(r.accuracy, 4),
                'flip_rate':       round(r.flip_rate, 4),
                'reasoning_tokens': round(r.reasoning_tokens, 1),
                'actual_ratio':    round(r.actual_ratio, 3),
                'latency_sec':     round(r.latency_sec, 3),
                'answer_found_rate': round(r.answer_found_rate, 4),
            }
            for r in results
        ],
    }
    diag_path = os.path.join(results_dir, 'phase3_diagnostics.json')
    with open(diag_path, 'w') as f:
        json.dump(diag, f, indent=2)
    print(f"  phase3_diagnostics.json → {diag_path}")

    _print_phase3_table(results)

    print(f"\n{bar}")
    print(f"  Phase 3 Evaluation complete  [{model_tag}]")
    print(f"  {len(results)} conditions evaluated  total={total_elapsed:.0f}s  "
          f"({total_elapsed/60:.1f} min)")
    print(f"  Saved: {ph3_path}")
    print(f"         {sv_path}")
    print(f"         {diag_path}")
    print(bar)

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
    W = 82
    HDR = (f"{'Condition':<38} {'Acc':>6} {'Flip':>6} "
           f"{'Tok':>7} {'ActR':>6} {'Lat':>7}")

    def _row(r: ConditionResult) -> str:
        return (f"{r.condition:<38} {r.accuracy:>6.3f} {r.flip_rate:>6.3f} "
                f"{r.reasoning_tokens:>7.1f} {r.actual_ratio:>6.3f} "
                f"{r.latency_sec:>6.2f}s")

    print("\n" + "═" * W)
    print("  Phase 3 Results Summary")
    print("─" * W)
    print("  " + HDR)
    print("─" * W)

    res_map = {r.condition: r for r in results}

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("  [ BASELINES ]")
    for cond in ('no_cot', 'full_cot'):
        if cond in res_map:
            print("  " + _row(res_map[cond]))

    # ── Per-ratio groups ──────────────────────────────────────────────────────
    for ratio in RATIOS:
        ri = int(ratio * 10)
        rtag = f"R{ri}"
        ratio_conds = [r for r in results if r.ratio == ratio]
        if not ratio_conds:
            continue
        print(f"\n  [ R={ratio} ({rtag}) ]")

        # Trimmed CoT and CCoT baseline first
        for prefix in (f'trimmed_{rtag}', f'ccot_{rtag}'):
            if prefix in res_map:
                print("  " + _row(res_map[prefix]))

        # Main steered (dom, cpca) per source
        for src in SOURCES:
            for method in ('dom', 'cpca'):
                key = f'{method}_{rtag}_{src}'
                if key in res_map:
                    r = res_map[key]
                    delta = r.accuracy - res_map.get(f'ccot_{rtag}', r).accuracy
                    print(f"  {_row(r)}   Δ={delta:+.3f}")

        # Trimmed+DoM
        key = f'trimmed_dom_{rtag}'
        if key in res_map:
            r = res_map[key]
            delta = r.accuracy - res_map.get(f'ccot_{rtag}', r).accuracy
            print(f"  {_row(r)}   Δ={delta:+.3f}")

        # Controls (noise, neg_*, shuf_*)
        ctrl_prefixes = [f'noise_{rtag}', f'neg_dom_{rtag}', f'neg_cpca_{rtag}',
                         f'shuf_dom_{rtag}', f'shuf_cpca_{rtag}']
        ctrl_rows = [r for r in ratio_conds
                     if any(r.condition.startswith(p) for p in ctrl_prefixes)]
        if ctrl_rows:
            print(f"  {'  controls':}")
            for r in ctrl_rows:
                print("    " + _row(r))

    # ── CCoT vs Trimmed gain summary ──────────────────────────────────────────
    print("\n" + "─" * W)
    print("  Mechanism Gain: best(CCoT+DoM/cPCA) − trimmed_cot  at same R")
    print("─" * W)
    for ratio in RATIOS:
        ri   = int(ratio * 10)
        rtag = f"R{ri}"
        base_acc = res_map.get(f'trimmed_{rtag}')
        if base_acc is None:
            continue
        base_acc = base_acc.accuracy
        best_steer = max(
            (res_map[k] for k in res_map
             if k.startswith(('dom_', 'cpca_')) and f'_{rtag}_' in k),
            key=lambda r: r.accuracy,
            default=None,
        )
        if best_steer:
            gain = best_steer.accuracy - base_acc
            label = ("CCoT-steered better" if gain > 0.01 else
                     "Trimmed better" if gain < -0.01 else "Roughly equal")
            print(f"  R={ratio}  trimmed={base_acc:.3f}  "
                  f"best_steered={best_steer.condition}({best_steer.accuracy:.3f})  "
                  f"gain={gain:+.3f}  → {label}")

    print("═" * W)
