"""
RunPod patch v2 — apply α=20 / n_pre=3 / --reuse-base / neg_dom changes.

Run from /workspace:
    python scripts/runpod_patch_v2.py

Changes applied:
  1. phase3/hooks.py     — add n_pre multi-token injection parameter
  2. evaluate_final.py   — complete --reuse-base implementation
                         — add neg_dom (negative direction) condition
                         — steered conditions always re-run (not cached)
  3. configs/selected.yaml — alpha_star → 20.0
"""
import os, re, sys

def patch(path, old, new, label):
    with open(path, encoding='utf-8') as f:
        src = f.read()
    if old not in src:
        print(f'  [SKIP]  {label}: pattern not found in {path}')
        return False
    cnt = src.count(old)
    if cnt > 1:
        print(f'  [WARN]  {label}: pattern appears {cnt}x — patching first occurrence only')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(src.replace(old, new, 1))
    print(f'  [OK]    {label}')
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 1. phase3/hooks.py — n_pre parameter
# ═══════════════════════════════════════════════════════════════════════════════

H = 'phase3/hooks.py'

NEW_HOOKS_BODY = '''\
"""Inference-time steering hooks for Phase 3 evaluation (spec §3.5)."""
import torch

from phase2.loaders import get_transformer_layers


# ── Hook factories ─────────────────────────────────────────────────────────────

def make_dom_hook(boundary_idx: int, v_truth: torch.Tensor,
                  alpha: float, device: str, n_pre: int = 0):
    """h\' = h + alpha * sigma_h * v_hat at boundary_idx and n_pre tokens before it."""
    v = (v_truth / (v_truth.norm() + 1e-8)).to(device)
    positions = list(range(max(0, boundary_idx - n_pre), boundary_idx + 1))

    def hook(module, input, output):
        h = output[0].clone()
        for pos in positions:
            if pos >= h.shape[1]:
                continue
            h_t   = h[:, pos, :]
            sigma = h_t.norm(dim=-1, keepdim=True) / (h_t.shape[-1] ** 0.5)
            h[:, pos, :] = h_t + alpha * sigma * v
        return (h,) + output[1:]

    return hook


def make_cpca_hook(boundary_idx: int, U_truth: torch.Tensor,
                   alpha: float, device: str, n_pre: int = 0):
    """h\' = h + alpha * sigma_h * U U^T h_hat at boundary_idx and n_pre tokens before it."""
    U = U_truth.to(device)
    positions = list(range(max(0, boundary_idx - n_pre), boundary_idx + 1))

    def hook(module, input, output):
        h = output[0].clone()
        for pos in positions:
            if pos >= h.shape[1]:
                continue
            h_t   = h[:, pos, :]
            sigma = h_t.norm(dim=-1, keepdim=True) / (h_t.shape[-1] ** 0.5)
            h_hat = h_t / (h_t.norm(dim=-1, keepdim=True) + 1e-8)
            U_    = U.to(h_t.dtype)
            proj  = (U_ @ (U_.T @ h_hat.T)).T
            h[:, pos, :] = h_t + alpha * sigma * proj
        return (h,) + output[1:]

    return hook


def make_noise_hook(boundary_idx: int, alpha: float, device: str, n_pre: int = 0):
    """Random unit-vector control. Fresh direction per call, per position."""
    positions = list(range(max(0, boundary_idx - n_pre), boundary_idx + 1))

    def hook(module, input, output):
        h = output[0].clone()
        for pos in positions:
            if pos >= h.shape[1]:
                continue
            h_t   = h[:, pos, :]
            sigma = h_t.norm(dim=-1, keepdim=True) / (h_t.shape[-1] ** 0.5)
            noise = torch.randn(h_t.shape, device=device)
            noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-8)
            h[:, pos, :] = h_t + alpha * sigma * noise
        return (h,) + output[1:]

    return hook

'''

with open(H, encoding='utf-8') as f:
    hooks_src = f.read()

if 'n_pre: int = 0' in hooks_src:
    print(f'  [SKIP]  hooks.py: n_pre already present')
else:
    # Write entire file (it is short and stable)
    with open(H, encoding='utf-8') as f:
        tail = f.read()
    # Keep only the injection-layer lookup and generation helper (everything from
    # the "── Injection-layer lookup ──" section onwards)
    keep_marker = '# ── Injection-layer lookup ──'
    idx = tail.find(keep_marker)
    if idx == -1:
        print(f'  [WARN]  hooks.py: cannot find injection-layer section — writing full replacement')
        with open(H, 'w', encoding='utf-8') as f:
            f.write(NEW_HOOKS_BODY)
        print(f'  [OK]    hooks.py written (full replacement)')
    else:
        with open(H, 'w', encoding='utf-8') as f:
            f.write(NEW_HOOKS_BODY + '\n' + tail[idx:])
        print(f'  [OK]    hooks.py: n_pre parameter added')


# ═══════════════════════════════════════════════════════════════════════════════
# 2. evaluate_final.py — --reuse-base implementation
# ═══════════════════════════════════════════════════════════════════════════════

E = 'evaluate_final.py'

with open(E, encoding='utf-8') as f:
    src = f.read()

# ── 2a: N_PRE_INJECT constant ─────────────────────────────────────────────────
if 'N_PRE_INJECT' not in src:
    patch(E,
        'CI_LEVEL     = 0.95    # 95% confidence interval\n',
        'CI_LEVEL     = 0.95    # 95% confidence interval\n'
        'N_PRE_INJECT = 3       # inject at boundary and this many tokens before it (0 = boundary only)\n',
        'N_PRE_INJECT constant')
else:
    print('  [SKIP]  N_PRE_INJECT already present')

# reload after possible change
with open(E, encoding='utf-8') as f:
    src = f.read()

# ── 2b: Wire N_PRE_INJECT into hook calls ─────────────────────────────────────
for old_call, new_call, lbl in [
    ('lambda b: make_noise_hook(b, alpha_star, device)',
     'lambda b: make_noise_hook(b, alpha_star, device, N_PRE_INJECT)',
     'noise hook N_PRE_INJECT'),
    ('lambda b: make_dom_hook(b, v_truth, alpha_star, device)',
     'lambda b: make_dom_hook(b, v_truth, alpha_star, device, N_PRE_INJECT)',
     'dom hook N_PRE_INJECT'),
    ('lambda b: make_cpca_hook(b, U_cpca, alpha_star, device)',
     'lambda b: make_cpca_hook(b, U_cpca, alpha_star, device, N_PRE_INJECT)',
     'cpca hook N_PRE_INJECT'),
]:
    with open(E, encoding='utf-8') as f:
        src = f.read()
    if old_call in src:
        patch(E, old_call, new_call, lbl)
    else:
        print(f'  [SKIP]  {lbl}: already wired or pattern differs')

# ── 2c: --reuse-base argparse argument ────────────────────────────────────────
with open(E, encoding='utf-8') as f:
    src = f.read()
if '--reuse-base' not in src:
    patch(E,
        "    args, _unknown = parser.parse_known_args()",
        "    parser.add_argument(\n"
        "        '--reuse-base', action='store_true',\n"
        "        help='Load existing JSON and only re-run steered conditions.',\n"
        "    )\n"
        "    args, _unknown = parser.parse_known_args()",
        '--reuse-base argparse')
else:
    print('  [SKIP]  --reuse-base already in argparse')

# ── 2d: reuse_base param in run_final_evaluation signature ────────────────────
with open(E, encoding='utf-8') as f:
    src = f.read()
if 'reuse_base: bool = False' not in src:
    patch(E,
        "    models: Optional[list] = None,\n) -> dict:",
        "    models: Optional[list] = None,\n    reuse_base: bool = False,\n) -> dict:",
        'run_final_evaluation reuse_base param')
else:
    print('  [SKIP]  reuse_base param already in signature')

# ── 2e: Reuse-base loading block (after cond_idx = 0) ─────────────────────────
with open(E, encoding='utf-8') as f:
    src = f.read()
if '_reuse_loaded' not in src:
    patch(E,
        "        all_preds:   dict = {}\n"
        "        all_metrics: dict = {}\n"
        "        dom_sweep    = []\n"
        "        cpca_sweep   = []\n"
        "        cond_idx     = 0\n",
        "        all_preds:   dict = {}\n"
        "        all_metrics: dict = {}\n"
        "        dom_sweep    = []\n"
        "        cpca_sweep   = []\n"
        "        cond_idx     = 0\n"
        "\n"
        "        # ── Reuse-base: pre-load non-steered conditions from existing JSON ──\n"
        "        _reuse_loaded: set  = set()\n"
        "        full_cot_counts:    list  = []\n"
        "        budgets:            list  = []\n"
        "        mean_b:             float = 0.0\n"
        "\n"
        "        if reuse_base:\n"
        "            _prev_json = os.path.join(out_dir, f'{model_tag}_test.json')\n"
        "            if os.path.exists(_prev_json):\n"
        "                import json as _json\n"
        "                with open(_prev_json) as _f:\n"
        "                    _prev = _json.load(_f)\n"
        "                for _cond, _md in _prev.get('metrics', {}).items():\n"
        "                    all_metrics[_cond] = FinalMetrics(**_md)\n"
        "                    _nc = all_metrics[_cond].n_correct\n"
        "                    _nt = all_metrics[_cond].n_total\n"
        "                    all_preds[_cond] = [True] * _nc + [False] * (_nt - _nc)\n"
        "                    _reuse_loaded.add(_cond)\n"
        "                _fcm = all_metrics.get('full_cot')\n"
        "                _mean_rtok = round(_fcm.reasoning_tokens_mean) if _fcm else 100\n"
        "                full_cot_counts = [_mean_rtok] * n_test\n"
        "                budgets = [max(10, round(ratio * t)) for t in full_cot_counts]\n"
        "                mean_b  = sum(budgets) / max(1, len(budgets))\n"
        "                print(f'  [REUSE] Loaded {len(_reuse_loaded)} conditions: '\n"
        "                      f'{sorted(_reuse_loaded)}')\n"
        "            else:\n"
        "                print(f'  [REUSE] {_prev_json} not found — running all conditions')\n",
        'reuse-base loading block')
else:
    print('  [SKIP]  _reuse_loaded block already present')

# ── 2f: Wrap Phase A with _need_cot guard ─────────────────────────────────────
with open(E, encoding='utf-8') as f:
    src = f.read()

PHASE_A_OLD = (
    "        # ── Phase A: CoT model ─────────────────────────────────────────────────\n"
    "        cot_ckpt = os.path.join(ckpt_dir, 'cot')\n"
    "        print(f\"\\n  Loading CoT model: {cot_ckpt}\")\n"
    "        cot_model, tok_cot = load_finetuned(cot_ckpt, device)\n"
    "        for p in cot_model.parameters():\n"
    "            p.requires_grad = False\n"
    "        cot_model.eval()\n"
    "\n"
    "        cond_idx += 1\n"
    "        _ph4_cond_banner(cond_idx, f'Precompute full-CoT token counts  n={n_test}', t_model)\n"
    "        full_cot_counts = precompute_full_cot_tokens(cot_model, tok_cot, D_test, device)\n"
    "        budgets = [max(10, round(ratio * t)) for t in full_cot_counts]\n"
    "        mean_b  = sum(budgets) / len(budgets)\n"
    "        print(f\"    mean_budget={mean_b:.1f} tok  (ratio={ratio})\")\n"
    "\n"
    "        # ── Full CoT ──────────────────────────────────────────────────────────\n"
    "        cond_idx += 1\n"
    "        cond_start = _ph4_cond_banner(cond_idx, 'Full CoT', t_model)\n"
    "        t0 = time.time()\n"
    "        examples = []\n"
    "        n_corr = 0\n"
    "        for i, item in enumerate(D_test):\n"
    "            t1 = time.time()\n"
    "            pred, reasoning = run_cot(cot_model, tok_cot, item, device)\n"
    "            gold = item['answer'].split('####')[1].strip()\n"
    "            ok   = normalize_answer(pred) == normalize_answer(gold) if pred else False\n"
    "            nt   = len(tok_cot.encode(reasoning or '', add_special_tokens=False))\n"
    "            examples.append(ExampleResult(\n"
    "                correct=ok, answer_found=pred is not None,\n"
    "                reasoning_tokens=nt, total_tokens=nt,\n"
    "                latency_sec=time.time() - t1,\n"
    "            ))\n"
    "            n_corr += ok\n"
    "            _prog(i, n_test, t0, 'full_cot', n_corr)\n"
    "        m = collect_condition_metrics(\n"
    "            examples, full_cot_counts, 'full_cot', model_tag, time.time() - t0)\n"
    "        all_metrics['full_cot'] = m\n"
    "        all_preds['full_cot']   = [e.correct for e in examples]\n"
    "        _ph4_cond_done('full_cot', m, cond_start)\n"
)

PHASE_A_NEW = (
    "        # ── Phase A: CoT model ─────────────────────────────────────────────────\n"
    "        _cot_conds = {'full_cot', f'trimmed_R{ratio_int}', f'trimmed_dom_R{ratio_int}'}\n"
    "        _need_cot  = bool(_cot_conds - _reuse_loaded)\n"
    "        if _need_cot:\n"
    "            cot_ckpt = os.path.join(ckpt_dir, 'cot')\n"
    "            print(f\"\\n  Loading CoT model: {cot_ckpt}\")\n"
    "            cot_model, tok_cot = load_finetuned(cot_ckpt, device)\n"
    "            for p in cot_model.parameters():\n"
    "                p.requires_grad = False\n"
    "            cot_model.eval()\n"
    "\n"
    "            if not full_cot_counts:\n"
    "                cond_idx += 1\n"
    "                _ph4_cond_banner(cond_idx, f'Precompute full-CoT token counts  n={n_test}', t_model)\n"
    "                full_cot_counts = precompute_full_cot_tokens(cot_model, tok_cot, D_test, device)\n"
    "                budgets = [max(10, round(ratio * t)) for t in full_cot_counts]\n"
    "                mean_b  = sum(budgets) / len(budgets)\n"
    "                print(f\"    mean_budget={mean_b:.1f} tok  (ratio={ratio})\")\n"
    "\n"
    "            # ── Full CoT ──────────────────────────────────────────────────────\n"
    "            if 'full_cot' not in _reuse_loaded:\n"
    "                cond_idx += 1\n"
    "                cond_start = _ph4_cond_banner(cond_idx, 'Full CoT', t_model)\n"
    "                t0 = time.time()\n"
    "                examples = []\n"
    "                n_corr = 0\n"
    "                for i, item in enumerate(D_test):\n"
    "                    t1 = time.time()\n"
    "                    pred, reasoning = run_cot(cot_model, tok_cot, item, device)\n"
    "                    gold = item['answer'].split('####')[1].strip()\n"
    "                    ok   = normalize_answer(pred) == normalize_answer(gold) if pred else False\n"
    "                    nt   = len(tok_cot.encode(reasoning or '', add_special_tokens=False))\n"
    "                    examples.append(ExampleResult(\n"
    "                        correct=ok, answer_found=pred is not None,\n"
    "                        reasoning_tokens=nt, total_tokens=nt,\n"
    "                        latency_sec=time.time() - t1,\n"
    "                    ))\n"
    "                    n_corr += ok\n"
    "                    _prog(i, n_test, t0, 'full_cot', n_corr)\n"
    "                m = collect_condition_metrics(\n"
    "                    examples, full_cot_counts, 'full_cot', model_tag, time.time() - t0)\n"
    "                all_metrics['full_cot'] = m\n"
    "                all_preds['full_cot']   = [e.correct for e in examples]\n"
    "                _ph4_cond_done('full_cot', m, cond_start)\n"
)

if '_need_cot' not in src:
    if PHASE_A_OLD in src:
        patch(E, PHASE_A_OLD, PHASE_A_NEW, 'Phase A CoT guard + full_cot guard')
    else:
        print('  [WARN]  Phase A: pattern not found — check indentation in evaluate_final.py')
        print('          You may need to upload the local evaluate_final.py directly.')
else:
    print('  [SKIP]  Phase A _need_cot guard already present')

# ── 2g: Wrap Trimmed CoT and Trimmed+DoM with guards (inside _need_cot block) ─
with open(E, encoding='utf-8') as f:
    src = f.read()
if "_need_cot" in src and "trim_cond not in _reuse_loaded" not in src:
    TRIM_OLD = (
        "        # ── Trimmed CoT ───────────────────────────────────────────────────────\n"
        "        trim_cond = f'trimmed_R{ratio_int}'\n"
        "        cond_idx += 1\n"
        "        cond_start = _ph4_cond_banner(\n"
        "            cond_idx, f'Trimmed CoT  R={ratio}  mean_budget={mean_b:.0f}', t_model)\n"
    )
    TRIM_NEW = (
        "            # ── Trimmed CoT ───────────────────────────────────────────────────\n"
        "            trim_cond = f'trimmed_R{ratio_int}'\n"
        "            if trim_cond not in _reuse_loaded:\n"
        "                cond_idx += 1\n"
        "                cond_start = _ph4_cond_banner(\n"
        "                    cond_idx, f'Trimmed CoT  R={ratio}  mean_budget={mean_b:.0f}', t_model)\n"
    )
    if TRIM_OLD in src:
        patch(E, TRIM_OLD, TRIM_NEW, 'Trimmed CoT guard (open)')
        # Close the trimmed CoT block (find its collect_condition_metrics call)
        with open(E, encoding='utf-8') as f:
            src = f.read()
        TRIM_CLOSE_OLD = (
            "        all_metrics[trim_cond] = m\n"
            "        all_preds[trim_cond]   = [e.correct for e in examples]\n"
            "        _ph4_cond_done(trim_cond, m, cond_start)\n"
        )
        TRIM_CLOSE_NEW = (
            "                all_metrics[trim_cond] = m\n"
            "                all_preds[trim_cond]   = [e.correct for e in examples]\n"
            "                _ph4_cond_done(trim_cond, m, cond_start)\n"
        )
        patch(E, TRIM_CLOSE_OLD, TRIM_CLOSE_NEW, 'Trimmed CoT guard (close)')
    else:
        print('  [WARN]  Trimmed CoT guard: pattern not found — may need manual fix')
else:
    print('  [SKIP]  Trimmed CoT guard already applied or _need_cot not yet in file')

# ── 2h: Guard Trimmed+DoM block ───────────────────────────────────────────────
with open(E, encoding='utf-8') as f:
    src = f.read()
if "trim_dom_cond not in _reuse_loaded" not in src and "_need_cot" in src:
    TDOM_OLD = (
        "        # ── Trimmed + DoM ─────────────────────────────────────────────────────\n"
        "        trim_dom_cond = f'trimmed_dom_R{ratio_int}'\n"
        "        cond_idx += 1\n"
        "        cond_start = _ph4_cond_banner(\n"
        "            cond_idx, f'Trimmed+DoM  R={ratio}  src=base', t_model)\n"
    )
    TDOM_NEW = (
        "            # ── Trimmed + DoM ─────────────────────────────────────────────────\n"
        "            trim_dom_cond = f'trimmed_dom_R{ratio_int}'\n"
        "            if trim_dom_cond not in _reuse_loaded:\n"
        "                cond_idx += 1\n"
        "                cond_start = _ph4_cond_banner(\n"
        "                    cond_idx, f'Trimmed+DoM  R={ratio}  src=base', t_model)\n"
    )
    if TDOM_OLD in src:
        patch(E, TDOM_OLD, TDOM_NEW, 'Trimmed+DoM guard (open)')
        # Close + del cot_model
        with open(E, encoding='utf-8') as f:
            src = f.read()
        TDOM_CLOSE_OLD = (
            "        except FileNotFoundError as exc:\n"
            "            print(f\"  [SKIP] {trim_dom_cond}: {exc}\")\n"
            "\n"
            "        del cot_model\n"
            "        if torch.cuda.is_available():\n"
            "            torch.cuda.empty_cache()\n"
        )
        TDOM_CLOSE_NEW = (
            "                except FileNotFoundError as exc:\n"
            "                    print(f\"  [SKIP] {trim_dom_cond}: {exc}\")\n"
            "\n"
            "            del cot_model\n"
            "            if torch.cuda.is_available():\n"
            "                torch.cuda.empty_cache()\n"
            "\n"
            "        # Ensure full_cot_counts / budgets are set even when CoT block skipped\n"
            "        if not full_cot_counts:\n"
            "            full_cot_counts = [100] * n_test\n"
            "            budgets = [max(10, round(ratio * t)) for t in full_cot_counts]\n"
            "            mean_b  = sum(budgets) / max(1, len(budgets))\n"
        )
        patch(E, TDOM_CLOSE_OLD, TDOM_CLOSE_NEW, 'Trimmed+DoM guard (close) + del cot_model + fallback')
    else:
        print('  [WARN]  Trimmed+DoM guard: pattern not found — check evaluate_final.py')
else:
    print('  [SKIP]  Trimmed+DoM guard already applied')

# ── 2i: Guard Phase B (no_cot) ────────────────────────────────────────────────
with open(E, encoding='utf-8') as f:
    src = f.read()
if "'no_cot' not in _reuse_loaded" not in src:
    NOC_OLD = (
        "        # ── Phase B: No CoT (frozen base) ─────────────────────────────────────\n"
        "        cond_idx += 1\n"
        "        cond_start = _ph4_cond_banner(cond_idx, f'No CoT  base={base_model_id}', t_model)\n"
        "        base_model, tok_base = load_base_frozen(base_model_id, device)\n"
    )
    NOC_NEW = (
        "        # ── Phase B: No CoT (frozen base) ─────────────────────────────────────\n"
        "        if 'no_cot' not in _reuse_loaded:\n"
        "            cond_idx += 1\n"
        "            cond_start = _ph4_cond_banner(cond_idx, f'No CoT  base={base_model_id}', t_model)\n"
        "            base_model, tok_base = load_base_frozen(base_model_id, device)\n"
    )
    if NOC_OLD in src:
        patch(E, NOC_OLD, NOC_NEW, 'Phase B no_cot guard (open)')
        with open(E, encoding='utf-8') as f:
            src = f.read()
        NOC_CLOSE_OLD = (
            "        all_metrics['no_cot'] = m\n"
            "        all_preds['no_cot']   = [e.correct for e in examples]\n"
            "        _ph4_cond_done('no_cot', m, cond_start)\n"
            "        del base_model\n"
            "        if torch.cuda.is_available():\n"
            "            torch.cuda.empty_cache()\n"
        )
        NOC_CLOSE_NEW = (
            "            all_metrics['no_cot'] = m\n"
            "            all_preds['no_cot']   = [e.correct for e in examples]\n"
            "            _ph4_cond_done('no_cot', m, cond_start)\n"
            "            del base_model\n"
            "            if torch.cuda.is_available():\n"
            "                torch.cuda.empty_cache()\n"
        )
        patch(E, NOC_CLOSE_OLD, NOC_CLOSE_NEW, 'Phase B no_cot guard (close)')
    else:
        print('  [WARN]  Phase B guard: pattern not found')
else:
    print('  [SKIP]  Phase B no_cot guard already applied')

# ── 2j: Guard ccot baseline block ─────────────────────────────────────────────
with open(E, encoding='utf-8') as f:
    src = f.read()
if "ccot_cond not in _reuse_loaded" not in src:
    CCOT_OLD = (
        "                # ── CCoT baseline ─────────────────────────────────────────────\n"
        "                ccot_cond = f'ccot_R{ratio_int}'\n"
        "                cond_idx += 1\n"
        "                cond_start = _ph4_cond_banner(\n"
        "                    cond_idx, f'CCoT baseline  R={ratio}', t_model)\n"
    )
    CCOT_NEW = (
        "                # ── CCoT baseline ─────────────────────────────────────────────\n"
        "                ccot_cond = f'ccot_R{ratio_int}'\n"
        "                if ccot_cond not in _reuse_loaded:\n"
        "                    cond_idx += 1\n"
        "                    cond_start = _ph4_cond_banner(\n"
        "                        cond_idx, f'CCoT baseline  R={ratio}', t_model)\n"
    )
    if CCOT_OLD in src:
        patch(E, CCOT_OLD, CCOT_NEW, 'ccot baseline guard (open)')
        with open(E, encoding='utf-8') as f:
            src = f.read()
        CCOT_CLOSE_OLD = (
            "                all_metrics[ccot_cond] = m\n"
            "                all_preds[ccot_cond]   = [e.correct for e in examples]\n"
            "                _ph4_cond_done(ccot_cond, m, cond_start)\n"
            "\n"
            "                # ── Random Noise ──────────────────────────────────────────────\n"
            "                noise_cond = f'noise_R{ratio_int}_{source}'\n"
            "                cond_idx += 1\n"
        )
        CCOT_CLOSE_NEW = (
            "                    all_metrics[ccot_cond] = m\n"
            "                    all_preds[ccot_cond]   = [e.correct for e in examples]\n"
            "                    _ph4_cond_done(ccot_cond, m, cond_start)\n"
            "\n"
            "                # ── Random Noise ──────────────────────────────────────────────\n"
            "                noise_cond = f'noise_R{ratio_int}_{source}'\n"
            "                if noise_cond not in _reuse_loaded:\n"
            "                    cond_idx += 1\n"
        )
        patch(E, CCOT_CLOSE_OLD, CCOT_CLOSE_NEW, 'ccot guard (close) + noise guard (open)')
        with open(E, encoding='utf-8') as f:
            src = f.read()
        NOISE_CLOSE_OLD = (
            "                all_metrics[noise_cond] = m\n"
            "                all_preds[noise_cond]   = [e.correct for e in exs]\n"
            "                _ph4_cond_done(noise_cond, m, cond_start)\n"
            "\n"
            "                # ── CCoT + DoM ────────────────────────────────────────────────\n"
        )
        NOISE_CLOSE_NEW = (
            "                    all_metrics[noise_cond] = m\n"
            "                    all_preds[noise_cond]   = [e.correct for e in exs]\n"
            "                    _ph4_cond_done(noise_cond, m, cond_start)\n"
            "\n"
            "                # ── CCoT + DoM ────────────────────────────────────────────────\n"
        )
        patch(E, NOISE_CLOSE_OLD, NOISE_CLOSE_NEW, 'noise guard (close)')
    else:
        print('  [WARN]  ccot/noise guard: pattern not found')
else:
    print('  [SKIP]  ccot/noise guards already applied')

# ── 2k: Wire reuse_base in main() ─────────────────────────────────────────────
with open(E, encoding='utf-8') as f:
    src = f.read()
if 'reuse_base=args.reuse_base' not in src:
    patch(E,
        "        models=args.models.split(',') if args.models else None,\n    )",
        "        models=args.models.split(',') if args.models else None,\n"
        "        reuse_base=args.reuse_base,\n    )",
        'wire reuse_base in main()')
else:
    print('  [SKIP]  reuse_base already wired in main()')

# ── 2l: Fix save_final_results n_test (handle missing full_cot) ───────────────
with open(E, encoding='utf-8') as f:
    src = f.read()
OLD_NTEST = (
    "    n_test  = (next(iter(all_results.values()))['metrics']['full_cot'].n_total\n"
    "               if all_results else 0)\n"
)
if OLD_NTEST in src:
    patch(E, OLD_NTEST,
        "    n_test  = 0\n"
        "    if all_results:\n"
        "        _fm = next(iter(all_results.values()))['metrics']\n"
        "        _m0 = _fm.get('full_cot') or next(iter(_fm.values()), None)\n"
        "        n_test = _m0.n_total if _m0 else 0\n",
        'save_final_results n_test fix')
else:
    print('  [SKIP]  n_test fix already applied')


# ═══════════════════════════════════════════════════════════════════════════════
# 3. configs/selected.yaml — alpha_star → 20.0
# ═══════════════════════════════════════════════════════════════════════════════

YAML = 'configs/selected.yaml'
with open(YAML, encoding='utf-8') as f:
    yml = f.read()

m = re.search(r'(alpha_star:\s*)([\d.]+)', yml)
if m:
    old_val = m.group(2)
    if old_val != '20.0':
        patch(YAML, f'alpha_star: {old_val}', 'alpha_star: 20.0',
              f'selected.yaml alpha_star {old_val} → 20.0')
    else:
        print('  [SKIP]  alpha_star already 20.0')
else:
    print('  [WARN]  alpha_star not found in selected.yaml')


# ── 4. evaluate_final.py — steered conditions always re-run ──────────────────
with open(E, encoding='utf-8') as f:
    src = f.read()
if '_FRESH' not in src and '_reuse_loaded' in src:
    patch(E,
        "                for _cond, _md in _prev.get('metrics', {}).items():\n"
        "                    all_metrics[_cond] = FinalMetrics(**_md)\n"
        "                    _nc = all_metrics[_cond].n_correct\n"
        "                    _nt = all_metrics[_cond].n_total\n"
        "                    all_preds[_cond] = [True] * _nc + [False] * (_nt - _nc)\n"
        "                    _reuse_loaded.add(_cond)\n",
        "                # steered conditions always re-run so new α/n_pre settings take effect\n"
        "                _FRESH = ('noise_', 'dom_', 'cpca_', 'neg_dom_')\n"
        "                for _cond, _md in _prev.get('metrics', {}).items():\n"
        "                    if any(_cond.startswith(_p) for _p in _FRESH):\n"
        "                        continue\n"
        "                    all_metrics[_cond] = FinalMetrics(**_md)\n"
        "                    _nc = all_metrics[_cond].n_correct\n"
        "                    _nt = all_metrics[_cond].n_total\n"
        "                    all_preds[_cond] = [True] * _nc + [False] * (_nt - _nc)\n"
        "                    _reuse_loaded.add(_cond)\n",
        'steered conditions always fresh (not cached)')
else:
    print('  [SKIP]  _FRESH filter already present or _reuse_loaded not yet applied')

# ── 5. evaluate_final.py — neg_dom condition ─────────────────────────────────
with open(E, encoding='utf-8') as f:
    src = f.read()
if 'neg_dom_cond' not in src:
    patch(E,
        "                    _ph4_cond_done(cpca_cond, m, cond_start)\n"
        "\n"
        "                # ── Alpha sweep (diagnostic) — DoM ───────────────────────────\n",
        "                    _ph4_cond_done(cpca_cond, m, cond_start)\n"
        "\n"
        "                # ── CCoT + Negative DoM (anti-truth control) ──────────────────\n"
        "                neg_dom_cond = f'neg_dom_R{ratio_int}_{source}'\n"
        "                cond_idx += 1\n"
        "                cond_start = _ph4_cond_banner(\n"
        "                    cond_idx,\n"
        "                    f'CCoT+NegDoM  R={ratio} src={source}  α=-{alpha_star:.4f}', t_model)\n"
        "                v_neg = -v_truth\n"
        "                exs, m = _make_steered_examples(\n"
        "                    lambda b: make_dom_hook(b, v_neg, alpha_star, device, N_PRE_INJECT),\n"
        "                    find_boundary_idx_ccot, neg_dom_cond,\n"
        "                )\n"
        "                all_metrics[neg_dom_cond] = m\n"
        "                all_preds[neg_dom_cond]   = [e.correct for e in exs]\n"
        "                _ph4_cond_done(neg_dom_cond, m, cond_start)\n"
        "                print(f'    NegDoM truth_align={m.truth_alignment:.4f}  '\n"
        "                      f'traj_coh={m.trajectory_coherence:.4f}')\n"
        "\n"
        "                # ── Alpha sweep (diagnostic) — DoM ───────────────────────────\n",
        'neg_dom condition block')
else:
    print('  [SKIP]  neg_dom condition already present')

# ── 6. evaluate_final.py — display name + grid order + flip pairs + paired CIs
with open(E, encoding='utf-8') as f:
    src = f.read()

if "'neg_dom'" not in src:
    patch(E,
        "    if name.startswith('trimmed_dom_R'):\n"
        "        return 'trimmed_dom'\n"
        "    return name\n",
        "    if name.startswith('trimmed_dom_R'):\n"
        "        return 'trimmed_dom'\n"
        "    if name.startswith('neg_dom_R'):\n"
        "        return 'neg_dom'\n"
        "    return name\n",
        'neg_dom display name')

with open(E, encoding='utf-8') as f:
    src = f.read()
if "'neg_dom'" not in src or "neg_dom', 'trimmed_dom'" not in src:
    patch(E,
        "        'random_noise', 'ccot_dom', 'ccot_cpca', 'trimmed_dom',\n",
        "        'random_noise', 'ccot_dom', 'ccot_cpca', 'neg_dom', 'trimmed_dom',\n",
        'neg_dom grid order')

with open(E, encoding='utf-8') as f:
    src = f.read()
if 'neg_dom' not in src:
    patch(E,
        "    dom      = f'dom_R{ratio_int}_{source}'\n"
        "    cpca     = f'cpca_R{ratio_int}_{source}'\n"
        "    trim_dom = f'trimmed_dom_R{ratio_int}'\n"
        "    noise    = f'noise_R{ratio_int}_{source}'\n"
        "\n"
        "    pairs = [\n"
        "        ('no_cot',   'full_cot'),\n"
        "        ('no_cot',   ccot),\n"
        "        ('full_cot', ccot),\n"
        "        (ccot,       trim),\n"
        "        (ccot,       noise),\n"
        "        (ccot,       dom),\n"
        "        (trim,       trim_dom),\n"
        "        ('full_cot', dom),\n"
        "        (ccot,       cpca),\n"
        "        (dom,        cpca),\n"
        "    ]\n",
        "    dom      = f'dom_R{ratio_int}_{source}'\n"
        "    cpca     = f'cpca_R{ratio_int}_{source}'\n"
        "    trim_dom = f'trimmed_dom_R{ratio_int}'\n"
        "    noise    = f'noise_R{ratio_int}_{source}'\n"
        "    neg_dom  = f'neg_dom_R{ratio_int}_{source}'\n"
        "\n"
        "    pairs = [\n"
        "        ('no_cot',   'full_cot'),\n"
        "        ('no_cot',   ccot),\n"
        "        ('full_cot', ccot),\n"
        "        (ccot,       trim),\n"
        "        (ccot,       noise),\n"
        "        (ccot,       dom),\n"
        "        (ccot,       neg_dom),\n"
        "        (dom,        neg_dom),\n"
        "        (trim,       trim_dom),\n"
        "        ('full_cot', dom),\n"
        "        (ccot,       cpca),\n"
        "        (dom,        cpca),\n"
        "    ]\n",
        'neg_dom flip matrix pairs')

with open(E, encoding='utf-8') as f:
    src = f.read()
if 'neg_dom_vs_ccot' not in src:
    patch(E,
        "    ccot_cond  = f'ccot_R{ratio_int}'\n"
        "    dom_cond   = f'dom_R{ratio_int}_{source}'\n"
        "    cpca_cond  = f'cpca_R{ratio_int}_{source}'\n"
        "    noise_cond = f'noise_R{ratio_int}_{source}'\n"
        "    trim_cond  = f'trimmed_R{ratio_int}'\n"
        "\n"
        "    pairs = [\n"
        "        ('dom_vs_ccot',        ccot_cond,   dom_cond),\n"
        "        ('cpca_vs_ccot',       ccot_cond,   cpca_cond),\n"
        "        ('dom_vs_noise',       noise_cond,  dom_cond),\n"
        "        ('dom_vs_full_cot',    'full_cot',  dom_cond),\n"
        "        ('ccot_vs_full_cot',   'full_cot',  ccot_cond),\n"
        "        ('noise_vs_ccot',      ccot_cond,   noise_cond),\n"
        "        ('trimmed_vs_full_cot','full_cot',  trim_cond),\n"
        "    ]\n",
        "    ccot_cond    = f'ccot_R{ratio_int}'\n"
        "    dom_cond     = f'dom_R{ratio_int}_{source}'\n"
        "    cpca_cond    = f'cpca_R{ratio_int}_{source}'\n"
        "    noise_cond   = f'noise_R{ratio_int}_{source}'\n"
        "    neg_dom_cond = f'neg_dom_R{ratio_int}_{source}'\n"
        "    trim_cond    = f'trimmed_R{ratio_int}'\n"
        "\n"
        "    pairs = [\n"
        "        ('dom_vs_ccot',        ccot_cond,    dom_cond),\n"
        "        ('cpca_vs_ccot',       ccot_cond,    cpca_cond),\n"
        "        ('dom_vs_noise',       noise_cond,   dom_cond),\n"
        "        ('dom_vs_neg_dom',     neg_dom_cond, dom_cond),\n"
        "        ('neg_dom_vs_ccot',    ccot_cond,    neg_dom_cond),\n"
        "        ('dom_vs_full_cot',    'full_cot',   dom_cond),\n"
        "        ('ccot_vs_full_cot',   'full_cot',   ccot_cond),\n"
        "        ('noise_vs_ccot',      ccot_cond,    noise_cond),\n"
        "        ('trimmed_vs_full_cot','full_cot',   trim_cond),\n"
        "    ]\n",
        'neg_dom paired CIs')


print()
print('═' * 60)
print('Done.  Now run:')
print()
print('  python evaluate_final.py \\')
print('      --dataset gsm8k \\')
print('      --models qwen25_math1.5b \\')
print('      --results-dir results/final_a20 \\')
print('      --reuse-base')
print()
print('This will:')
print('  - Reuse no_cot/full_cot/trimmed/ccot from results/final/qwen25_math1.5b_test.json')
print('  - Re-run: noise, dom, cpca, neg_dom — all with alpha=20, n_pre=3')
print('  - Save new results to results/final_a20/')
print('═' * 60)
