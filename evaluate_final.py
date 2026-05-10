"""Phase 4: single-pass D_test evaluation with locked Phase 3 configs.

D_test is loaded exactly once. This file is run exactly once.
No hyperparameter is changed after Phase 3. No result inspires a rerun.

Phase 5 (SVAMP transfer): use ``--dataset svamp`` and ``--results-dir`` (e.g.
``results/final_svamp_transfer``) so GSM8K final JSON is not overwritten; vectors
and checkpoints still come from ``vectors/{winner}/<model>/`` (GSM8K pipeline).
"""
import argparse
import glob as _glob
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from utils.data import load_test_set
from utils.dataset_paths import get_active_dataset_id, init_project_dataset
from phase1.inference import (
    extract_answer,
    extract_reasoning_span,
    load_base_frozen,
    load_finetuned,
    normalize_answer,
    run_cot,
    run_no_cot,
    run_trimmed_cot,
)
from phase2.loaders import (
    find_boundary_idx_base,
    find_boundary_idx_ccot,
    get_transformer_layers,
)
from phase3.hooks import (
    get_injection_layer,
    make_cpca_hook,
    make_dom_hook,
    make_noise_hook,
)

# ── Constants ──────────────────────────────────────────────────────────────────

N_BOOTSTRAP  = 1000    # resamples for all bootstrap CIs
CI_SEED      = 0       # fixed seed → reproducible CIs across re-runs
CI_LEVEL     = 0.95    # 95% confidence interval

MODEL_TAGS = ['llama32_3b', 'phi2', 'qwen25_3b', 'qwen25_math1.5b']
MODEL_ID_MAP = {
    'llama32_3b':      'meta-llama/Llama-3.2-3B',
    'phi2':            'microsoft/phi-2',
    'qwen25_3b':       'Qwen/Qwen2.5-3B',
    'qwen25_math1.5b': 'Qwen/Qwen2.5-Math-1.5B',
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class FlipMatrix:
    """Counts of correct/incorrect transitions from condition_a to condition_b."""
    F00: int   # stable correct  (a right, b right)
    F01: int   # degradation     (a right, b wrong)
    F10: int   # improvement     (a wrong, b right)
    F11: int   # stable wrong    (a wrong, b wrong)
    condition_a: str
    condition_b: str
    model_tag: str

    @property
    def total(self) -> int:
        return self.F00 + self.F01 + self.F10 + self.F11

    @property
    def improvement_rate(self) -> float:
        denom = self.F10 + self.F11
        return self.F10 / denom if denom else 0.0

    @property
    def degradation_rate(self) -> float:
        denom = self.F00 + self.F01
        return self.F01 / denom if denom else 0.0

    @property
    def net_gain(self) -> float:
        return self.F10 - self.F01

    @property
    def agreement(self) -> float:
        return (self.F00 + self.F11) / self.total if self.total else 0.0


@dataclass
class BootstrapResult:
    """95% bootstrap CI for a single accuracy estimate or a paired difference."""
    point:       float   # point estimate (accuracy or Δ-accuracy)
    lower:       float   # 2.5th percentile
    upper:       float   # 97.5th percentile
    significant: bool = False   # True when lower > 0 (difference CIs only)

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2.0

    def fmt(self) -> str:
        return f"{self.point:.3f}  [{self.lower:.3f}, {self.upper:.3f}]"

    def fmt_diff(self) -> str:
        sig = "  ✓" if self.significant else ""
        return f"{self.point:+.3f}  [{self.lower:+.3f}, {self.upper:+.3f}]{sig}"


@dataclass
class ExampleResult:
    correct: bool
    answer_found: bool
    reasoning_tokens: int
    total_tokens: int
    latency_sec: float
    traj_coherence: float = 0.0
    truth_align: float = 0.0


@dataclass
class FinalMetrics:
    condition: str
    model_tag: str
    accuracy: float
    n_correct: int
    n_total: int
    reasoning_tokens_mean: float
    reasoning_tokens_std: float
    reasoning_tokens_min: float
    reasoning_tokens_max: float
    actual_ratio_mean: float
    total_tokens_mean: float
    latency_mean: float
    latency_std: float
    latency_p50: float
    latency_p95: float
    wall_time_total: float
    answer_found_rate: float
    trajectory_coherence: float = 0.0
    truth_alignment: float = 0.0
    # Populated after model evaluation via compute_condition_cis — zero until then
    ci_lower_95: float = 0.0
    ci_upper_95: float = 0.0


# ── Latent metric functions ────────────────────────────────────────────────────

def trajectory_coherence(latent_states: list) -> float:
    """Mean cosine similarity between consecutive hidden states at L_star."""
    if len(latent_states) < 2:
        return 0.0
    sims = []
    for h_t, h_t1 in zip(latent_states[:-1], latent_states[1:]):
        cos = F.cosine_similarity(
            h_t.float().reshape(1, -1),
            h_t1.float().reshape(1, -1),
            dim=-1,
        ).item()
        sims.append(cos)
    return float(np.mean(sims)) if sims else 0.0


def truth_alignment(latent_states: list, v_hat: torch.Tensor) -> float:
    """Mean cosine similarity between hidden states at L_star and the truth direction."""
    if not latent_states:
        return 0.0
    sims = []
    for h_t in latent_states:
        cos = F.cosine_similarity(
            h_t.float().reshape(1, -1),
            v_hat.float().reshape(1, -1),
            dim=-1,
        ).item()
        sims.append(cos)
    return float(np.mean(sims)) if sims else 0.0


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    correct_array: list,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = CI_LEVEL,
    seed: int = CI_SEED,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for a single accuracy estimate.
    The model is run exactly once; the 1,000 resamples are pure numpy.

    Parameters
    ----------
    correct_array : list of int/bool  (1 = correct, 0 = wrong), length = n_test
    Returns (point_estimate, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(correct_array, dtype=float)
    n   = len(arr)
    point = float(arr.mean())

    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx       = rng.integers(0, n, size=n)
        boot[i]   = arr[idx].mean()

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot, 100.0 * alpha / 2.0))
    upper = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return point, lower, upper


def bootstrap_ci_difference(
    results_a: list,
    results_b: list,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = CI_LEVEL,
    seed: int = CI_SEED,
) -> tuple[float, float, float, bool]:
    """
    Paired percentile bootstrap CI on (accuracy_b − accuracy_a).
    Paired = same resampled indices for A and B, which accounts for the
    question-level correlation when both conditions see the same test items.

    Returns (point, lower, upper, significant) where
    significant = True iff the CI excludes 0 (i.e. lower > 0).
    """
    rng = np.random.default_rng(seed)
    a   = np.asarray(results_a, dtype=float)
    b   = np.asarray(results_b, dtype=float)
    n   = len(a)
    point = float(b.mean() - a.mean())

    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx     = rng.integers(0, n, size=n)
        boot[i] = b[idx].mean() - a[idx].mean()

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot, 100.0 * alpha / 2.0))
    upper = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return point, lower, upper, bool(lower > 0.0)


# ── CI aggregation ─────────────────────────────────────────────────────────────

def compute_condition_cis(
    all_preds: dict,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = CI_SEED,
) -> dict:
    """Per-condition bootstrap CI. Returns {condition_name: BootstrapResult}."""
    cis = {}
    for cond, preds in all_preds.items():
        pt, lo, hi = bootstrap_ci(preds, n_bootstrap=n_bootstrap, seed=seed)
        cis[cond]  = BootstrapResult(point=pt, lower=lo, upper=hi)
    return cis


def compute_paired_cis(
    all_preds: dict,
    ratio_int: int,
    source: str,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = CI_SEED,
) -> dict:
    """
    Paired bootstrap CIs on (acc_b − acc_a) for the key comparisons needed
    to support causal claims. Significant = 95% CI excludes 0.

    Covers:
      - steered vs unsteered (primary claim)
      - steered vs random noise (direction specificity)
      - steered vs full CoT   (efficiency claim)
      - ccot vs full CoT      (compression cost)
      - trimmed vs ccot       (mechanism baseline)
    """
    ccot_cond  = f'ccot_R{ratio_int}'
    dom_cond   = f'dom_R{ratio_int}_{source}'
    cpca_cond  = f'cpca_R{ratio_int}_{source}'
    noise_cond = f'noise_R{ratio_int}_{source}'
    trim_cond  = f'trimmed_R{ratio_int}'

    pairs = [
        ('dom_vs_ccot',        ccot_cond,   dom_cond),
        ('cpca_vs_ccot',       ccot_cond,   cpca_cond),
        ('dom_vs_noise',       noise_cond,  dom_cond),
        ('dom_vs_full_cot',    'full_cot',  dom_cond),
        ('ccot_vs_full_cot',   'full_cot',  ccot_cond),
        ('noise_vs_ccot',      ccot_cond,   noise_cond),
        ('trimmed_vs_full_cot','full_cot',  trim_cond),
    ]

    cis: dict = {}
    for name, cond_a, cond_b in pairs:
        if cond_a not in all_preds or cond_b not in all_preds:
            continue
        pt, lo, hi, sig = bootstrap_ci_difference(
            all_preds[cond_a], all_preds[cond_b],
            n_bootstrap=n_bootstrap, seed=seed,
        )
        cis[name] = BootstrapResult(point=pt, lower=lo, upper=hi, significant=sig)
    return cis


# ── Small helpers ──────────────────────────────────────────────────────────────

def _score_text(text: str, gold: str) -> bool:
    pred = extract_answer(text)
    return normalize_answer(pred) == normalize_answer(gold) if pred else False


def _load_best_config(results_dir: str) -> dict:
    yaml_path = os.path.join(results_dir, 'phase3_best_config.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"phase3_best_config.yaml not found: {yaml_path}")
    try:
        import yaml
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        cfg = {}
        with open(yaml_path) as f:
            for line in f:
                line = line.strip()
                if ':' not in line or line.startswith('#'):
                    continue
                k, _, v = line.partition(':')
                v = v.strip()
                if v == 'null' or v == '':
                    cfg[k.strip()] = None
                else:
                    try:
                        cfg[k.strip()] = float(v) if '.' in v else int(v)
                    except ValueError:
                        cfg[k.strip()] = v
        return cfg


def _load_phase3_best_configs(
    cfg: dict,
    winning_config: str,
    results_base: str,
) -> dict:
    """Return per-model locked Phase 3 configs, using cfg first and files as fallback."""
    phase3_best = dict(cfg.get('phase3_best') or {})
    if phase3_best:
        return phase3_best

    best_by_model = {}
    for model_tag in MODEL_TAGS:
        results_dir = os.path.join(results_base, winning_config, model_tag)
        best_path = os.path.join(results_dir, 'phase3_best_config.yaml')
        if not os.path.exists(best_path):
            continue
        best_by_model[model_tag] = _load_best_config(results_dir)
    return best_by_model


def _load_meta_file(vectors_dir: str) -> dict:
    path = os.path.join(vectors_dir, 'phase2_meta.json')
    with open(path) as f:
        return json.load(f)


def _load_dom(vectors_dir: str, source: str) -> torch.Tensor:
    path = os.path.join(vectors_dir, f'{source}_dom.pt')
    return torch.load(path, map_location='cpu')['v_truth']


def _load_cpca(vectors_dir: str, source: str, r_final: int) -> torch.Tensor:
    target = os.path.join(vectors_dir, f'{source}_cpca_r{r_final}.pt')
    if os.path.exists(target):
        return torch.load(target, map_location='cpu')['U_truth']
    files = sorted(_glob.glob(os.path.join(vectors_dir, f'{source}_cpca_r*.pt')))
    if not files:
        raise FileNotFoundError(f"No cPCA file for source={source} in {vectors_dir}")
    return torch.load(files[-1], map_location='cpu')['U_truth']


def _load_alpha_file(vectors_dir: str, source: str) -> float:
    path = os.path.join(vectors_dir, f'{source}_alpha_star.pt')
    return torch.load(path, map_location='cpu').item()


def _load_selected_yaml(path: str = 'configs/selected.yaml') -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        cfg = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if ':' not in line or line.startswith('#'):
                    continue
                k, _, v = line.partition(':')
                v = v.strip()
                try:
                    cfg[k.strip()] = int(v)
                except ValueError:
                    try:
                        cfg[k.strip()] = float(v)
                    except ValueError:
                        cfg[k.strip()] = v
        return cfg


def _display_condition_name(name: str) -> str:
    if name.startswith('ccot_R'):
        return 'ccot'
    if name.startswith('trimmed_R'):
        return 'trimmed_cot'
    if name.startswith('noise_R'):
        return 'random_noise'
    if name.startswith('dom_R'):
        return 'ccot_dom'
    if name.startswith('cpca_R'):
        return 'ccot_cpca'
    if name.startswith('trimmed_dom_R'):
        return 'trimmed_dom'
    return name


def _display_pair_name(cond_a: str, cond_b: str) -> str:
    return f"{_display_condition_name(cond_a)}_vs_{_display_condition_name(cond_b)}"


def _display_grid_order(conditions: list[str]) -> list[str]:
    preferred = [
        'no_cot', 'full_cot', 'ccot', 'trimmed_cot',
        'random_noise', 'ccot_dom', 'ccot_cpca', 'trimmed_dom',
    ]
    order = []
    for name in preferred:
        if name in conditions:
            order.append(name)
    for name in conditions:
        if name not in order:
            order.append(name)
    return order


def _metric_by_prefix(metrics: dict, prefix: str):
    for key, value in metrics.items():
        if key == prefix or key.startswith(prefix + '_'):
            return value
    return None


def _serialize_flip_matrix(fm: FlipMatrix) -> dict:
    return {
        'condition_a': fm.condition_a,
        'condition_b': fm.condition_b,
        'F00': fm.F00,
        'F01': fm.F01,
        'F10': fm.F10,
        'F11': fm.F11,
        'improvement_rate': fm.improvement_rate,
        'degradation_rate': fm.degradation_rate,
        'net_gain': fm.net_gain,
        'agreement': fm.agreement,
        'model_tag': fm.model_tag,
    }


def _print_flip_grid(grid: dict):
    conditions = grid['conditions']
    net_gain = grid['net_gain']
    print(f"\nFull net-gain flip grid ({grid['model_tag']}):")
    print(f"{'':>20}", end='')
    for c in conditions:
        print(f"{_display_condition_name(c)[:10]:>12}", end='')
    print()
    for ca in conditions:
        print(f"{_display_condition_name(ca)[:20]:<20}", end='')
        for cb in conditions:
            val = net_gain[ca][cb]
            marker = f"{val:+d}" if val != 0 else '  —'
            print(f"{marker:>12}", end='')
        print()


def _build_summary(all_results: dict, n_test: int) -> dict:
    summary = {
        'n_test': n_test,
        'models': list(all_results.keys()),
        'accuracy_table': {},
        'latent_metrics_table': {},
        'primary_flip_matrices': {},
        'mechanism_gain_table': {},
        'specificity_table': {},
        'flip_grids': {},
    }

    for model_tag, data in all_results.items():
        metrics = data['metrics']
        full_cot_metric = _metric_by_prefix(metrics, 'full_cot')
        full_cot_acc = full_cot_metric.accuracy if full_cot_metric else 0.0
        summary['accuracy_table'][model_tag] = {
            _display_condition_name(cond): {
                'accuracy': m.accuracy,
                'n_correct': m.n_correct,
                'n_total': m.n_total,
            }
            for cond, m in metrics.items()
        }
        summary['latent_metrics_table'][model_tag] = {
            _display_condition_name(cond): {
                'trajectory_coherence': m.trajectory_coherence,
                'truth_alignment': m.truth_alignment,
            }
            for cond, m in metrics.items()
        }
        summary['primary_flip_matrices'][model_tag] = {
            _display_pair_name(fm.condition_a, fm.condition_b): _serialize_flip_matrix(fm)
            for fm in data['flip_matrices']
        }
        summary['mechanism_gain_table'][model_tag] = {
            _display_condition_name(cond): round(m.accuracy - full_cot_acc, 4)
            for cond, m in metrics.items()
            if cond != 'full_cot'
        }
        summary['specificity_table'][model_tag] = {
            'random_noise_truth_alignment': (_metric_by_prefix(metrics, 'noise').truth_alignment
                                             if _metric_by_prefix(metrics, 'noise') else 0.0),
            'ccot_dom_truth_alignment': (_metric_by_prefix(metrics, 'dom').truth_alignment
                                         if _metric_by_prefix(metrics, 'dom') else 0.0),
            'ccot_cpca_truth_alignment': (_metric_by_prefix(metrics, 'cpca').truth_alignment
                                          if _metric_by_prefix(metrics, 'cpca') else 0.0),
        }
        summary['flip_grids'][model_tag] = data['flip_grid']

    # ── Confidence intervals ───────────────────────────────────────────────────
    summary['confidence_intervals'] = {
        model_tag: {
            'n_bootstrap':  N_BOOTSTRAP,
            'ci_level':     CI_LEVEL,
            'condition_cis': {
                _display_condition_name(k): {
                    'accuracy':   v.point,
                    'ci_lower':   v.lower,
                    'ci_upper':   v.upper,
                    'half_width': v.half_width,
                }
                for k, v in data.get('condition_cis', {}).items()
            },
            'paired_cis': {
                k: {
                    'delta':       v.point,
                    'ci_lower':    v.lower,
                    'ci_upper':    v.upper,
                    'significant': v.significant,
                }
                for k, v in data.get('paired_cis', {}).items()
            },
        }
        for model_tag, data in all_results.items()
    }

    return summary


# ── Core generation helpers ────────────────────────────────────────────────────

def precompute_full_cot_tokens(
    cot_model, tokenizer, D_test: list, device: str
) -> list[int]:
    """Run full CoT on D_test and return per-example reasoning token counts."""
    counts = []
    cot_model.eval()
    for item in D_test:
        enc = tokenizer(
            f"Question: {item['question']}\n\nReasoning:",
            return_tensors='pt',
        ).to(device)
        with torch.no_grad():
            out = cot_model.generate(
                **enc, do_sample=False, max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
        )
        reasoning = generated.split('\n\nAnswer:')[0].strip()
        counts.append(len(tokenizer.encode(reasoning, add_special_tokens=False)))
    return counts


def run_steered_with_metrics(
    model,
    tokenizer,
    prompt: str,
    item: dict,
    hook_fn,              # pre-created hook (None for unsteered baseline)
    L_star: int,
    v_truth: torch.Tensor,  # unit-normalised truth vector on correct device
    device: str,
    max_new_tokens: int = 256,
) -> ExampleResult:
    """
    Greedy decode with optional steering hook.
    Captures hidden states at L_star for trajectory_coherence and truth_alignment.
    hook_fn is expected to already embed the boundary_idx in its closure.
    """
    gold     = item['answer'].split('####')[1].strip()
    captured: list[torch.Tensor] = []
    v_hat    = (v_truth / (v_truth.norm() + 1e-8)) if v_truth is not None else None

    layers = get_transformer_layers(model)

    def _capture(module, input, output):
        h = output[0]
        captured.append(h[:, -1, :].detach().clone())
        return output

    handles = []
    if hook_fn is not None:
        handles.append(layers[L_star].register_forward_hook(hook_fn))
    handles.append(layers[L_star].register_forward_hook(_capture))

    t0 = time.time()
    try:
        enc = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][enc['input_ids'].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()

    latency    = time.time() - t0
    found      = extract_answer(text) is not None
    ok         = _score_text(text, gold)
    reasoning_text = extract_reasoning_span(text)
    n_reasoning = len(tokenizer.encode(reasoning_text, add_special_tokens=False))

    tc = trajectory_coherence(captured)
    ta = truth_alignment(captured, v_hat) if v_hat is not None else 0.0

    return ExampleResult(
        correct=ok,
        answer_found=found,
        reasoning_tokens=n_reasoning,
        total_tokens=int(len(enc['input_ids'][0]) + len(generated)),
        latency_sec=latency,
        traj_coherence=tc,
        truth_align=ta,
    )


def collect_condition_metrics(
    examples: list[ExampleResult],
    full_cot_counts: list[int],
    condition: str,
    model_tag: str,
    wall_time: float,
) -> FinalMetrics:
    n        = len(examples)
    n_correct = sum(e.correct for e in examples)
    r_toks   = [e.reasoning_tokens for e in examples]
    lats     = [e.latency_sec      for e in examples]
    ratios   = [
        e.reasoning_tokens / max(full, 1)
        for e, full in zip(examples, full_cot_counts)
    ]
    mean_r    = float(np.mean(r_toks)) if r_toks else 0.0

    return FinalMetrics(
        condition=condition,
        model_tag=model_tag,
        accuracy=n_correct / n if n else 0.0,
        n_correct=n_correct,
        n_total=n,
        reasoning_tokens_mean=mean_r,
        reasoning_tokens_std=float(np.std(r_toks))          if r_toks else 0.0,
        reasoning_tokens_min=float(np.min(r_toks))          if r_toks else 0.0,
        reasoning_tokens_max=float(np.max(r_toks))          if r_toks else 0.0,
        actual_ratio_mean=float(np.mean(ratios))             if ratios else 0.0,
        total_tokens_mean=float(np.mean([e.total_tokens for e in examples])) if examples else 0.0,
        latency_mean=float(np.mean(lats))                   if lats else 0.0,
        latency_std=float(np.std(lats))                     if lats else 0.0,
        latency_p50=float(np.percentile(lats, 50))          if lats else 0.0,
        latency_p95=float(np.percentile(lats, 95))          if lats else 0.0,
        wall_time_total=wall_time,
        answer_found_rate=sum(e.answer_found for e in examples) / n if n else 0.0,
        trajectory_coherence=float(np.mean([e.traj_coherence for e in examples])) if examples else 0.0,
        truth_alignment=float(np.mean([e.truth_align for e in examples]))         if examples else 0.0,
    )


# ── Flip matrix computation ────────────────────────────────────────────────────

def compute_flip_matrix(
    preds_a: list[bool],
    preds_b: list[bool],
    golds: list,
    condition_a: str,
    condition_b: str,
    model_tag: str,
) -> FlipMatrix:
    """2×2 flip matrix: condition_a is the reference, condition_b is compared."""
    F00 = sum(1 for a, b in zip(preds_a, preds_b) if     a and     b)
    F01 = sum(1 for a, b in zip(preds_a, preds_b) if     a and not b)
    F10 = sum(1 for a, b in zip(preds_a, preds_b) if not a and     b)
    F11 = sum(1 for a, b in zip(preds_a, preds_b) if not a and not b)
    return FlipMatrix(
        F00=F00, F01=F01, F10=F10, F11=F11,
        condition_a=condition_a, condition_b=condition_b,
        model_tag=model_tag,
    )


def compute_all_flip_matrices(
    all_preds: dict,
    golds: list,
    model_tag: str,
    ratio_int: int,
    source: str,
) -> list[FlipMatrix]:
    """10 primary comparison pairs."""
    ccot     = f'ccot_R{ratio_int}'
    trim     = f'trimmed_R{ratio_int}'
    dom      = f'dom_R{ratio_int}_{source}'
    cpca     = f'cpca_R{ratio_int}_{source}'
    trim_dom = f'trimmed_dom_R{ratio_int}'
    noise    = f'noise_R{ratio_int}_{source}'

    pairs = [
        ('no_cot',   'full_cot'),
        ('no_cot',   ccot),
        ('full_cot', ccot),
        (ccot,       trim),
        (ccot,       noise),
        (ccot,       dom),
        (trim,       trim_dom),
        ('full_cot', dom),
        (ccot,       cpca),
        (dom,        cpca),
    ]

    return [
        compute_flip_matrix(all_preds[ca], all_preds[cb], golds, ca, cb, model_tag)
        for ca, cb in pairs
        if ca in all_preds and cb in all_preds
    ]


def compute_full_flip_grid(
    all_preds: dict,
    golds: list,
    model_tag: str,
) -> dict:
    """N_cond × N_cond net-gain grid: grid[a][b] = net_gain(a→b)."""
    conditions = _display_grid_order(list(all_preds.keys()))
    net_gain = {}
    for ca in conditions:
        net_gain[ca] = {}
        for cb in conditions:
            if ca == cb:
                net_gain[ca][cb] = 0.0
            else:
                fm = compute_flip_matrix(
                    all_preds[ca], all_preds[cb], golds, ca, cb, model_tag
                )
                net_gain[ca][cb] = int(fm.net_gain)
    return {'model_tag': model_tag, 'conditions': conditions, 'net_gain': net_gain}


# ── Alpha sweep on D_test ──────────────────────────────────────────────────────

def run_alpha_sweep_test(
    model,
    tokenizer,
    D_test: list,
    v_truth: torch.Tensor,
    L_star: int,
    alpha_star: float,
    device: str,
    model_tag: str,
    prompt_fn,
    boundary_fn,
    alphas: list = None,
    n_sub: int = 100,
) -> list[dict]:
    """
    Diagnostic α sweep on D_test subset. No hyperparameter is changed after this.
    Uses DoM steering across a grid of alpha values.
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    D_sub = D_test[:min(n_sub, len(D_test))]
    print(f"\n[PH4] α sweep on D_test subset ({len(D_sub)} examples)...")

    sweep = []
    for a in alphas:
        c_list = []
        tc_list = []
        ta_list = []
        for item in D_sub:
            prompt = prompt_fn(item)
            if a == 0.0:
                ex = run_steered_with_metrics(
                    model, tokenizer, prompt, item,
                    None, L_star, v_truth, device,
                )
            else:
                enc = tokenizer(prompt, return_tensors='pt').to(device)
                with torch.no_grad():
                    probe_ids = model.generate(
                        **enc, do_sample=False, max_new_tokens=128,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                try:
                    b_idx = boundary_fn(probe_ids, tokenizer)
                except Exception:
                    b_idx = max(0, enc['input_ids'].shape[1] - 1)
                hook_fn = make_dom_hook(b_idx, v_truth, a, device)
                ex = run_steered_with_metrics(
                    model, tokenizer, prompt, item,
                    hook_fn, L_star, v_truth, device,
                )
            c_list.append(ex.correct)
            tc_list.append(ex.traj_coherence)
            ta_list.append(ex.truth_align)

        acc    = sum(c_list) / len(c_list)
        coh    = float(np.mean(tc_list)) if tc_list else 0.0
        aln    = float(np.mean(ta_list)) if ta_list else 0.0
        marker = " ← α*" if abs(a - alpha_star) < 0.5 else ""
        print(f"  α={a:>5.1f}  acc={acc:.3f}  align={aln:.4f}  coh={coh:.4f}{marker}")
        sweep.append({'alpha': a, 'accuracy': acc, 'truth_alignment': aln, 'trajectory_coherence': coh})

    return sweep


# ── Reporting tables ───────────────────────────────────────────────────────────

def print_accuracy_table(metrics: dict):
    w = 100
    print("\n" + "=" * w)
    print(f"{'Condition':<36} {'Acc':>7} {'95% CI':^22} {'N':>5} "
          f"{'AnswFnd%':>9} {'ActRatio':>9}")
    print("-" * w)
    for cond, m in sorted(metrics.items()):
        if m.ci_upper_95 > 0:
            ci_str = f"[{m.ci_lower_95:.3f}, {m.ci_upper_95:.3f}]"
        else:
            ci_str = "  (not computed)  "
        print(
            f"{_display_condition_name(cond):<36} {m.accuracy:>7.3f} "
            f"{ci_str:^22} {m.n_total:>5} "
            f"{m.answer_found_rate:>9.3f} {m.actual_ratio_mean:>9.3f}"
        )
    print("=" * w)


def print_latent_metrics_table(metrics: dict):
    w = 65
    print("\n" + "=" * w)
    print(f"{'Condition':<36} {'TrajCoh':>10} {'TruthAlign':>12}")
    print("-" * w)
    for cond, m in sorted(metrics.items()):
        print(f"{_display_condition_name(cond):<36} {m.trajectory_coherence:>10.4f} {m.truth_alignment:>12.4f}")
    print("=" * w)


def print_primary_flip_summary(flip_matrices: list):
    w = 84
    print("\n" + "=" * w)
    print(
        f"{'Pair (a → b)':<44} {'F00':>5} {'F01':>5} {'F10':>5} "
        f"{'F11':>5} {'NetGain':>8} {'Degrade':>8}"
    )
    print("-" * w)
    for fm in flip_matrices:
        pair = f"{_display_condition_name(fm.condition_a)} → {_display_condition_name(fm.condition_b)}"
        print(
            f"{pair:<44} {fm.F00:>5} {fm.F01:>5} {fm.F10:>5} {fm.F11:>5} "
            f"{fm.net_gain:>8d} {fm.degradation_rate:>8.3f}"
        )
    print("=" * w)


def print_mechanism_gain_table(metrics: dict, baseline_cond: str):
    print(f"\n--- Mechanism Gain (baseline = {_display_condition_name(baseline_cond)}) ---")
    if baseline_cond not in metrics:
        print("  (baseline not evaluated)")
        return
    base_acc = metrics[baseline_cond].accuracy
    print(f"{'Condition':<36} {'Acc':>7} {'Gain':>8}")
    print("-" * 55)
    for cond, m in sorted(metrics.items()):
        if cond == baseline_cond:
            continue
        print(f"{_display_condition_name(cond):<36} {m.accuracy:>7.3f} {m.accuracy - base_acc:>+8.3f}")


def print_specificity_table(
    metrics: dict,
    flip_matrices: list,
    dom_cond: str,
    noise_cond: str,
    ccot_cond: str,
):
    print("\n--- Steering Specificity (DoM vs Noise) ---")
    if dom_cond in metrics and noise_cond in metrics:
        dm = metrics[dom_cond]
        nm = metrics[noise_cond]
        print(f"  DoM   acc={dm.accuracy:.3f}  truth_align={dm.truth_alignment:.4f}")
        print(f"  Noise acc={nm.accuracy:.3f}  truth_align={nm.truth_alignment:.4f}")
        print(f"  Specificity gain: {dm.accuracy - nm.accuracy:+.3f}")
    for fm in flip_matrices:
        if fm.condition_a == ccot_cond and fm.condition_b in (dom_cond, noise_cond):
            print(
                f"  {_display_condition_name(fm.condition_a)} → {_display_condition_name(fm.condition_b)}: "
                f"improve={fm.improvement_rate:.3f}  "
                f"degrade={fm.degradation_rate:.3f}  "
                f"net={fm.net_gain:+d}"
            )


def print_efficiency_table(metrics: dict):
    w = 84
    print("\n" + "=" * w)
    print(
        f"{'Condition':<36} {'Acc':>7} {'RTok_mean':>10} "
        f"{'ActRatio':>9} {'Lat_mean':>9} {'Lat_p95':>8}"
    )
    print("-" * w)
    for cond, m in sorted(metrics.items()):
        print(
            f"{_display_condition_name(cond):<36} {m.accuracy:>7.3f} {m.reasoning_tokens_mean:>10.1f} "
            f"{m.actual_ratio_mean:>9.3f} {m.latency_mean:>9.2f} {m.latency_p95:>8.2f}"
        )
    print("=" * w)


def print_ci_table(cis: dict):
    """Per-condition 95% bootstrap CIs, sorted by accuracy descending."""
    if not cis:
        return
    w = 82
    print(f"\n{'═' * w}")
    print(f"  Bootstrap 95% CIs  "
          f"(n={N_BOOTSTRAP} resamples · paired resampling · seed={CI_SEED})")
    print(f"{'Condition':<34} {'Acc':>6}  {'95% CI':^22}  {'±HW':>6}")
    print(f"{'─' * w}")
    for cond, br in sorted(cis.items(), key=lambda x: -x[1].point):
        print(f"  {_display_condition_name(cond):<32} {br.point:>6.3f}  "
              f"[{br.lower:.3f}, {br.upper:.3f}]  {br.half_width:>6.3f}")
    print(f"{'═' * w}")


def print_paired_ci_table(paired_cis: dict):
    """Paired bootstrap CIs on accuracy difference; marks statistically significant gains."""
    if not paired_cis:
        return
    w = 82
    print(f"\n{'═' * w}")
    print(f"  Paired bootstrap CIs on accuracy difference  "
          f"(n={N_BOOTSTRAP} resamples)")
    print(f"  Significant (✓) = 95% CI excludes 0")
    print(f"{'Comparison':<34} {'Δ':>7}  {'95% CI':^22}  {'Sig':>4}")
    print(f"{'─' * w}")
    for name, br in paired_cis.items():
        sig = "✓" if br.significant else "—"
        print(f"  {name:<32} {br.point:>+7.3f}  "
              f"[{br.lower:+.3f}, {br.upper:+.3f}]  {sig:>4}")
    print(f"{'═' * w}")


# ── Persistence ────────────────────────────────────────────────────────────────

def save_final_results(
    all_results: dict,
    out_dir: str,
    provenance: Optional[dict] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    summary = _build_summary(all_results, next(iter(all_results.values()))['metrics']['full_cot'].n_total if all_results else 0)
    if provenance:
        summary = {'provenance': provenance, **summary}
    for model_tag, data in all_results.items():
        out_path = os.path.join(out_dir, f"{model_tag}_test.json")
        def _ser_br(br) -> dict:
            return {'point': br.point, 'lower': br.lower,
                    'upper': br.upper, 'significant': br.significant,
                    'half_width': br.half_width}

        serializable = {
            'model_tag':      model_tag,
            'metrics':        {k: asdict(v) for k, v in data['metrics'].items()},
            'flip_matrices':  [
                _serialize_flip_matrix(fm)
                for fm in data['flip_matrices']
            ],
            'flip_grid':      data.get('flip_grid', {}),
            'alpha_sweep':    data.get('alpha_sweep', []),
            'locked_config':  data.get('locked_config', {}),
            'condition_cis':  {
                k: _ser_br(v) for k, v in data.get('condition_cis', {}).items()
            },
            'paired_cis':     {
                k: _ser_br(v) for k, v in data.get('paired_cis', {}).items()
            },
        }
        if provenance:
            wc = provenance['winning_config']
            vb = provenance.get('vectors_base', 'vectors')
            cb = provenance.get('checkpoints_base', 'checkpoints')
            serializable['provenance'] = {
                **provenance,
                'vectors_dir': os.path.join(vb, wc, model_tag),
                'checkpoints_dir': os.path.join(cb, wc, model_tag),
            }
        with open(out_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"  Saved: {out_path}")

    summary_path = os.path.join(out_dir, 'summary_test.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")


# ── Main evaluation runner ─────────────────────────────────────────────────────

def run_final_evaluation(
    D_test: list,
    cfg: dict,
    device: str,
    results_base: str = 'results',
    vectors_base: str = 'vectors',
    checkpoints_base: str = 'checkpoints',
    max_new_tokens: int = 256,
) -> dict:
    """
    Single-pass D_test evaluation using locked Phase 3 configs.
    Must be called exactly once from this file. D_test is never re-loaded.
    """
    winning_config = cfg['winning_config']
    phase3_best = _load_phase3_best_configs(cfg, winning_config, results_base)
    all_results:  dict = {}
    golds = [item['answer'].split('####')[1].strip() for item in D_test]

    for model_tag in MODEL_TAGS:
        base_model_id = MODEL_ID_MAP[model_tag]
        results_dir   = os.path.join(results_base, winning_config, model_tag)
        vectors_dir   = os.path.join(vectors_base, winning_config, model_tag)
        ckpt_dir      = os.path.join(checkpoints_base, winning_config, model_tag)

        header = f"Phase 4 | {model_tag} | config={winning_config}"
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")

        # ── Load locked Phase 3 config ─────────────────────────────────────────
        best_cfg   = phase3_best.get(model_tag) or _load_best_config(results_dir)
        meta       = _load_meta_file(vectors_dir)
        ratio      = float(best_cfg.get('ratio') or 0.7)
        ratio_int  = int(round(ratio * 10))
        source     = str(best_cfg.get('vector_source') or 'ccot')
        alpha_star = float(best_cfg.get('alpha_star') or 1.0)
        r_final    = int(meta.get('ccot_r_final', 10))

        print(
            f"Locked: R={ratio}  src={source}  "
            f"method={best_cfg.get('vector_method')}  α*={alpha_star:.4f}"
        )

        all_preds:   dict = {}
        all_metrics: dict = {}
        sweep        = []   # default; populated if CCoT checkpoint exists

        # ── Phase A: CoT model ─────────────────────────────────────────────────
        cot_ckpt = os.path.join(ckpt_dir, 'cot')
        cot_model, tok_cot = load_finetuned(cot_ckpt, device)
        for p in cot_model.parameters():
            p.requires_grad = False
        cot_model.eval()

        print("[PH4] Precomputing full CoT token counts on D_test...")
        full_cot_counts = precompute_full_cot_tokens(cot_model, tok_cot, D_test, device)
        budgets = [max(10, round(ratio * t)) for t in full_cot_counts]

        # Condition: Full CoT
        print("[PH4] full_cot")
        t0 = time.time()
        examples = []
        for item in D_test:
            t1 = time.time()
            pred, reasoning = run_cot(cot_model, tok_cot, item, device)
            gold = item['answer'].split('####')[1].strip()
            ok   = normalize_answer(pred) == normalize_answer(gold) if pred else False
            nt   = len(tok_cot.encode(reasoning or '', add_special_tokens=False))
            examples.append(ExampleResult(
                correct=ok, answer_found=pred is not None,
                reasoning_tokens=nt, total_tokens=nt,
                latency_sec=time.time() - t1,
            ))
        all_metrics['full_cot'] = collect_condition_metrics(
            examples, full_cot_counts, 'full_cot', model_tag, time.time() - t0
        )
        all_preds['full_cot'] = [e.correct for e in examples]
        print(f"  acc={all_metrics['full_cot'].accuracy:.3f}")

        # Condition: Trimmed CoT
        trim_cond = f'trimmed_R{ratio_int}'
        print(f"[PH4] {trim_cond}")
        t0 = time.time()
        examples = []
        for i, item in enumerate(D_test):
            t1 = time.time()
            pred, reasoning = run_trimmed_cot(cot_model, tok_cot, item, budgets[i], device)
            gold = item['answer'].split('####')[1].strip()
            ok   = normalize_answer(pred) == normalize_answer(gold) if pred else False
            nt   = len(tok_cot.encode(reasoning or '', add_special_tokens=False))
            examples.append(ExampleResult(
                correct=ok, answer_found=pred is not None,
                reasoning_tokens=nt, total_tokens=nt,
                latency_sec=time.time() - t1,
            ))
        all_metrics[trim_cond] = collect_condition_metrics(
            examples, full_cot_counts, trim_cond, model_tag, time.time() - t0
        )
        all_preds[trim_cond] = [e.correct for e in examples]
        print(f"  acc={all_metrics[trim_cond].accuracy:.3f}")

        # Condition: Trimmed + DoM  (CoT model, base source, base L_star)
        trim_dom_cond = f'trimmed_dom_R{ratio_int}'
        print(f"[PH4] {trim_dom_cond}")
        try:
            v_base_dom  = _load_dom(vectors_dir, 'base').to(device)
            try:
                L_star_base = get_injection_layer(vectors_dir, 'base')
            except FileNotFoundError:
                L_star_base = meta.get('base_best_layer', meta.get('ccot_best_layer', 14))
            try:
                alpha_base = _load_alpha_file(vectors_dir, 'base')
            except FileNotFoundError:
                alpha_base = alpha_star

            cot_prompt_fn = lambda item: f"Question: {item['question']}\n\nReasoning:"
            t0 = time.time()
            examples = []
            for i, item in enumerate(D_test):
                prompt = cot_prompt_fn(item)
                enc = tok_cot(prompt, return_tensors='pt').to(device)
                with torch.no_grad():
                    probe_ids = cot_model.generate(
                        **enc, do_sample=False, max_new_tokens=128,
                        pad_token_id=tok_cot.eos_token_id,
                    )
                try:
                    b_idx = find_boundary_idx_base(probe_ids, tok_cot)
                except Exception:
                    b_idx = max(0, enc['input_ids'].shape[1] - 1)
                hook_fn = make_dom_hook(b_idx, v_base_dom, alpha_base, device)
                ex = run_steered_with_metrics(
                    cot_model, tok_cot, prompt, item, hook_fn,
                    L_star_base, v_base_dom, device, budgets[i],
                )
                examples.append(ex)
            all_metrics[trim_dom_cond] = collect_condition_metrics(
                examples, full_cot_counts, trim_dom_cond, model_tag, time.time() - t0
            )
            all_preds[trim_dom_cond] = [e.correct for e in examples]
            print(f"  acc={all_metrics[trim_dom_cond].accuracy:.3f}")
        except FileNotFoundError as exc:
            print(f"  [SKIP] {trim_dom_cond}: {exc}")

        del cot_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Phase B: Base model → No CoT ──────────────────────────────────────
        print("[PH4] no_cot")
        base_model, tok_base = load_base_frozen(base_model_id, device)
        t0 = time.time()
        examples = []
        for item in D_test:
            t1 = time.time()
            pred, _ = run_no_cot(base_model, tok_base, item, device)
            gold = item['answer'].split('####')[1].strip()
            ok   = normalize_answer(pred) == normalize_answer(gold) if pred else False
            examples.append(ExampleResult(
                correct=ok, answer_found=pred is not None,
                reasoning_tokens=0, total_tokens=0,
                latency_sec=time.time() - t1,
            ))
        all_metrics['no_cot'] = collect_condition_metrics(
            examples, full_cot_counts, 'no_cot', model_tag, time.time() - t0
        )
        all_preds['no_cot'] = [e.correct for e in examples]
        print(f"  acc={all_metrics['no_cot'].accuracy:.3f}")
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Phase C: CCoT model (locked ratio) ────────────────────────────────
        ccot_ckpt = os.path.join(ckpt_dir, f'ccot_R{ratio_int}')
        if not os.path.exists(os.path.join(ccot_ckpt, 'adapter_config.json')):
            print(f"[PH4] CCoT checkpoint missing: {ccot_ckpt} — skipping CCoT conditions")
        else:
            ccot_model, tok_ccot = load_finetuned(ccot_ckpt, device)
            for p in ccot_model.parameters():
                p.requires_grad = False
            ccot_model.eval()
            ccot_prompt_fn = lambda item: (
                f"Question: {item['question']}\n\n[compress:{ratio}]\n"
            )

            # Load best-source vectors
            try:
                v_truth = _load_dom(vectors_dir, source).to(device)
            except FileNotFoundError:
                print(f"  [SKIP] DoM vector missing for source={source}")
                goto_flip = True
            else:
                goto_flip = False

            if not goto_flip:
                try:
                    U_cpca   = _load_cpca(vectors_dir, source, r_final).to(device)
                    has_cpca = True
                except FileNotFoundError:
                    U_cpca   = None
                    has_cpca = False

                try:
                    L_star = get_injection_layer(vectors_dir, source)
                except FileNotFoundError:
                    L_star = meta.get(f'{source}_best_layer', meta.get('ccot_best_layer', 14))

                # Helper: probe-generate boundary, create hook, run + collect
                def _make_steered_examples(hook_factory, boundary_fn, cond_name):
                    t_inner = time.time()
                    exs = []
                    for item in D_test:
                        prompt = ccot_prompt_fn(item)
                        enc = tok_ccot(prompt, return_tensors='pt').to(device)
                        with torch.no_grad():
                            probe_ids = ccot_model.generate(
                                **enc, do_sample=False, max_new_tokens=128,
                                pad_token_id=tok_ccot.eos_token_id,
                            )
                        try:
                            b_idx = boundary_fn(probe_ids, tok_ccot)
                        except Exception:
                            b_idx = max(0, enc['input_ids'].shape[1] - 1)
                        hook_fn = hook_factory(b_idx)
                        ex = run_steered_with_metrics(
                            ccot_model, tok_ccot, prompt, item, hook_fn,
                            L_star, v_truth, device, max_new_tokens,
                        )
                        exs.append(ex)
                    m = collect_condition_metrics(
                        exs, full_cot_counts, cond_name, model_tag, time.time() - t_inner
                    )
                    return exs, m

                # Condition: CCoT baseline (no hook, but metrics captured)
                ccot_cond = f'ccot_R{ratio_int}'
                print(f"[PH4] {ccot_cond}")
                t0 = time.time()
                examples = []
                for item in D_test:
                    prompt = ccot_prompt_fn(item)
                    ex = run_steered_with_metrics(
                        ccot_model, tok_ccot, prompt, item, None,
                        L_star, v_truth, device, max_new_tokens,
                    )
                    examples.append(ex)
                all_metrics[ccot_cond] = collect_condition_metrics(
                    examples, full_cot_counts, ccot_cond, model_tag, time.time() - t0
                )
                all_preds[ccot_cond] = [e.correct for e in examples]
                print(f"  acc={all_metrics[ccot_cond].accuracy:.3f}")

                # Condition: Noise control
                noise_cond = f'noise_R{ratio_int}_{source}'
                print(f"[PH4] {noise_cond}")
                exs, m = _make_steered_examples(
                    lambda b: make_noise_hook(b, alpha_star, device),
                    find_boundary_idx_ccot,
                    noise_cond,
                )
                all_metrics[noise_cond] = m
                all_preds[noise_cond]   = [e.correct for e in exs]
                print(f"  acc={m.accuracy:.3f}")

                # Condition: DoM steering
                dom_cond = f'dom_R{ratio_int}_{source}'
                print(f"[PH4] {dom_cond}")
                exs, m = _make_steered_examples(
                    lambda b: make_dom_hook(b, v_truth, alpha_star, device),
                    find_boundary_idx_ccot,
                    dom_cond,
                )
                all_metrics[dom_cond] = m
                all_preds[dom_cond]   = [e.correct for e in exs]
                print(f"  acc={m.accuracy:.3f}  truth_align={m.truth_alignment:.4f}")

                # Condition: cPCA steering (if vector available)
                if has_cpca:
                    cpca_cond = f'cpca_R{ratio_int}_{source}'
                    print(f"[PH4] {cpca_cond}")
                    exs, m = _make_steered_examples(
                        lambda b: make_cpca_hook(b, U_cpca, alpha_star, device),
                        find_boundary_idx_ccot,
                        cpca_cond,
                    )
                    all_metrics[cpca_cond] = m
                    all_preds[cpca_cond]   = [e.correct for e in exs]
                    print(f"  acc={m.accuracy:.3f}  truth_align={m.truth_alignment:.4f}")

                # ── Diagnostic alpha sweep on D_test ──────────────────────────
                sweep = run_alpha_sweep_test(
                    ccot_model, tok_ccot, D_test,
                    v_truth, L_star, alpha_star, device, model_tag,
                    prompt_fn=ccot_prompt_fn,
                    boundary_fn=find_boundary_idx_ccot,
                )

            del ccot_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Bootstrap CIs (pure numpy, no model) ──────────────────────────────
        print(f"\n[PH4] Computing bootstrap CIs ({N_BOOTSTRAP} resamples)...")
        condition_cis = compute_condition_cis(all_preds, seed=CI_SEED)
        paired_cis    = compute_paired_cis(all_preds, ratio_int, source, seed=CI_SEED)

        # Attach per-condition CIs back to FinalMetrics for inline display
        for cond, br in condition_cis.items():
            if cond in all_metrics:
                all_metrics[cond].ci_lower_95 = br.lower
                all_metrics[cond].ci_upper_95 = br.upper

        # ── Compute flip matrices and grid ─────────────────────────────────────
        flip_matrices = compute_all_flip_matrices(
            all_preds, golds, model_tag, ratio_int, source
        )
        flip_grid = compute_full_flip_grid(all_preds, golds, model_tag)

        # ── Reporting ──────────────────────────────────────────────────────────
        ccot_c  = f'ccot_R{ratio_int}'
        dom_c   = f'dom_R{ratio_int}_{source}'
        noise_c = f'noise_R{ratio_int}_{source}'

        print_accuracy_table(all_metrics)
        print_ci_table(condition_cis)
        print_paired_ci_table(paired_cis)
        print_latent_metrics_table(all_metrics)
        print_primary_flip_summary(flip_matrices)
        print_mechanism_gain_table(all_metrics, baseline_cond=ccot_c)
        print_specificity_table(all_metrics, flip_matrices, dom_c, noise_c, ccot_c)
        print_efficiency_table(all_metrics)

        all_results[model_tag] = {
            'metrics':        all_metrics,
            'flip_matrices':  flip_matrices,
            'flip_grid':      flip_grid,
            'alpha_sweep':    sweep,
            'locked_config':  best_cfg,
            'condition_cis':  condition_cis,
            'paired_cis':     paired_cis,
        }

    return all_results


# ── Entry point ────────────────────────────────────────────────────────────────

def _print_transfer_artifact_banner(winning_config: str) -> None:
    bar = '=' * 72
    print(f"\n{bar}")
    print('  NON-GSM8K D_TEST: vectors and checkpoints are NOT dataset-suffixed.')
    print(f'  They load from vectors/{winning_config}/<model>/ and checkpoints/{winning_config}/<model>/')
    print('  For Phase 5 transfer, those directories MUST be from your GSM8K pipeline run')
    print('  (no SVAMP re-tuning of v_truth or alpha_star).')
    print(f"{bar}\n")


def main():
    parser = argparse.ArgumentParser(description='Phase 4 final evaluation on D_test.')
    parser.add_argument(
        '--dataset', default=None, choices=('gsm8k', 'svamp', 'prontoqa'),
        help='Dataset id (default: CCOT_DATASET env, configs/active_dataset.txt, or prompt)',
    )
    parser.add_argument(
        '--results-dir',
        default='results/final',
        help='Directory for summary_test.json and <model>_test.json (default: results/final)',
    )
    args, _unknown = parser.parse_known_args()

    init_project_dataset(args.dataset, interactive=sys.stdin.isatty())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[PH4] Device: {device}")

    torch.manual_seed(0)
    np.random.seed(0)

    cfg = _load_selected_yaml('configs/selected.yaml')
    winning = cfg['winning_config']
    print(f"[PH4] Winning config: {winning}")

    if get_active_dataset_id() != 'gsm8k':
        _print_transfer_artifact_banner(winning)

    D_test = load_test_set()
    print(f"[PH4] Loaded D_test: {len(D_test)} examples")

    all_results = run_final_evaluation(D_test, cfg, device)
    provenance = {
        'eval_dataset': get_active_dataset_id(),
        'steering_artifact_policy': 'frozen_from_gsm8k_pipeline',
        'winning_config': winning,
        'vectors_base': 'vectors',
        'checkpoints_base': 'checkpoints',
    }
    save_final_results(all_results, args.results_dir, provenance=provenance)
    print(f"\n[PH4] Done. Results saved to {args.results_dir}/")


if __name__ == '__main__':
    main()
