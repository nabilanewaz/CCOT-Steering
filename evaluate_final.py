"""Phase 4: single-pass D_test evaluation with locked Phase 3 configs.

D_test is loaded exactly once. This file is run exactly once.
No hyperparameter is changed after Phase 3. No result inspires a rerun.

Phase 5 (SVAMP transfer): use ``--dataset svamp --training-dataset gsm8k`` and ``--results-dir`` (e.g.
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
from utils.dataset_paths import get_active_dataset_id, init_project_dataset, artifact_root, selected_config_path
from utils.experiment_config import require_exact_count
from utils.artifacts import EXPERIMENT_VERSION, dataset_fingerprint, file_fingerprint, checkpoint_identity, fingerprint
from phase3.evaluate import load_source_artifacts, available_methods, _require_phase2_inputs
from phase3.hooks import condition_hooks, generation_scope, intervention_positions
from phase3.hook_utils import first_hidden
from contextlib import nullcontext
from phase1.inference import (
    extract_answer,
    extract_reasoning_span,
    cot_prompt,
    latent_prompt,
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
from phase3.hook_utils import last_token_state

# ── Constants ──────────────────────────────────────────────────────────────────

N_BOOTSTRAP  = 1000    # resamples for all bootstrap CIs
CI_SEED      = 0       # fixed seed → reproducible CIs across re-runs
CI_LEVEL     = 0.95    # 95% confidence interval

MODEL_TAGS = ['qwen25_0.5b', 'qwen25_3b', 'qwen25_math1.5b']
MODEL_ID_MAP = {
    'llama32_3b':      'meta-llama/Llama-3.2-3B',
    'phi2':            'microsoft/phi-2',
    'qwen25_0.5b':     'Qwen/Qwen2.5-0.5B',
    'qwen25_3b':       'Qwen/Qwen2.5-3B',
    'qwen25_math1.5b': 'Qwen/Qwen2.5-Math-1.5B',
}


def _last_hidden_state(output) -> torch.Tensor | None:
    return last_token_state(output)


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
    generated_text: str = ""
    question_hash: str = ""


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
    condition_tag: str,
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
    ccot_cond  = f'ccot_{condition_tag}'
    dom_cond   = f'dom_{condition_tag}_{source}'
    cpca_cond  = f'cpca_{condition_tag}_{source}'
    noise_cond = f'noise_{condition_tag}_{source}'
    trim_cond  = f'trimmed_{condition_tag}'

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
    for model_tag in MODEL_ID_MAP:
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
    if name.startswith('ccot_R') or name.startswith('ccot_L'):
        return 'ccot'
    if name.startswith('trimmed_R') or name.startswith('trimmed_L'):
        return 'trimmed_cot'
    if name.startswith('noise_R') or name.startswith('noise_L'):
        return 'random_noise'
    if name.startswith('dom_R') or name.startswith('dom_L'):
        return 'ccot_dom'
    if name.startswith('cpca_R') or name.startswith('cpca_L'):
        return 'ccot_cpca'
    if name.startswith('trimmed_dom_R') or name.startswith('trimmed_dom_L'):
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
        'token_budgeting_table': {},
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
        summary['token_budgeting_table'][model_tag] = data.get('token_budgeting', {})

    # ── Confidence intervals ───────────────────────────────────────────────────
    summary['confidence_intervals'] = {
        model_tag: {
            'n_bootstrap':  N_BOOTSTRAP,
            'ci_level':     CI_LEVEL,
            'condition_cis': {
                _display_condition_name(k): {
                    'accuracy':   v.point,
                    'point':      v.point,
                    'ci_lower':   v.lower,
                    'ci_upper':   v.upper,
                    'lower':      v.lower,
                    'upper':      v.upper,
                    'half_width': v.half_width,
                }
                for k, v in data.get('condition_cis', {}).items()
            },
            'paired_cis': {
                k: {
                    'delta':       v.point,
                    'point':       v.point,
                    'ci_lower':    v.lower,
                    'ci_upper':    v.upper,
                    'lower':       v.lower,
                    'upper':       v.upper,
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
        enc = tokenizer(cot_prompt(item['question']), return_tensors='pt').to(device)
        with torch.no_grad():
            out = cot_model.generate(
                **enc, do_sample=False, max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
        )
        reasoning = extract_reasoning_span(generated)
        counts.append(len(tokenizer.encode(reasoning, add_special_tokens=False)))
    return counts


def run_steered_with_metrics(
    model, tokenizer, prompt, item, hook_fn, L_star, v_truth, device,
    max_new_tokens=256, method=None, artifacts=None, alpha=0.0,
):
    gold = item["answer"].split("####", 1)[1].strip()
    captured = []
    direction = v_truth.to(device).float() if v_truth is not None else None
    layers = get_transformer_layers(model)

    def capture(module, inputs, output):
        hidden = first_hidden(output)
        positions = intervention_positions(hidden)
        if positions:
            captured.append(hidden[..., positions[-1], :].detach().float().cpu().clone())
        return output

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    length = encoded["input_ids"].shape[1]
    started = time.time()
    steering = condition_hooks(model, method, artifacts, alpha, device) if method else nullcontext()
    with generation_scope(model, length), steering:
        handles = []
        try:
            if hook_fn is not None:
                handles.append(layers[L_star].register_forward_hook(hook_fn))
            if direction is not None:
                handles.append(layers[L_star].register_forward_hook(capture))
            with torch.no_grad():
                output = model.generate(**encoded, max_new_tokens=max_new_tokens,
                                        do_sample=False, pad_token_id=tokenizer.pad_token_id)
        finally:
            for handle in handles:
                handle.remove()
    generated = output[0, length:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return ExampleResult(
        correct=_score_text(text, gold), answer_found=extract_answer(text) is not None,
        reasoning_tokens=len(tokenizer.encode(extract_reasoning_span(text), add_special_tokens=False)),
        total_tokens=length + len(generated), latency_sec=time.time() - started,
        traj_coherence=trajectory_coherence(captured),
        truth_align=truth_alignment(captured, direction.cpu()) if direction is not None else 0.0,
        generated_text=text, question_hash=fingerprint(item["question"]),
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


def coconut_thinking_token_counts(
    examples: list[ExampleResult],
    latent_tokens: int,
) -> list[int]:
    """
    Token budget used for the matched trimmed-CoT baseline.
    Counts fixed Coconut latent tokens plus any visible reasoning tokens emitted
    after the latent span.
    """
    return [
        max(1, int(latent_tokens) + int(e.reasoning_tokens))
        for e in examples
    ]


def build_token_budget_log(
    D_test: list,
    full_cot_counts: list[int],
    coconut_counts: list[int],
    trimmed_examples: list[ExampleResult],
    model_tag: str,
    condition_tag: str,
    latent_tokens: int,
) -> dict:
    full_mean = float(np.mean(full_cot_counts)) if full_cot_counts else 0.0
    coconut_mean = float(np.mean(coconut_counts)) if coconut_counts else 0.0
    token_budget_valid = full_mean > 0.0
    ratio = coconut_mean / full_mean if token_budget_valid else None
    trimmed_counts = [e.reasoning_tokens for e in trimmed_examples]
    trimmed_mean = float(np.mean(trimmed_counts)) if trimmed_counts else 0.0
    per_example = []
    for i, (item, full_n, coco_n, trim_ex) in enumerate(
        zip(D_test, full_cot_counts, coconut_counts, trimmed_examples)
    ):
        per_example.append({
            "id": item.get("id", i),
            "full_cot_reasoning_tokens": int(full_n),
            "coconut_matched_tokens": int(coco_n),
            "trimmed_cot_budget_tokens": int(coco_n),
            "trimmed_cot_actual_reasoning_tokens": int(trim_ex.reasoning_tokens),
            "trimmed_cot_correct": bool(trim_ex.correct),
            "per_example_ratio": float(coco_n / full_n) if full_n > 0 else None,
        })
    return {
        "model_tag": model_tag,
        "condition_tag": condition_tag,
        "latent_tokens": int(latent_tokens),
        "policy": "trim_full_cot_to_best_coconut_token_count",
        "x_coconut_tokens_mean": coconut_mean,
        "y_full_cot_tokens_mean": full_mean,
        "x_over_y_ratio": ratio,
        "token_budget_valid": token_budget_valid,
        "trimmed_cot_actual_tokens_mean": trimmed_mean,
        "n_examples": len(per_example),
        "per_example": per_example,
    }


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
    condition_tag: str,
    source: str,
) -> list[FlipMatrix]:
    """10 primary comparison pairs."""
    ccot     = f'ccot_{condition_tag}'
    trim     = f'trimmed_{condition_tag}'
    dom      = f'dom_{condition_tag}_{source}'
    cpca     = f'cpca_{condition_tag}_{source}'
    trim_dom = f'trimmed_dom_{condition_tag}'
    noise    = f'noise_{condition_tag}_{source}'

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

def merge_final_summaries(previous, current):
    """Preserve completed models when separate model runners share an output directory."""
    for key in ('eval_dataset', 'training_dataset', 'dataset_fingerprint', 'experiment_version', 'winning_config'):
        if previous.get('provenance', {}).get(key) != current.get('provenance', {}).get(key):
            raise RuntimeError(f'Cannot merge final results with different {key}')
    merged = dict(previous)
    for key, value in current.items():
        if key != 'provenance' and isinstance(value, dict):
            merged[key] = {**previous.get(key, {}), **value}
        else:
            merged[key] = value
    merged['models'] = sorted(set(previous['models']) | set(current['models']))
    merged['provenance'] = {**current['provenance'], 'models': merged['models']}
    return merged


def save_final_results(
    all_results: dict,
    out_dir: str,
    provenance: Optional[dict] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    summary = _build_summary(all_results, next(iter(all_results.values()))['metrics']['full_cot'].n_total if all_results else 0)
    if provenance:
        summary = {'provenance': provenance, **summary}
    summary_path = os.path.join(out_dir, 'summary_test.json')
    if os.path.exists(summary_path):
        with open(summary_path) as stream:
            summary = merge_final_summaries(json.load(stream), summary)
    for model_tag, data in all_results.items():
        out_path = os.path.join(out_dir, f"{model_tag}_test.json")
        def _ser_br(br) -> dict:
            return {
                'point': br.point,
                'lower': br.lower,
                'upper': br.upper,
                'delta': br.point,
                'ci_lower': br.lower,
                'ci_upper': br.upper,
                'accuracy': br.point,
                'significant': br.significant,
                'half_width': br.half_width,
            }

        serializable = {
            'model_tag':      model_tag,
            'metrics':        {k: asdict(v) for k, v in data['metrics'].items()},
            'examples':       data.get('examples', {}),
            'flip_matrices':  [
                _serialize_flip_matrix(fm)
                for fm in data['flip_matrices']
            ],
            'flip_grid':      data.get('flip_grid', {}),
            'alpha_sweep':    data.get('alpha_sweep', []),
            'locked_config':  data.get('locked_config', {}),
            'token_budgeting': data.get('token_budgeting', {}),
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

        budget_log = data.get('token_budget_log')
        if budget_log:
            budget_path = os.path.join(out_dir, f"{model_tag}_trimmed_cot_budget_log.json")
            with open(budget_path, 'w') as f:
                json.dump(budget_log, f, indent=2)
            print(f"  Saved: {budget_path}")

    summary_path = os.path.join(out_dir, 'summary_test.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")


# ── Main evaluation runner ─────────────────────────────────────────────────────

def run_final_evaluation(
    D_test, cfg, device, results_base="results", vectors_base="vectors",
    checkpoints_base="checkpoints", max_new_tokens=256, model_tags=None,
):
    require_exact_count(D_test, "D_test")
    if cfg.get("experiment_version") != EXPERIMENT_VERSION or cfg.get("winning_config") != "S3":
        raise RuntimeError("Phase 4 requires a full-experiment S3 selection, not legacy/smoke artifacts")
    selected = _load_phase3_best_configs(cfg, "S3", results_base)
    all_results = {}
    golds = [item["answer"].split("####", 1)[1].strip() for item in D_test]
    for model_tag in model_tags or MODEL_TAGS:
        base_id = MODEL_ID_MAP[model_tag]
        results_dir = os.path.join(results_base, "S3", model_tag)
        vectors_dir = os.path.join(vectors_base, "S3", model_tag)
        checkpoints_dir = os.path.join(checkpoints_base, "S3", model_tag)
        _require_phase2_inputs(vectors_dir)
        meta = _load_meta_file(vectors_dir)
        best = selected[model_tag]
        with open(os.path.join(results_dir, "phase3_run_meta.json")) as stream:
            run_meta = json.load(stream)
        if not run_meta.get("complete") or best.get("phase3_signature") != run_meta.get("signature"):
            raise RuntimeError(f"Unlocked or stale Phase 3 selection for {model_tag}")
        if best.get("phase3_val_sha256") != file_fingerprint(os.path.join(results_dir, "phase3_val.json")):
            raise RuntimeError("Phase 3 validation results changed after selection")
        if file_fingerprint(os.path.join(vectors_dir, "phase2_meta.json")) != run_meta["phase2"]:
            raise RuntimeError("Phase 2 artifacts changed after Phase 3 selection")
        for name, digest in run_meta["artifacts"].items():
            if file_fingerprint(os.path.join(vectors_dir, name)) != digest:
                raise RuntimeError(f"Steering artifact changed after lock: {name}")
        budget = int(best["latent_tokens"])
        tag = f"L{budget}"
        for key, suffix in (("cot", "cot"), ("ccot", f"ccot_{tag}")):
            if checkpoint_identity(os.path.join(checkpoints_dir, suffix)) != run_meta["checkpoints"][key]:
                raise RuntimeError(f"Checkpoint changed after lock: {suffix}")
        source_tag = best.get("vector_source") or "ccot"
        artifacts = load_source_artifacts(vectors_dir, source_tag, meta)
        with open(os.path.join(results_dir, "phase3_val.json")) as stream:
            validation = {row["condition"]: row for row in json.load(stream)}
        layer = artifacts["dom"]["best_layer"]
        direction = artifacts["dom"]["v_truth"]
        metrics, predictions, examples_by_condition = {}, {}, {}
        full_counts = []

        def record(name, examples):
            examples_by_condition[name] = [asdict(example) for example in examples]
            metrics[name] = collect_condition_metrics(
                examples, full_counts, name, model_tag, sum(example.latency_sec for example in examples),
            )
            predictions[name] = [example.correct for example in examples]
            print(f"[PH4] {model_tag}/{name}: acc={metrics[name].accuracy:.4f} n={len(examples)}")

        model, tokenizer = load_finetuned(os.path.join(checkpoints_dir, "cot"), device)
        try:
            full_examples = [
                run_steered_with_metrics(model, tokenizer, cot_prompt(item["question"]), item,
                                         None, layer, None, device, 512)
                for item in D_test
            ]
            full_counts = [example.reasoning_tokens for example in full_examples]
            record("full_cot", full_examples)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        model, tokenizer = load_base_frozen(base_id, device)
        try:
            direct = [
                run_steered_with_metrics(model, tokenizer, f"Question: {item['question']}\n\nAnswer:", item,
                                         None, layer, None, device, 32)
                for item in D_test
            ]
            for example in direct:
                example.reasoning_tokens = 0
            record("no_cot", direct)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        model, tokenizer = load_finetuned(os.path.join(checkpoints_dir, f"ccot_{tag}"), device)
        try:
            baseline = [
                run_steered_with_metrics(model, tokenizer, latent_prompt(item["question"], budget), item,
                                         None, layer, direction, device, max_new_tokens)
                for item in D_test
            ]
            record(f"ccot_{tag}", baseline)
            for method in available_methods(artifacts):
                name = f"{method}_{tag}_{source_tag}"
                if name not in validation:
                    raise RuntimeError(f"Condition was not validated before test: {name}")
                alpha = float(validation[name]["alpha"])
                torch.manual_seed(int(fingerprint([0, name])[:8], 16))
                examples = [
                    run_steered_with_metrics(
                        model, tokenizer, latent_prompt(item["question"], budget), item,
                        None, layer, direction, device, max_new_tokens,
                        method=method, artifacts=artifacts, alpha=alpha,
                    ) for item in D_test
                ]
                record(name, examples)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        budgets = coconut_thinking_token_counts(baseline, budget)
        model, tokenizer = load_finetuned(os.path.join(checkpoints_dir, "cot"), device)
        trimmed = []
        try:
            for item, token_budget in zip(D_test, budgets):
                started = time.time()
                prediction, reasoning = run_trimmed_cot(model, tokenizer, item, token_budget, device)
                gold = item["answer"].split("####", 1)[1].strip()
                text = reasoning + ("\n#### " + prediction if prediction is not None else "")
                trimmed.append(ExampleResult(
                    correct=prediction is not None and normalize_answer(prediction) == normalize_answer(gold),
                    answer_found=prediction is not None,
                    reasoning_tokens=len(tokenizer.encode(reasoning, add_special_tokens=False)),
                    total_tokens=len(tokenizer.encode(cot_prompt(item["question"]) + text)),
                    latency_sec=time.time() - started, generated_text=text,
                    question_hash=fingerprint(item["question"]),
                ))
            record(f"trimmed_{tag}", trimmed)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        condition_cis = compute_condition_cis(predictions)
        paired_cis = compute_paired_cis(predictions, tag, source_tag)
        for method in ("multilayer_dom", "multilayer_dom_mlp", "iti"):
            condition = f"{method}_{tag}_{source_tag}"
            if condition in predictions:
                point, lower, upper, significant = bootstrap_ci_difference(
                    predictions[f"ccot_{tag}"], predictions[condition],
                )
                paired_cis[f"{method}_vs_ccot"] = BootstrapResult(point, lower, upper, significant)
        for name, interval in condition_cis.items():
            metrics[name].ci_lower_95, metrics[name].ci_upper_95 = interval.lower, interval.upper
        all_results[model_tag] = {
            "metrics": metrics, "examples": examples_by_condition,
            "flip_matrices": compute_all_flip_matrices(predictions, golds, model_tag, tag, source_tag),
            "flip_grid": compute_full_flip_grid(predictions, golds, model_tag),
            "locked_config": best, "condition_cis": condition_cis, "paired_cis": paired_cis,
            "token_budgeting": {"latent_tokens": budget, "n_examples": len(D_test),
                                "mean_coconut_thinking_tokens": float(np.mean(budgets))},
            "token_budget_log": {"budgets": budgets, "dataset_fingerprint": dataset_fingerprint(D_test)},
        }
    return all_results


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
        default=None,
        help='Directory for summary_test.json and <model>_test.json (default: results/final)',
    )
    parser.add_argument(
        '--model',
        default='all',
        help='Model tag(s): all | qwen25_0.5b | qwen25_math1.5b | llama32_3b,phi2',
    )
    parser.add_argument('--overwrite', action='store_true', help='Explicitly allow repeated test evaluation')
    parser.add_argument('--training-dataset', choices=('gsm8k', 'svamp', 'prontoqa'),
                        help='Source of checkpoints and steering settings; defaults to --dataset. Set gsm8k for SVAMP transfer.')
    args = parser.parse_args()

    init_project_dataset(args.dataset, interactive=sys.stdin.isatty())
    training_dataset = args.training_dataset or get_active_dataset_id()
    transfer = training_dataset != get_active_dataset_id()

    if args.results_dir is None:
        args.results_dir = (f'results/final_{get_active_dataset_id()}_transfer' if training_dataset == 'gsm8k' and transfer
                            else f'results/{training_dataset}_to_{get_active_dataset_id()}/final' if transfer
                            else os.path.join(artifact_root('results'), 'final'))
    summary_path = os.path.join(args.results_dir, 'summary_test.json')
    models_to_run = MODEL_TAGS if args.model == 'all' else args.model.split(',')
    if os.path.exists(summary_path):
        with open(summary_path) as stream:
            previous = json.load(stream)
        if previous.get('provenance', {}).get('eval_dataset') != get_active_dataset_id():
            raise RuntimeError('Refusing to mix datasets in one result directory')
        if previous.get('provenance', {}).get('experiment_version') != EXPERIMENT_VERSION:
            raise RuntimeError('Legacy test results require a new --results-dir')
        if previous.get('provenance', {}).get('training_dataset') != training_dataset:
            raise RuntimeError('Refusing to mix training datasets in one result directory')
        if not args.overwrite and set(models_to_run) & set(previous.get('models', [])):
            raise RuntimeError('Test results already exist; --overwrite explicitly permits a repeated evaluation')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[PH4] Device: {device}")

    torch.manual_seed(0)
    np.random.seed(0)

    cfg = _load_selected_yaml(selected_config_path(training_dataset))
    if cfg.get('training_dataset', 'gsm8k') != training_dataset:
        raise RuntimeError('Selected configuration belongs to a different training dataset')
    winning = cfg['winning_config']
    print(f"[PH4] Winning config: {winning}")
    models_to_run = MODEL_TAGS if args.model == 'all' else args.model.split(',')
    bad_model = [m for m in models_to_run if m not in MODEL_ID_MAP]
    if bad_model:
        raise SystemExit(f"Unknown model(s): {bad_model}. Valid: {MODEL_TAGS}")
    print(f"[PH4] Models: {models_to_run}")

    if transfer:
        print(f'[TRANSFER] Frozen {training_dataset} artifacts; evaluating {get_active_dataset_id()}')

    D_test = load_test_set()
    print(f"[PH4] Loaded D_test: {len(D_test)} examples")

    vectors_base = artifact_root('vectors', training_dataset)
    checkpoints_base = artifact_root('checkpoints', training_dataset)
    all_results = run_final_evaluation(
        D_test, cfg, device, model_tags=models_to_run,
        results_base=artifact_root('results', training_dataset),
        vectors_base=vectors_base, checkpoints_base=checkpoints_base,
    )
    provenance = {
        'eval_dataset': get_active_dataset_id(),
        'training_dataset': training_dataset,
        'steering_artifact_policy': f'frozen_from_{training_dataset}_pipeline',
        'winning_config': winning,
        'models': models_to_run,
        'n_test': len(D_test),
        'dataset_fingerprint': dataset_fingerprint(D_test),
        'experiment_version': EXPERIMENT_VERSION,
        'injection': 'generated_positions_only',
        'test_tuning': False,
        'vectors_base': vectors_base,
        'checkpoints_base': checkpoints_base,
    }
    save_final_results(all_results, args.results_dir, provenance=provenance)
    print(f"\n[PH4] Done. Results saved to {args.results_dir}/")


if __name__ == '__main__':
    main()
