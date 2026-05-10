"""Summarize evaluate_final.py JSON: CCoT vs DoM accuracy/CIs and ccot→dom flip matrix.

Typical use after Phase 5:

  python scripts/print_transfer_summary.py results/final_svamp_transfer
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _ratio_int(locked: dict) -> int:
    r = locked.get("ratio")
    if r is None:
        return 7
    return int(round(float(r) * 10))


def _ccot_dom_keys(locked: dict) -> tuple[str, str]:
    src = str(locked.get("vector_source") or "ccot")
    ri = _ratio_int(locked)
    return f"ccot_R{ri}", f"dom_R{ri}_{src}"


def _print_model_report(data: dict) -> None:
    prov = data.get("provenance") or {}
    if prov:
        print("Provenance:")
        for k in (
            "eval_dataset",
            "steering_artifact_policy",
            "winning_config",
            "vectors_dir",
            "checkpoints_dir",
        ):
            if k in prov:
                print(f"  {k}: {prov[k]}")
        print()

    locked = data.get("locked_config") or {}
    print("Locked Phase 3 config (alpha_star and vectors from this lock):")
    for k in ("ratio", "vector_source", "vector_method", "alpha_star"):
        if k in locked:
            print(f"  {k}: {locked[k]}")
    print()

    ccot_k, dom_k = _ccot_dom_keys(locked)
    metrics = data.get("metrics") or {}
    cis = data.get("condition_cis") or {}

    if ccot_k not in metrics:
        print(f"  (no {ccot_k} in metrics — CCoT may have been skipped)", file=sys.stderr)
        return
    if dom_k not in metrics:
        print(f"  (no {dom_k} in metrics — DoM may have been skipped)", file=sys.stderr)
        return

    mc = metrics[ccot_k]
    md = metrics[dom_k]
    print("Accuracy:")
    print(
        f"  {ccot_k}: {mc['accuracy']:.4f}  "
        f"(n_correct={mc['n_correct']} / n_total={mc['n_total']})"
    )
    print(
        f"  {dom_k}:  {md['accuracy']:.4f}  "
        f"(n_correct={md['n_correct']} / n_total={md['n_total']})"
    )
    print(f"  delta (DoM - CCoT): {md['accuracy'] - mc['accuracy']:+.4f}")
    print()

    if ccot_k in cis and dom_k in cis:
        c_ci = cis[ccot_k]
        d_ci = cis[dom_k]
        print("95% bootstrap CI (accuracy):")
        print(
            f"  {ccot_k}: {c_ci['point']:.4f}  "
            f"[{c_ci['lower']:.4f}, {c_ci['upper']:.4f}]"
        )
        print(
            f"  {dom_k}:  {d_ci['point']:.4f}  "
            f"[{d_ci['lower']:.4f}, {d_ci['upper']:.4f}]"
        )
        print()

    fms = data.get("flip_matrices") or []
    pair = None
    for fm in fms:
        if fm.get("condition_a") == ccot_k and fm.get("condition_b") == dom_k:
            pair = fm
            break
    if pair:
        print(f"Flip matrix ({ccot_k} → {dom_k}; a=unsteered CCoT, b=DoM):")
        print(f"  F00 (stable correct):  {pair['F00']}")
        print(f"  F01 (Right→Wrong):      {pair['F01']}")
        print(f"  F10 (Wrong→Right):      {pair['F10']}")
        print(f"  F11 (stable wrong):      {pair['F11']}")
        print(
            f"  net_gain (F10 - F01): {pair['net_gain']}  "
            f"improvement_rate: {pair['improvement_rate']:.4f}  "
            f"degradation_rate: {pair['degradation_rate']:.4f}"
        )
    else:
        print("(No flip matrix for CCoT→DoM in file.)", file=sys.stderr)

    print()
    print(
        "Interpretation: if DoM improves SVAMP accuracy with a favorable "
        "(F10 vs F01) flip matrix, the GSM8K-derived direction may encode "
        "general arithmetic reasoning; otherwise it may be "
        "distribution-specific to GSM8K. Either outcome is publishable if stated clearly."
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Print CCoT vs DoM metrics and ccot→dom flip matrix from evaluate_final JSON."
    )
    p.add_argument(
        "path",
        nargs="?",
        default="results/final_svamp_transfer",
        help="Directory with *_test.json or a single *_test.json path",
    )
    args = p.parse_args()
    path = args.path

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*_test.json")))
        if not files:
            print(f"No *_test.json in {path!r}", file=sys.stderr)
            sys.exit(1)
    elif os.path.isfile(path):
        files = [path]
    else:
        print(f"Not a file or directory: {path!r}", file=sys.stderr)
        sys.exit(1)

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        print("=" * 70)
        print(os.path.basename(fp))
        print("=" * 70)
        _print_model_report(data)
        print()


if __name__ == "__main__":
    main()
