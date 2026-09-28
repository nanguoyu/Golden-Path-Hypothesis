#!/usr/bin/env python3
"""Attribute the four-factor single-step score to P and its calibrated profile.

This is a post-stage analysis of the existing Phase-1 probes.  For each cache
approximation and eligible (prompt, timestep) observation it compares

    P(i, k),  h(k) = mean(S(k)) Q(k) mean(A(k)),  and  P(i, k) h(k).

It also reports the timestep index alone, negated as -k, on the same rows, so
the profile can be compared with the plain denoising-time ordering.

It also evaluates the leave-one-out versions used by ``score_compare.py`` and
can verify that ``P * h_loo`` exactly reproduces the manuscript's Score_4.
No model inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis import score_compare as score


def _tie_expected_recall(x: np.ndarray, y: np.ndarray, top_p: float) -> float:
    """Expected top-p recall when ties at the prediction boundary are random."""
    if x.size == 0:
        return float("nan")
    n_top = max(1, int(round(x.size * top_p / 100.0)))
    true_top = set(np.argsort(-y, kind="mergesort")[:n_top].tolist())
    order = np.argsort(-x, kind="mergesort")
    selected = 0
    expected_hits = 0.0
    start = 0
    while start < order.size and selected < n_top:
        stop = start + 1
        value = x[order[start]]
        while stop < order.size and x[order[stop]] == value:
            stop += 1
        group = order[start:stop]
        slots = min(n_top - selected, group.size)
        hits = sum(int(idx) in true_top for idx in group)
        expected_hits += float(slots) * float(hits) / float(group.size)
        selected += slots
        start = stop
    return expected_hits / float(n_top)


def _metrics(x: Iterable[float], y: Iterable[float], top_p: float) -> Dict[str, float]:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    return {
        "n": int(xa.size),
        "spearman": score._spearman(xa, ya),
        "recall_at_top_pct": score._recall_at_top_p(xa, ya, top_p),
        "tie_expected_recall_at_top_pct": _tie_expected_recall(xa, ya, top_p),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s_k", required=True, type=Path)
    parser.add_argument("--q_k", required=True, type=Path)
    parser.add_argument("--a_k", required=True, type=Path)
    parser.add_argument("--oracle_seacache", required=True, type=Path)
    parser.add_argument("--oracle_hicache", required=True, type=Path)
    parser.add_argument("--oracle_taylorseer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top_p", type=float, default=10.0)
    parser.add_argument("--reference_score_json", type=Path, default=None)
    return parser.parse_args()


def _reference_metrics(path: Path, method: str) -> Tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload["summary"][method]["latent_L2_abs"]["Score_4"]
    return float(block["spearman"]), float(block["recall_at_top_10pct"])


def main() -> int:
    args = _parse_args()
    p_data = score._load_p_k_or_s_k(None, args.s_k)
    s_data = score._load_s_k_only(args.s_k)
    q_data = score._load_q_k(args.q_k)
    a_data = score._load_a_k(args.a_k)
    oracle_paths = {
        "seacache": args.oracle_seacache,
        "hicache": args.oracle_hicache,
        "taylorseer": args.oracle_taylorseer,
    }
    oracle_data = {method: score._load_oracle(path) for method, path in oracle_paths.items()}

    s_arr, s_stats = score._build_calib_table(s_data, score._sk_field)
    a_arr, a_stats = score._build_calib_table(a_data, score._ak_field)
    results: Dict[str, Any] = {}

    for method in score.METHODS:
        values: Dict[str, List[float]] = {
            "timestep_neg": [],
            "p": [], "p_times_q": [], "p_times_s_q_loo": [],
            "h_fixed": [], "p_times_h_fixed": [],
            "h_loo": [], "p_times_h_loo": [], "target": [],
        }
        step_index: List[int] = []
        common_prompts = sorted(set(p_data) & set(oracle_data[method]))
        for prompt_id in common_prompts:
            common_steps = sorted(set(p_data[prompt_id]) & set(oracle_data[method][prompt_id]))
            for k in common_steps:
                p_value = p_data[prompt_id][k].get(f"p_{method}_abs")
                target = oracle_data[method][prompt_id][k].get("latent_L2_abs")
                if p_value is None or target is None or q_data is None or k >= len(q_data):
                    continue
                s_loo = score._calib_lookup(s_arr, s_stats, prompt_id, k, method, "loo")
                a_loo = score._calib_lookup(a_arr, a_stats, prompt_id, k, method, "loo")
                s_fixed = score._calib_lookup(s_arr, s_stats, prompt_id, k, method, "mean")
                a_fixed = score._calib_lookup(a_arr, a_stats, prompt_id, k, method, "mean")
                raw = (p_value, target, s_loo, a_loo, s_fixed, a_fixed, q_data[k])
                if any(v is None or not math.isfinite(float(v)) for v in raw):
                    continue
                p_float = float(p_value)
                h_loo = float(s_loo) * float(q_data[k]) * float(a_loo)
                h_fixed = float(s_fixed) * float(q_data[k]) * float(a_fixed)
                step_index.append(int(k))
                values["timestep_neg"].append(-float(k))
                values["p"].append(p_float)
                values["p_times_q"].append(p_float * float(q_data[k]))
                values["p_times_s_q_loo"].append(
                    p_float * float(s_loo) * float(q_data[k])
                )
                values["h_fixed"].append(h_fixed)
                values["p_times_h_fixed"].append(p_float * h_fixed)
                values["h_loo"].append(h_loo)
                values["p_times_h_loo"].append(p_float * h_loo)
                values["target"].append(float(target))

        target_values = values.pop("target")
        n_common = len(target_values)
        for name, series in values.items():
            if len(series) != n_common:
                raise AssertionError(
                    f"{method}: {name} has {len(series)} rows, expected {n_common}"
                )
        for value, k in zip(values["timestep_neg"], step_index):
            if value != -float(k):
                raise AssertionError(
                    f"{method}: timestep_neg {value} does not match step {k}"
                )
        method_result = {
            name: _metrics(series, target_values, args.top_p)
            for name, series in values.items()
        }
        if args.reference_score_json is not None:
            ref_rho, ref_recall = _reference_metrics(args.reference_score_json, method)
            got = method_result["p_times_h_loo"]
            if not math.isclose(got["spearman"], ref_rho, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(f"{method}: Score_4 Spearman mismatch {got['spearman']} != {ref_rho}")
            if not math.isclose(got["recall_at_top_pct"], ref_recall, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(
                    f"{method}: Score_4 recall mismatch {got['recall_at_top_pct']} != {ref_recall}"
                )
        results[method] = method_result

    output = {
        "schema": "four_factor_profile_ablation.v1",
        "target": "latent_L2_abs",
        "top_p_percent": float(args.top_p),
        "definitions": {
            "timestep_neg": "-k; the timestep index alone, negated to match the sign",
            "h_fixed": "mean_prompt(S_k) * Q_k * mean_prompt(A_k); one fixed value per timestep",
            "h_loo": "leave_one_prompt_out(S_k) * Q_k * leave_one_prompt_out(A_k)",
            "scope": "isolated one-step cache interventions; not a multi-step schedule evaluation",
        },
        "sources": {
            "s_k": str(args.s_k), "q_k": str(args.q_k), "a_k": str(args.a_k),
            "oracles": {method: str(path) for method, path in oracle_paths.items()},
            "reference_score_json": (
                str(args.reference_score_json) if args.reference_score_json is not None else None
            ),
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    tsv_path = args.out.with_suffix(".tsv")
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["method", "score", "n", "spearman", "recall_at_top_pct",
                         "tie_expected_recall_at_top_pct"])
        for method, block in results.items():
            for name, metrics in block.items():
                writer.writerow([method, name, metrics["n"], metrics["spearman"],
                                 metrics["recall_at_top_pct"],
                                 metrics["tie_expected_recall_at_top_pct"]])
    print(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
