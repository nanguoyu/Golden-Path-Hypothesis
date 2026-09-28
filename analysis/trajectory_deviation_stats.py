#!/usr/bin/env python3
"""Statistical checks for trajectory-deviation audit outputs.

This script implements the non-causal diagnostics requested by
`docs/research_plan_trajectory_deviation.md` section 8.2.  It reads the flat
step rows produced by `analysis/trajectory_deviation.py` and asks how much of a
metric is explained by timestep-only structure, prompt-only structure, and a
two-way additive prompt+step model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_METRICS = [
    "latent_drift_pre",
    "latent_drift_post",
    "output_drift_rel",
    "action_defect_rel_to_full",
    "state_gap_rel",
]
DEFAULT_PER_PROMPT_XS = [
    "step_index",
    "gate_accumulator_after",
    "gate_increment_rescaled",
    "is_cached",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trajectory-deviation statistical diagnostics.")
    p.add_argument("--acc", type=Path, required=True,
                   help="Trajectory audit output dir containing trajectory_deviation_steps.csv.")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   help="Step-level metrics to diagnose.")
    p.add_argument("--output_json", type=Path, default=None,
                   help="Default: <acc>/trajectory_deviation_stats_checks.json.")
    p.add_argument("--output_csv", type=Path, default=None,
                   help="Default: <acc>/trajectory_deviation_stats_checks.csv.")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if mechanical acceptance checks fail.")
    p.add_argument("--closure_rel_tol", type=float, default=1e-6,
                   help="Tolerance for relative decomposition closure checks.")
    p.add_argument("--cf_l2_tol", type=float, default=1e-6,
                   help="Tolerance for with/without-CF final latent equality.")
    p.add_argument("--full_action_tol", type=float, default=1e-6,
                   help="Tolerance for action_defect on native full steps.")
    return p.parse_args()


def _coerce_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, str):
        low = v.lower()
        if low == "true":
            return 1.0
        if low == "false":
            return 0.0
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return None
    if isinstance(v, str):
        low = v.strip().lower()
        if low in {"1", "true", "t", "yes", "y"}:
            return True
        if low in {"0", "false", "f", "no", "n"}:
            return False
    return None


def _rankdata(xs: List[float]) -> np.ndarray:
    arr = np.asarray(xs, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = _rankdata(xs), _rankdata(ys)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _kendall(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            prod = (xs[i] - xs[j]) * (ys[i] - ys[j])
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return None
    return float((concordant - discordant) / denom)


def _mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _finite_values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        x = _coerce_float(row.get(key))
        if x is not None:
            vals.append(x)
    return vals


def _numeric_stats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p95": None, "p99": None, "max": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(len(vals)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _variance(xs: Iterable[float]) -> Optional[float]:
    vals = list(xs)
    if len(vals) < 2:
        return None
    arr = np.asarray(vals, dtype=float)
    return float(np.var(arr))


def _r2(y: np.ndarray, yhat: np.ndarray) -> Optional[float]:
    if len(y) < 2:
        return None
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 0.0:
        return None
    sse = float(np.sum((y - yhat) ** 2))
    return float(1.0 - sse / sst)


def _metric_arrays(rows: List[Dict[str, Any]], metric: str) -> Tuple[np.ndarray, List[int], List[int]]:
    ys: List[float] = []
    pids: List[int] = []
    steps: List[int] = []
    for row in rows:
        y = _coerce_float(row.get(metric))
        if y is None:
            continue
        ys.append(y)
        pids.append(int(row["prompt_id"]))
        steps.append(int(row["step_index"]))
    return np.asarray(ys, dtype=float), pids, steps


def _fixed_effects(rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    y, pids, steps = _metric_arrays(rows, metric)
    if len(y) == 0:
        return {"metric": metric, "n": 0}
    grand = float(np.mean(y))
    step_mean: Dict[int, float] = {}
    prompt_mean: Dict[int, float] = {}
    by_step: Dict[int, List[float]] = defaultdict(list)
    by_prompt: Dict[int, List[float]] = defaultdict(list)
    for val, pid, step in zip(y, pids, steps):
        by_step[step].append(float(val))
        by_prompt[pid].append(float(val))
    step_mean = {k: _mean(v) for k, v in by_step.items()}
    prompt_mean = {k: _mean(v) for k, v in by_prompt.items()}
    step_hat = np.asarray([step_mean[s] for s in steps], dtype=float)
    prompt_hat = np.asarray([prompt_mean[p] for p in pids], dtype=float)
    two_way_hat = np.asarray(
        [prompt_mean[p] + step_mean[s] - grand for p, s in zip(pids, steps)],
        dtype=float,
    )
    within_prompt_vars = [_variance(v) for v in by_prompt.values()]
    cross_prompt_step_vars = [_variance(v) for v in by_step.values()]
    within_prompt_vars = [v for v in within_prompt_vars if v is not None]
    cross_prompt_step_vars = [v for v in cross_prompt_step_vars if v is not None]
    return {
        "metric": metric,
        "n": int(len(y)),
        "n_prompts": int(len(by_prompt)),
        "n_steps": int(len(by_step)),
        "mean": grand,
        "std": float(np.std(y)),
        "step_fixed_r2": _r2(y, step_hat),
        "prompt_fixed_r2": _r2(y, prompt_hat),
        "prompt_plus_step_fixed_r2": _r2(y, two_way_hat),
        "mean_within_prompt_variance": _mean(within_prompt_vars),
        "mean_cross_prompt_same_step_variance": _mean(cross_prompt_step_vars),
    }


def _per_prompt_correlations(rows: List[Dict[str, Any]], metrics: List[str]) -> List[Dict[str, Any]]:
    by_prompt: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[int(row["prompt_id"])].append(row)
    out: List[Dict[str, Any]] = []
    for metric in metrics:
        for x_key in DEFAULT_PER_PROMPT_XS:
            spears: List[float] = []
            kendalls: List[float] = []
            n_valid = 0
            for prompt_rows in by_prompt.values():
                xs: List[float] = []
                ys: List[float] = []
                for row in prompt_rows:
                    x = _coerce_float(row.get(x_key))
                    y = _coerce_float(row.get(metric))
                    if x is None or y is None:
                        continue
                    xs.append(x)
                    ys.append(y)
                sp = _spearman(xs, ys)
                kd = _kendall(xs, ys)
                if sp is not None:
                    spears.append(sp)
                if kd is not None:
                    kendalls.append(kd)
                if sp is not None or kd is not None:
                    n_valid += 1
            out.append({
                "metric": metric,
                "x": x_key,
                "n_prompts_valid": int(n_valid),
                "spearman_mean": _mean(spears) if spears else None,
                "spearman_median": float(np.median(spears)) if spears else None,
                "kendall_mean": _mean(kendalls) if kendalls else None,
                "kendall_median": float(np.median(kendalls)) if kendalls else None,
            })
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_prompt_jsons(acc: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summaries: List[Dict[str, Any]] = []
    manifests: List[Dict[str, Any]] = []
    for prompt_dir in sorted(acc.glob("prompt_*")):
        if not prompt_dir.is_dir():
            continue
        summary = _read_json(prompt_dir / "trajectory_summary.json")
        manifest = _read_json(prompt_dir / "manifest.json")
        if summary is not None:
            summaries.append(summary)
        if manifest is not None:
            manifests.append(manifest)
    return summaries, manifests


def _completion_checks(
    acc: Path,
    rows: List[Dict[str, Any]],
    *,
    closure_rel_tol: float,
    cf_l2_tol: float,
    full_action_tol: float,
) -> Dict[str, Any]:
    summaries, manifests = _load_prompt_jsons(acc)
    by_prompt: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[int(row["prompt_id"])].append(row)
    counts = {pid: len(prs) for pid, prs in by_prompt.items()}
    unique_counts = sorted(set(counts.values()))
    expected_steps = unique_counts[0] if len(unique_counts) == 1 else None
    bad_prompt_row_counts = {
        str(pid): n for pid, n in counts.items()
        if expected_steps is not None and n != expected_steps
    }

    scheduler_mismatches = 0
    for row in rows:
        step = _coerce_float(row.get("step_index"))
        sched = _coerce_float(row.get("scheduler_step_index"))
        if step is None or sched is None or int(step) != int(sched):
            scheduler_mismatches += 1

    cached_rows = [r for r in rows if _as_bool(r.get("is_cached")) is True]
    full_rows = [r for r in rows if _as_bool(r.get("is_cached")) is False]

    def digest_equal_count(lhs: str, rhs: str, subset: List[Dict[str, Any]]) -> int:
        return sum(1 for row in subset if row.get(lhs) == row.get(rhs))

    full_action_defects = _finite_values(full_rows, "action_defect")
    cf_equal = []
    native_rows_equal = []
    cf_l2_vals: List[float] = []
    native_row_mismatch_counts: List[float] = []
    native_row_float_diffs: List[float] = []
    for summary in summaries:
        checks = summary.get("acceptance_checks") or {}
        v = _as_bool(checks.get("with_without_cf_gate_sequence_equal"))
        if v is not None:
            cf_equal.append(v)
        native_equal = _as_bool(checks.get("with_without_cf_native_rows_equal"))
        if native_equal is not None:
            native_rows_equal.append(native_equal)
        l2 = _coerce_float(checks.get("with_without_cf_final_l2"))
        if l2 is not None:
            cf_l2_vals.append(l2)
        n_mismatch = _coerce_float(checks.get("with_without_cf_native_mismatch_count"))
        if n_mismatch is not None:
            native_row_mismatch_counts.append(n_mismatch)
        max_float_diff = _coerce_float(checks.get("with_without_cf_native_max_float_abs_diff"))
        if max_float_diff is not None:
            native_row_float_diffs.append(max_float_diff)

    manifest_complete = sum(1 for m in manifests if _as_bool(m.get("complete")) is True)
    image_pairs_present = 0
    latent_pairs_present = 0
    for prompt_dir in sorted(acc.glob("prompt_*")):
        if not prompt_dir.is_dir():
            continue
        if (prompt_dir / "baseline.png").is_file() and (prompt_dir / "cached.png").is_file():
            image_pairs_present += 1
        if (prompt_dir / "baseline.pt").is_file() and (prompt_dir / "cached.pt").is_file():
            latent_pairs_present += 1

    closure_output = _numeric_stats(_finite_values(rows, "decomposition_closure_rel"))
    closure_latent = _numeric_stats(_finite_values(rows, "latent_decomposition_closure_rel"))
    full_action = _numeric_stats(full_action_defects)
    max_cf_l2 = max(cf_l2_vals) if cf_l2_vals else None
    checks = {
        "n_rows": int(len(rows)),
        "n_prompts_from_rows": int(len(by_prompt)),
        "n_prompt_summaries": int(len(summaries)),
        "n_manifests": int(len(manifests)),
        "expected_steps_per_prompt": expected_steps,
        "unique_row_counts_per_prompt": unique_counts,
        "bad_prompt_row_counts": bad_prompt_row_counts,
        "manifest_complete_count": int(manifest_complete),
        "image_pair_count": int(image_pairs_present),
        "latent_pair_count": int(latent_pairs_present),
        "scheduler_step_index_mismatches": int(scheduler_mismatches),
        "n_cached_rows": int(len(cached_rows)),
        "n_full_rows": int(len(full_rows)),
        "decomposition_closure_rel": closure_output,
        "latent_decomposition_closure_rel": closure_latent,
        "full_step_action_defect": full_action,
        "digest_stability": {
            "all_rows_cache_native_before_cf_equals_after_cf": {
                "n_equal": digest_equal_count(
                    "cache_state_digest_after_native_before_cf",
                    "cache_state_digest_after_cf",
                    rows,
                ),
                "n_total": int(len(rows)),
            },
            "cached_rows_cache_native_before_cf_equals_after_cf": {
                "n_equal": digest_equal_count(
                    "cache_state_digest_after_native_before_cf",
                    "cache_state_digest_after_cf",
                    cached_rows,
                ),
                "n_total": int(len(cached_rows)),
            },
            "scheduler_before_cf_equals_after_cf": {
                "n_equal": digest_equal_count(
                    "scheduler_state_digest_before_cf",
                    "scheduler_state_digest_after_cf",
                    rows,
                ),
                "n_total": int(len(rows)),
            },
            "rng_before_cf_equals_after_cf": {
                "n_equal": digest_equal_count(
                    "rng_state_digest_before_cf",
                    "rng_state_digest_after_cf",
                    rows,
                ),
                "n_total": int(len(rows)),
            },
        },
        "with_without_cf_gate_sequence_equal_count": int(sum(1 for v in cf_equal if v)),
        "with_without_cf_gate_sequence_total": int(len(cf_equal)),
        "with_without_cf_final_l2": _numeric_stats(cf_l2_vals),
        "with_without_cf_native_rows_equal_count": int(sum(1 for v in native_rows_equal if v)),
        "with_without_cf_native_rows_total": int(len(native_rows_equal)),
        "with_without_cf_native_mismatch_count": _numeric_stats(native_row_mismatch_counts),
        "with_without_cf_native_max_float_abs_diff": _numeric_stats(native_row_float_diffs),
        "pass": True,
        "fail_reasons": [],
    }

    fail_reasons: List[str] = []
    if expected_steps is None:
        fail_reasons.append("nonuniform row counts per prompt")
    if len(summaries) != len(by_prompt):
        fail_reasons.append("prompt summary count differs from prompt count in rows")
    if manifest_complete != len(by_prompt):
        fail_reasons.append("not all prompt manifests are complete")
    if scheduler_mismatches:
        fail_reasons.append("scheduler_step_index mismatches step_index")
    if closure_output["max"] is None or closure_output["max"] > closure_rel_tol:
        fail_reasons.append("output decomposition closure exceeds tolerance")
    if closure_latent["max"] is None or closure_latent["max"] > closure_rel_tol:
        fail_reasons.append("latent decomposition closure exceeds tolerance")
    if full_action["max"] is not None and full_action["max"] > full_action_tol:
        fail_reasons.append("native full steps have nonzero action_defect")
    if cf_equal and not all(cf_equal):
        fail_reasons.append("with/without-CF gate sequences differ")
    if native_rows_equal and not all(native_rows_equal):
        fail_reasons.append("with/without-CF native row fields differ")
    if max_cf_l2 is not None and max_cf_l2 > cf_l2_tol:
        fail_reasons.append("with/without-CF final latent differs")
    for key, rec in checks["digest_stability"].items():
        if rec["n_equal"] != rec["n_total"]:
            fail_reasons.append(f"digest instability: {key}")
    checks["pass"] = not fail_reasons
    checks["fail_reasons"] = fail_reasons
    return checks


def main() -> int:
    args = parse_args()
    rows_path = args.acc / "trajectory_deviation_steps.csv"
    if not rows_path.is_file():
        raise SystemExit(f"missing flat step rows: {rows_path}")
    rows = _read_rows(rows_path)
    if not rows:
        raise SystemExit(f"empty step rows: {rows_path}")

    completion_checks = _completion_checks(
        args.acc,
        rows,
        closure_rel_tol=args.closure_rel_tol,
        cf_l2_tol=args.cf_l2_tol,
        full_action_tol=args.full_action_tol,
    )
    fixed_effect_rows = [_fixed_effects(rows, metric) for metric in args.metrics]
    per_prompt_rows = _per_prompt_correlations(rows, args.metrics)
    payload = {
        "input_dir": str(args.acc),
        "n_rows": len(rows),
        "metrics": args.metrics,
        "completion_checks": completion_checks,
        "fixed_effects": fixed_effect_rows,
        "per_prompt_correlations": per_prompt_rows,
    }

    out_json = args.output_json or (args.acc / "trajectory_deviation_stats_checks.json")
    out_csv = args.output_csv or (args.acc / "trajectory_deviation_stats_checks.csv")
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_rows = []
    csv_rows.append({"kind": "completion_checks", **completion_checks})
    for row in fixed_effect_rows:
        rec = {"kind": "fixed_effects", **row}
        csv_rows.append(rec)
    for row in per_prompt_rows:
        rec = {"kind": "per_prompt_correlation", **row}
        csv_rows.append(rec)
    _write_csv(out_csv, csv_rows)

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_csv}")
    print(f"completion_checks: pass={completion_checks['pass']} "
          f"fail_reasons={completion_checks['fail_reasons']}")
    for row in fixed_effect_rows:
        print(
            f"{row['metric']}: step_R2={row.get('step_fixed_r2')} "
            f"prompt_R2={row.get('prompt_fixed_r2')} "
            f"two_way_R2={row.get('prompt_plus_step_fixed_r2')}"
        )
    if args.strict and not completion_checks["pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
