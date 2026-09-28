#!/usr/bin/env python3
"""Build the numbers file behind docs/suffix_reversal_results.md.

Reads the payload causal-fork runs (one per forecast payload) and the
closed-loop trajectory-deviation runs, and writes a single JSON holding the
four number groups the pre-registered plan names:

    G1  closed_loop_exposure_correlations
    G2  fork_design
    G3  suffix_reversal_by_step
    G4  local_vs_cross_correlations

Every statistic the results document prints comes from this file; the renderer
reads it and types nothing of its own.

Example:

    python analysis/suffix_reversal.py \
      --fork taylor_o1=$DATA/suffix_reversal/fork/suffix_taylor_o1 \
      --fork ensemble_mean=$DATA/suffix_reversal/fork/suffix_ensemble_mean \
      --exposure seacache_t029=$DATA/suffix_reversal/exposure/sea_t029 \
      --out resources/suffix_reversal/report_numbers.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]

# Per-fork-row columns staged for re-analysis. The two G4 predictors and the
# terminal target lead; the rest describe the branch pair that produced them.
ROW_COLUMNS = (
    "prompt_id",
    "fork_step",
    "tail_policy",
    "forecast_payload_mode",
    "improvement_reuse_minus_forecast",
    "reuse_final_drift",
    "forecast_final_drift",
    "native_final_drift",
    "forecast_reuse_final_l2",
    "first_improvement_action_sq",
    "first_improvement_action_state_cross",
    "first_improvement_state_sq",
    "reuse_first_action_defect",
    "forecast_first_action_defect",
    "reuse_first_state_gap",
    "forecast_first_state_gap",
    "reuse_first_dot_action_gap",
    "forecast_first_dot_action_gap",
    "future_action_hamming_forecast_vs_reuse",
    "reuse_cache_rate_suffix",
    "forecast_cache_rate_suffix",
)

EXPOSURE_COLUMNS = (
    "prompt_id",
    "total_action_exposure",
    "total_state_gap_exposure",
    "final_latent_drift",
    "cache_rate",
    "n_cached",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fork", action="append", default=[], metavar="LABEL=DIR",
                   help="payload causal-fork run directory, one per forecast payload.")
    p.add_argument("--exposure", action="append", default=[], metavar="LABEL=DIR",
                   help="closed-loop trajectory-deviation run directory.")
    p.add_argument("--schedule_file", type=Path,
                   default=REPO / "resources/sp_cross_schedules/flux_k29_meancache.txt")
    p.add_argument("--schedule_cell", default="",
                   help="name of the stored cell whose decisions locked the prefix.")
    p.add_argument("--fork_steps", default="9,20,33,37,41,46")
    p.add_argument("--tail_policy", default="locked-after")
    p.add_argument("--g3_prompt_ids", type=Path,
                   default=REPO / "resources/suffix_reversal/prompt_ids_test200.txt")
    p.add_argument("--g1_prompt_ids", type=Path,
                   default=REPO / "resources/suffix_reversal/prompt_ids_test100.txt")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--out", type=Path,
                   default=REPO / "resources/suffix_reversal/report_numbers.json")
    p.add_argument("--stage_dir", type=Path, default=None,
                   help="directory for the staged per-row tables [default: --out's parent].")
    return p.parse_args()


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _split_labelled(items: Sequence[str], what: str) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--{what} wants LABEL=DIR, got {item!r}")
        label, _, path = item.partition("=")
        label = label.strip()
        directory = Path(path.strip()).expanduser()
        if not label:
            raise SystemExit(f"--{what} has an empty label in {item!r}")
        if not directory.is_dir():
            raise SystemExit(f"--{what} {label}: missing directory {directory}")
        out.append((label, directory))
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(tok) for tok in text.replace(",", " ").split() if tok.strip()]


def _num(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, str):
        low = v.strip().lower()
        if low == "true":
            return 1.0
        if low == "false":
            return 0.0
        if low in {"", "none", "nan"}:
            return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _p_from_r(r: Optional[float], n: int) -> Optional[float]:
    """Two-sided p for a correlation of n pairs, from the t distribution."""
    if r is None or n < 4:
        return None
    r = max(min(float(r), 1.0 - 1e-15), -1.0 + 1e-15)
    t = abs(r) * math.sqrt((n - 2) / (1.0 - r * r))
    return float(2.0 * stats.t.sf(t, n - 2))


def _corr(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]) -> Dict[str, Any]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)
             if x is not None and y is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson": None, "spearman": None,
                "pearson_p": None, "spearman_p": None}
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    pearson = None if x.std() <= 0.0 or y.std() <= 0.0 else float(np.corrcoef(x, y)[0, 1])
    rx, ry = _rankdata(x), _rankdata(y)
    spearman = None if rx.std() <= 0.0 or ry.std() <= 0.0 else float(np.corrcoef(rx, ry)[0, 1])
    n = len(pairs)
    return {"n": n, "pearson": pearson, "spearman": spearman,
            "pearson_p": _p_from_r(pearson, n), "spearman_p": _p_from_r(spearman, n)}


def _bootstrap_ci(values: Sequence[float], *, n_boot: int, rng: np.random.Generator
                  ) -> Optional[List[float]]:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return None
    if vals.size == 1 or n_boot <= 0:
        return [float(vals[0]), float(vals[0])]
    draws = rng.choice(vals, size=(int(n_boot), vals.size), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _stats(values: Sequence[float], *, n_boot: int, rng: np.random.Generator) -> Dict[str, Any]:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "ci95_mean": None,
                "positive": 0, "negative": 0, "zero": 0}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "ci95_mean": _bootstrap_ci(vals, n_boot=n_boot, rng=rng),
        "positive": int((vals > 0).sum()),
        "negative": int((vals < 0).sum()),
        "zero": int((vals == 0).sum()),
    }


def _write_tsv(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), delimiter="\t",
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _write_tsv_gz(path: Path, columns: Sequence[str], rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    # mtime=0 so a re-run of the same rows produces the same bytes
    with path.open("wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), delimiter="\t",
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})
            written += 1
    return written


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def _derive(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add the squared first-step decomposition the plan's G3/G4 read.

    a = payload velocity minus the same-state full velocity, g = the same-state
    full velocity minus the full-trajectory velocity, both at the first step
    after the fork; the runner stores their norms and inner product per branch.
    """
    out = dict(row)
    for branch in ("reuse", "forecast"):
        action = _num(row.get(f"{branch}_first_action_defect"))
        state = _num(row.get(f"{branch}_first_state_gap"))
        dot = _num(row.get(f"{branch}_first_dot_action_gap"))
        out[f"{branch}_first_action_sq"] = None if action is None else action * action
        out[f"{branch}_first_state_sq"] = None if state is None else state * state
        out[f"{branch}_first_action_state_cross"] = None if dot is None else 2.0 * dot
    for field in ("action_sq", "state_sq", "action_state_cross"):
        r = _num(out.get(f"reuse_first_{field}"))
        f_ = _num(out.get(f"forecast_first_{field}"))
        out[f"first_improvement_{field}"] = None if r is None or f_ is None else r - f_
    return out


def _load_fork_rows(acc: Path) -> List[Dict[str, Any]]:
    """Read the per-prompt fork CSVs: those are the complete record."""
    paths = sorted(acc.glob("prompt_*/payload_causal_fork_rows.csv"))
    if not paths:
        paths = sorted(acc.glob("payload_causal_fork_rows_shard*of*.csv"))
    if not paths:
        raise SystemExit(f"no fork row CSV under {acc}")
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_csv(path))
    return [_derive(row) for row in rows]


def _load_exposure_rows(acc: Path) -> List[Dict[str, Any]]:
    """Read one closed-loop run's per-prompt exposures and terminal drift."""
    summary_csv = acc / "trajectory_deviation_prompt_summary.csv"
    if summary_csv.is_file():
        raw = _read_csv(summary_csv)
    else:
        raw = []
        for path in sorted(acc.glob("prompt_*/trajectory_summary.json")):
            raw.append(json.loads(path.read_text(encoding="utf-8")))
    out: List[Dict[str, Any]] = []
    for row in raw:
        pid = row.get("prompt_id", row.get("prompt_idx", row.get("idx")))
        if pid is None or pid == "":
            continue
        out.append({
            "prompt_id": int(pid),
            "total_action_exposure": _num(row.get("total_action_exposure")),
            "total_state_gap_exposure": _num(row.get("total_state_gap_exposure")),
            "final_latent_drift": _num(row.get("final_latent_drift")),
            "cache_rate": _num(row.get("cache_rate")),
            "n_cached": _num(row.get("num_cached_steps", row.get("n_cached"))),
        })
    out.sort(key=lambda r: r["prompt_id"])
    return out


def _run_manifest(acc: Path) -> Dict[str, Any]:
    for name in sorted(acc.glob("manifest_shard*of*.json")):
        payload = json.loads(name.read_text(encoding="utf-8"))
        return {
            "git_commit": payload.get("git_commit"),
            "num_steps": payload.get("num_steps"),
            "seed": payload.get("seed"),
            "forecast_payload_mode": payload.get("forecast_payload_mode"),
            "forecast_payload_blend": payload.get("forecast_payload_blend"),
            "tail_policies": payload.get("tail_policies"),
            "fork_steps": payload.get("fork_steps"),
            "prompt_file": payload.get("prompt_file"),
            "payload_schedule_dir": (payload.get("cache_params") or {}).get("payload_schedule_dir"),
        }
    manifest = acc / "manifest.json"
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))
    return {}


# --------------------------------------------------------------------------
# number groups
# --------------------------------------------------------------------------

def build_g2(*, schedule_file: Path, schedule_cell: str, fork_steps: Sequence[int],
             tail_policy: str, num_steps: int, payload_labels: Sequence[str],
             fork_rows: Dict[str, List[Dict[str, Any]]],
             g3_ids: Sequence[int]) -> Dict[str, Any]:
    bits = schedule_file.read_text(encoding="utf-8").split()[0].strip()
    if len(bits) != int(num_steps) or set(bits) - {"0", "1"}:
        raise SystemExit(f"schedule file {schedule_file} is not {num_steps} bits of 0/1")
    cached = [i for i, b in enumerate(bits) if b == "1"]
    per_step: List[Dict[str, Any]] = []
    for step in fork_steps:
        entry: Dict[str, Any] = {
            "fork_step": int(step),
            "cached_in_schedule": bits[int(step)] == "1",
            "suffix_length": int(num_steps) - int(step),
            "materialized": {},
        }
        for label in payload_labels:
            ids = {int(_num(r["prompt_id"]))
                   for r in fork_rows[label]
                   if int(_num(r["fork_step"])) == int(step)
                   and str(r.get("valid")).lower() == "true"}
            entry["materialized"][label] = len(ids)
        per_step.append(entry)
    return {
        "schedule_bits": bits,
        "num_steps": int(num_steps),
        "n_cached_steps": len(cached),
        "n_full_steps": int(num_steps) - len(cached),
        "cache_ratio": len(cached) / float(num_steps),
        "cached_steps": cached,
        "schedule_cell": schedule_cell,
        "fork_steps": [int(s) for s in fork_steps],
        "tail_policy": tail_policy,
        "reuse_branch": "reuse",
        "forecast_branches": list(payload_labels),
        "n_prompts_requested": len(g3_ids),
        "expected_rows_per_payload": len(g3_ids) * len(fork_steps),
        "per_step": per_step,
    }


def build_g3(fork_rows: Dict[str, List[Dict[str, Any]]], *, fork_steps: Sequence[int],
             tail_policy: str, n_boot: int, seed: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"tail_policy": tail_policy, "by_payload": {}}
    for label, rows in fork_rows.items():
        valid = [r for r in rows
                 if str(r.get("valid")).lower() == "true"
                 and str(r.get("tail_policy")) == tail_policy]
        rng = np.random.default_rng(int(seed))
        steps_out: List[Dict[str, Any]] = []
        for step in list(fork_steps) + ["pooled"]:
            if step == "pooled":
                sel = valid
            else:
                sel = [r for r in valid if int(_num(r["fork_step"])) == int(step)]
            i50 = [_num(r["improvement_reuse_minus_forecast"]) for r in sel]
            local = [_num(r["first_improvement_action_sq"]) for r in sel]
            paired = [(l, t) for l, t in zip(local, i50) if l is not None and t is not None]
            n_pair = len(paired)
            steps_out.append({
                "fork_step": step,
                "n": len(sel),
                "n_paired": n_pair,
                # forecast is locally better when its squared action defect is
                # the smaller one, i.e. the reuse-minus-forecast gap is positive
                "n_local_forecast_better": sum(1 for l, _ in paired if l > 0),
                "n_local_forecast_worse": sum(1 for l, _ in paired if l < 0),
                "n_terminal_forecast_better": sum(1 for _, t in paired if t > 0),
                "n_reversal": sum(1 for l, t in paired if l < 0 and t > 0),
                "n_reversal_opposite": sum(1 for l, t in paired if l > 0 and t < 0),
                "frac_local_forecast_worse": (sum(1 for l, _ in paired if l < 0) / n_pair
                                              if n_pair else None),
                "frac_terminal_forecast_better": (sum(1 for _, t in paired if t > 0) / n_pair
                                                  if n_pair else None),
                "frac_reversal": (sum(1 for l, t in paired if l < 0 and t > 0) / n_pair
                                  if n_pair else None),
                # of the rows where the forecast is locally worse, how many end
                # closer: zero locally-worse rows makes the ratio undefined
                # rather than zero
                "frac_reversal_given_local_worse": (
                    sum(1 for l, t in paired if l < 0 and t > 0)
                    / sum(1 for l, _ in paired if l < 0)
                    if sum(1 for l, _ in paired if l < 0) else None),
                # the accumulated state gap the fork state carries. It is
                # identically zero at the schedule's first cached step, which
                # makes the cross term degenerate there.
                "state_gap_at_fork_mean": (
                    float(np.mean([v for v in (_num(r["reuse_first_state_gap"])
                                               for r in sel) if v is not None]))
                    if sel else None),
                "state_gap_at_fork_max": (
                    float(np.max([v for v in (_num(r["reuse_first_state_gap"])
                                              for r in sel) if v is not None]))
                    if sel else None),
                "i50": _stats([v for v in i50 if v is not None], n_boot=n_boot, rng=rng),
                "local_gap": _stats([v for v in local if v is not None], n_boot=n_boot, rng=rng),
                "reuse_final_drift_mean": float(np.mean(
                    [_num(r["reuse_final_drift"]) for r in sel
                     if _num(r["reuse_final_drift"]) is not None])) if sel else None,
                "forecast_final_drift_mean": float(np.mean(
                    [_num(r["forecast_final_drift"]) for r in sel
                     if _num(r["forecast_final_drift"]) is not None])) if sel else None,
            })
        # per-prompt paired mean, so a prompt counts once instead of six times
        by_prompt: Dict[int, List[float]] = {}
        for r in valid:
            v = _num(r["improvement_reuse_minus_forecast"])
            if v is not None:
                by_prompt.setdefault(int(_num(r["prompt_id"])), []).append(v)
        prompt_means = [float(np.mean(v)) for v in by_prompt.values()]
        out["by_payload"][label] = {
            "steps": steps_out,
            "n_prompts": len(by_prompt),
            "prompt_mean_i50": _stats(prompt_means, n_boot=n_boot,
                                      rng=np.random.default_rng(int(seed))),
        }
    return out


def build_g4(fork_rows: Dict[str, List[Dict[str, Any]]], *, tail_policy: str) -> Dict[str, Any]:
    predictors = ("first_improvement_action_sq", "first_improvement_action_state_cross")
    out: Dict[str, Any] = {"target": "improvement_reuse_minus_forecast",
                           "tail_policy": tail_policy, "by_payload": {}}
    for label, rows in fork_rows.items():
        valid = [r for r in rows
                 if str(r.get("valid")).lower() == "true"
                 and str(r.get("tail_policy")) == tail_policy]
        target = [_num(r["improvement_reuse_minus_forecast"]) for r in valid]
        block: Dict[str, Any] = {"n_rows": len(valid), "predictors": {}}
        for predictor in predictors:
            block["predictors"][predictor] = _corr(
                [_num(r[predictor]) for r in valid], target)
        # the same two correlations inside each fork step, so the pooled value
        # can be read against the step-level ones
        per_step: List[Dict[str, Any]] = []
        for step in sorted({int(_num(r["fork_step"])) for r in valid}):
            sel = [r for r in valid if int(_num(r["fork_step"])) == step]
            tgt = [_num(r["improvement_reuse_minus_forecast"]) for r in sel]
            per_step.append({
                "fork_step": step,
                "n": len(sel),
                **{p: _corr([_num(r[p]) for r in sel], tgt) for p in predictors},
            })
        block["per_step"] = per_step
        out["by_payload"][label] = block
    return out


def build_g1(exposure_rows: Dict[str, List[Dict[str, Any]]], *,
             g1_ids: Sequence[int]) -> Dict[str, Any]:
    wanted = set(int(i) for i in g1_ids)
    out: Dict[str, Any] = {"target": "final_latent_drift",
                           "n_prompts_requested": len(wanted),
                           "by_config": {}}
    for label, rows in exposure_rows.items():
        sel = [r for r in rows if int(r["prompt_id"]) in wanted]
        drift = [r["final_latent_drift"] for r in sel]
        block = {
            "n_prompts": len(sel),
            "n_missing": len(wanted - {int(r["prompt_id"]) for r in sel}),
            "state_exposure": _corr([r["total_state_gap_exposure"] for r in sel], drift),
            "action_exposure": _corr([r["total_action_exposure"] for r in sel], drift),
            "mean_cache_rate": (float(np.mean([r["cache_rate"] for r in sel
                                               if r["cache_rate"] is not None]))
                                if sel else None),
            "mean_final_latent_drift": (float(np.mean([d for d in drift if d is not None]))
                                        if sel else None),
        }
        out["by_config"][label] = block
    return out


# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    forks = _split_labelled(args.fork, "fork")
    exposures = _split_labelled(args.exposure, "exposure")
    if not forks:
        raise SystemExit("at least one --fork LABEL=DIR is required")

    fork_steps = _parse_ints(args.fork_steps)
    g3_ids = _parse_ints(args.g3_prompt_ids.read_text(encoding="utf-8"))
    g1_ids = _parse_ints(args.g1_prompt_ids.read_text(encoding="utf-8"))

    fork_rows = {label: _load_fork_rows(path) for label, path in forks}
    exposure_rows = {label: _load_exposure_rows(path) for label, path in exposures}

    stage_dir = args.stage_dir or args.out.parent
    staged: Dict[str, Any] = {"fork": {}, "exposure": {}}
    for label, rows in fork_rows.items():
        target = stage_dir / f"fork_rows_{label}.tsv.gz"
        n = _write_tsv_gz(target, ROW_COLUMNS,
                          sorted(rows, key=lambda r: (int(_num(r["prompt_id"])),
                                                      int(_num(r["fork_step"])))))
        staged["fork"][label] = {"file": target.name, "rows": n}
    for label, rows in exposure_rows.items():
        target = stage_dir / f"exposure_rows_{label}.tsv.gz"
        n = _write_tsv_gz(target, EXPOSURE_COLUMNS, rows)
        staged["exposure"][label] = {"file": target.name, "rows": n}

    # coverage: every requested (prompt, fork step) pair present and valid
    coverage: Dict[str, Any] = {}
    problems: List[str] = []
    for label, rows in fork_rows.items():
        seen = {(int(_num(r["prompt_id"])), int(_num(r["fork_step"])))
                for r in rows
                if str(r.get("valid")).lower() == "true"
                and str(r.get("tail_policy")) == args.tail_policy}
        expected = {(pid, step) for pid in g3_ids for step in fork_steps}
        missing = sorted(expected - seen)
        invalid = [r for r in rows if str(r.get("valid")).lower() != "true"]
        coverage[label] = {
            "expected_rows": len(expected),
            "valid_rows": len(seen),
            "missing_rows": len(missing),
            "missing_examples": [list(m) for m in missing[:10]],
            "invalid_rows": len(invalid),
            "prompts_with_all_steps": sum(
                1 for pid in g3_ids
                if all((pid, s) in seen for s in fork_steps)),
        }
        if missing:
            problems.append(f"{label}: {len(missing)} missing fork rows")
        if invalid:
            problems.append(f"{label}: {len(invalid)} invalid fork rows")
    for label, rows in exposure_rows.items():
        got = {int(r["prompt_id"]) for r in rows}
        missing = sorted(set(g1_ids) - got)
        coverage[f"exposure:{label}"] = {
            "expected_prompts": len(g1_ids),
            "present_prompts": len(got & set(g1_ids)),
            "missing_prompts": len(missing),
            "missing_examples": missing[:10],
        }
        if missing:
            problems.append(f"exposure {label}: {len(missing)} missing prompts")

    payload_labels = [label for label, _ in forks]
    report = {
        "schema": "suffix_reversal.report_numbers.v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "protocol": {
            "model": "FLUX.1-dev",
            "num_steps": int(args.num_steps),
            "resolution": "1024x1024",
            "dtype": "bf16",
            "guidance": 3.5,
            "seed_base": 42,
            "seed_rule": "seed_for(42, global prompt index)",
            "prompt_file": "resources/prompts/partiprompts_full_eval1632_seed42.txt",
            "prompt_pool_size": 1632,
            "split_role": "test",
            "g3_population": len(g3_ids),
            "g1_population": len(g1_ids),
            "bootstrap": int(args.bootstrap),
            "bootstrap_seed": int(args.seed),
        },
        "runs": {
            "fork": {label: _run_manifest(path) for label, path in forks},
            "exposure": {label: str(path.name) for label, path in exposures},
        },
        "staged_tables": staged,
        "coverage": coverage,
        "fork_design": build_g2(
            schedule_file=args.schedule_file, schedule_cell=args.schedule_cell,
            fork_steps=fork_steps, tail_policy=args.tail_policy,
            num_steps=int(args.num_steps), payload_labels=payload_labels,
            fork_rows=fork_rows, g3_ids=g3_ids),
        "suffix_reversal_by_step": build_g3(
            fork_rows, fork_steps=fork_steps, tail_policy=args.tail_policy,
            n_boot=int(args.bootstrap), seed=int(args.seed)),
        "local_vs_cross_correlations": build_g4(fork_rows, tail_policy=args.tail_policy),
        "closed_loop_exposure_correlations": build_g1(exposure_rows, g1_ids=g1_ids),
    }

    # the same step summary and correlation table as flat TSVs, for readers who
    # want to sort them without going through the JSON
    step_rows: List[Dict[str, Any]] = []
    for label, block in sorted(report["suffix_reversal_by_step"]["by_payload"].items()):
        for s in block["steps"]:
            band = s["i50"]["ci95_mean"] or [None, None]
            step_rows.append({
                "payload": label, "fork_step": s["fork_step"], "n": s["n"],
                "n_local_forecast_better": s["n_local_forecast_better"],
                "n_local_forecast_worse": s["n_local_forecast_worse"],
                "n_terminal_forecast_better": s["n_terminal_forecast_better"],
                "n_reversal": s["n_reversal"],
                "n_reversal_opposite": s["n_reversal_opposite"],
                "mean_i50": s["i50"]["mean"], "median_i50": s["i50"]["median"],
                "ci95_lo": band[0], "ci95_hi": band[1],
                "mean_local_gap": s["local_gap"]["mean"],
            })
    _write_tsv(stage_dir / "step_summary.tsv", list(step_rows[0]) if step_rows else [],
               step_rows)
    corr_rows: List[Dict[str, Any]] = []
    for label, block in sorted(report["local_vs_cross_correlations"]["by_payload"].items()):
        for predictor, c in sorted(block["predictors"].items()):
            corr_rows.append({"group": "fork", "unit": label, "predictor": predictor,
                              "target": "improvement_reuse_minus_forecast", **c})
    for label, block in sorted(report["closed_loop_exposure_correlations"]["by_config"].items()):
        for predictor in ("state_exposure", "action_exposure"):
            corr_rows.append({"group": "closed_loop", "unit": label,
                              "predictor": predictor, "target": "final_latent_drift",
                              **block[predictor]})
    _write_tsv(stage_dir / "correlations.tsv",
               list(corr_rows[0]) if corr_rows else [], corr_rows)
    staged["tables"] = {"step_summary": len(step_rows), "correlations": len(corr_rows)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                   sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "coverage": coverage,
                      "problems": problems}, indent=2, ensure_ascii=False))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
