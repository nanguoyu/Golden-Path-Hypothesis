"""H2 / H3 analysis: compare Score_1..Score_4 risk scores against R_k^oracle.

Per research plan §10 Exp 2:
    Score_1(k) = P_k                       (local prediction error alone)
    Score_2(k) = P_k * Q_k                 (+ solver step size)
    Score_3(k) = P_k * S_k^dir * Q_k       (+ directional velocity sensitivity)
    Score_4(k) = P_k * S_k^dir * Q_k * A_k (+ future amplification — Phase 3)

Hypothesis H3:  corr(Score_n, R_k^oracle) is non-decreasing in n. Concretely,
Score_4 should be a meaningfully better predictor of R_k^oracle than Score_1.

This script joins all available factor sources, computes whichever Score_n
levels have full data, and produces:
  - score_comparison_summary.csv / .json
  - per-method per-target bar chart of Spearman / Recall@top-p across n
  - per-method per-target scatter Score_n vs target

Skips Score levels that have missing prerequisites; reports which.

Oracle pairing (two modes):
  - Shared (legacy): --oracle <dir>. All 3 methods compared against the same
    oracle. If that oracle was generated with --cache_mode seacache, the
    hicache/taylorseer rows are indirect (method-target mismatch per
    docs/phase1_results.md §7).
  - Per-method (strict H3): --oracle_seacache, --oracle_hicache,
    --oracle_taylorseer each pointing at the corresponding oracle_runner
    --cache_mode run. Each method's Score is paired against ITS OWN oracle.
    Variant tags get a "_methodmatch" suffix in output filenames so
    per-method outputs do not overwrite shared-oracle outputs in the same
    out_dir.

Input shape conventions (all 1-indexed by k):
  - P_k:  per (prompt, k, method) from p_k_metrics.json OR s_k_metrics.json
          (the latter is a superset, prefer it if both present)
          Fields: p_seacache_abs/rel, p_hicache_abs/rel, p_taylorseer_abs/rel,
                  p_teacache_gate (NB: gate, not residual error)
  - S_k:  per (prompt, k, method) from s_k_metrics.json
          Fields: s_k_seacache_dir, s_k_hicache_dir, s_k_taylorseer_dir
  - Q_k:  per k from q_k_schedule.json (Q_k array length N)
  - A_k:  per (prompt, k[, method]) from a_k_metrics.json (Phase 3, optional)
          Field: a_k_dir or a_k_<method>_dir
  - R_k^oracle: per (prompt, k) from oracle_metrics.json
          Fields: latent_L2_abs, lpips, d_ir (sign-flipped for "larger=worse")

Method universe: {seacache, hicache, taylorseer}. TeaCache excluded from
Score_2/3/4 because p_teacache_gate is a different formulation (poly-rescaled
modulated-input rel_L1, not a residual prediction error vector); its true
residual P_k equals SeaCache's by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("seacache", "hicache", "taylorseer")

TARGETS = [
    ("latent_L2_abs", "‖z_N^oracle − z_N^baseline‖₂  (canonical R_k^oracle)"),
    ("lpips",         "LPIPS(oracle, baseline) — perceptual harm"),
    ("d_ir",          "ΔImageReward (oracle − baseline)"),
]


# ---------------------------------------------------------------------------
# Stats helpers (kept self-contained — same as correlate_p_k_oracle.py)
# ---------------------------------------------------------------------------
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float((xm * xm).sum() * (ym * ym).sum()))
    return float((xm * ym).sum() / denom) if denom > 0 else float("nan")


def _rankdata(a: np.ndarray) -> np.ndarray:
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(a.size, dtype=np.int64)
    inv[sorter] = np.arange(a.size)
    sorted_a = a[sorter]
    obs = np.r_[True, sorted_a[1:] != sorted_a[:-1]]
    dense_sorted = obs.cumsum()
    G = int(dense_sorted[-1])
    sizes = np.bincount(dense_sorted, minlength=G + 1)[1:]
    ends = sizes.cumsum()
    starts = ends - sizes + 1
    avg = (starts + ends) / 2.0
    dense_orig = dense_sorted[inv]
    return avg[dense_orig - 1]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    """R^2 of linear regression y = a*x + b. = pearson^2 for OLS."""
    r = _pearson(x, y)
    return r * r if math.isfinite(r) else float("nan")


def _recall_at_top_p(x: np.ndarray, y: np.ndarray, top_p: float) -> float:
    if x.size == 0:
        return float("nan")
    n_top = max(1, int(round(x.size * top_p / 100.0)))
    true_top = set(np.argsort(-y, kind="mergesort")[:n_top].tolist())
    pred_top = set(np.argsort(-x, kind="mergesort")[:n_top].tolist())
    return len(true_top & pred_top) / float(n_top)


# ---------------------------------------------------------------------------
# Loaders (per (prompt, k) dicts)
# ---------------------------------------------------------------------------
def _load_p_k_or_s_k(
    p_k_dir: Optional[Path], s_k_dir: Optional[Path]
) -> Dict[int, Dict[int, Dict[str, Optional[float]]]]:
    """Returns {prompt_idx: {k: {field: value}}} for P_k fields.

    Prefers s_k_metrics.json when both are available (superset). Required for
    Score_1+ for any method. Returns empty dict if neither source given.
    """
    out: Dict[int, Dict[int, Dict[str, Optional[float]]]] = {}
    primary = s_k_dir if s_k_dir is not None else p_k_dir
    if primary is None:
        return out
    fname = "s_k_metrics.json" if s_k_dir is not None else "p_k_metrics.json"
    for d in sorted(primary.iterdir()):
        if not (d.is_dir() and d.name.startswith("prompt_")):
            continue
        f = d / fname
        if not f.is_file():
            continue
        try:
            m = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not m.get("complete", False):
            continue
        idx = int(m["prompt_idx"])
        per_k: Dict[int, Dict[str, Optional[float]]] = {}
        for rec in m["per_k"]:
            k = int(rec["k"])
            per_k[k] = {
                "p_seacache_abs": rec.get("p_seacache_abs"),
                "p_hicache_abs": rec.get("p_hicache_abs"),
                "p_taylorseer_abs": rec.get("p_taylorseer_abs"),
                "p_teacache_gate": rec.get("p_teacache_gate"),
                "r_actual_norm": rec.get("r_actual_norm"),
            }
        out[idx] = per_k
    return out


def _load_s_k_only(s_k_dir: Optional[Path]) -> Dict[int, Dict[int, Dict[str, Optional[float]]]]:
    """Returns {prompt_idx: {k: {s_k_method_dir: value}}}. Empty if no s_k_dir."""
    out: Dict[int, Dict[int, Dict[str, Optional[float]]]] = {}
    if s_k_dir is None:
        return out
    for d in sorted(s_k_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith("prompt_")):
            continue
        f = d / "s_k_metrics.json"
        if not f.is_file():
            continue
        try:
            m = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not m.get("complete", False):
            continue
        idx = int(m["prompt_idx"])
        per_k: Dict[int, Dict[str, Optional[float]]] = {}
        for rec in m["per_k"]:
            k = int(rec["k"])
            per_k[k] = {
                f"s_k_{M}_dir": rec.get(f"s_k_{M}_dir") for M in METHODS
            }
        out[idx] = per_k
    return out


def _load_q_k(q_k_path: Optional[Path]) -> Optional[List[float]]:
    """Returns Q_k array of length N (or None if not provided)."""
    if q_k_path is None:
        return None
    payload = json.loads(q_k_path.read_text())
    return list(payload["Q_k"])


def _load_a_k(a_k_dir: Optional[Path]) -> Dict[int, Dict[int, Dict[str, Optional[float]]]]:
    """Per-(prompt, k[, method]) A_k. Returns {} if not provided.

    Supports two field naming conventions:
      - "a_k_dir": single direction-agnostic A_k per (prompt, k)
      - "a_k_<method>_dir": per-method A_k (paired with method's r_k direction)
    """
    out: Dict[int, Dict[int, Dict[str, Optional[float]]]] = {}
    if a_k_dir is None:
        return out
    for d in sorted(a_k_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith("prompt_")):
            continue
        f = d / "a_k_metrics.json"
        if not f.is_file():
            continue
        try:
            m = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not m.get("complete", False):
            continue
        idx = int(m["prompt_idx"])
        per_k: Dict[int, Dict[str, Optional[float]]] = {}
        for rec in m["per_k"]:
            k = int(rec["k"])
            slot: Dict[str, Optional[float]] = {"a_k_dir": rec.get("a_k_dir")}
            for M in METHODS:
                slot[f"a_k_{M}_dir"] = rec.get(f"a_k_{M}_dir")
            per_k[k] = slot
        out[idx] = per_k
    return out


def _load_oracle(oracle_dir: Path) -> Dict[int, Dict[int, Dict[str, Optional[float]]]]:
    """Returns {prompt_idx: {k: {target: value}}}."""
    out: Dict[int, Dict[int, Dict[str, Optional[float]]]] = {}
    for d in sorted(oracle_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith("prompt_")):
            continue
        f = d / "oracle_metrics.json"
        if not f.is_file():
            continue
        try:
            m = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        idx = int(m["prompt_idx"])
        ks = m["k_values"]
        per_k_list = m["per_k"]
        per_k: Dict[int, Dict[str, Optional[float]]] = {}
        for k, rec in zip(ks, per_k_list):
            per_k[int(k)] = {key: rec.get(key) for key, _ in TARGETS}
        out[idx] = per_k
    return out


# ---------------------------------------------------------------------------
# Calibration tables (LOO / mean) for deployable Score_4
# ---------------------------------------------------------------------------
def _build_calib_table(
    data_dict: Dict[int, Dict[int, Dict[str, Optional[float]]]],
    field_name_fn,
) -> Tuple[Dict[Tuple[int, str], Dict[int, float]],
           Dict[Tuple[int, str], Tuple[float, int]]]:
    """Index per-prompt per-(k, method) values into a calibration table.

    Returns (table, stats):
      table[(k, M)] = {prompt_id: value}
      stats[(k, M)] = (sum, count)
    """
    arr: Dict[Tuple[int, str], Dict[int, float]] = {}
    for pid, per_k in (data_dict or {}).items():
        for k, slot in per_k.items():
            for M in METHODS:
                v = field_name_fn(slot, M)
                if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
                    arr.setdefault((k, M), {})[int(pid)] = float(v)
    stats = {key: (sum(d.values()), len(d)) for key, d in arr.items()}
    return arr, stats


def _calib_lookup(
    arr: Dict[Tuple[int, str], Dict[int, float]],
    stats: Dict[Tuple[int, str], Tuple[float, int]],
    pid: int, k: int, M: str, mode: str,
) -> Optional[float]:
    """Return calibrated S_k / A_k value for (pid, k, M) under mode.

    mode = "loo":  leave-one-out mean across other prompts (purest;
                   for n=100 the LOO bias relative to full mean is <= 1%)
    mode = "mean": full mean across all prompts (slightly tautological for
                   the contributing prompt; fine when n is large)
    """
    d = arr.get((k, M))
    if d is None:
        return None
    total, n = stats[(k, M)]
    if mode == "mean":
        return total / n if n > 0 else None
    if mode == "loo":
        v_p = d.get(pid)
        if v_p is None:
            return total / n if n > 0 else None
        if n <= 1:
            return None
        return (total - v_p) / (n - 1)
    raise ValueError(f"unknown calibration mode: {mode}")


def _sk_field(slot: Dict[str, Optional[float]], M: str) -> Optional[float]:
    return slot.get(f"s_k_{M}_dir")


def _ak_field(slot: Dict[str, Optional[float]], M: str) -> Optional[float]:
    v = slot.get(f"a_k_{M}_dir")
    if v is None or not (isinstance(v, (int, float)) and math.isfinite(v)):
        v = slot.get("a_k_dir")
    return v if isinstance(v, (int, float)) else None


# ---------------------------------------------------------------------------
# Score assembly
# ---------------------------------------------------------------------------
def _build_scores(
    p_k_data: Dict[int, Dict[int, Dict[str, Optional[float]]]],
    s_k_data: Dict[int, Dict[int, Dict[str, Optional[float]]]],
    q_k: Optional[List[float]],
    a_k_data: Dict[int, Dict[int, Dict[str, Optional[float]]]],
    oracle_data_per_method: Dict[str, Dict[int, Dict[int, Dict[str, Optional[float]]]]],
    calibration_mode: str = "none",
) -> Tuple[Dict[Tuple[str, int, str], List[float]],
           Dict[Tuple[str, int, str], List[float]],
           Dict[Tuple[str, int, str], List[Tuple[int, float, float]]]]:
    """For each (method, score_level, target) build pooled (x, y) arrays.

    oracle_data_per_method: maps method name ("seacache" / "hicache" /
    "taylorseer") to that method's oracle dict {prompt_idx: {k: {target: v}}}.
    Each method's Score_n is paired against ITS OWN oracle. To run a single
    shared oracle for all methods (legacy behavior + method-target mismatch
    caveat per docs/phase1_results.md §7), pass the same dict under all three
    method keys.

    calibration_mode:
      "none": per-prompt S_k(p, k, M) and A_k(p, k, M) (DIAGNOSTIC variant —
              Score_4 collapses by chain identity to ||z_N^injection - z_N||,
              so its correlation with R_k^oracle is structural ~1.0 modulo
              direction; useful as sanity check, NOT as predictive evidence
              for H3).
      "loo":  leave-one-out per-k mean (S_k_calib(k, M), A_k_calib(k, M)).
              DEPLOYABLE variant — breaks the tautological chain because
              S_k_calib and A_k_calib are not derived from prompt p's own
              cache direction. Score_4 is no longer algebraically equal to
              the injection final error.
      "mean": full mean (simpler; small contamination for the contributing
              prompt but irrelevant at n>=100).

    Returns:
      pooled_x: {(method, level, target): list of Score_n values}
      pooled_y: {(method, level, target): list of corresponding target values}
      per_prompt_pairs: {(method, level, target): list of (prompt_idx, x, y)}
                       — used for per-prompt rank correlation.

    Levels included depend on which inputs are present:
      level 1: always (needs P_k)
      level 2: if q_k provided
      level 3: if q_k AND s_k provided
      level 4: if q_k AND s_k AND a_k provided
    """
    levels = [1]
    if q_k is not None:
        levels.append(2)
    if q_k is not None and s_k_data:
        levels.append(3)
    if q_k is not None and s_k_data and a_k_data:
        levels.append(4)

    if calibration_mode not in ("none", "loo", "mean"):
        raise ValueError(f"unknown calibration mode: {calibration_mode}")

    # Pre-build calibration tables (only needed if calibration_mode != "none")
    if calibration_mode != "none":
        s_arr, s_stats = _build_calib_table(s_k_data, _sk_field)
        a_arr, a_stats = _build_calib_table(a_k_data, _ak_field)
    else:
        s_arr = s_stats = a_arr = a_stats = None  # unused

    pooled_x: Dict[Tuple[str, int, str], List[float]] = {}
    pooled_y: Dict[Tuple[str, int, str], List[float]] = {}
    per_prompt_pairs: Dict[Tuple[str, int, str], List[Tuple[int, float, float]]] = {}
    for M in METHODS:
        for L in levels:
            for T, _ in TARGETS:
                key = (M, L, T)
                pooled_x[key] = []
                pooled_y[key] = []
                per_prompt_pairs[key] = []

    # Iterate per method first; each method uses its own oracle.
    for M in METHODS:
        oracle_data = oracle_data_per_method.get(M)
        if oracle_data is None:
            continue  # this method's oracle wasn't provided — skip it entirely
        common_prompts = sorted(set(p_k_data.keys()) & set(oracle_data.keys()))
        for pid in common_prompts:
            p_per_k = p_k_data[pid]
            o_per_k = oracle_data[pid]
            s_per_k = s_k_data.get(pid, {}) if s_k_data else {}
            a_per_k = a_k_data.get(pid, {}) if a_k_data else {}
            common_k = sorted(set(p_per_k.keys()) & set(o_per_k.keys()))
            for L in levels:
                for T, _ in TARGETS:
                    for k in common_k:
                        p_val = p_per_k[k].get(f"p_{M}_abs")
                        if p_val is None or not math.isfinite(p_val):
                            continue

                        # Build Score_n step by step
                        x = float(p_val)  # Score_1
                        if L >= 2:
                            if q_k is None or k >= len(q_k):
                                continue
                            x *= float(q_k[k])  # Score_2
                        if L >= 3:
                            if calibration_mode == "none":
                                s_val = (s_per_k.get(k, {}).get(f"s_k_{M}_dir")
                                         if s_per_k else None)
                            else:
                                s_val = _calib_lookup(s_arr, s_stats,
                                                      pid, k, M, calibration_mode)
                            if s_val is None or not math.isfinite(s_val):
                                continue
                            x *= float(s_val)  # Score_3
                        if L >= 4:
                            if calibration_mode == "none":
                                a_slot = a_per_k.get(k, {}) if a_per_k else {}
                                a_val = a_slot.get(f"a_k_{M}_dir")
                                if a_val is None or not math.isfinite(a_val):
                                    a_val = a_slot.get("a_k_dir")
                            else:
                                a_val = _calib_lookup(a_arr, a_stats,
                                                      pid, k, M, calibration_mode)
                            if a_val is None or not math.isfinite(a_val):
                                continue
                            x *= float(a_val)  # Score_4

                        y_raw = o_per_k[k].get(T)
                        if y_raw is None or not math.isfinite(y_raw):
                            continue
                        # Sign-flip d_ir so "larger = worse" semantics matches latent_L2/lpips
                        y = -float(y_raw) if T == "d_ir" else float(y_raw)

                        pooled_x[(M, L, T)].append(x)
                        pooled_y[(M, L, T)].append(y)
                        per_prompt_pairs[(M, L, T)].append((pid, x, y))

    return pooled_x, pooled_y, per_prompt_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score_1..4 vs R_k^oracle comparison (H2 baseline + H3 ladder)."
    )
    p.add_argument("--p_k", type=Path, default=None,
                   help="p_k_probe output dir (contains prompt_XXXXX/p_k_metrics.json). "
                        "Required if --s_k not provided; ignored if --s_k provided "
                        "(s_k_metrics.json is a superset).")
    p.add_argument("--s_k", type=Path, default=None,
                   help="s_k_probe output dir. Provides both P_k and S_k. "
                        "Preferred over --p_k.")
    p.add_argument("--q_k", type=Path, default=None,
                   help="Q_k schedule JSON from analysis/extract_q_k.py. "
                        "Required for Score_2/3/4.")
    p.add_argument("--a_k", type=Path, default=None,
                   help="A_k probe output dir (Phase 3, optional). "
                        "Required for Score_4.")
    p.add_argument("--oracle", type=Path, default=None,
                   help="Single oracle output dir. When provided, all 3 methods "
                        "are compared against the same oracle (method-target "
                        "mismatch for non-seacache methods if it's a seacache "
                        "oracle — see docs/phase1_results.md §7). Mutually "
                        "exclusive with the --oracle_<method> flags.")
    p.add_argument("--oracle_seacache", type=Path, default=None,
                   help="SeaCache-specific oracle dir (oracle_runner.py "
                        "--cache_mode seacache). Paired only with seacache Score.")
    p.add_argument("--oracle_hicache", type=Path, default=None,
                   help="HiCache-specific oracle dir (--cache_mode hicache). "
                        "Paired only with hicache Score.")
    p.add_argument("--oracle_taylorseer", type=Path, default=None,
                   help="TaylorSeer-specific oracle dir (--cache_mode taylorseer). "
                        "Paired only with taylorseer Score.")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Output dir for summary + plots. Default: --s_k or --p_k dir.")
    p.add_argument("--top_p", type=float, default=10.0,
                   help="Top-p%% for Recall@top-p%% (default 10).")
    p.add_argument("--max_scatter", type=int, default=2000,
                   help="Subsample for scatter plots (default 2000).")
    p.add_argument("--calibration_mode", choices=["none", "loo", "mean", "both"],
                   default="both",
                   help="How to derive S_k / A_k for Score_3/4. "
                        "'none' = per-prompt (DIAGNOSTIC, tautological with R_oracle; "
                        "the chain P*S*Q*A then equals ||z_N^injection - z_N|| modulo sign). "
                        "'loo' = leave-one-out per-k mean across other prompts "
                        "(DEPLOYABLE, breaks tautology). "
                        "'mean' = full per-k mean (cheap, ~1pp LOO bias at n=100). "
                        "'both' = run none + loo, write two summaries (default).")
    return p.parse_args()


def _summarize_and_write(
    *, pooled_x, pooled_y, per_prompt_pairs,
    levels_built: List[int], out_dir: Path, tag: str,
    n_common_prompts: int, top_p: float, max_scatter: int,
    sources: Dict[str, Optional[str]], rng,
) -> Dict[str, Any]:
    """Compute summary + write JSON/CSV/plots for one variant (diagnostic or
    calibrated). `tag` is a short suffix appended to all output filenames
    (e.g. 'diagnostic', 'deployable_loo')."""
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for M in METHODS:
        summary[M] = {}
        for T, _ in TARGETS:
            summary[M][T] = {}
            for L in levels_built:
                xs = np.array(pooled_x[(M, L, T)], dtype=np.float64)
                ys = np.array(pooled_y[(M, L, T)], dtype=np.float64)
                n = int(xs.size)

                per_prompt_rho: List[float] = []
                groups: Dict[int, List[Tuple[float, float]]] = {}
                for (pid, x, y) in per_prompt_pairs[(M, L, T)]:
                    groups.setdefault(pid, []).append((x, y))
                for pid, pairs in groups.items():
                    if len(pairs) < 5:
                        continue
                    arr = np.array(pairs, dtype=np.float64)
                    per_prompt_rho.append(_spearman(arr[:, 0], arr[:, 1]))
                per_prompt_rho_arr = np.array(per_prompt_rho, dtype=np.float64)

                summary[M][T][f"Score_{L}"] = {
                    "n_samples": n,
                    "pearson": _pearson(xs, ys) if n >= 2 else float("nan"),
                    "spearman": _spearman(xs, ys) if n >= 2 else float("nan"),
                    "r2": _r2(xs, ys) if n >= 2 else float("nan"),
                    f"recall_at_top_{int(top_p)}pct": (
                        _recall_at_top_p(xs, ys, top_p) if n > 0 else float("nan")
                    ),
                    "per_prompt_spearman_mean": (
                        float(np.nanmean(per_prompt_rho_arr))
                        if per_prompt_rho_arr.size else float("nan")
                    ),
                    "per_prompt_spearman_std": (
                        float(np.nanstd(per_prompt_rho_arr))
                        if per_prompt_rho_arr.size else float("nan")
                    ),
                    "n_prompts_for_per_prompt": int(per_prompt_rho_arr.size),
                }

    json_path = out_dir / f"score_comparison_{tag}.json"
    json_path.write_text(json.dumps({
        "variant": tag,
        "n_common_prompts": n_common_prompts,
        "top_p_percent": float(top_p),
        "targets": [t for t, _ in TARGETS],
        "methods": list(METHODS),
        "levels": levels_built,
        "sources": sources,
        "sign_flips": {"d_ir": "flipped (more negative = worse)"},
        "summary": summary,
    }, indent=2))
    print(f"[INFO] wrote {json_path}")

    csv_path = out_dir / f"score_comparison_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "score_level", "target",
            "n_samples", "pearson", "spearman", "r2",
            f"recall_at_top_{int(top_p)}pct",
            "per_prompt_spearman_mean", "per_prompt_spearman_std",
            "n_prompts",
        ])
        for M in METHODS:
            for T, _ in TARGETS:
                for L in levels_built:
                    s = summary[M][T][f"Score_{L}"]
                    w.writerow([
                        M, f"Score_{L}", T,
                        s["n_samples"],
                        f"{s['pearson']:.4f}",
                        f"{s['spearman']:.4f}",
                        f"{s['r2']:.4f}",
                        f"{s[f'recall_at_top_{int(top_p)}pct']:.4f}",
                        f"{s['per_prompt_spearman_mean']:.4f}",
                        f"{s['per_prompt_spearman_std']:.4f}",
                        s["n_prompts_for_per_prompt"],
                    ])
    print(f"[INFO] wrote {csv_path}")

    for T, _tlabel in TARGETS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        x_pos = np.arange(len(levels_built))
        width = 0.25
        for mi, M in enumerate(METHODS):
            spear = [summary[M][T][f"Score_{L}"]["spearman"] for L in levels_built]
            rec = [summary[M][T][f"Score_{L}"][f"recall_at_top_{int(top_p)}pct"]
                   for L in levels_built]
            axes[0].bar(x_pos + mi * width, spear, width, label=M)
            axes[1].bar(x_pos + mi * width, rec, width, label=M)
        for ax, ylabel in zip(axes, ("Spearman ρ (pooled)",
                                     f"Recall@top-{int(top_p)}% (pooled)")):
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels([f"Score_{L}" for L in levels_built])
            ax.set_ylabel(ylabel)
            ax.set_title(f"vs target = {T}" + (" (sign-flipped)" if T == "d_ir" else ""))
            ax.grid(alpha=0.3, axis="y")
            ax.legend(loc="best", fontsize=9)
            ax.axhline(0, color="black", linewidth=0.5)
        fig.suptitle(f"Score_n vs R_k^oracle({T})  [{tag}]  n_prompts={n_common_prompts}")
        fig.tight_layout()
        png_path = out_dir / f"score_ladder_{T}_{tag}.png"
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"[INFO] wrote {png_path}")

    for M in METHODS:
        for L in levels_built:
            for T, tlabel in TARGETS:
                xs = np.array(pooled_x[(M, L, T)], dtype=np.float64)
                ys = np.array(pooled_y[(M, L, T)], dtype=np.float64)
                if xs.size == 0:
                    continue
                if xs.size > max_scatter:
                    idx = rng.choice(xs.size, size=max_scatter, replace=False)
                    xp, yp = xs[idx], ys[idx]
                else:
                    xp, yp = xs, ys
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                ax.scatter(xp, yp, s=4, alpha=0.3, color="tab:blue")
                s = summary[M][T][f"Score_{L}"]
                ax.set_xlabel(f"Score_{L}({M})  [{tag}]")
                ax.set_ylabel(T + (" (sign-flipped)" if T == "d_ir" else ""))
                ax.set_title(
                    f"{M}  Score_{L}  vs  {T}  [{tag}]\n"
                    f"pearson={s['pearson']:.3f}, spearman={s['spearman']:.3f}, "
                    f"recall@top{int(top_p)}%={s[f'recall_at_top_{int(top_p)}pct']:.3f}, "
                    f"n={s['n_samples']}"
                )
                ax.grid(alpha=0.3)
                fig.tight_layout()
                png_path = out_dir / f"scatter_{M}_Score_{L}_vs_{T}_{tag}.png"
                fig.savefig(png_path, dpi=100)
                plt.close(fig)

    return summary


def _print_console_table(summary, levels_built, top_p, tag):
    print(f"\n=== SCORE LADDER (Spearman ρ, pooled) — {tag} ===")
    header = ["method", "target"] + [f"Score_{L}" for L in levels_built]
    print("  " + "  ".join(f"{h:>14s}" for h in header))
    for M in METHODS:
        for T, _ in TARGETS:
            row = [M, T] + [f"{summary[M][T][f'Score_{L}']['spearman']:+.3f}"
                            for L in levels_built]
            print("  " + "  ".join(f"{c:>14s}" for c in row))
    print(f"\n=== SCORE LADDER (Recall@top-{int(top_p)}%) — {tag} ===")
    print("  " + "  ".join(f"{h:>14s}" for h in header))
    for M in METHODS:
        for T, _ in TARGETS:
            row = [M, T] + [
                f"{summary[M][T][f'Score_{L}'][f'recall_at_top_{int(top_p)}pct']:.3f}"
                for L in levels_built
            ]
            print("  " + "  ".join(f"{c:>14s}" for c in row))


def main() -> int:
    args = parse_args()
    if args.p_k is None and args.s_k is None:
        raise SystemExit("--p_k or --s_k required")
    out_dir = (args.out_dir if args.out_dir is not None
               else (args.s_k if args.s_k is not None else args.p_k))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading p_k/s_k from {args.s_k or args.p_k}")
    p_k_data = _load_p_k_or_s_k(args.p_k, args.s_k)
    print(f"  loaded {len(p_k_data)} prompt(s)")

    print(f"[INFO] loading s_k from {args.s_k}" if args.s_k else "[INFO] no --s_k, Score_3/4 skipped")
    s_k_data = _load_s_k_only(args.s_k)
    print(f"  loaded {len(s_k_data)} s_k prompt(s)")

    print(f"[INFO] loading q_k from {args.q_k}" if args.q_k else "[INFO] no --q_k, Score_2/3/4 skipped")
    q_k = _load_q_k(args.q_k)
    if q_k is not None:
        print(f"  Q_k length {len(q_k)}, range [{min(q_k):.4f}, {max(q_k):.4f}]")

    print(f"[INFO] loading a_k from {args.a_k}" if args.a_k else "[INFO] no --a_k, Score_4 skipped")
    a_k_data = _load_a_k(args.a_k)
    print(f"  loaded {len(a_k_data)} a_k prompt(s)")

    # Resolve oracle mode: either single shared oracle (--oracle) OR per-method
    # oracles (--oracle_{seacache,hicache,taylorseer}). Mutually exclusive.
    per_method_paths: Dict[str, Optional[Path]] = {
        "seacache": args.oracle_seacache,
        "hicache": args.oracle_hicache,
        "taylorseer": args.oracle_taylorseer,
    }
    n_per_method = sum(1 for v in per_method_paths.values() if v is not None)
    if args.oracle is not None and n_per_method > 0:
        raise SystemExit("--oracle is mutually exclusive with --oracle_<method> flags")
    if args.oracle is None and n_per_method == 0:
        raise SystemExit("require --oracle OR at least one --oracle_<method> flag")

    is_per_method = n_per_method > 0
    oracle_data_per_method: Dict[str, Dict[int, Dict[int, Dict[str, Optional[float]]]]] = {}
    sources_oracle: Any
    if args.oracle is not None:
        print(f"[INFO] loading single shared oracle from {args.oracle}")
        shared = _load_oracle(args.oracle)
        print(f"  loaded {len(shared)} oracle prompt(s)")
        for M in METHODS:
            oracle_data_per_method[M] = shared
        sources_oracle = str(args.oracle)
        n_oracle_prompts_per_method = {M: len(shared) for M in METHODS}
    else:
        sources_oracle = {}
        n_oracle_prompts_per_method = {}
        for M in METHODS:
            p = per_method_paths[M]
            if p is None:
                print(f"[INFO] no --oracle_{M}; method '{M}' will be skipped")
                sources_oracle[M] = None
                continue
            print(f"[INFO] loading {M} oracle from {p}")
            data = _load_oracle(p)
            print(f"  loaded {len(data)} {M} oracle prompt(s)")
            oracle_data_per_method[M] = data
            sources_oracle[M] = str(p)
            n_oracle_prompts_per_method[M] = len(data)

    # Common prompts (intersection across P_k + every provided oracle's prompts)
    all_oracle_prompt_sets = [set(d.keys()) for d in oracle_data_per_method.values()
                              if d is not None]
    if not all_oracle_prompt_sets:
        raise SystemExit("no oracle data loaded")
    common_prompts = sorted(set(p_k_data.keys()).intersection(*all_oracle_prompt_sets))
    print(f"[INFO] {len(common_prompts)} prompts in P_k ∩ (all provided oracles)")
    if not common_prompts:
        raise SystemExit("no overlap between P_k and oracle prompts")

    sources = {
        "p_k": str(args.p_k) if args.p_k else None,
        "s_k": str(args.s_k) if args.s_k else None,
        "q_k": str(args.q_k) if args.q_k else None,
        "a_k": str(args.a_k) if args.a_k else None,
        "oracle": sources_oracle,
        "oracle_mode": "per_method" if is_per_method else "shared",
    }
    rng = np.random.default_rng(0)

    # Decide which variants to run. "both" runs none (diagnostic) + loo (deployable).
    if args.calibration_mode == "both":
        variants = [("none", "diagnostic"), ("loo", "deployable_loo")]
    elif args.calibration_mode == "none":
        variants = [("none", "diagnostic")]
    elif args.calibration_mode == "loo":
        variants = [("loo", "deployable_loo")]
    elif args.calibration_mode == "mean":
        variants = [("mean", "deployable_mean")]
    else:
        raise SystemExit(f"unknown calibration_mode: {args.calibration_mode}")

    # When per-method oracles are in use, suffix the variant tag so per-method
    # outputs do not overwrite a previous shared-oracle run in the same dir.
    if is_per_method:
        variants = [(c, f"{t}_methodmatch") for c, t in variants]

    for cmode, tag in variants:
        print(f"\n[INFO] === computing variant '{tag}' (calibration_mode={cmode}) ===")
        pooled_x, pooled_y, per_prompt_pairs = _build_scores(
            p_k_data, s_k_data, q_k, a_k_data, oracle_data_per_method,
            calibration_mode=cmode,
        )
        levels_built = sorted({key[1] for key in pooled_x.keys()})
        print(f"[INFO] Score levels computed: {levels_built}")
        summary = _summarize_and_write(
            pooled_x=pooled_x, pooled_y=pooled_y,
            per_prompt_pairs=per_prompt_pairs,
            levels_built=levels_built, out_dir=out_dir, tag=tag,
            n_common_prompts=len(common_prompts),
            top_p=args.top_p, max_scatter=args.max_scatter,
            sources=sources, rng=rng,
        )
        _print_console_table(summary, levels_built, args.top_p, tag)

    print(f"\n[INFO] done. n_common_prompts={len(common_prompts)}, "
          f"outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
