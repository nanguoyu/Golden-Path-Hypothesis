#!/usr/bin/env python3
"""Q1-Q4 of the image cache-bend layer: does the early bend carry the quality?

`docs/image_cached_trajectory_plan_zh.md` sections 1 and 7.2. The statistical
protocol is the plan's, written before any number was read and not changed
afterwards:

Q2 (the main test)
    1. row level -- inside each (model, K) partition, the row's median early
       offset against the row's mean PSNR over the full 1632-prompt seed-42
       slice of ``resources/spx/perprompt_spx_<model>.tsv.gz``, as Spearman;
       ``k0`` and ``D[50]`` are put in the same table as the two comparison
       univariates. 13-33 rows per partition, so the pooled regression is run
       as well, on within-partition z-scores, with its p marked non-independent.
    2. mediation -- quality on ``k0``, then ``k0`` plus the early offset, and
       the reverse direction in the same table, neither one chosen. Run on two
       populations: all rows, and the rows with ``k0 < 10`` (the primary one --
       a row with ``k0 >= 10`` has ``D[10]`` exactly 0 by definition, and those
       degenerate rows alone flip the joint-regression ``k0`` coefficient); and
       with two early readings, ``D[10]`` / early area and the event-aligned
       post-jump slope, which is defined for every row. The pre-registered
       absorbed-share numbers are still emitted; a coefficient that crosses
       zero makes the share exceed 1, which is why the coefficients, their
       signs and the incremental R^2 are the readings to quote (the protocol
       change is recorded in the plan's section 9).
    3. within cell -- inside one cell, the kept prompts' own ``D[10]`` against
       their own PSNR, as a per-cell Spearman whose distribution is reported.
       A row-level correlation can come from between-row confounding; this one
       cannot. Beside it, the same cells' ``rho(D[50], PSNR)`` and the partial
       rank correlation of ``D[10]`` given ``D[50]``: the within-cell evidence
       rules out between-row confounding, and those two numbers say how much of
       it is early-specific rather than the prompt's overall bend.
    4. every statistic is computed twice, on all kept prompts (49 of the 50
       drawn) and on the kept held-out ones (32), and printed side by side.

Q1  the D[n] profile's shape and whether the D[50] order matches the PSNR order.
Q3  the three direction shares by payload family.
Q4  the first jump: Delta D[k0] and the slope after it against the sigma
    geometry at k0, over every row including the random and Hamming ladders.

Reads only staged artefacts, writes one JSON. CPU, seconds.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODELS = ("flux", "qwen")
KS = (29, 37, 41)
EARLY_N = 10
PAYLOAD_FAMILY = {
    "reuse": "zero-order reuse",
    "taylor_o1": "first-order extrapolation",
    "hermite_o2": "second-order extrapolation",
    "di_two_anchor": "two anchor",
    "mean_avg_vel": "interval average velocity",
}
PREDICTORS = ("D10_over_chord_ref", "early_auc_over_chord_ref", "k0",
              "D50_over_chord_ref", "slope_k0_plus_5")
CAV_POOLED = (
    "the pooled row-level statistic is taken over within-partition z-scores of "
    "13-33 rows each; the rows of one partition share a schedule family and a "
    "budget, so the pooled p is not an independent-sample p and is printed as a "
    "descriptive number"
)
CAV_COLLINEAR = (
    "a row whose first cache step is at or after state 10 has an early offset "
    "of exactly 0, which makes the early variable collinear with k0 over those "
    "rows; the k0 <= 10 subset is reported beside the full one for that reason"
)
CAV_ASYMMETRIC = (
    "the bend readings are medians over 49 prompts and the quality readings are "
    "means over all 1632; the two sample sizes differ by design, the quality "
    "table having been measured before this layer existed"
)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def read_bend(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            values = line.rstrip("\n").split("\t")
            row = dict(zip(header, values))
            for key in ("k", "prompt_idx", "k0", "n_cached",
                        "structurally_zero_early"):
                row[key] = int(row[key]) if row[key] != "" else None
            for key in header:
                if key in ("model", "schedule", "payload", "role") or key in (
                        "k", "prompt_idx", "k0", "n_cached",
                        "structurally_zero_early"):
                    continue
                row[key] = float(row[key]) if row[key] != "" else None
            rows.append(row)
    return rows


def read_quality(model: str, *, metric: str) -> dict[tuple[str, str, int], float]:
    """Row -> mean metric over the full seed-42 slice (1632 prompts)."""
    path = _ROOT / "resources" / "spx" / f"perprompt_spx_{model}.tsv.gz"
    total: dict[tuple[str, str, int], list[float]] = {}
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        col = header.index(metric)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if parts[3] != "42":
                continue
            value = float(parts[col])
            if math.isfinite(value):
                total.setdefault((parts[0], parts[1], int(parts[2])), []).append(value)
    return {key: float(np.mean(values)) for key, values in total.items()}


def read_quality_perprompt(model: str, *, metric: str
                           ) -> dict[tuple[str, str, int, int], float]:
    path = _ROOT / "resources" / "spx" / f"perprompt_spx_{model}.tsv.gz"
    out: dict[tuple[str, str, int, int], float] = {}
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        col = header.index(metric)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if parts[3] != "42":
                continue
            out[(parts[0], parts[1], int(parts[2]), int(parts[4]))] = float(parts[col])
    return out


# ---------------------------------------------------------------------------
# small statistics
# ---------------------------------------------------------------------------


def spearman(x: Iterable[Any], y: Iterable[Any]) -> dict[str, Any]:
    xs, ys = [], []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        if math.isfinite(a) and math.isfinite(b):
            xs.append(a)
            ys.append(b)
    if len(xs) < 4 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"n": len(xs), "rho": None, "p": None}
    rho, p = stats.spearmanr(xs, ys)
    return {"n": len(xs), "rho": float(rho), "p": float(p)}


def _z(values: np.ndarray) -> np.ndarray:
    sd = float(np.std(values))
    return (values - float(np.mean(values))) / sd if sd > 0 else values * 0.0


def ols(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    """Standardised OLS with an intercept; coefficients, R^2, and t on each."""
    n, p = X.shape
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = n - p - 1
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else None
    ts: list[float | None] = []
    if dof > 0:
        sigma2 = float((resid ** 2).sum()) / dof
        try:
            cov = sigma2 * np.linalg.inv(A.T @ A)
            ts = [float(beta[i] / math.sqrt(cov[i, i])) if cov[i, i] > 0 else None
                  for i in range(1, p + 1)]
        except np.linalg.LinAlgError:
            ts = [None] * p
    else:
        ts = [None] * p
    return {"n": n, "coef": [float(v) for v in beta[1:]], "t": ts, "r2": r2,
            "dof": dof}


# ---------------------------------------------------------------------------
# row-level aggregation
# ---------------------------------------------------------------------------


def row_table(bend: list[dict[str, Any]], *, roles: str) -> dict[tuple, dict[str, Any]]:
    """(model, k, schedule, payload) -> the row's median bend readings."""
    keep = [r for r in bend if roles == "all" or r["role"] != "discovery"]
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in keep:
        groups.setdefault(
            (row["model"], row["k"], row["schedule"], row["payload"]), []).append(row)
    out: dict[tuple, dict[str, Any]] = {}
    for key, rows in groups.items():
        entry: dict[str, Any] = {"n_pairs": len(rows)}
        for field in ("D10_over_chord_ref", "early_auc_over_chord_ref",
                      "D50_over_chord_ref", "delta_D_k0", "slope_k0_plus_5",
                      "chord_ratio", "straightness_diff", "max_dev_ratio_diff",
                      "pca_evr_top2_diff", "chord_angle_deg", "plane_angle1_deg",
                      "share_chord_50", "share_in_plane_50", "share_off_plane_50",
                      "share_off_plane_k0p1"):
            values = np.asarray([r[field] for r in rows if r[field] is not None],
                                dtype=np.float64)
            values = values[np.isfinite(values)]
            entry[field] = float(np.median(values)) if values.size else None
        k0s = {r["k0"] for r in rows}
        entry["k0"] = None if len(k0s) != 1 else k0s.pop()
        entry["n_cached"] = rows[0]["n_cached"]
        out[key] = entry
    return out


def q2_row_level(rows: dict[tuple, dict[str, Any]],
                 quality: dict[str, dict[tuple[str, str, int], float]],
                 metric: str) -> dict[str, Any]:
    partitions: dict[str, dict[str, Any]] = {}
    pooled: dict[str, list[float]] = {p: [] for p in PREDICTORS}
    pooled["quality"] = []
    pooled_small: dict[str, list[float]] = {p: [] for p in PREDICTORS}
    pooled_small["quality"] = []
    for model in MODELS:
        for k in KS:
            keys = sorted(kk for kk in rows if kk[0] == model and kk[1] == k)
            y, cols = [], {p: [] for p in PREDICTORS}
            small_mask = []
            for key in keys:
                entry = rows[key]
                q = quality[model].get((key[2], key[3], key[1]))
                if q is None:
                    continue
                y.append(q)
                for p in PREDICTORS:
                    cols[p].append(entry[p])
                small_mask.append(entry["k0"] is not None and entry["k0"] < EARLY_N)
            if len(y) < 5:
                partitions[f"{model}_k{k}"] = {"n_rows": len(y)}
                continue
            block: dict[str, Any] = {"n_rows": len(y), "metric": metric}
            for p in PREDICTORS:
                block[p] = spearman(cols[p], y)
            small = [i for i, m in enumerate(small_mask) if m]
            block["k0_lt_early_subset"] = {
                "n_rows": len(small),
                **{p: spearman([cols[p][i] for i in small], [y[i] for i in small])
                   for p in PREDICTORS},
            }
            partitions[f"{model}_k{k}"] = block

            # pooling is over WITHIN-partition z-scores, so a row's pooled
            # value depends only on its own partition; the k0 < 10 pool is the
            # same z-scores restricted to those rows, never re-standardised
            zed: dict[str, np.ndarray] = {
                "quality": _z(np.asarray(y, dtype=np.float64))}
            for p in PREDICTORS:
                vals = np.asarray([np.nan if v is None else float(v) for v in cols[p]],
                                  dtype=np.float64)
                fill = float(np.nanmedian(vals)) if np.isfinite(vals).any() else 0.0
                zed[p] = _z(np.nan_to_num(vals, nan=fill))
            for name, arr in zed.items():
                pooled[name].extend(arr.tolist())
                pooled_small[name].extend(float(arr[i]) for i in small)

    pooled_block = {"n_rows": len(pooled["quality"]), "caveat": CAV_POOLED}
    for p in PREDICTORS:
        pooled_block[p] = spearman(pooled[p], pooled["quality"])
    return {"partitions": partitions, "pooled": pooled_block,
            "pooled_k0_lt_early": {
                "n_rows": len(pooled_small["quality"]),
                **{p: spearman(pooled_small[p], pooled_small["quality"])
                   for p in PREDICTORS}}}


def q2_mediation(rows: dict[tuple, dict[str, Any]],
                 quality: dict[str, dict[tuple[str, str, int], float]],
                 early_field: str,
                 *, restrict_k0_lt: int | None = None) -> dict[str, Any]:
    """Both directions of the k0 / early-offset mediation, on pooled z-scores.

    With ``restrict_k0_lt`` the rows whose ``k0`` is at or past that state are
    dropped BEFORE the within-partition z-scoring, so the restricted population
    is standardised on itself. That population is the primary one for the
    ``D[10]`` reading: the dropped rows have ``D[10]`` exactly 0 by definition,
    and on the full population they alone drive the joint-regression ``k0``
    coefficient negative.
    """
    y_all, k0_all, e_all = [], [], []
    for model in MODELS:
        for k in KS:
            keys = sorted(kk for kk in rows if kk[0] == model and kk[1] == k)
            y, k0s, es = [], [], []
            for key in keys:
                entry = rows[key]
                q = quality[model].get((key[2], key[3], key[1]))
                if q is None or entry["k0"] is None or entry[early_field] is None:
                    continue
                if restrict_k0_lt is not None and entry["k0"] >= restrict_k0_lt:
                    continue
                y.append(q)
                k0s.append(float(entry["k0"]))
                es.append(float(entry[early_field]))
            if len(y) < 5:
                continue
            y_all.extend(_z(np.asarray(y)).tolist())
            k0_all.extend(_z(np.asarray(k0s)).tolist())
            e_all.extend(_z(np.asarray(es)).tolist())
    if len(y_all) < 10:
        return {"n_rows": len(y_all)}
    y = np.asarray(y_all)
    k0 = np.asarray(k0_all)
    e = np.asarray(e_all)
    only_k0 = ols(y, k0[:, None])
    only_e = ols(y, e[:, None])
    both = ols(y, np.column_stack([k0, e]))
    return {
        "n_rows": int(y.size),
        "early_field": early_field,
        "restricted_to_k0_lt": restrict_k0_lt,
        "quality_on_k0": only_k0,
        "quality_on_early": only_e,
        "quality_on_both": both,
        "k0_absorbed_fraction": (
            None if abs(only_k0["coef"][0]) < 1e-12
            else 1.0 - both["coef"][0] / only_k0["coef"][0]),
        "early_absorbed_fraction": (
            None if abs(only_e["coef"][0]) < 1e-12
            else 1.0 - both["coef"][1] / only_e["coef"][0]),
        "caveats": [CAV_POOLED, CAV_COLLINEAR],
    }


def q2_within_cell(bend: list[dict[str, Any]],
                   perprompt: dict[str, dict[tuple[str, str, int, int], float]],
                   *, roles: str, field: str,
                   control_field: str = "D50_over_chord_ref") -> dict[str, Any]:
    """Per-cell Spearman of ``field`` against quality, over the kept prompts.

    ``share_p_below_05`` is computed over the cells whose rho exists -- the
    same denominator as ``share_negative``; the cells whose ``field`` is
    constant (the structurally zero ``D[10]`` rows) are counted in ``n_cells``
    only. Beside the main reading, the same cells' rho of ``control_field``
    against quality and the partial rank correlation of ``field`` given the
    control, both restricted to the cells whose main rho exists.
    """
    groups: dict[tuple, list[tuple[float, float, float | None]]] = {}
    for row in bend:
        if roles != "all" and row["role"] == "discovery":
            continue
        if row[field] is None:
            continue
        q = perprompt[row["model"]].get(
            (row["schedule"], row["payload"], row["k"], row["prompt_idx"]))
        if q is None:
            continue
        groups.setdefault(
            (row["model"], row["k"], row["schedule"], row["payload"]), []
        ).append((float(row[field]), float(q), row.get(control_field)))
    rhos: list[float] = []
    p_below: list[bool] = []
    control_rhos: list[float] = []
    partials: list[float] = []
    per_cell: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        res = spearman([v[0] for v in values], [v[1] for v in values])
        cell = {"model": key[0], "k": key[1], "schedule": key[2],
                "payload": key[3], **res}
        if res["rho"] is not None:
            rhos.append(res["rho"])
            p_below.append(res["p"] is not None and res["p"] < 0.05)
            ctrl = [v[2] for v in values]
            if all(c is not None for c in ctrl):
                r_cq = spearman(ctrl, [v[1] for v in values])
                r_fc = spearman([v[0] for v in values], ctrl)
                cell["rho_control"] = r_cq["rho"]
                if r_cq["rho"] is not None and r_fc["rho"] is not None:
                    control_rhos.append(r_cq["rho"])
                    den = math.sqrt((1.0 - r_fc["rho"] ** 2)
                                    * (1.0 - r_cq["rho"] ** 2))
                    if den > 0:
                        partial = (res["rho"] - r_fc["rho"] * r_cq["rho"]) / den
                        cell["partial_rho_given_control"] = partial
                        partials.append(partial)
        per_cell.append(cell)
    arr = np.asarray(rhos, dtype=np.float64)
    ctrl_arr = np.asarray(control_rhos, dtype=np.float64)
    part_arr = np.asarray(partials, dtype=np.float64)
    return {
        "field": field,
        "control_field": control_field,
        "n_cells": len(per_cell),
        "n_cells_with_rho": int(arr.size),
        "rho_median": float(np.median(arr)) if arr.size else None,
        "rho_p25": float(np.percentile(arr, 25)) if arr.size else None,
        "rho_p75": float(np.percentile(arr, 75)) if arr.size else None,
        "share_negative": float(np.mean(arr < 0)) if arr.size else None,
        "share_p_below_05": float(np.mean(p_below)) if p_below else None,
        "share_p_below_05_denominator": "cells_with_rho",
        "control_rho_median": (float(np.median(ctrl_arr))
                               if ctrl_arr.size else None),
        "partial_rho_median": (float(np.median(part_arr))
                               if part_arr.size else None),
        "partial_share_negative": (float(np.mean(part_arr < 0))
                                   if part_arr.size else None),
        "n_cells_with_partial": int(part_arr.size),
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# Q1 / Q3 / Q4
# ---------------------------------------------------------------------------


def q1_profile_and_order(bend_json: dict[str, dict[str, Any]],
                         rows: dict[tuple, dict[str, Any]],
                         quality: dict[str, dict[tuple[str, str, int], float]],
                         metric: str) -> dict[str, Any]:
    out: dict[str, Any] = {"metric": metric, "models": {}}
    for model, payload in bend_json.items():
        profiles = []
        for cell in payload["cells"]:
            med = cell["D_over_chord_ref_profile"]["median"]
            arr = np.asarray([np.nan if v is None else v for v in med],
                             dtype=np.float64)
            if not np.isfinite(arr).all():
                continue
            profiles.append(arr)
        mat = np.asarray(profiles) if profiles else np.zeros((0, 51))
        monotone = (float(np.mean([bool(np.all(np.diff(p) >= -1e-12)) for p in mat]))
                    if mat.size else None)
        late = (float(np.median([(p[50] - p[40]) / max(p[40] - p[30], 1e-30)
                                 for p in mat if p[40] > p[30]]))
                if mat.size else None)
        order = {}
        for k in KS:
            keys = sorted(kk for kk in rows if kk[0] == model and kk[1] == k)
            xs = [rows[kk]["D50_over_chord_ref"] for kk in keys]
            ys = [quality[model].get((kk[2], kk[3], kk[1])) for kk in keys]
            order[f"k{k}"] = spearman(xs, ys)
        out["models"][model] = {
            "n_cells_profiled": int(mat.shape[0]),
            "share_monotone_nondecreasing": monotone,
            "late_over_mid_growth_ratio_median": late,
            "D50_vs_quality_spearman": order,
        }
    return out


def q3_direction(rows: dict[tuple, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"families": PAYLOAD_FAMILY, "by_payload": {}}
    for model in MODELS:
        block: dict[str, Any] = {}
        for payload in PAYLOAD_FAMILY:
            keys = [kk for kk in rows if kk[0] == model and kk[3] == payload]
            if not keys:
                continue
            entry = {}
            for field in ("share_chord_50", "share_in_plane_50",
                          "share_off_plane_50", "share_off_plane_k0p1"):
                vals = np.asarray([rows[kk][field] for kk in keys
                                   if rows[kk][field] is not None], dtype=np.float64)
                entry[field] = {
                    "n_rows": int(vals.size),
                    "median": float(np.median(vals)) if vals.size else None,
                    "p25": float(np.percentile(vals, 25)) if vals.size else None,
                    "p75": float(np.percentile(vals, 75)) if vals.size else None,
                }
            block[payload] = entry
        out["by_payload"][model] = block
    return out


def q4_first_jump(rows: dict[tuple, dict[str, Any]],
                  quality: dict[str, dict[tuple[str, str, int], float]],
                  metric: str) -> dict[str, Any]:
    out: dict[str, Any] = {"metric": metric, "partitions": {}}
    for model in MODELS:
        for k in KS:
            keys = sorted(kk for kk in rows if kk[0] == model and kk[1] == k)
            k0s = [rows[kk]["k0"] for kk in keys]
            djump = [rows[kk]["delta_D_k0"] for kk in keys]
            slope = [rows[kk]["slope_k0_plus_5"] for kk in keys]
            qual = [quality[model].get((kk[2], kk[3], kk[1])) for kk in keys]
            out["partitions"][f"{model}_k{k}"] = {
                "n_rows": len(keys),
                "delta_D_k0_vs_k0": spearman(k0s, djump),
                "slope_vs_k0": spearman(k0s, slope),
                "delta_D_k0_vs_quality": spearman(djump, qual),
                "slope_vs_quality": spearman(slope, qual),
            }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path,
                        default=_ROOT / "resources" / "image_trajectory")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--metric", default="psnr")
    parser.add_argument("--secondary_metric", default="lpips")
    args = parser.parse_args()

    bend = read_bend(args.tables / "perprompt_bend.tsv.gz")
    bend_json = {}
    for model in MODELS:
        path = args.tables / f"cached_bend_{model}.json"
        if path.is_file():
            bend_json[model] = json.loads(path.read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "schema": "image_trajectory.early_quality_link.v1",
        "metric": args.metric,
        "secondary_metric": args.secondary_metric,
        "early_n": EARLY_N,
        "caveats": [CAV_POOLED, CAV_COLLINEAR, CAV_ASYMMETRIC],
        "slices": {},
    }
    quality = {m: read_quality(m, metric=args.metric) for m in MODELS}
    quality2 = {m: read_quality(m, metric=args.secondary_metric) for m in MODELS}
    perprompt = {m: read_quality_perprompt(m, metric=args.metric) for m in MODELS}

    for slice_name in ("all", "held_out"):
        rows = row_table(bend, roles=slice_name)
        block = {
            "n_rows": len(rows),
            "q1": q1_profile_and_order(bend_json, rows, quality, args.metric),
            "q2_row_level": q2_row_level(rows, quality, args.metric),
            "q2_row_level_secondary": q2_row_level(rows, quality2,
                                                   args.secondary_metric),
            "q2_mediation_D10": q2_mediation(rows, quality, "D10_over_chord_ref"),
            "q2_mediation_D10_k0_lt_early": q2_mediation(
                rows, quality, "D10_over_chord_ref", restrict_k0_lt=EARLY_N),
            "q2_mediation_auc": q2_mediation(rows, quality,
                                             "early_auc_over_chord_ref"),
            "q2_mediation_auc_k0_lt_early": q2_mediation(
                rows, quality, "early_auc_over_chord_ref",
                restrict_k0_lt=EARLY_N),
            "q2_mediation_slope": q2_mediation(rows, quality, "slope_k0_plus_5"),
            "q2_mediation_slope_k0_lt_early": q2_mediation(
                rows, quality, "slope_k0_plus_5", restrict_k0_lt=EARLY_N),
            "q2_within_cell": q2_within_cell(bend, perprompt, roles=slice_name,
                                             field="D10_over_chord_ref"),
            "q3_direction": q3_direction(rows),
            "q4_first_jump": q4_first_jump(rows, quality, args.metric),
        }
        report["slices"][slice_name] = block

    out = args.out or (args.tables / "early_quality_link.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out)
    print(f"[image-traj] wrote {out}")
    for slice_name, block in report["slices"].items():
        pooled = block["q2_row_level"]["pooled"]
        print(f"  {slice_name}: rows={block['n_rows']} "
              f"pooled D10 rho={pooled['D10_over_chord_ref']['rho']} "
              f"k0 rho={pooled['k0']['rho']} "
              f"D50 rho={pooled['D50_over_chord_ref']['rho']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
