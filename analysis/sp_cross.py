#!/usr/bin/env python3
"""SPX analysis: two-way schedule x payload decomposition + P1-P4 statistics.

Reads the per-cell metrics JSON written by `evaluation/eval_metrics.py` under
the SPX output tree

    <root>/<model>/k<K>/<schedule>x<payload>_s<seed>/metrics.json

averages the seeds of each cell, and fits the additive model

    y_sp = mu + alpha_s + beta_p + gamma_sp

by least squares with sum-to-zero coding, so `gamma_sp` is the interaction
residual. Values are oriented so that larger is better (LPIPS is negated);
the orientation is recorded in the output.

The fit runs on whatever cells exist. W1 is the full 4 schedules x 5 payloads
subgrid (plan section 3, revised after the original ragged shape proved
gamma-unidentifiable); W1b adds single-payload rows for P4, which ARE ragged,
so `residual_dof` and the per-cell identifiability counts (row / column
residual dof) are reported next to every decomposition; confounded diagonal
cells are excluded from the P1 sign test and the panel is scored on the rest.

Pre-registered statistics (`docs/research_plan_schedule_payload_cross.md` §2):

P1 diagonal advantage   sign test + mean magnitude of gamma on the homologous
                        (schedule, payload) cells. A diagonal cell that is the
                        only observation of its payload column or schedule row
                        (W1b single-cell rows) has its advantage absorbed by
                        beta_p / alpha_s, so its gamma is ~0 by construction;
                        such CELLS are listed in `confounded_cells` and left
                        out of the test (`n_identifiable` counts the rest).
P2 order matching       dpcache: hermite_o2 >= reuse; dicache_top1:
                        di_two_anchor >= taylor_o1 >= reuse; budcache: reuse is
                        the row argmax.
P3 zero-order at high   at K=41, median payload rank within each schedule row
   compression          should read reuse >= taylor_o1 >= hermite_o2 (ties
                        count; `strict_order_holds` reports the strict form).
P4 modal-path capacity  each gate's top-1 schedule under its native payload
                        against that gate's native dynamic run, compared to a
                        seed-noise band; dicache additionally reports the
                        mispaired reuse cell. The native run is external data:
                        pass `--native_reference` with a TSV of
                        (model, budget_k, method, metric, value) extracted from
                        the matrix results, or P4 reports reference=None.

`--split` restricts every statistic to one frozen prompt split
(`resources/sp_cross_schedules/parti_spx_splits.v1.json`) by averaging the
per-prompt `per_image` values instead of reading `summary`; `all` (default)
reads the summary mean and is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PAYLOADS = ("reuse", "taylor_o1", "hermite_o2", "mean_avg_vel", "di_two_anchor")
HIGHER_IS_BETTER = {
    "psnr": True,
    "ssim": True,
    "image_reward": True,
    "clip": True,
    "lpips": False,
}
# Schedule -> the payload(s) that share its source method. This mapping IS the
# pre-registered diagonal of P1 (plan section 2b lists it verbatim); two rows
# are declared proxies rather than literal sources: dpcache's own payload is
# its order-2 forecast, mapped to hermite_o2 as the same-order stand-in, and
# uniform is the schedule TaylorSeer O1 / HiCache O2 share, so both payloads
# are homologous to it. The gpf rows come from plan section 8.2b: each family
# member is constructed FOR one payload order and its manifest records it.
HOMOLOGOUS = {
    "budcache": ("reuse",),
    "meancache": ("mean_avg_vel",),
    "dpcache": ("hermite_o2",),
    "uniform": ("taylor_o1", "hermite_o2"),
    "seacache_top1": ("reuse",),
    "teacache_top1": ("reuse",),
    "sencache_top1": ("reuse",),
    "dicache_top1": ("di_two_anchor",),
    "gpf_reuse_e05_1": ("reuse",),
    "gpf_o1_e15_1": ("taylor_o1",),
    "gpf_o1_e20_1": ("taylor_o1",),
}
# P2 ordered pairs: (schedule, expected_better_payload, expected_worse_payload)
P2_PAIRS = (
    ("dpcache", "hermite_o2", "reuse"),
    ("dicache_top1", "di_two_anchor", "taylor_o1"),
    ("dicache_top1", "taylor_o1", "reuse"),
)
P2_ROW_ARGMAX = (("budcache", "reuse"),)
P3_PAYLOAD_ORDER = ("reuse", "taylor_o1", "hermite_o2")
#: The balanced sub-grid: these four schedule rows carry all five payloads, so
#: the additive fit on them is full rank and every gamma is identified cell by
#: cell. The formal P1 reading uses this grid; the ragged whole-panel fit (which
#: also contains rows observed in one or two columns) is kept as an appendix,
#: because a row seen in a single column has its diagonal advantage absorbed by
#: beta_p and its gamma is then ~0 by construction rather than by measurement.
W1_ROWS = ("budcache", "dpcache", "uniform", "dicache_top1")
P4_GATES = {
    "seacache_top1": "seacache",
    "teacache_top1": "teacache",
    "sencache_top1": "sencache",
    "dicache_top1": "dicache",
}


# ----- cell discovery --------------------------------------------------------


def parse_cell_name(name: str) -> tuple[str, str, int] | None:
    """`<schedule>x<payload>_s<seed>` -> (schedule, payload, seed)."""

    head, _, seed_text = name.rpartition("_s")
    if not head or not seed_text.isdigit():
        return None
    for payload in PAYLOADS:
        suffix = f"x{payload}"
        if head.endswith(suffix):
            schedule = head[: -len(suffix)]
            if schedule:
                return schedule, payload, int(seed_text)
    return None


SPLITS = ("all", "discovery", "validation", "test", "heldout")


def load_split(path: Path, split: str) -> frozenset[int] | None:
    """Prompt indices of `split` from the frozen parti splits file.

    `all` returns None (no restriction: the summary mean over every prompt is
    used, exactly as before the flag existed); `heldout` is validation + test.
    """

    if split == "all":
        return None
    roles = json.loads(path.read_text(encoding="utf-8"))["roles"]
    names = ("validation", "test") if split == "heldout" else (split,)
    return frozenset(int(i) for name in names for i in roles[name])


def read_metric(
    path: Path,
    metric: str,
    prompt_indices: frozenset[int] | None = None,
) -> tuple[float, int] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if prompt_indices is None:
        entry = payload.get("summary", {}).get(metric)
        if not isinstance(entry, Mapping) or entry.get("mean") is None:
            return None
        return float(entry["mean"]), int(entry.get("n", payload.get("n_pairs", 0)))
    values = payload.get("per_image", {}).get(metric)
    indices = payload.get("indices")
    if not values or not indices or len(values) != len(indices):
        return None
    kept = [
        float(value)
        for value, index in zip(values, indices)
        if int(index) in prompt_indices and value is not None
    ]
    if not kept:
        return None
    return float(statistics.fmean(kept)), len(kept)


def discover_cells(
    root: Path,
    *,
    metric: str,
    metrics_name: str = "metrics.json",
    prompt_indices: frozenset[int] | None = None,
) -> list[dict[str, Any]]:
    """Walk `<root>/<model>/k<K>/<cell>/<metrics_name>` into flat records."""

    cells: list[dict[str, Any]] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for budget_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if not budget_dir.name.startswith("k") or not budget_dir.name[1:].isdigit():
                continue
            budget_k = int(budget_dir.name[1:])
            for cell_dir in sorted(p for p in budget_dir.iterdir() if p.is_dir()):
                parsed = parse_cell_name(cell_dir.name)
                if parsed is None:
                    continue
                metrics_path = cell_dir / metrics_name
                if not metrics_path.is_file():
                    continue
                value = read_metric(metrics_path, metric, prompt_indices)
                if value is None:
                    continue
                schedule, payload, seed = parsed
                cells.append(
                    {
                        "model": model_dir.name,
                        "budget_k": budget_k,
                        "schedule": schedule,
                        "payload": payload,
                        "seed": seed,
                        "metric": metric,
                        "value": value[0],
                        "n_pairs": value[1],
                        "path": str(cell_dir),
                    }
                )
    return cells


def orient(value: float, metric: str) -> float:
    return float(value) if HIGHER_IS_BETTER.get(metric, True) else -float(value)


def group_cells(
    cells: Sequence[Mapping[str, Any]], *, metric: str
) -> dict[tuple[str, int], dict[tuple[str, str], dict[str, Any]]]:
    """(model, K) -> (schedule, payload) -> seed-averaged oriented value."""

    grouped: dict[tuple[str, int], dict[tuple[str, str], dict[str, Any]]] = {}
    for cell in cells:
        key = (str(cell["model"]), int(cell["budget_k"]))
        inner = grouped.setdefault(key, {})
        entry = inner.setdefault(
            (str(cell["schedule"]), str(cell["payload"])), {"seeds": {}}
        )
        entry["seeds"][int(cell["seed"])] = orient(cell["value"], metric)
    for inner in grouped.values():
        for entry in inner.values():
            values = list(entry["seeds"].values())
            entry["value"] = float(statistics.fmean(values))
            entry["n_seeds"] = len(values)
            entry["seed_sd"] = (
                float(statistics.stdev(values)) if len(values) >= 2 else None
            )
    return grouped


# ----- two-way decomposition -------------------------------------------------


def decompose(
    matrix: Mapping[tuple[str, str], float]
) -> dict[str, Any]:
    """Least-squares additive fit with sum-to-zero coding.

    On a complete grid this reproduces the classical means decomposition
    (alpha_s = row mean - grand mean, and so on).
    """

    keys = sorted(matrix)
    schedules = sorted({key[0] for key in keys})
    payloads = sorted({key[1] for key in keys})
    n_obs = len(keys)
    n_params = 1 + max(len(schedules) - 1, 0) + max(len(payloads) - 1, 0)
    design = np.zeros((n_obs, n_params), dtype=float)
    target = np.array([float(matrix[key]) for key in keys], dtype=float)
    design[:, 0] = 1.0
    for row, (schedule, payload) in enumerate(keys):
        for index, name in enumerate(schedules[:-1]):
            if schedule == name:
                design[row, 1 + index] = 1.0
            elif schedule == schedules[-1]:
                design[row, 1 + index] = -1.0
        offset = 1 + max(len(schedules) - 1, 0)
        for index, name in enumerate(payloads[:-1]):
            if payload == name:
                design[row, offset + index] = 1.0
            elif payload == payloads[-1]:
                design[row, offset + index] = -1.0
    rank = int(np.linalg.matrix_rank(design))
    if rank < n_params:
        raise ValueError(
            f"SPX design is rank deficient ({rank} < {n_params}); the observed "
            "cells cannot identify the additive effects"
        )
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    alpha = {name: float(coefficients[1 + i]) for i, name in enumerate(schedules[:-1])}
    if schedules:
        alpha[schedules[-1]] = -float(sum(alpha.values()))
    offset = 1 + max(len(schedules) - 1, 0)
    beta = {name: float(coefficients[offset + i]) for i, name in enumerate(payloads[:-1])}
    if payloads:
        beta[payloads[-1]] = -float(sum(beta.values()))
    gamma = {
        key: float(target[row] - fitted[row]) for row, key in enumerate(keys)
    }
    return {
        "mu": float(coefficients[0]),
        "alpha": alpha,
        "beta": beta,
        "gamma": {f"{s}x{p}": value for (s, p), value in gamma.items()},
        "gamma_pairs": gamma,
        "schedules": schedules,
        "payloads": payloads,
        "n_obs": n_obs,
        "n_params": n_params,
        "residual_dof": n_obs - n_params,
    }


# ----- pre-registered statistics --------------------------------------------


def two_sided_sign_p(n_pos: int, n_total: int) -> float | None:
    """Exact two-sided binomial test against p = 0.5."""

    if n_total <= 0:
        return None
    deviation = abs(n_pos - n_total / 2.0)
    tail = sum(
        math.comb(n_total, k)
        for k in range(n_total + 1)
        if abs(k - n_total / 2.0) >= deviation
    )
    return float(tail / (2**n_total))


def p1_diagonal(decomposition: Mapping[str, Any]) -> dict[str, Any]:
    gamma = decomposition["gamma_pairs"]
    # Identifiability is a per-CELL property. A diagonal cell that is the only
    # observation of its payload column or schedule row (the W1b single-cell
    # rows sea/tea/sen_top1 x reuse; the W1 shape's lone diagonal payloads)
    # has its advantage absorbed by beta_p / alpha_s, so its gamma is ~0 by
    # construction and it carries no information about diagonal advantage.
    # Such cells are listed but excluded from the sign test; the panel is
    # scored on the remaining identifiable diagonal cells. Because the row
    # (column) residuals sum to zero under the additive fit, a row of n cells
    # leaves n-1 residual dof; a 2-cell row (meancache x {reuse, mean_avg_vel})
    # is identifiable but pins gamma[mean x avg_vel] == -gamma[mean x reuse],
    # so the per-cell dof are reported for the reader to weigh.
    column_counts: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    for schedule, payload in gamma:
        column_counts[payload] = column_counts.get(payload, 0) + 1
        row_counts[schedule] = row_counts.get(schedule, 0) + 1
    entries = [
        {
            "schedule": schedule,
            "payload": payload,
            "gamma": float(gamma[(schedule, payload)]),
            "n_in_payload_column": column_counts[payload],
            "n_in_schedule_row": row_counts[schedule],
            "row_residual_dof": row_counts[schedule] - 1,
            "column_residual_dof": column_counts[payload] - 1,
            "identifiable": row_counts[schedule] >= 2 and column_counts[payload] >= 2,
        }
        for schedule, payloads in HOMOLOGOUS.items()
        for payload in payloads
        if (schedule, payload) in gamma
    ]
    off = [
        float(value)
        for (schedule, payload), value in gamma.items()
        if payload not in HOMOLOGOUS.get(schedule, ())
    ]
    values = [entry["gamma"] for entry in entries if entry["identifiable"]]
    # A structurally-absorbed diagonal cell comes back from lstsq as gamma on
    # the order of 1e-14 -- solver noise, not a trial. Counting it hands the
    # sign test coin flips whose sign depends on the BLAS build. The tolerance
    # is scaled to the panel's own gamma magnitudes so a genuinely tiny but
    # real interaction on a small-valued metric still counts.
    scale = max((abs(value) for value in gamma.values()), default=0.0)
    tolerance = max(scale, 1.0) * 1e-9
    n_pos = sum(1 for value in values if value > tolerance)
    n_nonzero = sum(1 for value in values if abs(value) > tolerance)
    confounded = [
        f"{entry['schedule']}x{entry['payload']}"
        for entry in entries
        if not entry["identifiable"]
    ]
    return {
        "cells": entries,
        "n_diagonal": len(entries),
        "n_identifiable": len(values),
        "n_positive": n_pos,
        "n_nonzero": n_nonzero,
        "structurally_identifiable": bool(values),
        "confounded_cells": confounded,
        "mean_gamma": float(statistics.fmean(values)) if values else None,
        "mean_abs_gamma": (
            float(statistics.fmean([abs(value) for value in values])) if values else None
        ),
        "mean_offdiagonal_gamma": float(statistics.fmean(off)) if off else None,
        "sign_test_p_two_sided": two_sided_sign_p(n_pos, n_nonzero),
    }


def p2_order(matrix: Mapping[tuple[str, str], float]) -> dict[str, Any]:
    pairs = []
    for schedule, better, worse in P2_PAIRS:
        if (schedule, better) not in matrix or (schedule, worse) not in matrix:
            continue
        delta = float(matrix[(schedule, better)] - matrix[(schedule, worse)])
        pairs.append(
            {
                "schedule": schedule,
                "better": better,
                "worse": worse,
                "delta": delta,
                "holds": delta >= 0.0,
            }
        )
    argmax = []
    for schedule, expected in P2_ROW_ARGMAX:
        row = {
            payload: value
            for (name, payload), value in matrix.items()
            if name == schedule
        }
        if not row:
            continue
        best = max(row, key=lambda payload: row[payload])
        argmax.append(
            {
                "schedule": schedule,
                "expected_argmax": expected,
                "observed_argmax": best,
                "holds": best == expected,
                "row": row,
            }
        )
    return {"ordered_pairs": pairs, "row_argmax": argmax}


def p3_high_compression(matrix: Mapping[tuple[str, str], float]) -> dict[str, Any]:
    ranks: dict[str, list[int]] = {payload: [] for payload in P3_PAYLOAD_ORDER}
    rows = []
    for schedule in sorted({key[0] for key in matrix}):
        row = {
            payload: matrix[(schedule, payload)]
            for payload in P3_PAYLOAD_ORDER
            if (schedule, payload) in matrix
        }
        if len(row) != len(P3_PAYLOAD_ORDER):
            continue
        order = sorted(row, key=lambda payload: row[payload], reverse=True)
        row_ranks = {payload: order.index(payload) + 1 for payload in row}
        for payload, rank in row_ranks.items():
            ranks[payload].append(rank)
        rows.append({"schedule": schedule, "values": row, "ranks": row_ranks})
    medians = {
        payload: (float(statistics.median(values)) if values else None)
        for payload, values in ranks.items()
    }
    # `median_order_holds` follows the plan wording ("reuse >= taylor_o1 >=
    # hermite_o2"): tied median ranks count as holding. `strict_order_holds`
    # demands strictly better median ranks so a tie is not read as an order.
    ordered = None
    strict = None
    if all(medians[payload] is not None for payload in P3_PAYLOAD_ORDER):
        steps = [
            (medians[P3_PAYLOAD_ORDER[i]], medians[P3_PAYLOAD_ORDER[i + 1]])
            for i in range(len(P3_PAYLOAD_ORDER) - 1)
        ]
        ordered = all(left <= right for left, right in steps)
        strict = all(left < right for left, right in steps)
    return {
        "rows": rows,
        "median_rank": medians,
        "median_order_holds": ordered,
        "strict_order_holds": strict,
    }


def p4_modal_capacity(
    entries: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    metric: str,
    reference: Mapping[tuple[str, str], float],
    pooled_band: float | None,
) -> dict[str, Any]:
    rows = []
    for schedule, method in P4_GATES.items():
        native_payload = HOMOLOGOUS[schedule][0]
        row: dict[str, Any] = {
            "schedule": schedule,
            "method": method,
            "native_payload_name": native_payload,
            "reference": reference.get((method, metric)),
            "pooled_band": pooled_band,
        }
        # The reuse comparison is the plan's mispairing control for gates
        # whose native payload is NOT reuse (dicache); when the native payload
        # is reuse the two would be the same cell, so it is emitted once.
        comparisons = [("native", native_payload)]
        if native_payload != "reuse":
            comparisons.append(("reuse", "reuse"))
        else:
            row["reuse"] = None
        for label, payload in comparisons:
            entry = entries.get((schedule, payload))
            if entry is None:
                row[label] = None
                continue
            delta = (
                None
                if row["reference"] is None
                else float(entry["value"] - orient(row["reference"], metric))
            )
            row[label] = {
                "payload": payload,
                "value": entry["value"],
                "n_seeds": entry["n_seeds"],
                "seed_sd": entry["seed_sd"],
                "delta_vs_reference": delta,
                "within_band": (
                    None
                    if delta is None or pooled_band is None
                    else bool(abs(delta) <= pooled_band)
                ),
            }
        rows.append(row)
    return {"rows": rows}


def pooled_seed_band(
    entries: Mapping[tuple[str, str], Mapping[str, Any]]
) -> float | None:
    """2 x pooled across-seed standard deviation of the cells in one panel."""

    variances = [
        float(entry["seed_sd"]) ** 2
        for entry in entries.values()
        if entry.get("seed_sd") is not None
    ]
    if not variances:
        return None
    return 2.0 * math.sqrt(sum(variances) / len(variances))


def clustered_paired_se(
    diffs: Mapping[Any, float], *, cluster_of=lambda key: key[1]
) -> dict[str, Any] | None:
    """Paired SE with prompts, not (seed, prompt) pairs, as the sampling unit.

    The i.i.d. paired SE divides by the number of pairs, and every prompt
    contributes one pair per seed stream. Those repeats are strongly correlated
    -- a hard prompt is hard on all three streams -- so the i.i.d. band is too
    narrow. This is the usual cluster-robust variance of a sample mean,

        Var = sum_g ( sum_{i in g} (d_i - dbar) )^2 / n^2

    with one cluster per prompt, which is the interval a reader should hold a
    "distinguishable" claim to.
    """

    if len(diffs) < 2:
        return None
    values = list(diffs.values())
    mean = statistics.fmean(values)
    groups: dict[Any, list[float]] = {}
    for key, value in diffs.items():
        groups.setdefault(cluster_of(key), []).append(value)
    if len(groups) < 2:
        return None
    total = sum((sum(value - mean for value in group)) ** 2 for group in groups.values())
    se = math.sqrt(total) / len(values)
    return {"se": se, "n_clusters": len(groups)}


def read_reference(path: Path | None) -> dict[tuple[str, str], float]:
    """Optional TSV: model, budget_k, method, metric, value.

    Leading `#` lines carry provenance (the shipped reference file records
    which table its values came from), so they are stripped before parsing —
    otherwise the first comment becomes the header and every lookup fails.
    """

    if path is None:
        return {}
    table: dict[tuple[str, str, int, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = (line for line in handle if not line.lstrip().startswith("#"))
        for row in csv.DictReader(rows, delimiter="\t"):
            table[
                (
                    row["model"],
                    row["method"],
                    int(row["budget_k"]),
                    row["metric"],
                )
            ] = float(row["value"])
    return table


def analyse(
    cells: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    high_k: int,
    reference_table: Mapping[tuple[str, str, int, str], float],
    split: str = "all",
) -> dict[str, Any]:
    # No off-target handling: the builder hard-fails on popcount != K and the
    # frozen manifest is all-zero k_vs_target (plan section 3, 严格算力约束),
    # so an off-budget schedule cannot exist to be excluded.
    grouped = group_cells(cells, metric=metric)
    panels = []
    for (model, budget_k) in sorted(grouped):
        entries = grouped[(model, budget_k)]
        matrix = {key: entry["value"] for key, entry in entries.items()}
        decomposition = decompose(matrix)
        w1_matrix = {
            key: value for key, value in matrix.items()
            if key[0] in W1_ROWS and key[1] in PAYLOADS
        }
        w1_decomposition = (
            decompose(w1_matrix)
            if len(w1_matrix) == len(W1_ROWS) * len(PAYLOADS)
            else None
        )
        band = pooled_seed_band(entries)
        reference = {
            (method, metric_name): value
            for (ref_model, method, ref_k, metric_name), value in reference_table.items()
            if ref_model == model and ref_k == int(budget_k)
        }
        panel = {
            "model": model,
            "budget_k": int(budget_k),
            "metric": metric,
            "orientation": "higher_is_better"
            if HIGHER_IS_BETTER.get(metric, True)
            else "lower_is_better_negated",
            "n_cells": len(matrix),
            "pooled_seed_band": band,
            "cells": {
                f"{schedule}x{payload}": {
                    "value": entry["value"],
                    "n_seeds": entry["n_seeds"],
                    "seed_sd": entry["seed_sd"],
                }
                for (schedule, payload), entry in sorted(entries.items())
            },
            "decomposition": {
                key: value
                for key, value in decomposition.items()
                if key != "gamma_pairs"
            },
            # Formal reading first, ragged panel second; `analysis/spx_supplement.py`
            # runs the same W1 fit over the staged per-image tables.
            "P1_diagonal_w1": (
                None if w1_decomposition is None else p1_diagonal(w1_decomposition)
            ),
            "w1_cells_present": len(w1_matrix),
            "P1_diagonal": p1_diagonal(decomposition),
            "P2_order": p2_order(matrix),
            # The shipped reference is a pooled mean over all 1,632 prompts, so
            # subtracting it from a subset's mean compares different prompt
            # populations. On a subset P4 is simply not computed here; the
            # per-prompt paired form in `analysis/spx_supplement.py` is the one
            # that takes a split, and it is the reported form either way.
            "P4_modal_capacity": (
                p4_modal_capacity(
                    entries, metric=metric, reference=reference, pooled_band=band,
                )
                if split == "all"
                else {"rows": [], "skipped_reason": (
                    "the native reference is a pooled 1,632-prompt mean; "
                    "use analysis/spx_supplement.py for a split-aware, "
                    "per-prompt paired P4")}
            ),
        }
        if int(budget_k) == int(high_k):
            panel["P3_high_compression"] = p3_high_compression(matrix)
        panels.append(panel)
    return {"metric": metric, "high_k": int(high_k), "split": split, "panels": panels}


def print_report(report: Mapping[str, Any]) -> None:
    for panel in report["panels"]:
        print(
            f"\n=== {panel['model']} K={panel['budget_k']} "
            f"metric={panel['metric']} ({panel['orientation']}) ==="
        )
        decomposition = panel["decomposition"]
        print(
            f"  cells={panel['n_cells']}  mu={decomposition['mu']:.4f}  "
            f"residual_dof={decomposition['residual_dof']}  "
            f"seed_band={panel['pooled_seed_band']}"
        )
        print("  alpha (schedule): " + ", ".join(
            f"{name}={value:+.4f}" for name, value in sorted(decomposition["alpha"].items())
        ))
        print("  beta  (payload):  " + ", ".join(
            f"{name}={value:+.4f}" for name, value in sorted(decomposition["beta"].items())
        ))
        formal = panel.get("P1_diagonal_w1")
        if formal is not None:
            print(
                f"  P1 (formal, W1 {panel['w1_cells_present']}-cell grid): "
                f"n={formal['n_diagonal']} positive={formal['n_positive']} "
                f"mean_gamma={formal['mean_gamma']} p={formal['sign_test_p_two_sided']}"
            )
        else:
            print("  P1 (formal): W1 grid incomplete on this panel; ragged fit only")
        p1 = panel["P1_diagonal"]
        print(
            f"  P1 (appendix, ragged panel): n={p1['n_diagonal']} identifiable={p1['n_identifiable']} "
            f"positive={p1['n_positive']} mean_gamma={p1['mean_gamma']} "
            f"p={p1['sign_test_p_two_sided']}"
        )
        if p1["confounded_cells"]:
            print(
                f"  P1 note: {', '.join(p1['confounded_cells'])} excluded — alone in "
                "their payload column or schedule row, so beta/alpha absorbs the "
                "diagonal effect and gamma is ~0 by construction; the sign test "
                "runs on the identifiable cells only."
            )
        if not p1["structurally_identifiable"]:
            print(
                "  P1 WARNING: no identifiable diagonal cell on this panel. Do not "
                "read it as evidence for or against diagonal advantage."
            )
        for entry in p1["cells"]:
            if entry["identifiable"] and entry["row_residual_dof"] < 2:
                print(
                    f"  P1 note: {entry['schedule']}x{entry['payload']} sits in a "
                    f"{entry['n_in_schedule_row']}-cell row (row_residual_dof="
                    f"{entry['row_residual_dof']}); its gamma is pinned to the "
                    "negative of its row-mate's."
                )
        for entry in panel["P2_order"]["ordered_pairs"]:
            print(
                f"  P2 {entry['schedule']}: {entry['better']} - {entry['worse']} = "
                f"{entry['delta']:+.4f} holds={entry['holds']}"
            )
        for entry in panel["P2_order"]["row_argmax"]:
            print(
                f"  P2 {entry['schedule']} row argmax: {entry['observed_argmax']} "
                f"(expected {entry['expected_argmax']}) holds={entry['holds']}"
            )
        if "P3_high_compression" in panel:
            p3 = panel["P3_high_compression"]
            print(
                f"  P3 median ranks: {p3['median_rank']} "
                f"order_holds={p3['median_order_holds']} "
                f"strict_order_holds={p3['strict_order_holds']}"
            )
        if panel["P4_modal_capacity"].get("skipped_reason"):
            print(f"  P4 skipped: {panel['P4_modal_capacity']['skipped_reason']}")
        rows = panel["P4_modal_capacity"]["rows"]
        if rows and all(row["reference"] is None for row in rows):
            print(
                "  P4 WARNING: no native reference — pass --native_reference with a "
                "TSV of (model, budget_k, method, metric, value) holding each gate's "
                "native dynamic-run metric from the matrix results, or P4 stays "
                "uncomputed."
            )
        for row in rows:
            line = (
                f"  P4 {row['schedule']}: reference={row['reference']} "
                f"{row['native_payload_name']}={_p4_brief(row.get('native'))}"
            )
            if row["native_payload_name"] != "reuse":
                line += f" reuse(mispaired)={_p4_brief(row.get('reuse'))}"
            print(line)


def _p4_brief(entry: Any) -> str:
    if not isinstance(entry, Mapping):
        return "n/a"
    delta = entry.get("delta_vs_reference")
    delta_text = "n/a" if delta is None else f"{delta:+.4f}"
    return f"{entry['value']:.4f} (delta={delta_text}, within={entry['within_band']})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metric", default="psnr", choices=sorted(HIGHER_IS_BETTER))
    parser.add_argument("--metrics_name", default="metrics.json")
    parser.add_argument("--high_k", type=int, default=41)
    parser.add_argument(
        "--native_reference",
        type=Path,
        default=None,
        help="optional native comparison TSV with model, budget_k, method, metric, value; "
        "omit to compute schedule-policy statistics without native-gate comparisons",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=SPLITS,
        help="restrict every statistic to the prompts of this frozen split "
        "(heldout = validation + test); needs per_image + indices in metrics.json",
    )
    parser.add_argument(
        "--splits_json",
        type=Path,
        default=Path("resources/sp_cross_schedules/parti_spx_splits.v1.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cells_tsv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt_indices = load_split(args.splits_json, args.split)
    cells = discover_cells(
        args.root,
        metric=args.metric,
        metrics_name=args.metrics_name,
        prompt_indices=prompt_indices,
    )
    if not cells:
        raise SystemExit(
            f"no SPX cells with metric {args.metric!r} (split {args.split!r}) "
            f"under {args.root}"
        )
    report = analyse(
        cells,
        metric=args.metric,
        high_k=args.high_k,
        reference_table=read_reference(args.native_reference),
        split=args.split,
    )
    report["root"] = str(args.root)
    report["split"] = args.split
    report["n_split_prompts"] = None if prompt_indices is None else len(prompt_indices)
    report["n_cell_runs"] = len(cells)
    print_report(report)
    if args.cells_tsv is not None:
        args.cells_tsv.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            "model",
            "budget_k",
            "schedule",
            "payload",
            "seed",
            "metric",
            "value",
            "n_pairs",
            "path",
        )
        with args.cells_tsv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fields),
                delimiter="\t",
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(cells)
        print(f"\n[sp-cross] wrote {args.cells_tsv}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[sp-cross] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
