#!/usr/bin/env python3
"""Video SPX analysis: schedule x payload cross on the video backbones.

Reads the per-cell evaluation JSONs staged under

    resources/video_spx/<backbone>/<row>x<payload>_<dataset>_K<K>_s<seed>.json

(each with `summary` and a 150-entry `per_video` list carrying `idx` plus the
four fidelity metrics), joins them to the frozen schedule manifest
(`resources/video_spx_schedules/manifest.tsv`), the native-gate per-video
quality of the baseline matrix (`resources/video_full_results/pervideo_<T>.tsv.gz`)
and the trajectory geometry (`resources/video_full_trajectory/<T>/density_form_<T>.json`,
`resources/video_native_gate_paths/<T>/per_seed_path_counts.tsv`), and writes

    resources/video_spx/video_spx_results.json
    resources/video_spx/video_spx_tables.md

Everything is per backbone; a backbone with no cell directory is skipped, so
the same command re-runs cleanly when the second backbone's JSONs land.

Pre-registered questions (docs/video_sp_cross_plan_zh.md section 1):

P1  path dominance      two-way decomposition y_sp = mu + alpha_s + beta_p +
                        gamma_sp on the full feasible grid and on the 7-row W1
                        subgrid, with the variance share of each term.
P2  few good paths      per-prompt paired differences against the MeanCache row
                        inside the reuse column, mean +- 2 x paired SE; how many
                        rows sit in the band and where the design controls land.
P3  method convergence  (i) each gate's modal path run as a fixed schedule
                        against that gate's own native per-prompt run, paired by
                        prompt, against the matrix three-stream pooled sd x 2;
                        (ii) the same row against the BudCache / MeanCache rows.
P4  geometry            Spearman of row quality against each geometry predictor,
                        per partition and pooled over partitions with partition
                        fixed effects (rank within partition, then Spearman),
                        plus the transposition dose curve away from MeanCache.
P5  payload dependence  argmax payload per row and the rows that disagree with
                        the partition majority.
P6  rho2 positive       the DP-rho2 row against the MeanCache row and against
    control             the random rows.

The two-way fit, the sum-to-zero coding and the variance-share definition are
`analysis/sp_cross.py`'s (`decompose`, and the ss/share formulas of
`analysis/plot_sp_cross.py`); the gap statistics are
`analysis/density_form_test.py`'s `measure` / `gaps_of`; the transport distance
and the basin components are `analysis/video_path_universe.py`'s `wasserstein` /
`components`. None of that maths is re-derived here.

    python analysis/video_spx.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.density_form_test import measure  # noqa: E402
from analysis.sp_cross import decompose  # noqa: E402
from analysis.video_path_universe import components, hamming, wasserstein  # noqa: E402

REPO = _REPO
BACKBONES = ("hunyuan_video", "wan21")
PAYLOADS = ("reuse", "taylor_o1", "hermite_o2", "mean_vel", "di_two_anchor")
#: Payload ids that are not a column of the cross, but a same-row variant of one.
#: `mean_vel_global` is the interval-mean-velocity payload run with the global
#: span every non-MeanCache row already uses, so that the `mean_vel` column can
#: be made one single payload instead of two (the MeanCache row otherwise runs
#: the per-edge spans its own offline search solved). When those cells exist the
#: column is homogenised onto them and the searched-span cell is reported apart.
SUPPLEMENT_PAYLOADS = ("mean_vel_global",)
MEAN_VEL = "mean_vel"
MEAN_VEL_GLOBAL = "mean_vel_global"
#: The row whose `mean_vel` cell is the heterogeneous one.
MEAN_VEL_NATIVE_ROW = "meancache"
#: Where the searched-span cell is parked once the column is homogenised.
MEAN_VEL_SEARCHED = "mean_vel_searched_span"
METRICS = ("psnr", "ssim", "lpips", "temporal_lpips_delta")
PRIMARY_METRIC = "psnr"
DATASETS = ("penguin599", "vbench944")
KS = ("K29", "K37", "K41")
NUM_STEPS = 50

#: Row display order; every row of the frozen manifest appears here. The `*f`
#: ladder rows are the first-cached-step-preserving twins of `ham2/4/8`; they
#: are absent until their cells land and are simply skipped while so.
ROW_ORDER = (
    "shared", "budcache", "meancache",
    "sea_top1", "tea_top1", "sen_top1", "di_top1",
    "sea_top1_off", "tea_top1_off", "sen_top1_off",
    "uniform", "dp_rho2", "rand_1", "rand_2",
    "ham2", "ham4", "ham8", "ham2f", "ham4f", "ham8f",
)

#: How many leading full steps each payload needs before its own construction is
#: available. A cell whose row caches earlier than this silently degrades to
#: zero-order reuse on those early steps, so it is not the payload it is labelled
#: with. Read off the adapters:
#:   HunyuanVideo TaylorSeer/HiCache clamp the extrapolation order to 0 while
#:   their post-incremented step counter is below `first_enhance` = 3
#:   (`hunyuan_video/methods/taylorseer.py`, `hunyuan_video/methods/hicache.py`),
#:   so steps 0-2 must be full for a first-order difference to exist at all.
#:   Wan2.1's fine payloads default to `first_enhance` = 1
#:   (`wan21/taylorseer_fine.py`, `wan21/fine_payload.py`), so a first-order
#:   difference exists after two full steps; the second-order Hermite still
#:   needs three.
#: The frozen schedule set already forbids caching steps {0,1,2} for
#: `hermite_o2` and {0,1} for `mean_vel` / `di_two_anchor`, so only the
#: first-order column can go impure, and only on HunyuanVideo.
PAYLOAD_WARMUP = {
    "hunyuan_video": {"reuse": 1, "taylor_o1": 3, "hermite_o2": 3,
                      "mean_vel": 2, "di_two_anchor": 2},
    "wan21": {"reuse": 1, "taylor_o1": 2, "hermite_o2": 3,
              "mean_vel": 2, "di_two_anchor": 2},
}
#: Plan section 5.1: the full-rank subgrid every variance statistic needs.
W1_ROWS = ("shared", "budcache", "meancache", "di_top1", "dp_rho2", "ham4", "uniform")
#: Gate rows -> the matrix method whose native per-prompt run they are compared with.
GATE_ROW_METHOD = {
    "sea_top1": "seacache", "tea_top1": "teacache",
    "sen_top1": "sencache", "di_top1": "dicache",
    "sea_top1_off": "seacache", "tea_top1_off": "teacache",
    "sen_top1_off": "sencache",
}
#: Gate rows -> the payload the gate itself runs (plan section 3).
GATE_NATIVE_PAYLOAD = {
    "sea_top1": "reuse", "tea_top1": "reuse", "sen_top1": "reuse",
    "di_top1": "di_two_anchor",
    "sea_top1_off": "reuse", "tea_top1_off": "reuse", "sen_top1_off": "reuse",
}
GATES = ("seacache", "teacache", "sencache", "dicache")
#: The matrix seed streams of each dataset; the first is the SPX stream.
MATRIX_STREAMS = {"penguin599": (54, 55, 56), "vbench944": (42, 43, 44)}
ANCHOR_ROW = "meancache"
CONTROL_ROWS = ("uniform", "rand_1", "rand_2")
#: The original ladder swaps cached and full steps anywhere in the 3-48 window,
#: which almost always moves the first cached step earlier; it is therefore a
#: confounded dose. The `*f` ladder holds the first cached step fixed, so it is
#: the dose of distance alone. Both are computed; the second only when present.
DOSE_LADDERS = {
    "free": (("meancache", 0), ("ham2", 2), ("ham4", 4), ("ham8", 8)),
    "first_preserving": (("meancache", 0), ("ham2f", 2), ("ham4f", 4), ("ham8f", 8)),
}
DOSE_ROWS = DOSE_LADDERS["free"]
#: rho2 dataset used for the geometry predictors; the other one is recomputed
#: as the robustness read (density_form_<T>.json holds both).
MAIN_RHO2_DATASET = "vbench944"
BASIN_RADII = (4, 6, 8)
GAMMA_SHARE_FLAG = 0.25

#: Larger is better after orientation. `temporal_lpips_delta` is a signed
#: deviation from the reference's own frame-to-frame LPIPS, so its quality
#: reading is minus its absolute value: either sign is a departure from the
#: full-step reference, which is what this experiment measures.
ORIENTATION = {
    "psnr": "as_is", "ssim": "as_is",
    "lpips": "negated", "temporal_lpips_delta": "negated_absolute",
}


def orient(metric: str, value: float) -> float:
    mode = ORIENTATION[metric]
    if mode == "as_is":
        return float(value)
    if mode == "negated":
        return -float(value)
    return -abs(float(value))


# ----- inputs ----------------------------------------------------------------


def parse_cell_name(name: str) -> tuple[str, str, str, str, int] | None:
    """`<row>x<payload>_<dataset>_K<K>_s<seed>` -> (row, payload, dataset, K, seed)."""

    match = re.match(
        r"^(?P<head>.+)_(?P<dataset>" + "|".join(DATASETS) + r")_K(?P<k>\d+)_s(?P<seed>\d+)$",
        name,
    )
    if match is None:
        return None
    head = match.group("head")
    # Longest suffix first, so `mean_vel_global` is not read as `mean_vel`.
    for payload in sorted(PAYLOADS + SUPPLEMENT_PAYLOADS, key=len, reverse=True):
        suffix = f"x{payload}"
        if head.endswith(suffix):
            row = head[: -len(suffix)]
            if not row:
                return None
            return row, payload, match.group("dataset"), f"K{match.group('k')}", int(match.group("seed"))
    return None


def load_cells(backbone: str) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """(row, payload, dataset, K) -> {"seed", "per": {idx: {metric: value}}, "summary"}."""

    directory = REPO / "resources/video_spx" / backbone
    cells: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if not directory.is_dir():
        return cells
    for path in sorted(directory.glob("*.json")):
        parsed = parse_cell_name(path.stem)
        if parsed is None:
            raise SystemExit(f"{path.name}: not a <row>x<payload>_<dataset>_K<K>_s<seed> cell name")
        row, payload, dataset, budget, seed = parsed
        key = (row, payload, dataset, budget)
        if key in cells:
            raise SystemExit(f"{path.name}: duplicate cell {key}")
        blob = json.loads(path.read_text(encoding="utf-8"))
        per: dict[int, dict[str, float]] = {}
        for entry in blob.get("per_video", []):
            values = {}
            for metric in METRICS:
                value = entry.get(metric)
                if value is None or not math.isfinite(float(value)):
                    values = {}
                    break
                values[metric] = float(value)
            if values:
                per[int(entry["idx"])] = values
        cells[key] = {
            "seed": seed,
            "per": per,
            "n_per_video": len(per),
            "summary": {
                metric: blob.get("summary", {}).get(metric, {}).get("mean")
                for metric in METRICS
            },
        }
    return cells


def homogenise_mean_vel(cells: dict[tuple[str, str, str, str], dict[str, Any]]
                        ) -> dict[str, Any]:
    """Make the interval-mean-velocity column one single payload, in place.

    The offline search solved per-edge spans for the MeanCache row only; every
    other row runs the global span, so as staged the column mixes two payloads.
    When the same row's global-span cells exist, they become the column and the
    searched-span cells move to `MEAN_VEL_SEARCHED`, which nothing but the
    dedicated comparison reads.
    """

    swapped: list[dict[str, Any]] = []
    for key in sorted(cells):
        row, payload, dataset, budget = key
        if payload != MEAN_VEL_GLOBAL:
            continue
        native_key = (row, MEAN_VEL, dataset, budget)
        native = cells.get(native_key)
        if native is None:
            continue
        cells[(row, MEAN_VEL_SEARCHED, dataset, budget)] = native
        cells[native_key] = cells[key]
        swapped.append({"row": row, "dataset": dataset, "budget": budget})
    return {
        "applied": bool(swapped),
        "cells_substituted": swapped,
        "rows": sorted({entry["row"] for entry in swapped}),
        "budgets": sorted({entry["budget"] for entry in swapped}),
    }


def load_perprompt_paths(backbone: str) -> dict[tuple[str, str, str, int, int], str]:
    """(method, dataset, K, seed, prompt index) -> the 50-bit path that run walked.

    This is the baseline matrix's realised gate decisions; P3 needs it to say
    which prompts the gate happened to walk its own modal path on, because on
    those the fixed-path run and the gate run are the same computation and their
    paired difference is a deterministic zero.
    """

    path = REPO / f"resources/video_full_results/perprompt_paths_{backbone}.tsv.gz"
    out: dict[tuple[str, str, str, int, int], str] = {}
    if not path.is_file():
        return out
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = next(handle).rstrip("\n").split("\t")
        column = {name: i for i, name in enumerate(header)}
        needed = ("method", "dataset", "K", "seed", "prompt_idx", "path")
        if any(name not in column for name in needed):
            return {}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            key = (fields[column["method"]], fields[column["dataset"]],
                   f"K{fields[column['K']]}", int(fields[column["seed"]]),
                   int(fields[column["prompt_idx"]]))
            out[key] = fields[column["path"]]
    return out


def load_manifest(backbone: str) -> dict[tuple[str, str], dict[str, Any]]:
    """(K, row) -> manifest record, for one backbone."""

    path = REPO / "resources/video_spx_schedules/manifest.tsv"
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            if record["backbone"] != backbone:
                continue
            out[(record["budget"], record["row"])] = record
    return out


def load_ladder_rejections(backbone: str) -> dict[str, Any]:
    """Per (budget, ladder row), how many draws the builder rejected.

    The ladder construction redraws any swap whose result opens a full-step gap
    wider than the builder's cap, so a surviving rung is a sample from the
    constrained neighbourhood intersected with that cap, not from the
    neighbourhood alone. The counts and the cap both come from the frozen
    schedules' provenance, so the report can state the actual filter instead of
    asserting one.
    """

    out: dict[str, dict[str, int]] = {}
    gap_cap: int | None = None
    directory = REPO / "resources/video_spx_schedules" / backbone
    for path in sorted(directory.glob("ham*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        provenance = blob.get("provenance") or {}
        rejected = provenance.get("draws_rejected")
        if rejected is None:
            continue
        out.setdefault(blob["budget"], {})[blob["row"]] = int(rejected)
        for rule in provenance.get("rejection_rules", []):
            found = re.fullmatch(r"full-step gap > (\d+)", rule)
            if found:
                gap_cap = int(found.group(1))
    return {"gap_cap": gap_cap, "by_budget": out}


def load_native_pervideo(backbone: str) -> dict[tuple[str, str, str, int, int], dict[str, float]]:
    """(method, dataset, K, seed, prompt_idx) -> oriented metric values.

    Rows whose PSNR is blank (the matrix's excluded identical-output pairs) are
    dropped whole, so a comparison that needs them loses that prompt on both
    sides rather than half of it.
    """

    path = REPO / f"resources/video_full_results/pervideo_{backbone}.tsv.gz"
    out: dict[tuple[str, str, str, int, int], dict[str, float]] = {}
    if not path.is_file():
        return out
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = next(handle).rstrip("\n").split("\t")
        column = {name: i for i, name in enumerate(header)}
        source = {"psnr": "psnr", "ssim": "ssim", "lpips": "lpips",
                  "temporal_lpips_delta": "temporal_delta"}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            index = int(fields[column["prompt_idx"]])
            values: dict[str, float] = {}
            for metric, name in source.items():
                text = fields[column[name]]
                try:
                    value = float(text)
                except ValueError:
                    values = {}
                    break
                if not math.isfinite(value):
                    values = {}
                    break
                values[metric] = orient(metric, value)
            if not values:
                continue
            key = (fields[column["method"]], fields[column["dataset"]],
                   f"K{fields[column['K']]}", int(fields[column["seed"]]), index)
            out[key] = values
    return out


def load_geometry_inputs(backbone: str) -> dict[str, Any]:
    """rho2 profiles + sigma grid + the pooled native gate path distributions."""

    out: dict[str, Any] = {"rho2": {}, "sigmas": None, "gate_dists": {}}
    density = REPO / f"resources/video_full_trajectory/{backbone}/density_form_{backbone}.json"
    if density.is_file():
        blob = json.loads(density.read_text(encoding="utf-8"))
        out["sigmas"] = np.asarray(blob["sigmas"], dtype=float)
        out["rho2"] = {name: np.asarray(values, dtype=float)
                       for name, values in blob["rho2"].items()}
    counts = REPO / f"resources/video_native_gate_paths/{backbone}/per_seed_path_counts.tsv"
    if counts.is_file():
        pooled: dict[tuple[str, str], Counter] = defaultdict(Counter)
        with counts.open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle, delimiter="\t"):
                pooled[(record["method"], record["budget"])][record["schedule"]] += int(record["count"])
        out["gate_dists"] = {
            key: {path: value / total for path, value in counter.items()}
            for key, counter in pooled.items()
            for total in (sum(counter.values()),)
        }
    return out


def load_wall_seconds(backbone: str) -> dict[str, Any] | None:
    """Generation seconds per video, from the staged per-video table if present."""

    path = REPO / f"resources/video_spx/{backbone}/pervideo_spx_{backbone}.tsv.gz"
    if not path.is_file():
        return None
    per_payload: dict[str, list[float]] = defaultdict(list)
    total = 0.0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = next(handle).rstrip("\n").split("\t")
        column = {name: i for i, name in enumerate(header)}
        if "wall_s" not in column:
            return None
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            try:
                seconds = float(fields[column["wall_s"]])
            except (ValueError, IndexError):
                continue
            per_payload[fields[column["payload"]]].append(seconds)
            total += seconds
    if not per_payload:
        return None
    return {
        "generation_gpu_hours": total / 3600.0,
        "seconds_per_video_median": {
            payload: float(statistics.median(values)) for payload, values in sorted(per_payload.items())
        },
        "n_videos": sum(len(values) for values in per_payload.values()),
    }


# ----- series and paired statistics -----------------------------------------


def series(cells: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
           row: str, payload: str, budget: str, metric: str) -> dict[tuple[str, int], float]:
    """(dataset, prompt index) -> oriented value, over every dataset of one cell."""

    out: dict[tuple[str, int], float] = {}
    for dataset in DATASETS:
        entry = cells.get((row, payload, dataset, budget))
        if entry is None:
            continue
        for index, values in entry["per"].items():
            out[(dataset, index)] = orient(metric, values[metric])
    return out


def dataset_means(cells: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
                  row: str, payload: str, budget: str, metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for dataset in DATASETS:
        entry = cells.get((row, payload, dataset, budget))
        if entry is None or not entry["per"]:
            continue
        out[dataset] = float(statistics.fmean(
            orient(metric, values[metric]) for values in entry["per"].values()))
    return out


def paired_difference(left: Mapping[tuple[str, int], float],
                      right: Mapping[tuple[str, int], float],
                      *, datasets: Sequence[str] | None = None,
                      restrict: set[tuple[str, int]] | None = None
                      ) -> dict[str, Any] | None:
    """Per-prompt paired difference `left - right`, mean +- 2 x paired SE."""

    keys = sorted(set(left) & set(right))
    if datasets is not None:
        keys = [key for key in keys if key[0] in datasets]
    if restrict is not None:
        keys = [key for key in keys if key in restrict]
    if len(keys) < 2:
        return None
    diffs = [left[key] - right[key] for key in keys]
    mean = float(statistics.fmean(diffs))
    sd = float(statistics.stdev(diffs))
    se = sd / math.sqrt(len(diffs))
    return {
        "mean": mean, "sd": sd, "se": se, "n_pairs": len(diffs),
        "ci_low": mean - 2.0 * se, "ci_high": mean + 2.0 * se,
        "n_pos": sum(1 for value in diffs if value > 0.0),
    }


def classify(entry: Mapping[str, Any] | None) -> str | None:
    """Where a paired difference sits relative to zero at +- 2 paired SE."""

    if entry is None:
        return None
    if entry["ci_low"] > 0.0:
        return "better"
    if entry["ci_high"] < 0.0:
        return "worse"
    return "indistinguishable"


def spearman(xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any] | None:
    """Rank correlation; None when either side has no spread."""

    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None
             and math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 4:
        return None
    xv = [pair[0] for pair in pairs]
    yv = [pair[1] for pair in pairs]
    if max(xv) - min(xv) <= 1e-12 or max(yv) - min(yv) <= 1e-12:
        return None
    from scipy.stats import spearmanr

    result = spearmanr(xv, yv)
    return {"rho": float(result.statistic), "p": float(result.pvalue), "n": len(pairs)}


def rank_within(values: Sequence[float]) -> list[float]:
    """Average ranks, so a partition's rows can be pooled with the others."""

    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


# ----- P1: two-way decomposition --------------------------------------------


def variance_shares(matrix: Mapping[tuple[str, str], float]) -> dict[str, Any]:
    """`decompose` plus the sum-of-squares split of `analysis/plot_sp_cross.py`.

    The sums are taken per observation, which on a complete grid is exactly the
    balanced formula (`n_payloads * sum alpha^2`) and on a ragged grid is its
    generalisation; `share_sum` says how far the three terms are from
    partitioning the total, which they do exactly only when the grid is
    complete.
    """

    fit = decompose(matrix)
    keys = sorted(matrix)
    ss_schedule = sum(fit["alpha"][row] ** 2 for row, _ in keys)
    ss_payload = sum(fit["beta"][payload] ** 2 for _, payload in keys)
    ss_gamma = sum(value ** 2 for value in fit["gamma_pairs"].values())
    ss_total = sum((matrix[key] - fit["mu"]) ** 2 for key in keys)
    shares = {
        "schedule": ss_schedule / ss_total if ss_total > 0 else None,
        "payload": ss_payload / ss_total if ss_total > 0 else None,
        "interaction": ss_gamma / ss_total if ss_total > 0 else None,
    }
    out = {key: value for key, value in fit.items() if key != "gamma_pairs"}
    out["ss"] = {"schedule": ss_schedule, "payload": ss_payload,
                 "interaction": ss_gamma, "total": ss_total}
    out["share"] = shares
    out["share_sum"] = (
        None if any(value is None for value in shares.values())
        else float(sum(shares.values()))
    )
    out["interaction_share_above_flag"] = (
        None if shares["interaction"] is None else bool(shares["interaction"] > GAMMA_SHARE_FLAG)
    )
    out["gamma_flag_threshold"] = GAMMA_SHARE_FLAG
    return out


def cross_matrix(cells, rows: Sequence[str], budget: str, metric: str,
                 *, payloads: Sequence[str] = PAYLOADS,
                 datasets: Sequence[str] = DATASETS) -> dict[tuple[str, str], float]:
    """(row, payload) -> mean over the datasets of the per-video means."""

    out: dict[tuple[str, str], float] = {}
    for row in rows:
        for payload in payloads:
            means = dataset_means(cells, row, payload, budget, metric)
            means = {key: value for key, value in means.items() if key in datasets}
            if len(means) != len(datasets):
                continue
            out[(row, payload)] = float(statistics.fmean(means.values()))
    return out


# ----- P4: geometry predictors ----------------------------------------------


def gap_costs(bits: str, rho: np.ndarray, sigmas: np.ndarray) -> list[float]:
    """Per-gap cost `sum_n rho2[n] * |sigma_anchor - sigma_n|`, the definition
    `analysis/density_form_test.measure` uses for its coefficient of variation."""

    from analysis.density_form_test import gaps_of

    costs = []
    for first, last in gaps_of(bits):
        index = np.arange(first, last + 1)
        anchor = sigmas[max(first - 1, 0)]
        costs.append(float(np.sum(rho[index] * np.abs(anchor - sigmas[index]))))
    return costs


def geometry_predictors(bits: str, record: Mapping[str, Any], budget: str,
                        geometry: Mapping[str, Any],
                        universe: Sequence[str], meancache_bits: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {"bits": bits, "cache_count": bits.count("1")}
    sigmas = geometry.get("sigmas")
    for dataset, rho in sorted(geometry.get("rho2", {}).items()):
        if sigmas is None:
            continue
        stats = measure(bits, rho, sigmas)
        costs = gap_costs(bits, rho, sigmas)
        suffix = "" if dataset == MAIN_RHO2_DATASET else f"_{dataset}"
        out[f"rho2_cost_cv{suffix}"] = float(stats["cost_cv"])
        out[f"rho2_total_cost{suffix}"] = float(sum(costs))
        out[f"n_gaps{suffix}"] = int(stats["n_gaps"])
        out[f"longest_gap_sigma_span{suffix}"] = float(max(stats["span"])) if len(stats["span"]) else None
    for name in ("hamming_to_meancache", "transpositions_to_meancache", "hamming_to_budcache",
                 "first_cache_step", "longest_cached_run", "max_full_step_gap"):
        text = record.get(name, "")
        out[name] = int(text) if str(text).strip() != "" else None
    if sigmas is not None:
        cached = [i for i, bit in enumerate(bits) if bit == "1"]
        out["cache_sigma_centroid"] = (
            float(np.mean([sigmas[i] for i in cached])) if cached else None
        )
    point = {bits: 1.0}
    for gate in GATES:
        distribution = geometry.get("gate_dists", {}).get((gate, budget))
        out[f"w_to_{gate}"] = None if not distribution else float(wasserstein(point, distribution))
    if universe:
        out["min_hamming_to_universe"] = int(min(hamming(bits, path) for path in universe))
        if meancache_bits is not None:
            nodes = sorted(set(universe) | {bits, meancache_bits})
            index = {path: i for i, path in enumerate(nodes)}
            for radius in BASIN_RADII:
                groups = components(nodes, radius)
                same = any(index[bits] in group and index[meancache_bits] in group
                           for group in groups)
                out[f"same_basin_as_meancache_r{radius}"] = int(same)
    return out


PREDICTOR_KEYS = (
    "rho2_cost_cv", "rho2_total_cost", "n_gaps", "longest_gap_sigma_span",
    "hamming_to_meancache", "transpositions_to_meancache", "hamming_to_budcache",
    "first_cache_step", "longest_cached_run", "max_full_step_gap", "cache_sigma_centroid",
    "w_to_seacache", "w_to_teacache", "w_to_sencache", "w_to_dicache",
    "min_hamming_to_universe", "same_basin_as_meancache_r4",
    "same_basin_as_meancache_r6", "same_basin_as_meancache_r8",
    # The same four gap statistics read off the other reference population, so
    # the choice of population can be seen rather than assumed.
    "rho2_cost_cv_penguin599", "rho2_total_cost_penguin599",
    "n_gaps_penguin599", "longest_gap_sigma_span_penguin599",
)


# ----- per-partition analysis ------------------------------------------------


def analyse_partition(backbone: str, budget: str, cells, manifest, native,
                      geometry, paths=None, mean_vel_note=None) -> dict[str, Any]:
    rows_present = [row for row in ROW_ORDER
                    if any(key[0] == row and key[3] == budget for key in cells)]
    on_budget = [row for row in rows_present
                 if manifest.get((budget, row), {}).get("off_budget", "0") == "0"]
    off_budget = [row for row in rows_present if row not in on_budget]
    payloads_present = [payload for payload in PAYLOADS
                        if any(key[1] == payload and key[3] == budget for key in cells)]
    paths = paths or {}

    # ---- payload purity: does the row leave the payload its warm-up steps?
    warmup = PAYLOAD_WARMUP.get(backbone, {})
    impure: dict[str, list[str]] = {}
    first_steps: dict[str, int] = {}
    for row in rows_present:
        record = manifest.get((budget, row)) or {}
        text = str(record.get("first_cache_step", "")).strip()
        if text == "":
            continue
        first_steps[row] = int(text)
        for payload in PAYLOADS:
            need = warmup.get(payload)
            if need is None or need <= first_steps[row]:
                continue
            if not any((row, payload, dataset, budget) in cells for dataset in DATASETS):
                continue
            impure.setdefault(row, []).append(payload)

    panel: dict[str, Any] = {
        "backbone": backbone,
        "budget": budget,
        "rows": rows_present,
        "rows_off_budget": off_budget,
        "payloads": payloads_present,
        "n_cells": sum(1 for key in cells
                       if key[3] == budget and key[1] in PAYLOADS) // len(DATASETS),
        "first_cache_step": first_steps,
        "payload_warmup": {payload: warmup.get(payload) for payload in PAYLOADS},
        "impure_cells": impure,
        "n_impure_cells": sum(len(values) for values in impure.values()),
        "mean_vel_column": {
            "homogeneous": bool((mean_vel_note or {}).get("applied")),
            "global_span_rows": sorted(
                entry["row"] for entry in (mean_vel_note or {}).get("cells_substituted", [])
                if entry["budget"] == budget) or None,
        },
    }

    def is_impure(row: str, payload: str) -> bool:
        return payload in impure.get(row, ())

    # ---- cross matrices, all metrics
    cross: dict[str, Any] = {}
    for metric in METRICS:
        table: dict[str, Any] = {}
        for row in rows_present:
            entry: dict[str, Any] = {}
            for payload in PAYLOADS:
                means = dataset_means(cells, row, payload, budget, metric)
                if not means:
                    continue
                joint = series(cells, row, payload, budget, metric)
                entry[payload] = {
                    "mean": float(statistics.fmean(means.values())),
                    "per_dataset": means,
                    "n_videos": len(joint),
                    "impure": is_impure(row, payload),
                }
            if entry:
                table[row] = entry
        cross[metric] = table
    panel["cross"] = cross

    # ---- the searched-span variant of the interval-mean-velocity payload, when
    # the column has been homogenised onto the global span.
    searched: dict[str, Any] = {}
    for metric in METRICS:
        left = series(cells, MEAN_VEL_NATIVE_ROW, MEAN_VEL_SEARCHED, budget, metric)
        right = series(cells, MEAN_VEL_NATIVE_ROW, MEAN_VEL, budget, metric)
        if not left or not right:
            continue
        diff = paired_difference(left, right)
        if diff is None:
            continue
        diff["verdict"] = classify(diff)
        searched[metric] = {
            "row": MEAN_VEL_NATIVE_ROW,
            "searched_span_mean": float(statistics.fmean(
                dataset_means(cells, MEAN_VEL_NATIVE_ROW, MEAN_VEL_SEARCHED, budget, metric).values())),
            "global_span_mean": float(statistics.fmean(
                dataset_means(cells, MEAN_VEL_NATIVE_ROW, MEAN_VEL, budget, metric).values())),
            "paired_diff": diff,
        }
    panel["mean_vel_searched_span"] = searched

    # ---- P1
    p1: dict[str, Any] = {"grids": {}}
    for grid_name, grid_rows in (("full", on_budget), ("w1", list(W1_ROWS))):
        grid_rows = [row for row in grid_rows if row in rows_present]
        per_metric: dict[str, Any] = {}
        for metric in METRICS:
            matrix = cross_matrix(cells, grid_rows, budget, metric)
            if len(matrix) < 6:
                continue
            try:
                fit = variance_shares(matrix)
            except ValueError as error:
                per_metric[metric] = {"error": str(error)}
                continue
            fit["matrix"] = {f"{row}x{payload}": value for (row, payload), value in sorted(matrix.items())}
            per_metric[metric] = fit
        p1["grids"][grid_name] = {"rows": grid_rows, "metrics": per_metric}
    panel["P1"] = p1

    alpha_by_metric = {
        metric: p1["grids"]["full"]["metrics"].get(metric, {}).get("alpha", {})
        for metric in METRICS
    }

    # ---- the matrix three-stream band, shared by P2 and P3
    def stream_band(metric: str) -> dict[str, Any]:
        variances = []
        detail: dict[str, float] = {}
        for method in GATES:
            for dataset in DATASETS:
                streams = MATRIX_STREAMS[dataset]
                common = [
                    index for index in range(150)
                    if all(native.get((method, dataset, budget, seed, index)) is not None
                           for seed in streams)
                ]
                if len(common) < 50:
                    continue
                means = [
                    float(statistics.fmean(
                        native[(method, dataset, budget, seed, index)][metric] for index in common))
                    for seed in streams
                ]
                sd = float(statistics.stdev(means))
                detail[f"{method}/{dataset}"] = sd
                variances.append(sd ** 2)
        if not variances:
            return {"band": None, "per_cell_sd": detail, "n_cells": 0}
        pooled = math.sqrt(sum(variances) / len(variances))
        return {"band": 2.0 * pooled, "pooled_sd": pooled,
                "per_cell_sd": detail, "n_cells": len(variances),
                "n_streams": 3}

    stream_bands = {metric: stream_band(metric) for metric in METRICS}

    # ---- P2: band around the MeanCache row inside the reuse column
    def band_block(metric: str, datasets: Sequence[str]) -> dict[str, Any] | None:
        anchor = series(cells, ANCHOR_ROW, "reuse", budget, metric)
        if not anchor:
            return None
        entries: dict[str, Any] = {}
        for row in rows_present:
            if row == ANCHOR_ROW:
                continue
            other = series(cells, row, "reuse", budget, metric)
            if not other:
                continue
            diff = paired_difference(other, anchor, datasets=datasets)
            if diff is None:
                continue
            diff["verdict"] = classify(diff)
            diff["off_budget"] = row in off_budget
            entries[row] = diff
        scored = {row: entry for row, entry in entries.items() if not entry["off_budget"]}
        in_band = [row for row, entry in scored.items() if entry["verdict"] == "indistinguishable"]
        better = [row for row, entry in scored.items() if entry["verdict"] == "better"]
        worse = [row for row, entry in scored.items() if entry["verdict"] == "worse"]
        # The +- 2 paired SE band is the resolution of 300 paired prompts; it is
        # much narrower than the matrix's own seed-stream spread, so the second
        # count says how many rows sit within THAT width of the anchor.
        stream = stream_bands[metric]["band"]
        for row, entry in entries.items():
            entry["within_stream_band"] = (
                None if stream is None else bool(entry["mean"] >= -stream))
        near = [row for row, entry in scored.items() if entry.get("within_stream_band")]
        return {
            "anchor": ANCHOR_ROW,
            "datasets": list(datasets),
            "rows": entries,
            "n_rows_scored": len(scored) + 1,
            # Counts excluding the anchor itself, which is trivially in its own
            # band; the `*_with_anchor` variants keep the older convention.
            "n_rows_compared": len(scored),
            "n_indistinguishable_excl_anchor": len(in_band),
            "n_within_stream_band_excl_anchor": len(near),
            "n_indistinguishable": len(in_band) + 1,
            "n_not_worse": len(in_band) + len(better) + 1,
            "rows_indistinguishable": sorted(in_band),
            "rows_better": sorted(better),
            "rows_worse": sorted(worse),
            "stream_band": stream,
            "n_within_stream_band": None if stream is None else len(near) + 1,
            "rows_within_stream_band": sorted(near),
            "controls": {
                row: entries[row]["verdict"] for row in CONTROL_ROWS if row in entries
            },
            "controls_within_stream_band": {
                row: entries[row].get("within_stream_band")
                for row in CONTROL_ROWS if row in entries
            },
            "controls_all_outside": all(
                entries[row]["verdict"] == "worse" for row in CONTROL_ROWS if row in entries
            ) if any(row in entries for row in CONTROL_ROWS) else None,
        }

    panel["P2"] = {
        "pooled": {metric: band_block(metric, DATASETS) for metric in METRICS},
        "by_dataset": {
            dataset: {metric: band_block(metric, (dataset,)) for metric in METRICS}
            for dataset in DATASETS
        },
    }

    # ---- P3: gate rows against their native per-prompt runs
    def native_series(method: str, metric: str, seeds: Mapping[str, int] | None = None
                      ) -> dict[tuple[str, int], float]:
        out: dict[tuple[str, int], float] = {}
        for dataset in DATASETS:
            seed = (seeds or {dataset: MATRIX_STREAMS[dataset][0]})[dataset]
            for index in range(150):
                values = native.get((method, dataset, budget, seed, index))
                if values is not None:
                    out[(dataset, index)] = values[metric]
        return out

    # Which prompts did each gate happen to walk its own modal path on? On those
    # the fixed-schedule run and the gate run are the same computation, so their
    # paired difference is a deterministic zero that dilutes the pooled mean.
    modal_keys: dict[str, set[tuple[str, int]]] = {}
    walked_keys: dict[str, set[tuple[str, int]]] = {}
    for row in rows_present:
        method = GATE_ROW_METHOD.get(row)
        bits = (manifest.get((budget, row)) or {}).get("bits")
        if method is None or not bits or not paths:
            continue
        same: set[tuple[str, int]] = set()
        seen: set[tuple[str, int]] = set()
        for dataset in DATASETS:
            seed = MATRIX_STREAMS[dataset][0]
            for index in range(150):
                walked = paths.get((method, dataset, budget, seed, index))
                if walked is None:
                    continue
                seen.add((dataset, index))
                if walked == bits:
                    same.add((dataset, index))
        modal_keys[row] = same
        walked_keys[row] = seen

    def p3_block(metric: str, datasets: Sequence[str]) -> dict[str, Any]:
        band_info = stream_bands[metric]
        band = band_info["band"]
        rows_out: dict[str, Any] = {}
        for row in rows_present:
            method = GATE_ROW_METHOD.get(row)
            if method is None:
                continue
            reference = native_series(method, metric)
            seen = {key for key in walked_keys.get(row, set()) if key[0] in datasets}
            same = {key for key in modal_keys.get(row, set()) if key[0] in datasets}
            entry: dict[str, Any] = {
                "method": method,
                "native_payload": GATE_NATIVE_PAYLOAD[row],
                "off_budget": row in off_budget,
                "modal_path_known": bool(seen),
                "n_prompts_with_path": len(seen),
                "n_prompts_on_modal_path": len(same),
                "modal_share": (len(same) / len(seen)) if seen else None,
            }
            for label, payload in (("native_payload_cell", GATE_NATIVE_PAYLOAD[row]),
                                   ("reuse_cell", "reuse")):
                if label == "reuse_cell" and GATE_NATIVE_PAYLOAD[row] == "reuse":
                    entry[label] = None
                    continue
                fixed = series(cells, row, payload, budget, metric)
                diff = paired_difference(fixed, reference, datasets=datasets)
                if diff is None:
                    entry[label] = None
                    continue
                diff["payload"] = payload
                diff["within_band"] = None if band is None else bool(abs(diff["mean"]) <= band)
                diff["verdict"] = classify(diff)
                entry[label] = diff
                if label != "native_payload_cell" or not seen:
                    continue
                # The primary reading: only the prompts the gate did NOT walk
                # its modal path on. The complement is reported beside it, and
                # is a deterministic zero whenever the two runs coincide.
                for name, subset in (("off_modal_cell", seen - same),
                                     ("on_modal_cell", same)):
                    part = paired_difference(fixed, reference, datasets=datasets,
                                             restrict=subset)
                    if part is None:
                        entry[name] = None
                        continue
                    part["payload"] = payload
                    part["within_band"] = (
                        None if band is None else bool(abs(part["mean"]) <= band))
                    part["verdict"] = classify(part)
                    part["max_abs"] = max(
                        abs(fixed[key] - reference[key])
                        for key in sorted(subset & set(fixed) & set(reference))
                    ) if subset & set(fixed) & set(reference) else None
                    entry[name] = part
            for anchor_row in ("budcache", ANCHOR_ROW):
                payload = GATE_NATIVE_PAYLOAD[row]
                left = series(cells, row, payload, budget, metric)
                right = series(cells, anchor_row, payload, budget, metric)
                diff = paired_difference(left, right, datasets=datasets)
                if diff is not None:
                    diff["payload"] = payload
                    diff["verdict"] = classify(diff)
                entry[f"vs_{anchor_row}"] = diff
            rows_out[row] = entry
        return {"band_2sd": band, "band_detail": band_info,
                "datasets": list(datasets), "rows": rows_out}

    panel["P3"] = {
        "pooled": {metric: p3_block(metric, DATASETS) for metric in METRICS},
        "by_dataset": {
            dataset: {metric: p3_block(metric, (dataset,)) for metric in METRICS}
            for dataset in DATASETS
        },
    }

    # ---- the two datasets read separately: does any verdict flip?
    def robustness() -> dict[str, Any]:
        out: dict[str, Any] = {"P2": {}, "P3": {}}
        for metric in METRICS:
            left = panel["P2"]["by_dataset"][DATASETS[0]][metric]
            right = panel["P2"]["by_dataset"][DATASETS[1]][metric]
            if left and right:
                shared_rows = sorted(set(left["rows"]) & set(right["rows"]))
                flips = [row for row in shared_rows
                         if left["rows"][row]["verdict"] != right["rows"][row]["verdict"]]
                stream_flips = [row for row in shared_rows
                                if left["rows"][row].get("within_stream_band")
                                != right["rows"][row].get("within_stream_band")]
                order_left = [left["rows"][row]["mean"] for row in shared_rows]
                order_right = [right["rows"][row]["mean"] for row in shared_rows]
                out["P2"][metric] = {
                    "n_rows": len(shared_rows),
                    "verdict_flips": flips,
                    "stream_band_flips": stream_flips,
                    "row_order_spearman": spearman(order_left, order_right),
                    "n_within_stream_band": {
                        DATASETS[0]: left["n_within_stream_band"],
                        DATASETS[1]: right["n_within_stream_band"],
                    },
                }
            left3 = panel["P3"]["by_dataset"][DATASETS[0]][metric]["rows"]
            right3 = panel["P3"]["by_dataset"][DATASETS[1]][metric]["rows"]
            shared_gates = sorted(set(left3) & set(right3))
            flips = []
            off_modal_flips = []
            for row in shared_gates:
                a = (left3[row].get("native_payload_cell") or {}).get("within_band")
                b = (right3[row].get("native_payload_cell") or {}).get("within_band")
                if a != b:
                    flips.append(row)
                a = (left3[row].get("off_modal_cell") or {}).get("within_band")
                b = (right3[row].get("off_modal_cell") or {}).get("within_band")
                if a is not None and b is not None and a != b:
                    off_modal_flips.append(row)
            out["P3"][metric] = {
                "n_rows": len(shared_gates),
                "within_band_flips": flips,
                "off_modal_within_band_flips": off_modal_flips,
                "per_dataset_mean": {
                    row: {
                        DATASETS[0]: (left3[row].get("native_payload_cell") or {}).get("mean"),
                        DATASETS[1]: (right3[row].get("native_payload_cell") or {}).get("mean"),
                    }
                    for row in shared_gates
                },
            }
        return out

    panel["robustness"] = robustness()

    # ---- P4 predictors + within-partition Spearman
    universe = sorted({
        path
        for (gate, gate_budget), distribution in geometry.get("gate_dists", {}).items()
        if gate_budget == budget
        for path in distribution
    })
    meancache_bits = manifest.get((budget, ANCHOR_ROW), {}).get("bits")
    predictors: dict[str, Any] = {}
    for row in rows_present:
        record = manifest.get((budget, row))
        if record is None or not record.get("bits"):
            continue
        predictors[row] = geometry_predictors(
            record["bits"], record, budget, geometry, universe, meancache_bits)

    quality: dict[str, Any] = {}
    for metric in METRICS:
        reuse_mean = {}
        for row in rows_present:
            means = dataset_means(cells, row, "reuse", budget, metric)
            if len(means) == len(DATASETS):
                reuse_mean[row] = float(statistics.fmean(means.values()))
        quality[metric] = {"alpha": alpha_by_metric.get(metric, {}), "reuse_column_mean": reuse_mean}

    spearman_block: dict[str, Any] = {}
    for metric in METRICS:
        per_quality: dict[str, Any] = {}
        for quality_name in ("alpha", "reuse_column_mean"):
            values = quality[metric][quality_name]
            rows_used = [row for row in sorted(values) if row in predictors and row not in off_budget]
            per_predictor: dict[str, Any] = {}
            for predictor in PREDICTOR_KEYS:
                xs = [predictors[row].get(predictor) for row in rows_used]
                ys = [values[row] for row in rows_used]
                per_predictor[predictor] = spearman(xs, ys)
            per_quality[quality_name] = {"rows": rows_used, "spearman": per_predictor}
        spearman_block[metric] = per_quality
    panel["P4"] = {"predictors": predictors, "quality": quality, "spearman": spearman_block,
                   "universe_size": len(universe)}

    # ---- P4 dose curves: transpositions away from the MeanCache path. Two
    # ladders, because the free one also moves the first cached step earlier.
    ladders: dict[str, Any] = {}
    for ladder_name, ladder_rows in DOSE_LADDERS.items():
        if not any(row in rows_present for row, _ in ladder_rows if row != ANCHOR_ROW):
            continue
        dose: dict[str, Any] = {}
        for metric in METRICS:
            anchor = series(cells, ANCHOR_ROW, "reuse", budget, metric)
            points = []
            for row, distance in ladder_rows:
                if row not in rows_present:
                    continue
                record = manifest.get((budget, row), {})
                fixed = series(cells, row, "reuse", budget, metric)
                diff = paired_difference(fixed, anchor)
                points.append({
                    "row": row,
                    "hamming_planned": distance,
                    "hamming_actual": int(record["hamming_to_meancache"]) if record.get("hamming_to_meancache") else 0,
                    "first_cache_step": first_steps.get(row),
                    "first_cache_step_shift": (
                        None if row not in first_steps or ANCHOR_ROW not in first_steps
                        else first_steps[ANCHOR_ROW] - first_steps[row]),
                    "paired_diff": diff,
                    "payload_means": {
                        payload: (lambda means: float(statistics.fmean(means.values())) if len(means) == len(DATASETS) else None)(
                            dataset_means(cells, row, payload, budget, metric))
                        for payload in PAYLOADS
                    },
                })
            dose[metric] = points
        shifts = [point["first_cache_step_shift"]
                  for point in dose[PRIMARY_METRIC]
                  if point["first_cache_step_shift"] is not None]
        ladders[ladder_name] = {
            "rows": [row for row, _ in ladder_rows if row in rows_present],
            "metrics": dose,
            "first_cache_step_preserved": bool(shifts) and all(value == 0 for value in shifts),
            "max_first_cache_step_shift": max(shifts) if shifts else None,
        }
    panel["P4"]["dose_ladders"] = ladders
    # Kept under its old name so nothing that reads the free ladder breaks.
    panel["P4"]["dose_curve"] = (
        ladders.get("free", {}).get("metrics") or {metric: [] for metric in METRICS})

    # ---- P5: the argmax payload of every row
    p5: dict[str, Any] = {}
    for metric in METRICS:
        per_row: dict[str, Any] = {}
        for row in rows_present:
            values = {}
            dropped = []
            for payload in PAYLOADS:
                means = dataset_means(cells, row, payload, budget, metric)
                if len(means) != len(DATASETS):
                    continue
                if is_impure(row, payload):
                    # The row caches before this payload's construction exists,
                    # so the cell is not the payload it is labelled with and
                    # cannot compete for this row's best payload.
                    dropped.append(payload)
                    continue
                values[payload] = float(statistics.fmean(means.values()))
            if len(values) < 2:
                continue
            best = max(values, key=lambda payload: values[payload])
            ordered = sorted(values, key=lambda payload: values[payload], reverse=True)
            per_row[row] = {
                "argmax": best,
                "n_payloads": len(values),
                "values": values,
                "payloads_excluded_impure": dropped,
                "margin_over_second": (
                    float(values[ordered[0]] - values[ordered[1]]) if len(ordered) >= 2 else None
                ),
            }
        if not per_row:
            continue
        votes = Counter(entry["argmax"] for entry in per_row.values())
        majority, majority_count = votes.most_common(1)[0]
        p5[metric] = {
            "rows": per_row,
            "votes": dict(votes),
            "majority": majority,
            "n_majority": majority_count,
            "n_rows": len(per_row),
            "rows_disagreeing": sorted(row for row, entry in per_row.items()
                                       if entry["argmax"] != majority),
            "payload_marginal": {
                payload: (lambda vals: float(statistics.fmean(vals)) if vals else None)(
                    [entry["values"][payload] for entry in per_row.values()
                     if payload in entry["values"]])
                for payload in PAYLOADS
            },
        }
    panel["P5"] = p5

    # ---- P6: the DP-rho2 row against MeanCache and against the random rows
    p6: dict[str, Any] = {}
    for metric in METRICS:
        if "dp_rho2" not in rows_present:
            continue
        left = series(cells, "dp_rho2", "reuse", budget, metric)
        entry: dict[str, Any] = {"payload": "reuse"}
        for other in (ANCHOR_ROW, "budcache", "rand_1", "rand_2", "uniform"):
            if other not in rows_present:
                continue
            diff = paired_difference(left, series(cells, other, "reuse", budget, metric))
            if diff is not None:
                diff["verdict"] = classify(diff)
            entry[f"vs_{other}"] = diff
        p6[metric] = entry
    panel["P6"] = p6
    return panel


def pooled_geometry(panels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """P4 across partitions: rank within partition, then Spearman over all rows."""

    out: dict[str, Any] = {}
    for metric in METRICS:
        per_quality: dict[str, Any] = {}
        for quality_name in ("alpha", "reuse_column_mean"):
            pooled_quality: list[float] = []
            pooled_predictor: dict[str, list[float]] = {key: [] for key in PREDICTOR_KEYS}
            partitions_used = []
            for panel in panels:
                values = panel["P4"]["quality"][metric][quality_name]
                predictors = panel["P4"]["predictors"]
                rows = [row for row in sorted(values)
                        if row in predictors and row not in panel["rows_off_budget"]]
                if len(rows) < 4:
                    continue
                partitions_used.append(panel["budget"])
                quality_ranks = rank_within([values[row] for row in rows])
                pooled_quality.extend(quality_ranks)
                for key in PREDICTOR_KEYS:
                    xs = [predictors[row].get(key) for row in rows]
                    if any(value is None for value in xs):
                        pooled_predictor[key].extend([None] * len(rows))
                    else:
                        pooled_predictor[key].extend(rank_within(xs))
            per_quality[quality_name] = {
                "partitions": partitions_used,
                "n_rows": len(pooled_quality),
                "spearman": {
                    key: spearman(values, pooled_quality)
                    for key, values in pooled_predictor.items()
                },
            }
        out[metric] = per_quality
    return out


def analyse_backbone(backbone: str) -> dict[str, Any] | None:
    cells = load_cells(backbone)
    if not cells:
        return None
    mean_vel_note = homogenise_mean_vel(cells)
    manifest = load_manifest(backbone)
    native = load_native_pervideo(backbone)
    geometry = load_geometry_inputs(backbone)
    paths = load_perprompt_paths(backbone)
    budgets = [budget for budget in KS if any(key[3] == budget for key in cells)]
    panels = [analyse_partition(backbone, budget, cells, manifest, native, geometry,
                                paths=paths, mean_vel_note=mean_vel_note)
              for budget in budgets]
    prompts = sorted({index for entry in cells.values() for index in entry["per"]})
    out: dict[str, Any] = {
        "backbone": backbone,
        "mean_vel_column": mean_vel_note,
        "native_paths_available": bool(paths),
        # Evaluation files behind the cross itself: one per (cell, dataset),
        # primary payloads only. The searched-span cells the mean_vel
        # homogenisation parked aside are counted apart, not twice.
        "n_cell_files": sum(1 for key in cells if key[1] in PAYLOADS),
        "n_searched_span_files": sum(1 for key in cells if key[1] == MEAN_VEL_SEARCHED),
        "n_schedule_payload_cells": len({(key[0], key[1], key[3]) for key in cells
                                         if key[1] in PAYLOADS}),
        "ladder_rejections": load_ladder_rejections(backbone),
        "n_videos": sum(entry["n_per_video"] for key, entry in cells.items()
                        if key[1] in PAYLOADS),
        "datasets": sorted({key[2] for key in cells}),
        "prompt_indices": {"min": min(prompts), "max": max(prompts), "n": len(prompts)},
        "seed_streams": {
            dataset: sorted({entry["seed"] for key, entry in cells.items() if key[2] == dataset})
            for dataset in sorted({key[2] for key in cells})
        },
        "native_reference_available": bool(native),
        "geometry_available": bool(geometry.get("rho2")),
        "partitions": {panel["budget"]: panel for panel in panels},
        "P4_pooled": pooled_geometry(panels),
        "impure_cells": {
            "n_total": sum(panel["n_impure_cells"] for panel in panels),
            "by_partition": {panel["budget"]: panel["impure_cells"]
                             for panel in panels if panel["impure_cells"]},
            "warmup": PAYLOAD_WARMUP.get(backbone, {}),
        },
    }
    wall = load_wall_seconds(backbone)
    if wall is not None:
        out["wall"] = wall
    return out


# ----- markdown tables -------------------------------------------------------


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_tables(report: Mapping[str, Any], path: Path) -> None:
    lines: list[str] = ["# Video SPX numeric tables", ""]
    for backbone, block in sorted(report["backbones"].items()):
        lines.append(f"## {backbone}")
        lines.append("")
        lines.append(f"cells: {block['n_schedule_payload_cells']} schedule x payload, "
                     f"{block['n_cell_files']} evaluation files, {block['n_videos']} videos")
        lines.append("")
        for budget, panel in sorted(panel_items(block)):
            lines.append(f"### {backbone} {budget}")
            lines.append("")
            table = panel["cross"][PRIMARY_METRIC]
            lines.append("PSNR cross matrix (dB, mean over both datasets of the per-video means, "
                         "n = 300 videos per cell):")
            lines.append("")
            lines.append("| row | " + " | ".join(PAYLOADS) + " |")
            lines.append("|---|" + "---|" * len(PAYLOADS))
            for row in panel["rows"]:
                entry = table.get(row, {})
                lines.append("| " + row + " | " + " | ".join(
                    format_number(entry.get(payload, {}).get("mean"), 2) for payload in PAYLOADS) + " |")
            lines.append("")
            for grid in ("full", "w1"):
                fit = panel["P1"]["grids"][grid]["metrics"].get(PRIMARY_METRIC)
                if not fit or "share" not in fit:
                    continue
                share = fit["share"]
                lines.append(
                    f"variance shares ({grid} grid, PSNR): schedule "
                    f"{share['schedule'] * 100:.1f} %, payload {share['payload'] * 100:.1f} %, "
                    f"interaction {share['interaction'] * 100:.1f} %, total sum of squares "
                    f"{fit['ss']['total']:.1f} dB^2, cells {fit['n_obs']}")
                lines.append("")
            band = panel["P2"]["pooled"][PRIMARY_METRIC]
            if band:
                lines.append("P2 paired differences against the MeanCache row, reuse column, PSNR dB:")
                lines.append("")
                lines.append("| row | mean | 2 x SE | verdict | n pairs |")
                lines.append("|---|---|---|---|---|")
                for row, entry in sorted(band["rows"].items(),
                                         key=lambda item: -item[1]["mean"]):
                    lines.append(f"| {row} | {entry['mean']:+.3f} | {2 * entry['se']:.3f} | "
                                 f"{entry['verdict']} | {entry['n_pairs']} |")
                lines.append("")
            p3 = panel["P3"]["pooled"][PRIMARY_METRIC]
            if p3["rows"]:
                lines.append(f"P3 fixed modal path minus native gate, PSNR dB "
                             f"(band = {format_number(p3['band_2sd'])} dB):")
                lines.append("")
                lines.append("| row | payload | modal share | off-modal mean | off-modal n | "
                             "off-modal in band | all mean | all in band | vs BudCache | vs MeanCache |")
                lines.append("|---|" + "---|" * 9)
                for row, entry in sorted(p3["rows"].items()):
                    cell = entry.get("native_payload_cell")
                    if cell is None:
                        continue
                    off = entry.get("off_modal_cell") or {}
                    lines.append(
                        f"| {row} | {cell['payload']} | {format_number(entry.get('modal_share'))} | "
                        f"{format_number(off.get('mean'))} | {format_number(off.get('n_pairs'))} | "
                        f"{format_number(off.get('within_band'))} | "
                        f"{cell['mean']:+.3f} | {format_number(cell['within_band'])} | "
                        f"{format_number((entry.get('vs_budcache') or {}).get('mean'))} | "
                        f"{format_number((entry.get('vs_meancache') or {}).get('mean'))} |")
                lines.append("")
            p5 = panel["P5"].get(PRIMARY_METRIC)
            if p5:
                lines.append(f"P5 argmax payload: majority {p5['majority']} "
                             f"({p5['n_majority']}/{p5['n_rows']} rows); disagreeing rows: "
                             f"{', '.join(p5['rows_disagreeing']) or 'none'}")
                lines.append("")
            p6 = panel["P6"].get(PRIMARY_METRIC)
            if p6:
                parts = []
                for key, entry in sorted(p6.items()):
                    if key.startswith("vs_") and entry:
                        parts.append(f"{key[3:]} {entry['mean']:+.3f} ({entry['verdict']})")
                lines.append("P6 DP-rho2 row paired differences, reuse column, PSNR dB: " + "; ".join(parts))
                lines.append("")
        pooled = block["P4_pooled"][PRIMARY_METRIC]["reuse_column_mean"]
        lines.append(f"### {backbone} P4 pooled over {len(pooled['partitions'])} partitions "
                     f"({pooled['n_rows']} rows, ranks within partition)")
        lines.append("")
        lines.append("| predictor | Spearman | p |")
        lines.append("|---|---|---|")
        for key, entry in pooled["spearman"].items():
            if entry is None:
                lines.append(f"| {key} | - | - |")
            else:
                lines.append(f"| {key} | {entry['rho']:+.3f} | {entry['p']:.4f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def panel_items(block: Mapping[str, Any]):
    order = {budget: i for i, budget in enumerate(KS)}
    return sorted(block["partitions"].items(), key=lambda item: order.get(item[0], 99))


# ----- entry point -----------------------------------------------------------


def manifest_totals() -> dict[str, Any]:
    """What the frozen schedule manifest defines, over every backbone in it.

    `n_cells` is the manifest's own per-row cell count, which is the frozen
    design: it already excludes both payload-infeasible cells and the columns a
    row deliberately does not define (off-budget rows and the reuse-only
    supplement rows). The two exclusions are staged apart so the report can
    tell a cell that cannot exist from a column the design chose not to run.
    """

    path = REPO / "resources/video_spx_schedules/manifest.tsv"
    per_backbone: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            entry = per_backbone.setdefault(
                record["backbone"],
                {"n_schedules": 0, "n_cells": 0, "infeasible": {}, "reuse_only": {}})
            entry["n_schedules"] += 1
            entry["n_cells"] += int(record["n_cells"])
            if record.get("off_budget") == "1":
                continue
            infeasible = [payload for payload in PAYLOADS
                          if record.get(f"feasible_{payload}") == "0"]
            if infeasible:
                entry["infeasible"].setdefault(record["budget"], {})[record["row"]] = infeasible
            elif int(record["n_cells"]) < len(PAYLOADS):
                # Every payload is feasible yet the design defines fewer
                # columns: a reuse-only supplement row (the *f ladder).
                entry["reuse_only"].setdefault(record["budget"], []).append(record["row"])
    return {
        "per_backbone": per_backbone,
        "n_schedules": sum(entry["n_schedules"] for entry in per_backbone.values()),
        "n_cells": sum(entry["n_cells"] for entry in per_backbone.values()),
    }


def build_report(backbones: Sequence[str]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "metrics": list(METRICS),
        "primary_metric": PRIMARY_METRIC,
        "orientation": ORIENTATION,
        "payloads": list(PAYLOADS),
        "row_order": list(ROW_ORDER),
        "w1_rows": list(W1_ROWS),
        "gate_row_method": GATE_ROW_METHOD,
        "gate_native_payload": GATE_NATIVE_PAYLOAD,
        "matrix_streams": MATRIX_STREAMS,
        "main_rho2_dataset": MAIN_RHO2_DATASET,
        "manifest": manifest_totals(),
        "backbones": {},
        "backbones_missing": [],
    }
    for backbone in backbones:
        block = analyse_backbone(backbone)
        if block is None:
            report["backbones_missing"].append(backbone)
            continue
        report["backbones"][backbone] = block
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones", nargs="*", default=list(BACKBONES))
    parser.add_argument("--output", type=Path,
                        default=REPO / "resources/video_spx/video_spx_results.json")
    parser.add_argument("--tables", type=Path,
                        default=REPO / "resources/video_spx/video_spx_tables.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.backbones)
    if not report["backbones"]:
        raise SystemExit("no video SPX cells found for any backbone")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False, sort_keys=False),
                           encoding="utf-8")
    write_tables(report, args.tables)
    for backbone, block in sorted(report["backbones"].items()):
        print(f"[video-spx] {backbone}: {block['n_schedule_payload_cells']} cells, "
              f"{block['n_videos']} videos, partitions {sorted(block['partitions'])}")
        for budget, panel in panel_items(block):
            fit = panel["P1"]["grids"]["full"]["metrics"].get(PRIMARY_METRIC, {})
            share = fit.get("share", {})
            band = panel["P2"]["pooled"][PRIMARY_METRIC]
            print(f"  {budget}: shares schedule/payload/interaction = "
                  f"{share.get('schedule', float('nan')) * 100:.1f} / "
                  f"{share.get('payload', float('nan')) * 100:.1f} / "
                  f"{share.get('interaction', float('nan')) * 100:.1f} %; "
                  f"band rows {band['n_indistinguishable'] if band else '-'} / "
                  f"{band['n_rows_scored'] if band else '-'}")
    if report["backbones_missing"]:
        print(f"[video-spx] no cells yet for: {', '.join(report['backbones_missing'])}")
    print(f"[video-spx] wrote {args.output}")
    print(f"[video-spx] wrote {args.tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
