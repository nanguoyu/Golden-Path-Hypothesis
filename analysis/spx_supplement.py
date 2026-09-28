#!/usr/bin/env python3
"""Image SPX supplement: P1 re-fit, per-prompt P4, and the four P-S questions.

Everything here runs off the two staged per-image tables
(`analysis/stage_spx_perprompt.py`), so a split, a subset or an error band is a
row filter rather than a re-read of the cluster:

    resources/spx/perprompt_spx_<model>.tsv.gz
    resources/spx/perprompt_native_<model>.tsv.gz

What it computes, per (model, K) partition and per metric, on the full 1,632
PartiPrompts and again on the 1,088 held-out ones:

P1   diagonal advantage. The formal fit is the balanced W1 grid (4 schedules x
     5 payloads, 20 cells, full rank); the ragged whole-panel fit and the GPF
     rows are kept beside it as appendix readings, because the GPF members were
     chosen after looking at held-out prompts and one of them starts caching at
     step 1, where the extrapolating payloads degenerate to reuse
     (`payload_purity` counts those cells rather than asserting how many).
     `random_reference` restates the schedule-axis spread against the two
     random exact-K rows, which is the level the original wave had no way to
     quote: the same alpha_s spread splits into what design buys over a random
     path and what a bad gate path loses against one.

P4   modal-path capacity, per prompt. A gate's fixed top-1 path and the gate's
     own dynamic run are the *same computation* on any prompt where the gate
     happens to walk the modal path, so those pairs contribute a deterministic
     zero and only dilute the mean. The pairs are split on the realised path
     and the off-modal subset is reported with its own paired 2 SE, next to the
     modal mass that says how much of the population was dropped.

P-S1 few good paths: every reuse-column row against the MeanCache row, paired
     per prompt; the two random rows give the no-design level.
P-S2 dose: the Hamming ladder's decay away from MeanCache.
P-S3 rho2 positive control: where the unscreened rho2-DP row lands.
P-S4 geometry: row quality against the same predictors the video side used
     (`analysis/video_spx.geometry_predictors`), per partition and pooled.

    python analysis/spx_supplement.py --out resources/sp_cross_supplement/sp_cross_results.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis.build_golden_path_family import read_population, risk_profiles  # noqa: E402
from analysis.density_form_test import measure  # noqa: E402
from analysis.sp_cross import (  # noqa: E402
    HIGHER_IS_BETTER,
    HOMOLOGOUS,
    clustered_paired_se,
    decompose,
    orient,
    p2_order,
    p3_high_compression,
    pooled_seed_band,
    two_sided_sign_p,
)
from analysis.video_path_universe import components, hamming, wasserstein  # noqa: E402
from analysis.video_spx import (  # noqa: E402
    gap_costs,
    paired_difference,
    rank_within,
    spearman,
)

# ----- fixed vocabulary ------------------------------------------------------

MODELS = ("flux", "qwen")
MODEL_LABEL = {"flux": "FLUX.1-dev", "qwen": "Qwen-Image"}
BUDGETS = (29, 37, 41)
METRICS = ("psnr", "ssim", "lpips", "image_reward", "clip")
MAIN_METRIC = "psnr"
PAYLOADS = ("reuse", "taylor_o1", "hermite_o2", "mean_avg_vel", "di_two_anchor")
#: The balanced sub-grid. Every one of these four rows has all five payloads,
#: so the additive fit is full rank and gamma is identified cell by cell.
W1 = ("budcache", "dpcache", "uniform", "dicache_top1")
GATES = ("seacache", "teacache", "sencache", "dicache")
GATE_ROW = {gate: f"{gate}_top1" for gate in GATES}
GPF_ROWS = ("gpf_reuse_e05_1", "gpf_o1_e15_1", "gpf_o1_e20_1")
#: The unconstrained reference level. Two draws per partition everywhere; the
#: four densified partitions (flux/qwen K29/K37) have five, which is what turns
#: "the random level is here" into "the random level is here, plus or minus".
RANDOM_ROWS = ("rand_1", "rand_2", "rand_3", "rand_4", "rand_5")
#: The dose ladders, as RUNGS rather than rows. A rung is a Hamming distance
#: from the anchor and holds every independent draw taken at that distance:
#: one on the six frozen partitions, three (`ham*f`) or two (`ham*`) on the
#: four densified ones. Reading a rung as a single number was the thing the
#: round-3 audit called a rough reading -- with n = 1 nothing separates the
#: dose from which draw came out of the stream.
LADDER_RUNGS = {
    "preserving": {
        2: ("ham2f", "ham2f_d2", "ham2f_d3"),
        4: ("ham4f", "ham4f_d2", "ham4f_d3"),
        8: ("ham8f", "ham8f_d2", "ham8f_d3"),
    },
    #: The free ladder exists only where it was densified. Its swaps may land
    #: before the anchor's first cached step, so it carries the confound the
    #: `f` ladder was built to remove -- both curves are reported, neither is
    #: treated as the other's correction.
    "free": {
        2: ("ham2_d1", "ham2_d2"),
        4: ("ham4_d1", "ham4_d2"),
        8: ("ham8_d1", "ham8_d2"),
    },
}
LADDER_FAMILIES = tuple(LADDER_RUNGS)
#: Flat membership, for the places that only need "is this a ladder row".
LADDER_ROWS = tuple(
    row for family in LADDER_RUNGS.values() for rows in family.values() for row in rows
)
ANCHOR_ROW = "meancache"
NUM_STEPS = 50
BASIN_RADII = (4, 6, 8)

SPX_DIR = _ROOT / "resources" / "spx"
SCHEDULE_DIR = _ROOT / "resources" / "sp_cross_schedules"
SUPPLEMENT_DIR = _ROOT / "resources" / "spx_supplement_schedules"
SPLITS_JSON = SCHEDULE_DIR / "parti_spx_splits.v1.json"
DISCOVERY = SCHEDULE_DIR / "discovery_path_counts.tsv"
TRAJ_TABLE = {
    model: _ROOT / f"resources/full_trajectory/tables_jsonl/full_traj_{model}_parti_full.jsonl"
    for model in MODELS
}
#: The baseline matrix's own per-cell means on the same prompt set. Two claims
#: in the report are cross-table comparisons and have nowhere else to come from:
#: how close the MeanCache homologous cell lands to the matrix row it is meant
#: to reproduce (per-edge spans), and how the coarse extrapolating payloads
#: compare with the matrix's fine-grained implementations of the same order.
MATRIX_TABLE = _ROOT / "resources/full_results_local_analysis/tables/m0_env240.tsv"
MATRIX_DATASET = "parti_full"
#: Coarse payload column -> the matrix method that runs the same order finely.
FINE_COUNTERPART = {"taylor_o1": "taylorseer_o1", "hermite_o2": "hicache_o2"}
#: How many full steps of history a payload needs before its first cached step.
#: Below that the cell is not the payload its label says: order-1 and order-2
#: extrapolation degenerate to plain reuse, and the two-anchor forecast has only
#: one anchor.
PAYLOAD_HISTORY = {
    "reuse": 1,
    "taylor_o1": 2,
    "hermite_o2": 3,
    "mean_avg_vel": 1,
    "di_two_anchor": 2,
}

#: Geometry predictors, in the order the report reads them.
PREDICTOR_KEYS = (
    "first_cache_step",
    "cache_sigma_centroid",
    "hamming_to_meancache",
    "transpositions_to_meancache",
    "hamming_to_budcache",
    "transpositions_to_budcache",
    "rho2_cost_cv",
    "rho2_total_cost",
    "longest_gap_sigma_span",
    "n_gaps",
    "longest_cached_run",
    "w_to_seacache",
    "w_to_teacache",
    "w_to_sencache",
    "w_to_dicache",
    "min_hamming_to_universe",
    "same_basin_as_meancache_r4",
    "same_basin_as_meancache_r6",
    "same_basin_as_meancache_r8",
)


# ----- staged tables ---------------------------------------------------------


def read_tsv_gz(path: Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def load_spx(model: str) -> dict[tuple[str, str, int, int], dict[int, dict[str, float]]]:
    """(schedule, payload, K, seed) -> prompt_idx -> metric -> value."""

    out: dict[tuple[str, str, int, int], dict[int, dict[str, float]]] = defaultdict(dict)
    for row in read_tsv_gz(SPX_DIR / f"perprompt_spx_{model}.tsv.gz"):
        key = (row["schedule"], row["payload"], int(row["k"]), int(row["seed"]))
        out[key][int(row["prompt_idx"])] = {
            metric: float(row[metric]) for metric in METRICS if row.get(metric)
        }
    return dict(out)


def load_native(model: str) -> dict[tuple[str, int, int], dict[int, dict[str, Any]]]:
    """(gate, K, seed) -> prompt_idx -> {metrics..., 'path': bits}."""

    out: dict[tuple[str, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in read_tsv_gz(SPX_DIR / f"perprompt_native_{model}.tsv.gz"):
        key = (row["method"], int(row["k"]), int(row["seed"]))
        entry: dict[str, Any] = {
            metric: float(row[metric]) for metric in METRICS if row.get(metric)
        }
        entry["path"] = row.get("path") or None
        out[key][int(row["prompt_idx"])] = entry
    return dict(out)


def load_split(name: str) -> frozenset[int] | None:
    if name == "all":
        return None
    roles = json.loads(SPLITS_JSON.read_text(encoding="utf-8"))["roles"]
    names = ("validation", "test") if name == "heldout" else (name,)
    return frozenset(int(i) for role in names for i in roles[role])


# ----- schedules -------------------------------------------------------------


def read_bits(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_schedules() -> dict[tuple[str, int, str], str]:
    """(model, K, row) -> bitstring, over both frozen schedule directories."""

    out: dict[tuple[str, int, str], str] = {}
    for directory in (SCHEDULE_DIR, SUPPLEMENT_DIR):
        for path in sorted(directory.glob("*_k*_*.txt")):
            model, budget, name = path.stem.split("_", 2)
            if model not in MODELS or not budget.startswith("k"):
                continue
            out[(model, int(budget[1:]), name)] = read_bits(path)
    return out


def load_manifest_records() -> dict[tuple[str, int, str], dict[str, str]]:
    records: dict[tuple[str, int, str], dict[str, str]] = {}
    for name in ("manifest.tsv", "gpf_manifest.tsv"):
        path = SCHEDULE_DIR / name
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                records[(row["model"], int(row["target_k"]), row["name"])] = row
    with (SUPPLEMENT_DIR / "manifest.tsv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            records[(row["model"], int(row["target_k"]), row["name"])] = row
    return records


def load_discovery_counts() -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    """(model, K, gate) -> ranked realised paths with their mass."""

    out: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    with DISCOVERY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            out[(row["model"], int(row["target_k"]), row["method"])].append(
                {
                    "rank": int(row["rank"]),
                    "schedule": row["schedule"],
                    "count": int(row["count"]),
                    "mass": float(row["mass"]),
                    "cache_count": int(row["cache_count"]),
                }
            )
    for entries in out.values():
        entries.sort(key=lambda entry: entry["rank"])
    return dict(out)


def load_geometry(model: str) -> dict[str, Any]:
    """sigma ladder + component rho2 of the model's Parti reference population.

    Same table and same `risk_profiles` call the dp_rho2 row was solved from,
    so the predictor and the schedule it scores cannot come from different
    populations.
    """

    population = read_population(TRAJ_TABLE[model], num_steps=NUM_STEPS)
    profiles = risk_profiles(population)
    return {
        "sigmas": np.asarray(population.sigmas[:NUM_STEPS], dtype=float),
        "rho2": np.asarray(profiles.rho2, dtype=float),
        "n_rows": int(population.n_rows),
        "window": profiles.rho2_window,
    }


# ----- per-cell values -------------------------------------------------------


def cell_series(
    spx: Mapping[tuple[str, str, int, int], Mapping[int, Mapping[str, float]]],
    *,
    schedule: str,
    payload: str,
    budget_k: int,
    metric: str,
    seeds: Sequence[int],
    prompts: frozenset[int] | None,
) -> dict[tuple[int, int], float] | None:
    """(seed, prompt) -> oriented value over the seeds this cell actually has.

    A row is read on the seeds it was run with, not dropped for missing one.
    The densification rows run ONE seed by design, and every downstream reading
    of them is a per-(seed, prompt) paired difference against the anchor, so
    the pairing restricts itself to the shared seed automatically. What must
    not be lost is the count: `summarise` reports `n_seeds` per cell and the
    panel carries `seed_coverage`, so a row measured on fewer seeds than its
    neighbours is visible rather than silently absent.
    """

    out: dict[tuple[int, int], float] = {}
    for seed in seeds:
        per_prompt = spx.get((schedule, payload, budget_k, seed))
        if not per_prompt:
            continue
        for prompt_idx, values in per_prompt.items():
            if prompts is not None and prompt_idx not in prompts:
                continue
            if metric not in values:
                continue
            out[(seed, prompt_idx)] = orient(values[metric], metric)
    return out or None


def paired(
    left: Mapping[tuple[int, int], float],
    right: Mapping[tuple[int, int], float],
    *,
    restrict: set[tuple[int, int]] | None = None,
) -> dict[str, Any] | None:
    """`paired_difference` plus the prompt-clustered band beside the i.i.d. one.

    Every prompt appears once per seed stream, so the i.i.d. paired SE treats
    three correlated readings of the same prompt as three independent pairs.
    Both bands are carried: the narrow one because the rest of the report and
    the video side quote it, the clustered one because that is the interval a
    "distinguishable" claim has to clear.
    """

    keys = sorted(set(left) & set(right))
    if restrict is not None:
        keys = [key for key in keys if key in restrict]
    entry = paired_difference(left, right, restrict=restrict)
    if entry is None:
        return None
    cluster = clustered_paired_se({key: left[key] - right[key] for key in keys})
    if cluster is not None:
        entry["se_clustered"] = cluster["se"]
        entry["n_clusters"] = cluster["n_clusters"]
        entry["ci_low_clustered"] = entry["mean"] - 2.0 * cluster["se"]
        entry["ci_high_clustered"] = entry["mean"] + 2.0 * cluster["se"]
    return entry


def load_matrix_cells() -> dict[tuple[str, int, str], float]:
    """(model, K, method) -> baseline-matrix PSNR mean on the same prompt set."""

    if not MATRIX_TABLE.is_file():
        return {}
    lines = [
        line for line in MATRIX_TABLE.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    out: dict[tuple[str, int, str], float] = {}
    for row in csv.DictReader(lines, delimiter="\t"):
        if row.get("dataset") != MATRIX_DATASET or not row.get("psnr_mean"):
            continue
        out[(row["model"], int(row["target_k"]), row["method"])] = float(row["psnr_mean"])
    return out


def summarise(series: Mapping[tuple[int, int], float]) -> dict[str, Any]:
    per_seed: dict[int, list[float]] = defaultdict(list)
    for (seed, _), value in series.items():
        per_seed[seed].append(value)
    seed_means = {seed: float(statistics.fmean(values)) for seed, values in per_seed.items()}
    values = list(seed_means.values())
    return {
        "value": float(statistics.fmean(values)),
        "n_seeds": len(values),
        "seed_sd": float(statistics.stdev(values)) if len(values) >= 2 else None,
        "seed_means": seed_means,
        "n_pairs": len(series),
    }


# ----- P1 --------------------------------------------------------------------


def variance_shares(matrix: Mapping[tuple[str, str], float]) -> dict[str, Any]:
    fit = decompose(matrix)
    keys = sorted(matrix)
    ss_schedule = sum(fit["alpha"][row] ** 2 for row, _ in keys)
    ss_payload = sum(fit["beta"][payload] ** 2 for _, payload in keys)
    ss_gamma = sum(value**2 for value in fit["gamma_pairs"].values())
    ss_total = sum((matrix[key] - fit["mu"]) ** 2 for key in keys)
    out = {key: value for key, value in fit.items() if key != "gamma_pairs"}
    out["gamma_by_cell"] = {f"{s}x{p}": value for (s, p), value in fit["gamma_pairs"].items()}
    out["ss"] = {
        "schedule": ss_schedule,
        "payload": ss_payload,
        "interaction": ss_gamma,
        "total": ss_total,
    }
    out["share"] = {
        "schedule": ss_schedule / ss_total if ss_total > 0 else None,
        "payload": ss_payload / ss_total if ss_total > 0 else None,
        "interaction": ss_gamma / ss_total if ss_total > 0 else None,
    }
    return out


def diagonal_gammas(fit: Mapping[str, Any], rows: Sequence[str]) -> list[dict[str, Any]]:
    gamma = fit["gamma_by_cell"]
    out = []
    for schedule in rows:
        for payload in HOMOLOGOUS.get(schedule, ()):
            key = f"{schedule}x{payload}"
            if key in gamma:
                out.append(
                    {"schedule": schedule, "payload": payload, "gamma": float(gamma[key])}
                )
    return out


# ----- P4 --------------------------------------------------------------------


def p4_gate(
    *,
    gate: str,
    fixed_series: Mapping[tuple[int, int], float],
    native: Mapping[int, Mapping[str, Any]],
    modal_bits: str,
    metric: str,
    seeds: Sequence[int],
    native_by_seed: Mapping[int, Mapping[int, Mapping[str, Any]]],
    prompts: frozenset[int] | None,
) -> dict[str, Any]:
    """Fixed modal path minus the gate's own run, split on the realised path."""

    left: dict[tuple[int, int], float] = {}
    right: dict[tuple[int, int], float] = {}
    modal_keys: set[tuple[int, int]] = set()
    off_keys: set[tuple[int, int]] = set()
    unknown = 0
    for seed in seeds:
        per_prompt = native_by_seed.get(seed) or {}
        for prompt_idx, entry in per_prompt.items():
            if prompts is not None and prompt_idx not in prompts:
                continue
            key = (seed, prompt_idx)
            if key not in fixed_series or metric not in entry:
                continue
            left[key] = fixed_series[key]
            right[key] = orient(entry[metric], metric)
            path = entry.get("path")
            if not path:
                unknown += 1
            elif path == modal_bits:
                modal_keys.add(key)
            else:
                off_keys.add(key)
    total = len(left)
    result: dict[str, Any] = {
        "gate": gate,
        "modal_bits": modal_bits,
        "n_pairs": total,
        "n_modal": len(modal_keys),
        "n_offmodal": len(off_keys),
        "n_unknown_path": unknown,
        "modal_mass": (len(modal_keys) / total) if total else None,
        "all": paired(left, right),
        "modal": paired(left, right, restrict=modal_keys),
        "offmodal": paired(left, right, restrict=off_keys),
    }
    if result["modal"] is not None:
        # The two runs are the same computation on a modal prompt; anything but
        # zero here means the fixed path and the realised path disagree.
        result["modal_max_abs"] = float(
            max(abs(left[key] - right[key]) for key in modal_keys)
        )
    return result


# ----- geometry --------------------------------------------------------------

def transpositions(a: str, b: str) -> int:
    return hamming(a, b) // 2


def geometry_predictors(
    bits: str,
    *,
    geometry: Mapping[str, Any],
    anchors: Mapping[str, str],
    gate_distributions: Mapping[str, Mapping[str, float]],
    universe: Sequence[str],
) -> dict[str, Any]:
    sigmas = geometry["sigmas"]
    rho = geometry["rho2"]
    stats = measure(bits, rho, sigmas)
    costs = gap_costs(bits, rho, sigmas)
    cached = [index for index, bit in enumerate(bits) if bit == "1"]
    runs = [len(part) for part in bits.split("0")]
    out: dict[str, Any] = {
        "bits": bits,
        "cache_count": bits.count("1"),
        "first_cache_step": bits.index("1") if "1" in bits else None,
        "longest_cached_run": max(runs) if runs else 0,
        "cache_sigma_centroid": float(np.mean([sigmas[i] for i in cached])) if cached else None,
        "rho2_cost_cv": float(stats["cost_cv"]),
        "rho2_total_cost": float(sum(costs)),
        "n_gaps": int(stats["n_gaps"]),
        "longest_gap_sigma_span": (
            float(max(stats["span"])) if len(stats["span"]) else None
        ),
    }
    for name, anchor in anchors.items():
        out[f"hamming_to_{name}"] = hamming(bits, anchor)
        out[f"transpositions_to_{name}"] = transpositions(bits, anchor)
    point = {bits: 1.0}
    for gate in GATES:
        distribution = gate_distributions.get(gate)
        out[f"w_to_{gate}"] = (
            None if not distribution else float(wasserstein(point, distribution))
        )
    if universe:
        out["min_hamming_to_universe"] = int(min(hamming(bits, path) for path in universe))
        anchor = anchors.get(ANCHOR_ROW)
        if anchor is not None:
            nodes = sorted(set(universe) | {bits, anchor})
            index = {path: i for i, path in enumerate(nodes)}
            for radius in BASIN_RADII:
                groups = components(nodes, radius)
                out[f"same_basin_as_meancache_r{radius}"] = int(
                    any(
                        index[bits] in group and index[anchor] in group
                        for group in groups
                    )
                )
    return out


# ----- per-partition ---------------------------------------------------------


def build_ladder(
    by_row: Mapping[str, Mapping[str, Any]],
    rungs: Mapping[int, Sequence[str]],
) -> list[dict[str, Any]]:
    """One entry per rung, aggregating however many draws that rung holds.

    `delta` is the MEDIAN over draws, not the mean: with three draws the median
    is the draw that is neither the lucky nor the unlucky one, and it does not
    move when one draw lands far out. With an EVEN number of draws the lower
    (worse) of the two middle draws is taken rather than their average, so
    `delta` is always an actual draw that `row` names and `ci_low`/`ci_high`
    measure -- and the tie is broken conservatively. `delta_min` / `delta_max`
    are the range over draws: the spread the round-3 audit asked for, and the
    thing a basin width has to be quoted against.

    `ci_low` / `ci_high` remain the PAIRED interval of the median draw, so the
    two error sources stay apart: the CI says how well that draw's difference
    is measured over 1,632 prompts, the range says how much the draw itself
    matters. A rung with one draw reports the same numbers it always did, with
    `delta_min == delta_max == delta`.
    """

    ladder = [
        {"hamming": 0, "row": ANCHOR_ROW, "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0,
         "delta_min": 0.0, "delta_max": 0.0, "n_draws": 0, "members": []}
    ]
    for distance in sorted(rungs):
        members = []
        for row in rungs[distance]:
            entry = by_row.get(row)
            if entry is None:
                continue
            difference = entry["paired_vs_meancache"]
            members.append(
                {
                    "row": row,
                    "delta": difference["mean"],
                    "ci_low": difference["ci_low"],
                    "ci_high": difference["ci_high"],
                    "n_pairs": difference["n_pairs"],
                }
            )
        if not members:
            continue
        ordered = sorted(members, key=lambda m: m["delta"])
        median = ordered[(len(ordered) - 1) // 2]
        ladder.append(
            {
                "hamming": distance,
                # The representative row name, kept so a rung still prints as a
                # row where a table expects one.
                "row": median["row"],
                "delta": median["delta"],
                "ci_low": median["ci_low"],
                "ci_high": median["ci_high"],
                "n_pairs": median["n_pairs"],
                "delta_min": ordered[0]["delta"],
                "delta_max": ordered[-1]["delta"],
                "delta_range": ordered[-1]["delta"] - ordered[0]["delta"],
                "n_draws": len(ordered),
                "members": ordered,
            }
        )
    # A family with no rows at all is absent, not a lone anchor point: the free
    # ladder only exists on the densified partitions and a one-point curve
    # would render as if it did exist everywhere.
    return ladder if len(ladder) > 1 else []


def analyse_partition(
    *,
    model: str,
    budget_k: int,
    spx,
    native,
    schedules,
    manifest,
    discovery,
    geometry,
    seeds: Sequence[int],
    prompts: frozenset[int] | None,
) -> dict[str, Any]:
    rows_present = sorted(
        {key[0] for key in spx if key[2] == budget_k}
    )
    series_cache: dict[tuple[str, str, str], dict[tuple[int, int], float] | None] = {}

    def series(schedule: str, payload: str, metric: str):
        key = (schedule, payload, metric)
        if key not in series_cache:
            series_cache[key] = cell_series(
                spx,
                schedule=schedule,
                payload=payload,
                budget_k=budget_k,
                metric=metric,
                seeds=seeds,
                prompts=prompts,
            )
        return series_cache[key]

    panel: dict[str, Any] = {
        "model": model,
        "budget_k": budget_k,
        "rows": rows_present,
        "n_seeds": len(seeds),
        "cells": {},
    }

    # ---- cell table, every metric
    for metric in METRICS:
        table: dict[str, Any] = {}
        for schedule in rows_present:
            for payload in PAYLOADS:
                values = series(schedule, payload, metric)
                if values is None:
                    continue
                table[f"{schedule}x{payload}"] = summarise(values)
        panel["cells"][metric] = table

    # ---- how many seed streams each row was measured on. Everything in the
    # densification runs one; everything frozen runs three. A reader comparing
    # two rows' absolute means is comparing different numbers of streams, and
    # this is where that shows.
    panel["seed_coverage"] = {
        key: entry["n_seeds"] for key, entry in panel["cells"][MAIN_METRIC].items()
    }

    # ---- the second error band section 1.7 promises: how far a cell mean
    # moves when the same configuration is re-run on another seed stream.
    panel["seed_band"] = {
        metric: pooled_seed_band(
            {tuple(key.split("x", 1)): entry for key, entry in panel["cells"][metric].items()}
        )
        for metric in METRICS
    }

    # ---- which payload wins, and whether that is a fact about the budget or
    # about the schedule it is paired with.
    best: dict[str, Any] = {}
    for schedule in W1:
        values = {
            payload: panel["cells"][MAIN_METRIC][f"{schedule}x{payload}"]["value"]
            for payload in PAYLOADS
            if f"{schedule}x{payload}" in panel["cells"][MAIN_METRIC]
        }
        if values:
            winner = max(values, key=lambda payload: values[payload])
            runner = sorted(values.values(), reverse=True)
            best[schedule] = {
                "payload": winner,
                "margin": (runner[0] - runner[1]) if len(runner) > 1 else None,
            }
    if best:
        counts = defaultdict(int)
        for entry in best.values():
            counts[entry["payload"]] += 1
        modal = max(sorted(counts), key=lambda payload: counts[payload])
        panel["best_payload"] = {
            "by_schedule": best,
            "modal": modal,
            "n_agree": counts[modal],
            "n_rows": len(best),
        }

    # ---- payload purity: a cell whose schedule starts caching before the
    # payload has enough history is not the payload its label says.
    impure = []
    for schedule in rows_present:
        bits = schedules.get((model, budget_k, schedule))
        if bits is None or "1" not in bits:
            continue
        first_cache_step = bits.index("1")
        for payload in PAYLOADS:
            if f"{schedule}x{payload}" not in panel["cells"][MAIN_METRIC]:
                continue
            if first_cache_step < PAYLOAD_HISTORY[payload]:
                impure.append({
                    "schedule": schedule,
                    "payload": payload,
                    "first_cache_step": first_cache_step,
                })
    w1_first = [
        schedules[(model, budget_k, schedule)].index("1")
        for schedule in W1
        if (model, budget_k, schedule) in schedules
        and "1" in schedules[(model, budget_k, schedule)]
    ]
    panel["payload_purity"] = {
        "impure_cells": impure,
        "earliest_w1_first_cache_step": min(w1_first) if w1_first else None,
    }

    # ---- P1
    panel["P1"] = {}
    for metric in METRICS:
        table = panel["cells"][metric]
        w1_matrix = {
            (schedule, payload): table[f"{schedule}x{payload}"]["value"]
            for schedule in W1
            for payload in PAYLOADS
            if f"{schedule}x{payload}" in table
        }
        block: dict[str, Any] = {"n_w1_cells": len(w1_matrix)}
        if len(w1_matrix) == len(W1) * len(PAYLOADS):
            fit = variance_shares(w1_matrix)
            block["w1"] = fit
            block["diagonal"] = diagonal_gammas(fit, W1)
        ragged = {
            tuple(key.split("x", 1)): entry["value"]
            for key, entry in table.items()
            if key.split("x", 1)[0] not in GPF_ROWS
        }
        ragged = {
            key: value
            for key, value in ragged.items()
            if key[0] in W1 + (ANCHOR_ROW,) + tuple(GATE_ROW.values())
        }
        try:
            ragged_fit = variance_shares(ragged)
        except ValueError:
            ragged_fit = None
        if ragged_fit is not None:
            block["ragged_appendix"] = ragged_fit
            block["diagonal_ragged"] = diagonal_gammas(
                ragged_fit, W1 + (ANCHOR_ROW,) + tuple(GATE_ROW.values())
            )
        gpf_present = [row for row in GPF_ROWS if any(key.startswith(row + "x") for key in table)]
        if gpf_present:
            gpf_matrix = {
                tuple(key.split("x", 1)): entry["value"]
                for key, entry in table.items()
                if key.split("x", 1)[0] in W1 + tuple(gpf_present)
            }
            try:
                gpf_fit = variance_shares(gpf_matrix)
            except ValueError:
                gpf_fit = None
            if gpf_fit is not None:
                block["gpf_appendix"] = {
                    "rows": gpf_present,
                    "diagonal": diagonal_gammas(gpf_fit, tuple(gpf_present)),
                }
        panel["P1"][metric] = block

    # ---- P2 / P3, on the same balanced grid as P1
    panel["P2"] = {}
    panel["P3"] = {}
    for metric in METRICS:
        table = panel["cells"][metric]
        matrix = {
            (schedule, payload): table[f"{schedule}x{payload}"]["value"]
            for schedule in W1
            for payload in PAYLOADS
            if f"{schedule}x{payload}" in table
        }
        panel["P2"][metric] = p2_order(matrix) if matrix else None
        # P3 is a statement about the highest-compression budget only.
        panel["P3"][metric] = (
            p3_high_compression(matrix) if matrix and budget_k == max(BUDGETS) else None
        )

    # ---- P4, per prompt, split on the realised path
    panel["P4"] = {}
    for metric in METRICS:
        rows = []
        for gate in GATES:
            row_name = GATE_ROW[gate]
            payload = HOMOLOGOUS.get(row_name, ("reuse",))[0]
            fixed = series(row_name, payload, metric)
            ranked = discovery.get((model, budget_k, gate)) or []
            exact = [entry for entry in ranked if entry["cache_count"] == budget_k]
            modal_bits = schedules.get((model, budget_k, row_name))
            if fixed is None or modal_bits is None:
                continue
            native_by_seed = {
                seed: native.get((gate, budget_k, seed)) or {} for seed in seeds
            }
            entry = p4_gate(
                gate=gate,
                fixed_series=fixed,
                native=native,
                modal_bits=modal_bits,
                metric=metric,
                seeds=seeds,
                native_by_seed=native_by_seed,
                prompts=prompts,
            )
            entry["payload"] = payload
            entry["discovery_mass"] = exact[0]["mass"] if exact else None
            if len(exact) >= 2:
                margin = exact[0]["count"] - exact[1]["count"]
                entry["rank2_margin_votes"] = margin
                entry["near_tie"] = bool(margin <= 30)
                entry["rank2_bits"] = exact[1]["schedule"]
            rows.append(entry)
        # robustness + off-budget rows read against the same gate run
        extras = []
        for row_name in sorted(rows_present):
            if not (row_name.endswith("_r2") or row_name.endswith("_off")):
                continue
            gate = row_name.split("cache")[0] + "cache"
            payload = "di_two_anchor" if gate == "dicache" else "reuse"
            fixed = series(row_name, payload, metric)
            bits = schedules.get((model, budget_k, row_name))
            if fixed is None or bits is None:
                continue
            native_by_seed = {
                seed: native.get((gate, budget_k, seed)) or {} for seed in seeds
            }
            entry = p4_gate(
                gate=gate,
                fixed_series=fixed,
                native=native,
                modal_bits=bits,
                metric=metric,
                seeds=seeds,
                native_by_seed=native_by_seed,
                prompts=prompts,
            )
            entry["row"] = row_name
            entry["payload"] = payload
            entry["kind"] = "robustness" if row_name.endswith("_r2") else "off_budget"
            extras.append(entry)
        panel["P4"][metric] = {"gates": rows, "extras": extras}

    # ---- P-S1 / P-S2 / P-S3: the reuse column against the MeanCache anchor
    panel["PS1"] = {}
    for metric in METRICS:
        anchor_series = series(ANCHOR_ROW, "reuse", metric)
        if anchor_series is None:
            panel["PS1"][metric] = None
            continue
        entries = []
        for schedule in rows_present:
            values = series(schedule, "reuse", metric)
            if values is None or schedule == ANCHOR_ROW:
                continue
            bits = schedules.get((model, budget_k, schedule))
            if bits is not None and bits.count("1") != budget_k:
                # An off-budget row skips a different number of steps, so a
                # paired difference against the anchor is not a comparison of
                # where to skip. It belongs to the gate-convergence reading
                # only (P4), which pairs it against its own gate.
                continue
            difference = paired(values, anchor_series)
            if difference is None:
                continue
            record = manifest.get((model, budget_k, schedule)) or {}
            anchor_bits = schedules.get((model, budget_k, ANCHOR_ROW))
            summary = summarise(values)
            entries.append(
                {
                    "row": schedule,
                    "group": record.get("group") or record.get("axis") or "original",
                    "paired_vs_meancache": difference,
                    "mean_value": summary["value"],
                    "n_seeds": summary["n_seeds"],
                    # The dose figure puts every row on a distance axis, so the
                    # distance travels with the row rather than being looked up
                    # again from a manifest that does not cover the old rows.
                    "hamming_to_meancache": (
                        None if anchor_bits is None or bits is None
                        else hamming(bits, anchor_bits)
                    ),
                }
            )
        random_band = [
            entry["paired_vs_meancache"]["mean"]
            for entry in entries
            if entry["row"] in RANDOM_ROWS
        ]
        for entry in entries:
            if random_band:
                entry["above_random_band"] = bool(
                    entry["paired_vs_meancache"]["ci_low"] > max(random_band)
                )
        panel["PS1"][metric] = {
            "anchor": ANCHOR_ROW,
            "anchor_value": summarise(anchor_series)["value"],
            "rows": sorted(entries, key=lambda e: -e["paired_vs_meancache"]["mean"]),
            "random_rows": sorted(random_band, reverse=True) or None,
        }

    panel["PS2"] = {}
    for metric in METRICS:
        block = panel["PS1"].get(metric)
        if block is None:
            panel["PS2"][metric] = None
            continue
        by_row = {entry["row"]: entry for entry in block["rows"]}
        families = {
            family: build_ladder(by_row, LADDER_RUNGS[family])
            for family in LADDER_FAMILIES
        }
        panel["PS2"][metric] = {
            # `ladder` stays the first-step-preserving family under its old
            # name: it is the curve every existing reading is written against.
            "ladder": families["preserving"],
            "ladder_free": families["free"],
            "n_rungs": len(families["preserving"]),
            "n_rungs_free": len(families["free"]),
            "max_draws_per_rung": max(
                (rung["n_draws"] for rung in families["preserving"] + families["free"]),
                default=0,
            ),
        }

    panel["PS3"] = {}
    for metric in METRICS:
        block = panel["PS1"].get(metric)
        if block is None:
            panel["PS3"][metric] = None
            continue
        by_row = {entry["row"]: entry for entry in block["rows"]}
        panel["PS3"][metric] = {
            "dp_rho2": by_row.get("dp_rho2", {}).get("paired_vs_meancache"),
            "budcache": by_row.get("budcache", {}).get("paired_vs_meancache"),
            "random": {
                row: by_row.get(row, {}).get("paired_vs_meancache") for row in RANDOM_ROWS
            },
            "gpf_reuse": by_row.get("gpf_reuse_e05_1", {}).get("paired_vs_meancache"),
        }

    return panel


# ----- P-S4: geometry regression --------------------------------------------


def partition_geometry(
    *,
    model: str,
    budget_k: int,
    panel: Mapping[str, Any],
    schedules,
    geometry,
    discovery,
    metric: str,
) -> dict[str, Any]:
    anchors = {
        name: schedules[(model, budget_k, name)]
        for name in (ANCHOR_ROW, "budcache")
        if (model, budget_k, name) in schedules
    }
    gate_distributions = {}
    universe: list[str] = []
    for gate in GATES:
        entries = discovery.get((model, budget_k, gate)) or []
        if entries:
            total = sum(entry["mass"] for entry in entries)
            gate_distributions[gate] = {
                entry["schedule"]: entry["mass"] / total for entry in entries
            }
            universe.extend(entry["schedule"] for entry in entries)
    universe = sorted(set(universe))

    table = panel["cells"][metric]
    rows = []
    for key, entry in sorted(table.items()):
        schedule, payload = key.split("x", 1)
        if payload != "reuse":
            continue
        bits = schedules.get((model, budget_k, schedule))
        if bits is None or bits.count("1") != budget_k:
            continue  # off-budget rows do not belong on a quality-vs-geometry axis
        predictors = geometry_predictors(
            bits,
            geometry=geometry,
            anchors=anchors,
            gate_distributions=gate_distributions,
            universe=universe,
        )
        predictors["row"] = schedule
        predictors["quality"] = entry["value"]
        rows.append(predictors)

    correlations = {}
    for name in PREDICTOR_KEYS:
        xs = [row.get(name) for row in rows]
        ys = [row["quality"] for row in rows]
        correlations[name] = spearman(xs, ys)
    return {"n_rows": len(rows), "rows": rows, "spearman": correlations}


def pooled_geometry(
    partitions: Sequence[Mapping[str, Any]], *, drop_rows: Sequence[str] = ()
) -> dict[str, Any]:
    """Within-partition standardised quality against standardised predictors.

    `drop_rows` supports the one disclosure the GPF isolation owes: those rows
    were chosen after looking at held-out prompts, so the report has to be able
    to say how much this regression moves without them rather than assert that
    it does not.
    """

    pooled: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": []})
    for block in partitions:
        rows = [row for row in block["rows"] if row["row"] not in drop_rows]
        if len(rows) < 4:
            continue
        quality_ranks = rank_within([row["quality"] for row in rows])
        for name in PREDICTOR_KEYS:
            values = [row.get(name) for row in rows]
            if any(value is None for value in values):
                continue
            if max(values) - min(values) <= 1e-12:
                continue
            pooled[name]["x"].extend(rank_within([float(v) for v in values]))
            pooled[name]["y"].extend(quality_ranks)
    out = {}
    for name, block in pooled.items():
        out[name] = spearman(block["x"], block["y"])
        if out[name] is not None:
            out[name]["n_partitions"] = len(
                [p for p in partitions if all(row.get(name) is not None for row in p["rows"])]
            )
    return out


# ----- assembly --------------------------------------------------------------


def analyse_split(
    split: str,
    *,
    spx_by_model,
    native_by_model,
    schedules,
    manifest,
    discovery,
    geometry_by_model,
    seeds,
) -> dict[str, Any]:
    prompts = load_split(split)
    panels = []
    geometry_blocks = []
    for model in MODELS:
        for budget_k in BUDGETS:
            panel = analyse_partition(
                model=model,
                budget_k=budget_k,
                spx=spx_by_model[model],
                native=native_by_model[model],
                schedules=schedules,
                manifest=manifest,
                discovery=discovery,
                geometry=geometry_by_model[model],
                seeds=seeds[model],
                prompts=prompts,
            )
            block = partition_geometry(
                model=model,
                budget_k=budget_k,
                panel=panel,
                schedules=schedules,
                geometry=geometry_by_model[model],
                discovery=discovery,
                metric=MAIN_METRIC,
            )
            block["model"] = model
            block["budget_k"] = budget_k
            panel["PS4"] = {
                "n_rows": block["n_rows"],
                "spearman": block["spearman"],
            }
            geometry_blocks.append(block)
            panels.append(panel)

    # P1 sign test over the formal W1 diagonals of all six panels.
    sign_tests = {}
    for metric in METRICS:
        gammas = [
            entry["gamma"]
            for panel in panels
            for entry in panel["P1"][metric].get("diagonal", [])
        ]
        positive = sum(1 for value in gammas if value > 0)
        sign_tests[metric] = {
            "n_trials": len(gammas),
            "n_positive": positive,
            "p_two_sided": two_sided_sign_p(positive, len(gammas)),
            "mean_gamma": float(statistics.fmean(gammas)) if gammas else None,
            "mean_abs_gamma": (
                float(statistics.fmean([abs(v) for v in gammas])) if gammas else None
            ),
        }
        ragged = [
            entry["gamma"]
            for panel in panels
            for entry in panel["P1"][metric].get("diagonal_ragged", [])
        ]
        gpf = [
            entry["gamma"]
            for panel in panels
            for entry in panel["P1"][metric].get("gpf_appendix", {}).get("diagonal", [])
        ]
        sign_tests[metric]["appendix_ragged"] = {
            "n_trials": len(ragged),
            "n_positive": sum(1 for value in ragged if value > 0),
            "p_two_sided": two_sided_sign_p(
                sum(1 for value in ragged if value > 0), len(ragged)
            ),
        }
        sign_tests[metric]["appendix_gpf"] = {
            "n_trials": len(gpf),
            "n_positive": sum(1 for value in gpf if value > 0),
            "p_two_sided": two_sided_sign_p(sum(1 for value in gpf if value > 0), len(gpf)),
        }

    # The random reference level for the schedule axis.
    random_reference = []
    for panel in panels:
        block = panel["PS1"][MAIN_METRIC]
        if block is None:
            continue
        by_row = {entry["row"]: entry for entry in block["rows"]}
        randoms = [by_row[row]["mean_value"] for row in RANDOM_ROWS if row in by_row]
        gates = [
            by_row[GATE_ROW[gate]]["mean_value"]
            for gate in GATES
            if GATE_ROW[gate] in by_row
        ]
        if not randoms:
            continue
        random_mean = float(statistics.fmean(randoms))
        random_reference.append(
            {
                "model": panel["model"],
                "budget_k": panel["budget_k"],
                "meancache": block["anchor_value"],
                "random_mean": random_mean,
                # The spread across draws of the SAME reference level. Two
                # draws on the frozen partitions, five on the densified ones;
                # a design gain quoted without it is a difference of two
                # numbers only one of which has a stated uncertainty.
                "n_random": len(randoms),
                "random_min": min(randoms),
                "random_max": max(randoms),
                "random_spread": max(randoms) - min(randoms),
                "random_sd": (
                    float(statistics.stdev(randoms)) if len(randoms) > 1 else None
                ),
                "random_rows": {row: by_row[row]["mean_value"] for row in RANDOM_ROWS if row in by_row},
                # The frozen random rows run three seed streams and the
                # densified ones run one, so the level below is a mean over
                # rows measured on different numbers of streams.
                "random_row_seeds": {
                    row: by_row[row].get("n_seeds") for row in RANDOM_ROWS if row in by_row
                },
                "design_gain_over_random": block["anchor_value"] - random_mean,
                # The same gain measured against the best and the worst draw of
                # the reference, i.e. how much of the gain is the design and how
                # much could be which random row was drawn.
                "design_gain_vs_best_random": block["anchor_value"] - max(randoms),
                "design_gain_vs_worst_random": block["anchor_value"] - min(randoms),
                "worst_gate": min(gates) if gates else None,
                "worst_gate_loss_vs_random": (min(gates) - random_mean) if gates else None,
                "row_spread": (
                    max([block["anchor_value"], *randoms, *gates])
                    - min([block["anchor_value"], *randoms, *gates])
                ),
            }
        )

    return {
        "split": split,
        "n_prompts": None if prompts is None else len(prompts),
        "panels": panels,
        "P1_sign_tests": sign_tests,
        "random_reference": random_reference,
        "PS4_pooled": pooled_geometry(geometry_blocks),
        "PS4_pooled_no_gpf": pooled_geometry(geometry_blocks, drop_rows=GPF_ROWS),
        "PS4_partitions": [
            {
                "model": block["model"],
                "budget_k": block["budget_k"],
                "n_rows": block["n_rows"],
                "rows": [
                    {key: row.get(key) for key in ("row", "quality", *PREDICTOR_KEYS)}
                    for row in block["rows"]
                ],
                "spearman": block["spearman"],
            }
            for block in geometry_blocks
        ],
    }


def matrix_comparisons(
    report: Mapping[str, Any], matrix: Mapping[tuple[str, int, str], float]
) -> dict[str, Any]:
    """The two cross-table readings the report makes, against the matrix itself.

    `per_edge_span`: the MeanCache homologous cell runs the interval-mean payload
    with this schedule's own per-edge spans, so it should land on the matrix's
    MeanCache row. How far it misses is the only check that the spans were
    carried across, and it is a number, not an assurance.

    `coarse_vs_fine`: the extrapolating payload columns cache the whole
    transformer residual, while the matrix runs the same orders per block and
    per sub-module. Same schedule (the uniform table both sides use), so the
    difference is granularity alone.
    """

    if not matrix:
        return {}
    spans, granularity = [], []
    for panel in report["splits"]["all"]["panels"]:
        model, budget_k = panel["model"], panel["budget_k"]
        cells = panel["cells"][MAIN_METRIC]
        cell = cells.get(f"{ANCHOR_ROW}xmean_avg_vel")
        reference = matrix.get((model, budget_k, ANCHOR_ROW))
        if cell and reference is not None:
            spans.append({
                "model": model, "budget_k": budget_k,
                "spx": cell["value"], "matrix": reference,
                "delta": cell["value"] - reference,
            })
        for payload, method in sorted(FINE_COUNTERPART.items()):
            coarse = cells.get(f"uniformx{payload}")
            fine = matrix.get((model, budget_k, method))
            if coarse and fine is not None:
                granularity.append({
                    "model": model, "budget_k": budget_k, "payload": payload,
                    "matrix_method": method,
                    "coarse": coarse["value"], "fine": fine,
                    "delta": coarse["value"] - fine,
                })
    return {
        "source": str(MATRIX_TABLE.relative_to(_ROOT)),
        "dataset": MATRIX_DATASET,
        "per_edge_span": spans,
        "coarse_vs_fine": granularity,
    }


def row_coincidences(schedules: Mapping[tuple[str, int, str], str]) -> list[dict[str, Any]]:
    """Frozen rows that turn out to be the same bitstring.

    Worth knowing because the rho2-DP control was frozen as a *new* row, and if
    it lands on top of an existing one the two are not independent evidence.
    """

    out = []
    by_partition: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for (model, budget_k, name), bits in schedules.items():
        by_partition[(model, budget_k)][name] = bits
    for (model, budget_k), rows in sorted(by_partition.items()):
        seen: dict[str, list[str]] = defaultdict(list)
        for name, bits in sorted(rows.items()):
            seen[bits].append(name)
        for bits, names in seen.items():
            if len(names) > 1:
                out.append({
                    "model": model, "budget_k": budget_k,
                    "rows": names, "bits": bits,
                })
    return out


def build_report(splits: Sequence[str]) -> dict[str, Any]:
    schedules = load_schedules()
    manifest = load_manifest_records()
    discovery = load_discovery_counts()
    spx_by_model = {model: load_spx(model) for model in MODELS}
    native_by_model = {model: load_native(model) for model in MODELS}
    geometry_by_model = {model: load_geometry(model) for model in MODELS}
    seeds = {
        model: sorted({key[3] for key in spx_by_model[model]}) for model in MODELS
    }
    report: dict[str, Any] = {
        "models": list(MODELS),
        "budgets": list(BUDGETS),
        "metrics": list(METRICS),
        "main_metric": MAIN_METRIC,
        # Values in `cells` are oriented so larger is better, which negates
        # LPIPS. A table of levels has to undo that; a table of differences
        # must not, because there "+ means better" is the point.
        "lower_is_better": [
            metric for metric in METRICS if not HIGHER_IS_BETTER.get(metric, True)
        ],
        "w1_rows": list(W1),
        "payloads": list(PAYLOADS),
        "seeds": seeds,
        "n_runs": {
            model: len(spx_by_model[model]) for model in MODELS
        },
        "reference_population": {
            model: {
                "n_rows": geometry_by_model[model]["n_rows"],
                "rho2_window": geometry_by_model[model]["window"],
            }
            for model in MODELS
        },
        "row_coincidences": row_coincidences(schedules),
        "splits": {},
    }
    for split in splits:
        report["splits"][split] = analyse_split(
            split,
            spx_by_model=spx_by_model,
            native_by_model=native_by_model,
            schedules=schedules,
            manifest=manifest,
            discovery=discovery,
            geometry_by_model=geometry_by_model,
            seeds=seeds,
        )
    report["matrix_comparisons"] = matrix_comparisons(report, load_matrix_cells())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "resources" / "sp_cross_supplement" / "sp_cross_results.json",
    )
    parser.add_argument("--splits", nargs="+", default=["all", "heldout"])
    args = parser.parse_args()
    report = build_report(args.splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True, default=float),
        encoding="utf-8",
    )
    print(f"[spx-supplement] wrote {args.out}")
    for split, block in report["splits"].items():
        tests = block["P1_sign_tests"][MAIN_METRIC]
        print(
            f"  {split}: {len(block['panels'])} panels, "
            f"P1 {tests['n_positive']}/{tests['n_trials']} positive "
            f"(p={tests['p_two_sided']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
