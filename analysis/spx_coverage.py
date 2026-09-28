#!/usr/bin/env python3
"""Golden-path coverage test over the staged SPX per-prompt tables.

Hypothesis 1 (paper/main.tex, "The Golden Path Hypothesis") states that under
fixed (model, sampler, time grid, cache ratio K, cached-step payload, prompt
population, seed assignments) there exists a prompt-independent path u* whose
seed-averaged oriented losses satisfy

    Pr_x[ Delta_j(u*; x) <= eps_j  for all j in J_R ] >= 1 - delta,

with J_R = {PSNR, SSIM, LPIPS}, L_PSNR = -PSNR, L_SSIM = 1 - SSIM,
L_LPIPS = LPIPS, and prompts (not prompt-seed pairs) as the independent units.
This script estimates that joint coverage and its one-sided exact binomial
(Clopper-Pearson) 95% lower confidence bound for every staged fixed-schedule
cell of the image SPX wave (FLUX.1-dev, Qwen-Image) and the video SPX wave
(HunyuanVideo, Wan2.1), entirely from the staged per-prompt tables - zero GPU.

Inputs (all read-only)
    resources/spx/perprompt_spx_{flux,qwen}.tsv.gz          image SPX cells
    resources/spx/perprompt_native_{flux,qwen}.tsv.gz       native gate runs
    resources/sp_cross_schedules/parti_spx_splits.v1.json   1632-prompt splits
    resources/{sp_cross_schedules,spx_supplement_schedules}/<model>_k<K>_<row>.txt
    resources/video_spx/<b>/pervideo_spx_<b>.tsv.gz         video SPX cells
    resources/video_full_results/pervideo_<b>.tsv.gz        video native gates
    resources/video_spx_schedules/manifest.tsv              video row metadata

Outputs
    resources/spx/coverage_results.json
    resources/video_spx/coverage_results.json
    docs/figures/spx_coverage/fig_cov_<model>_K<K>.png      one per partition

The working points (anchor thresholds and delta) are conventional fidelity
levels declared in ANCHORS below before any coverage number was computed; the
complete sweep curves keep the conclusion threshold-independent. Cell means
had been seen in the SPX reports before this test; coverage had not.

Usage
    python analysis/spx_coverage.py            # both modalities + figures
    python analysis/spx_coverage.py --no-figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

REPO = Path(__file__).resolve().parents[1]

# ----- pre-declared working points and sweep grids ---------------------------

#: Conventional fidelity levels, declared before computing any coverage number.
#: PSNR floors in dB, SSIM floors, LPIPS ceilings; every combination is one
#: joint working point (3 x 2 x 2 = 12), each read at both delta values.
ANCHORS = {
    "psnr_min": (20.0, 25.0, 30.0),
    "ssim_min": (0.90, 0.95),
    "lpips_max": (0.10, 0.05),
    "delta": (0.05, 0.10),
    #: The single named point used for headline sentences.
    "middle": {"psnr_min": 25.0, "ssim_min": 0.90, "lpips_max": 0.10},
    "alpha": 0.05,  # one-sided exact binomial lower bound level
}

#: Sweep grids: one threshold moves, the other two sit at the middle anchors.
SWEEPS = {
    "psnr": [round(15.0 + 0.5 * i, 1) for i in range(51)],   # 15 .. 40 dB
    "ssim": [round(0.80 + 0.005 * i, 3) for i in range(41)],  # 0.80 .. 1.00
    "lpips": [round(0.005 * i, 3) for i in range(61)],        # 0.00 .. 0.30
}

METRICS = ("psnr", "ssim", "lpips")

# ----- image-side design (mirrors analysis/sp_cross.py) ----------------------

IMAGE_MODELS = ("flux", "qwen")
IMAGE_SEEDS = {"flux": (41, 42, 43), "qwen": (42, 100042, 200042)}
IMAGE_KS = (29, 37, 41)
N_PROMPTS = 1632

#: Schedule -> payload(s) sharing its source method (sp_cross.py HOMOLOGOUS).
IMAGE_HOMOLOGOUS = {
    "budcache": ("reuse",),
    "meancache": ("mean_avg_vel",),
    "dpcache": ("hermite_o2",),
    "uniform": ("taylor_o1", "hermite_o2"),
    "seacache_top1": ("reuse",),
    "teacache_top1": ("reuse",),
    "sencache_top1": ("reuse",),
    "dicache_top1": ("di_two_anchor",),
}

#: role -> schedules. `fixed` + `gate_top1` + `ladder_first` are the verdict
#: candidates; `random` is the collapse control; `ladder_free`, `rank2` are
#: extra context rows; `off_budget` rows do not lie in S_K at the labelled
#: budget and are excluded entirely.  Two roles are excluded from the verdict
#: because their construction saw held-out prompts: `geometry` rows were
#: screened on held-out quality, and `trajectory_fit` (the rho_2 table) is
#: solved on a trajectory table spanning all 1632 prompt indices.  The same
#: exposure rule applies to both.
IMAGE_ROLES = {
    "fixed": ("budcache", "dpcache", "meancache", "uniform"),
    "trajectory_fit": ("dp_rho2",),
    "gate_top1": ("seacache_top1", "teacache_top1", "sencache_top1", "dicache_top1"),
    "ladder_first": ("ham2f", "ham4f", "ham8f",
                     "ham2f_d2", "ham2f_d3", "ham4f_d2", "ham4f_d3",
                     "ham8f_d2", "ham8f_d3"),
    "ladder_free": ("ham2_d1", "ham2_d2", "ham4_d1", "ham4_d2", "ham8_d1", "ham8_d2"),
    "random": ("rand_1", "rand_2", "rand_3", "rand_4", "rand_5"),
    "geometry": ("gpf_reuse_e05_1", "gpf_o1_e15_1", "gpf_o1_e20_1"),
    "rank2": ("teacache_top1_r2", "dicache_top1_r2"),
    "off_budget": ("sencache_top1_off",),
}
VERDICT_ROLES = ("fixed", "gate_top1", "ladder_first")

IMAGE_GATES = ("seacache", "teacache", "sencache", "dicache")

# ----- video-side design (mirrors analysis/video_spx.py) ---------------------

VIDEO_BACKBONES = ("hunyuan_video", "wan21")
VIDEO_DATASETS = ("penguin599", "vbench944")
#: The SPX stream of each dataset (video_spx.py MATRIX_STREAMS[...][0]).
VIDEO_STREAM = {"penguin599": 54, "vbench944": 42}
VIDEO_N_PROMPTS = 150

VIDEO_HOMOLOGOUS = {
    "shared": ("taylor_o1", "hermite_o2"),
    "budcache": ("reuse",),
    "meancache": ("mean_vel",),
    "sea_top1": ("reuse",),
    "tea_top1": ("reuse",),
    "sen_top1": ("reuse",),
    "di_top1": ("di_two_anchor",),
}

VIDEO_ROLES = {
    "fixed": ("shared", "budcache", "meancache", "uniform", "dp_rho2"),
    "gate_top1": ("sea_top1", "tea_top1", "sen_top1", "di_top1"),
    "ladder_first": ("ham2f", "ham4f", "ham8f"),
    "ladder_free": ("ham2", "ham4", "ham8"),
    "random": ("rand_1", "rand_2"),
    "geometry": (),
    "rank2": (),
    "off_budget": ("sea_top1_off", "tea_top1_off", "sen_top1_off"),
}

VIDEO_GATE_METHOD = {"sea_top1": "seacache", "tea_top1": "teacache",
                     "sen_top1": "sencache", "di_top1": "dicache"}


# ----- shared statistics -----------------------------------------------------


def cp_lower(k: int, n: int, alpha: float) -> float:
    """One-sided exact binomial (Clopper-Pearson) lower bound for k/n."""
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    return float(beta_dist.ppf(alpha, k, n - k + 1))


def joint_event(psnr: np.ndarray, ssim: np.ndarray, lpips: np.ndarray,
                psnr_min: float, ssim_min: float, lpips_max: float) -> np.ndarray:
    """Seed-averaged joint reconstruction event per prompt.

    Delta_PSNR = -mean(PSNR) <= -psnr_min  <=>  mean PSNR >= psnr_min;
    Delta_SSIM = 1 - mean(SSIM) <= 1 - ssim_min  <=>  mean SSIM >= ssim_min;
    Delta_LPIPS = mean(LPIPS) <= lpips_max.
    """
    return (psnr >= psnr_min) & (ssim >= ssim_min) & (lpips <= lpips_max)


def coverage_entry(event: np.ndarray, alpha: float, m_candidates: int | None = None,
                   n_points: int | None = None) -> dict:
    """Coverage with three lower-bound levels.

    `lcb95` is the plain one-sided bound; `lcb_bonferroni` tightens the level
    to alpha/m over the candidates tested inside one partition; and
    `lcb_bonferroni_grid` tightens it further to alpha/(m * n_points), the
    family that also counts the working-point grid the candidate is allowed to
    pass at.  A pass that survives only the first two is reported as such.
    """
    n = int(event.size)
    k = int(event.sum())
    entry = {
        "n": n,
        "k": k,
        "coverage": round(k / n, 6) if n else None,
        "lcb95": round(cp_lower(k, n, alpha), 6),
    }
    if m_candidates:
        entry["lcb_bonferroni"] = round(cp_lower(k, n, alpha / m_candidates), 6)
        if n_points:
            entry["lcb_bonferroni_grid"] = round(
                cp_lower(k, n, alpha / (m_candidates * n_points)), 6)
    return entry


def grid_points() -> list[dict]:
    points = []
    for p in ANCHORS["psnr_min"]:
        for s in ANCHORS["ssim_min"]:
            for l in ANCHORS["lpips_max"]:
                points.append({"psnr_min": p, "ssim_min": s, "lpips_max": l})
    return points


def sweep_curves(psnr: np.ndarray, ssim: np.ndarray, lpips: np.ndarray) -> dict:
    mid = ANCHORS["middle"]
    out = {}
    base_ss = ssim >= mid["ssim_min"]
    base_lp = lpips <= mid["lpips_max"]
    base_ps = psnr >= mid["psnr_min"]
    n = psnr.size
    out["psnr"] = [round(float(((psnr >= t) & base_ss & base_lp).sum() / n), 6)
                   for t in SWEEPS["psnr"]]
    out["ssim"] = [round(float((base_ps & (ssim >= t) & base_lp).sum() / n), 6)
                   for t in SWEEPS["ssim"]]
    out["lpips"] = [round(float((base_ps & base_ss & (lpips <= t)).sum() / n), 6)
                    for t in SWEEPS["lpips"]]
    return out


def role_of(schedule: str, roles: dict) -> str:
    for role, names in roles.items():
        if schedule in names:
            return role
    return "other"


def candidate_payloads(schedule: str, available: set[str], homologous: dict) -> list[str]:
    """The reported payload set: homologous payload(s) plus zero-order reuse."""
    ordered = list(homologous.get(schedule, ())) + ["reuse"]
    out = []
    for payload in ordered:
        if payload in available and payload not in out:
            out.append(payload)
    return out


# ----- image side ------------------------------------------------------------


def load_splits() -> dict[str, list[int]]:
    blob = json.loads((REPO / "resources/sp_cross_schedules/parti_spx_splits.v1.json")
                      .read_text(encoding="utf-8"))
    roles = blob["roles"]
    heldout = sorted(set(roles["validation"]) | set(roles["test"]))
    return {
        "discovery": sorted(roles["discovery"]),
        "validation": sorted(roles["validation"]),
        "test": sorted(roles["test"]),
        "heldout": heldout,
        "full": list(range(blob["prompt_count"])),
    }


def image_schedule_bits(model: str, k: int, schedule: str) -> str | None:
    for directory in ("sp_cross_schedules", "spx_supplement_schedules"):
        path = REPO / "resources" / directory / f"{model}_k{k}_{schedule}.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    return None


def seed_mean_frame(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per-prompt seed means plus the seed count actually averaged."""
    grouped = frame.groupby(keys + ["prompt_idx"], sort=True)
    means = grouped[list(METRICS)].mean()
    counts = grouped["psnr"].size().rename("n_seeds")
    return pd.concat([means, counts], axis=1).reset_index()


def analyse_image(figures: bool) -> dict:
    splits = load_splits()
    populations = {name: np.asarray(idx, dtype=int)
                   for name, idx in splits.items() if name != "discovery"}
    report_pops = ("full", "heldout", "test")
    alpha = ANCHORS["alpha"]
    points = grid_points()

    result = {"models": {}}
    for model in IMAGE_MODELS:
        table = pd.read_csv(REPO / f"resources/spx/perprompt_spx_{model}.tsv.gz",
                            sep="\t", compression="gzip")
        native = pd.read_csv(REPO / f"resources/spx/perprompt_native_{model}.tsv.gz",
                             sep="\t", compression="gzip")
        for frame in (table, native):
            for metric in METRICS:
                if not np.isfinite(frame[metric].to_numpy(dtype=float)).all():
                    raise SystemExit(f"non-finite {metric} in a {model} table")
        cells = seed_mean_frame(table, ["schedule", "payload", "k"])
        gates = seed_mean_frame(native, ["method", "k"])

        model_block = {"partitions": {}}
        for k in IMAGE_KS:
            at_k = cells[cells["k"] == k]
            cell_names = sorted({(s, p) for s, p in
                                 zip(at_k["schedule"], at_k["payload"])})
            available = {}
            for schedule in sorted({s for s, _ in cell_names}):
                available[schedule] = {p for s, p in cell_names if s == schedule}

            # verdict candidate cells (fixed rows + gate top-1 + first-preserving
            # ladder, each with homologous payload(s) plus reuse)
            candidates = []
            for role in VERDICT_ROLES:
                for schedule in IMAGE_ROLES[role]:
                    if schedule not in available:
                        continue
                    for payload in candidate_payloads(schedule, available[schedule],
                                                      IMAGE_HOMOLOGOUS):
                        candidates.append((schedule, payload))
            controls = [(s, "reuse") for s in IMAGE_ROLES["random"] if s in available]
            m_candidates = len(candidates)

            cell_blocks = {}
            for schedule, payload in cell_names:
                sub = at_k[(at_k["schedule"] == schedule) & (at_k["payload"] == payload)]
                sub = sub.set_index("prompt_idx").sort_index()
                if len(sub) != N_PROMPTS:
                    raise SystemExit(f"{model} K{k} {schedule}x{payload}: "
                                     f"{len(sub)} prompts, expected {N_PROMPTS}")
                psnr = sub["psnr"].to_numpy(dtype=float)
                ssim = sub["ssim"].to_numpy(dtype=float)
                lpips = sub["lpips"].to_numpy(dtype=float)
                n_seeds = int(sub["n_seeds"].iloc[0])
                role = role_of(schedule, IMAGE_ROLES)
                is_candidate = (schedule, payload) in candidates
                block = {
                    "role": role,
                    "verdict_candidate": is_candidate,
                    "n_seeds": n_seeds,
                    "first_cache_step": None,
                    "populations": {},
                }
                bits = image_schedule_bits(model, k, schedule)
                if bits is not None:
                    block["first_cache_step"] = bits.find("1")
                for pop in report_pops:
                    idx = populations[pop]
                    p_, s_, l_ = psnr[idx], ssim[idx], lpips[idx]
                    pop_block = {
                        "mean_psnr": round(float(p_.mean()), 4),
                        "mean_ssim": round(float(s_.mean()), 6),
                        "mean_lpips": round(float(l_.mean()), 6),
                        "grid": [],
                    }
                    for point in points:
                        event = joint_event(p_, s_, l_, point["psnr_min"],
                                            point["ssim_min"], point["lpips_max"])
                        entry = dict(point)
                        entry.update(coverage_entry(
                            event, alpha,
                            m_candidates if is_candidate else None,
                            len(points) if is_candidate else None))
                        pop_block["grid"].append(entry)
                    block["populations"][pop] = pop_block
                if is_candidate or role in ("random", "ladder_free", "geometry",
                                           "trajectory_fit", "rank2"):
                    idx = populations["heldout"]
                    block["curves_heldout"] = sweep_curves(psnr[idx], ssim[idx], lpips[idx])
                cell_blocks[f"{schedule}|{payload}"] = block

            gate_blocks = {}
            for method in IMAGE_GATES:
                sub = gates[(gates["method"] == method) & (gates["k"] == k)]
                sub = sub.set_index("prompt_idx").sort_index()
                if len(sub) != N_PROMPTS:
                    raise SystemExit(f"{model} K{k} native {method}: {len(sub)} prompts")
                psnr = sub["psnr"].to_numpy(dtype=float)
                ssim = sub["ssim"].to_numpy(dtype=float)
                lpips = sub["lpips"].to_numpy(dtype=float)
                block = {"n_seeds": int(sub["n_seeds"].iloc[0]), "populations": {}}
                for pop in report_pops:
                    idx = populations[pop]
                    p_, s_, l_ = psnr[idx], ssim[idx], lpips[idx]
                    pop_block = {
                        "mean_psnr": round(float(p_.mean()), 4),
                        "grid": [],
                    }
                    for point in points:
                        event = joint_event(p_, s_, l_, point["psnr_min"],
                                            point["ssim_min"], point["lpips_max"])
                        entry = dict(point)
                        entry.update(coverage_entry(event, alpha))
                        pop_block["grid"].append(entry)
                    block["populations"][pop] = pop_block
                idx = populations["heldout"]
                block["curves_heldout"] = sweep_curves(psnr[idx], ssim[idx], lpips[idx])
                gate_blocks[method] = block

            model_block["partitions"][f"K{k}"] = {
                "m_verdict_candidates": m_candidates,
                "candidate_cells": [f"{s}|{p}" for s, p in candidates],
                "control_cells": [f"{s}|{p}" for s, p in controls],
                "cells": cell_blocks,
                "native_gates": gate_blocks,
                "verdicts": build_verdicts(cell_blocks, candidates, controls,
                                           report_pops, points, m_candidates),
            }
        result["models"][model] = model_block

    result["populations"] = {name: len(idx) for name, idx in populations.items()}
    result["splits_source"] = "resources/sp_cross_schedules/parti_spx_splits.v1.json"
    if figures:
        make_figures_image(result)
    return result


def build_verdicts(cell_blocks: dict, candidates: list, controls: list,
                   report_pops: tuple, points: list[dict], m_candidates: int) -> dict:
    """Existential verdict per (population, working point, delta)."""
    verdicts = {}
    for pop in report_pops:
        pop_verdicts = []
        for i, point in enumerate(points):
            rows = []
            for schedule, payload in candidates:
                block = cell_blocks[f"{schedule}|{payload}"]
                entry = block["populations"][pop]["grid"][i]
                rows.append((f"{schedule}|{payload}", entry))
            best = max(rows, key=lambda r: (r[1]["lcb95"], r[1]["coverage"]))
            control_cov = [
                (f"{s}|{p}", cell_blocks[f"{s}|{p}"]["populations"][pop]["grid"][i])
                for s, p in controls]
            best_control = (max(control_cov, key=lambda r: r[1]["coverage"])
                            if control_cov else None)
            record = {
                **point,
                "best_candidate": best[0],
                "best_coverage": best[1]["coverage"],
                "best_lcb95": best[1]["lcb95"],
                "best_lcb_bonferroni": best[1].get("lcb_bonferroni"),
                "best_lcb_bonferroni_grid": best[1].get("lcb_bonferroni_grid"),
                "best_control": best_control[0] if best_control else None,
                "best_control_coverage": (best_control[1]["coverage"]
                                          if best_control else None),
            }
            for delta in ANCHORS["delta"]:
                passed = best[1]["lcb95"] >= 1.0 - delta
                witnesses = sorted([name for name, entry in rows
                                    if entry["lcb95"] >= 1.0 - delta])
                record[f"pass_delta_{delta}"] = bool(passed)
                record[f"witnesses_delta_{delta}"] = witnesses
                record[f"pass_bonferroni_delta_{delta}"] = bool(
                    (best[1].get("lcb_bonferroni") or 0.0) >= 1.0 - delta)
                record[f"pass_bonferroni_grid_delta_{delta}"] = bool(
                    (best[1].get("lcb_bonferroni_grid") or 0.0) >= 1.0 - delta)
            pop_verdicts.append(record)
        verdicts[pop] = pop_verdicts
    return verdicts


# ----- video side ------------------------------------------------------------


def load_video_manifest(backbone: str) -> dict[tuple[str, str], dict]:
    out = {}
    path = REPO / "resources/video_spx_schedules/manifest.tsv"
    with path.open(encoding="utf-8") as handle:
        import csv as _csv
        for record in _csv.DictReader(handle, delimiter="\t"):
            if record["backbone"] == backbone:
                out[(record["budget"], record["row"])] = record
    return out


def analyse_video(figures: bool) -> dict:
    alpha = ANCHORS["alpha"]
    points = grid_points()
    report_pops = ("pooled", "penguin599", "vbench944")

    result = {"models": {}}
    for backbone in VIDEO_BACKBONES:
        table = pd.read_csv(
            REPO / f"resources/video_spx/{backbone}/pervideo_spx_{backbone}.tsv.gz",
            sep="\t", compression="gzip")
        native = pd.read_csv(
            REPO / f"resources/video_full_results/pervideo_{backbone}.tsv.gz",
            sep="\t", compression="gzip")
        manifest = load_video_manifest(backbone)
        for metric in METRICS:
            if not np.isfinite(table[metric].to_numpy(dtype=float)).all():
                raise SystemExit(f"non-finite {metric} in the {backbone} SPX table")

        # native rows restricted to the SPX streams and the first 150 prompts
        native = native[native["method"].isin(VIDEO_GATE_METHOD.values())]
        native = native[native["prompt_idx"] < VIDEO_N_PROMPTS]
        keep = ((native["dataset"] == "penguin599") & (native["seed"] == 54)) | \
               ((native["dataset"] == "vbench944") & (native["seed"] == 42))
        native = native[keep].copy()
        for metric in METRICS:
            native = native[np.isfinite(native[metric].astype(float))]

        model_block = {"partitions": {}}
        ks = sorted({int(k) for k in table["K"]})
        for k in ks:
            at_k = table[table["K"] == k]
            cell_names = sorted({(r, p) for r, p in zip(at_k["row"], at_k["payload"])})
            available = {}
            for row in sorted({r for r, _ in cell_names}):
                available[row] = {p for r, p in cell_names if r == row}

            candidates = []
            for role in VERDICT_ROLES:
                for row in VIDEO_ROLES[role]:
                    if row not in available:
                        continue
                    for payload in candidate_payloads(row, available[row],
                                                      VIDEO_HOMOLOGOUS):
                        candidates.append((row, payload))
            controls = [(r, "reuse") for r in VIDEO_ROLES["random"] if r in available]
            m_candidates = len(candidates)

            cell_blocks = {}
            for row, payload in cell_names:
                sub = at_k[(at_k["row"] == row) & (at_k["payload"] == payload)]
                if len(sub) != 2 * VIDEO_N_PROMPTS:
                    raise SystemExit(f"{backbone} K{k} {row}x{payload}: {len(sub)} rows")
                record = manifest.get((f"K{k}", row), {})
                role = role_of(row, VIDEO_ROLES)
                if record.get("off_budget") == "1":
                    role = "off_budget"
                is_candidate = (row, payload) in candidates and role != "off_budget"
                block = {
                    "role": role,
                    "verdict_candidate": is_candidate,
                    "n_seeds": 1,
                    "first_cache_step": (int(record["first_cache_step"])
                                         if record.get("first_cache_step") else None),
                    "payload_feasible": (record.get(f"feasible_{payload}") != "0"
                                         if record else None),
                    "populations": {},
                }
                arrays = {}
                for pop in report_pops:
                    part = sub if pop == "pooled" else sub[sub["dataset"] == pop]
                    part = part.sort_values(["dataset", "prompt_idx"])
                    arrays[pop] = tuple(part[m].to_numpy(dtype=float) for m in METRICS)
                for pop in report_pops:
                    p_, s_, l_ = arrays[pop]
                    pop_block = {
                        "mean_psnr": round(float(p_.mean()), 4),
                        "mean_ssim": round(float(s_.mean()), 6),
                        "mean_lpips": round(float(l_.mean()), 6),
                        "grid": [],
                    }
                    for point in points:
                        event = joint_event(p_, s_, l_, point["psnr_min"],
                                            point["ssim_min"], point["lpips_max"])
                        entry = dict(point)
                        entry.update(coverage_entry(
                            event, alpha,
                            m_candidates if is_candidate else None,
                            len(points) if is_candidate else None))
                        pop_block["grid"].append(entry)
                    block["populations"][pop] = pop_block
                if is_candidate or role in ("random", "ladder_free", "off_budget"):
                    p_, s_, l_ = arrays["pooled"]
                    block["curves_pooled"] = sweep_curves(p_, s_, l_)
                cell_blocks[f"{row}|{payload}"] = block

            gate_blocks = {}
            for row_name, method in VIDEO_GATE_METHOD.items():
                sub = native[(native["method"] == method) & (native["K"] == k)]
                if len(sub) == 0:
                    continue
                block = {"n_seeds": 1, "n_dropped": 2 * VIDEO_N_PROMPTS - len(sub),
                         "populations": {}}
                for pop in report_pops:
                    part = sub if pop == "pooled" else sub[sub["dataset"] == pop]
                    part = part.sort_values(["dataset", "prompt_idx"])
                    p_ = part["psnr"].to_numpy(dtype=float)
                    s_ = part["ssim"].to_numpy(dtype=float)
                    l_ = part["lpips"].to_numpy(dtype=float)
                    pop_block = {"mean_psnr": round(float(p_.mean()), 4), "grid": []}
                    for point in points:
                        event = joint_event(p_, s_, l_, point["psnr_min"],
                                            point["ssim_min"], point["lpips_max"])
                        entry = dict(point)
                        entry.update(coverage_entry(event, alpha))
                        pop_block["grid"].append(entry)
                    block["populations"][pop] = pop_block
                p_ = sub["psnr"].to_numpy(dtype=float)
                s_ = sub["ssim"].to_numpy(dtype=float)
                l_ = sub["lpips"].to_numpy(dtype=float)
                block["curves_pooled"] = sweep_curves(p_, s_, l_)
                gate_blocks[method] = block

            model_block["partitions"][f"K{k}"] = {
                "m_verdict_candidates": m_candidates,
                "candidate_cells": [f"{r}|{p}" for r, p in candidates],
                "control_cells": [f"{r}|{p}" for r, p in controls],
                "cells": cell_blocks,
                "native_gates": gate_blocks,
                "verdicts": build_verdicts(cell_blocks, candidates, controls,
                                           report_pops, points, m_candidates),
            }
        result["models"][backbone] = model_block

    result["populations"] = {"pooled": 2 * VIDEO_N_PROMPTS,
                             "penguin599": VIDEO_N_PROMPTS,
                             "vbench944": VIDEO_N_PROMPTS}
    result["streams"] = {ds: VIDEO_STREAM[ds] for ds in VIDEO_DATASETS}
    if figures:
        make_figures_video(result)
    return result


# ----- figures ---------------------------------------------------------------

FIG_DIR = REPO / "docs/figures/spx_coverage"

ROLE_STYLE = {
    "fixed": {"color": "#1f77b4", "ls": "-", "lw": 1.6},
    "gate_top1": {"color": "#d62728", "ls": "-", "lw": 1.3},
    "ladder_first": {"color": "#2ca02c", "ls": "-", "lw": 1.0},
    "ladder_free": {"color": "#98df8a", "ls": "-", "lw": 0.8},
    "random": {"color": "#7f7f7f", "ls": "--", "lw": 1.6},
    "geometry": {"color": "#9467bd", "ls": ":", "lw": 1.0},
    "rank2": {"color": "#ff7f0e", "ls": "-.", "lw": 0.9},
    "off_budget": {"color": "#bcbd22", "ls": ":", "lw": 0.8},
    "native": {"color": "black", "ls": ":", "lw": 1.6},
}

MODEL_TITLES = {"flux": "FLUX.1-dev", "qwen": "Qwen-Image",
                "hunyuan_video": "HunyuanVideo", "wan21": "Wan2.1"}


def _plot_partition(partition: dict, curve_key: str, title: str, path: Path,
                    pop_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), sharey=True)
    sweep_defs = (
        ("psnr", "PSNR floor (dB)", False),
        ("ssim", "SSIM floor", False),
        ("lpips", "LPIPS ceiling", True),
    )
    mid = ANCHORS["middle"]
    fixed_note = {
        "psnr": f"SSIM>={mid['ssim_min']}, LPIPS<={mid['lpips_max']}",
        "ssim": f"PSNR>={mid['psnr_min']:g}, LPIPS<={mid['lpips_max']}",
        "lpips": f"PSNR>={mid['psnr_min']:g}, SSIM>={mid['ssim_min']}",
    }
    entries = []
    for name, block in sorted(partition["cells"].items()):
        if curve_key in block:
            entries.append((name, block["role"], block[curve_key]))
    for method, block in sorted(partition["native_gates"].items()):
        if curve_key in block:
            entries.append((f"native {method}", "native", block[curve_key]))

    for ax, (sweep, xlabel, _) in zip(axes, sweep_defs):
        for name, role, curves in entries:
            style = ROLE_STYLE.get(role, {"color": "0.6", "ls": "-", "lw": 0.8})
            ax.plot(SWEEPS[sweep], curves[sweep], label=name,
                    color=style["color"], linestyle=style["ls"],
                    linewidth=style["lw"], alpha=0.85)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(f"sweep {sweep} | {fixed_note[sweep]}", fontsize=9)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        if sweep == "lpips":
            ax.invert_xaxis()
    axes[0].set_ylabel("joint coverage", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=6, frameon=False)
    fig.suptitle(f"{title} | population: {pop_label}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.99, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_figures_image(result: dict) -> None:
    for model, block in result["models"].items():
        for k_name, partition in block["partitions"].items():
            _plot_partition(
                partition, "curves_heldout",
                f"{MODEL_TITLES[model]} {k_name} joint coverage",
                FIG_DIR / f"fig_cov_{model}_{k_name}.png",
                "held-out 1088 prompts (validation+test), seed-averaged")


def make_figures_video(result: dict) -> None:
    for backbone, block in result["models"].items():
        for k_name, partition in block["partitions"].items():
            _plot_partition(
                partition, "curves_pooled",
                f"{MODEL_TITLES[backbone]} {k_name} joint coverage",
                FIG_DIR / f"fig_cov_{backbone}_{k_name}.png",
                "pooled 300 prompts (penguin599 s54 + vbench944 s42), single stream")


# ----- entry point -----------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--modality", choices=("image", "video", "all"), default="all")
    args = parser.parse_args()
    figures = not args.no_figures

    header = {
        "schema": "spx_coverage.v1",
        "generated_by": "analysis/spx_coverage.py",
        "hypothesis": ("paper/main.tex Hypothesis 1 (Golden Path): exists u* with "
                       "Pr_x[Delta_j(u*;x) <= eps_j for all j in J_R] >= 1-delta; "
                       "J_R = {PSNR, SSIM, LPIPS}; prompts are the independent units"),
        "anchors": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in ANCHORS.items()},
        "sweeps": SWEEPS,
        "anchor_declaration": (
            "The anchor thresholds and delta values are conventional fidelity "
            "levels fixed in this script before any coverage number was "
            "computed. Cell MEANS had been seen in the SPX reports before this "
            "test; per-prompt joint coverage had not. The complete sweep curves "
            "keep the conclusion threshold-independent."),
    }

    if args.modality in ("image", "all"):
        image = analyse_image(figures)
        blob = dict(header)
        blob["modality"] = "image"
        blob.update(image)
        out = REPO / "resources/spx/coverage_results.json"
        out.write_text(json.dumps(blob, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")

    if args.modality in ("video", "all"):
        video = analyse_video(figures)
        blob = dict(header)
        blob["modality"] = "video"
        blob.update(video)
        out = REPO / "resources/video_spx/coverage_results.json"
        out.write_text(json.dumps(blob, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
