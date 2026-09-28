#!/usr/bin/env python3
"""Figures and tables for the SPX (schedule x payload cross) results report.

Inputs
------
* `--perprompt` -- the staged per-image tables
  `resources/spx/perprompt_spx_<model>.tsv.gz`
  (`analysis/stage_spx_perprompt.py`), one row per
  (schedule, payload, K, seed, prompt).
* `--supplement` -- `resources/sp_cross_supplement/sp_cross_results.json`
  (`analysis/spx_supplement.py`), which carries the paired statistics the
  control rows exist for. Figures 4-7 are drawn only when it is present.
* `--reference` -- P4 native-gate reference TSV
  (`resources/sp_cross_schedules/native_reference_parti.tsv`); the pooled
  reference is an appendix reading now that P4 is paired per prompt.

The cell census is a list, not a count: `expected_cells()` names every
(model, K, schedule, payload) the frozen manifests declare, so a missing cell
is reported by name and a new control row does not trip an equality assert.

Outputs (`--out`, default `docs/figures/spx/`)
----------------------------------------------
Only three figures are drawn; everything a table can say stays a table.

fig1_cross_psnr        seed-averaged PSNR heatmaps, all schedules x 5 payloads
fig2_gamma_psnr        interaction residual gamma_sp on the balanced W1 grid
fig3_payload_vs_k      payload marginal means (over W1 schedules) vs K
tables.md              numeric tables: Fig 1 cell means, Fig 2 gamma grids,
                       W1 variance shares, alpha_s / beta_p main effects,
                       Fig 3 payload marginals, P4 fixed-vs-native deltas
                       (incl. the DiCache-path x reuse mispairing) with the
                       2-pooled-seed-SD band, diagonal gamma per metric
full_results_table.md  all 180 (model, K, schedule, payload) cells: n_seeds
                       and seed mean +- seed SD for PSNR / SSIM / LPIPS /
                       ImageReward / CLIP (per-seed value = mean over 1632
                       PartiPrompts)

The two-way decomposition and the seed band are imported from
`analysis/sp_cross.py` (`decompose`, `pooled_seed_band`, `HOMOLOGOUS`,
`orient`) so the numbers use exactly the fit the pre-registered analysis used.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from analysis.sp_cross import (  # noqa: E402
    HOMOLOGOUS,
    P4_GATES,
    PAYLOADS,
    decompose,
    orient,
    parse_cell_name,
    pooled_seed_band,
    read_reference,
)

# ----- display conventions ---------------------------------------------------

MODELS = ("flux", "qwen")
MODEL_LABEL = {"flux": "FLUX.1-dev", "qwen": "Qwen-Image"}
KS = (29, 37, 41)
#: The random exact-K reference rows: two everywhere, five on the four
#: densified partitions (`analysis/spx_supplement.RANDOM_ROWS`).
RANDOM_ROWS = ("rand_1", "rand_2", "rand_3", "rand_4", "rand_5")
W1 = ("budcache", "dpcache", "uniform", "dicache_top1")
# The two gpf_o1 family members are one display row (e15 at K29/K37, e20 at K41).
GPF_ALIAS = {"gpf_reuse_e05_1": "gpf_reuse", "gpf_o1_e15_1": "gpf_o1", "gpf_o1_e20_1": "gpf_o1"}
SCHEDULE_ORDER = (
    "budcache", "meancache", "dpcache", "uniform",
    "seacache_top1", "teacache_top1", "sencache_top1", "dicache_top1",
    "gpf_reuse", "gpf_o1",
    # supplement control rows, then the two off-budget / rank-2 rows
    "ham2f", "ham4f", "ham8f", "dp_rho2", "rand_1", "rand_2",
    "sencache_top1_off", "dicache_top1_r2", "teacache_top1_r2",
)
SCHEDULE_LABEL = {
    "budcache": "BudCache", "meancache": "MeanCache", "dpcache": "DPCache",
    "uniform": "uniform", "seacache_top1": "SeaCache top-1",
    "teacache_top1": "TeaCache top-1", "sencache_top1": "SenCache top-1",
    "dicache_top1": "DiCache top-1", "gpf_reuse": "GPF (reuse)", "gpf_o1": "GPF (O1)",
    "ham2f": "MeanCache d=2", "ham4f": "MeanCache d=4", "ham8f": "MeanCache d=8",
    "dp_rho2": "ρ₂-DP", "rand_1": "random 1", "rand_2": "random 2",
    "sencache_top1_off": "SenCache path (28)", "dicache_top1_r2": "DiCache rank-2",
    "teacache_top1_r2": "TeaCache rank-2",
}
PAYLOAD_LABEL = {
    "reuse": "reuse", "taylor_o1": "Taylor O1", "hermite_o2": "Hermite O2",
    "mean_avg_vel": "mean-vel", "di_two_anchor": "DI 2-anchor",
}
GATE_LABEL = {"seacache_top1": "SeaCache", "teacache_top1": "TeaCache",
              "sencache_top1": "SenCache", "dicache_top1": "DiCache"}
DISPLAY_HOMOLOGOUS = {GPF_ALIAS.get(k, k): v for k, v in HOMOLOGOUS.items()}
# Okabe-Ito, fixed assignment per entity (never cycled).
OI = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermil": "#D55E00",
    "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000",
}
PAYLOAD_COLOR = {"reuse": OI["black"], "taylor_o1": OI["sky"], "hermite_o2": OI["vermil"],
                 "mean_avg_vel": OI["green"], "di_two_anchor": OI["purple"]}
PAYLOAD_MARKER = {"reuse": "o", "taylor_o1": "s", "hermite_o2": "^", "mean_avg_vel": "D",
                  "di_two_anchor": "v"}
# full results table: (metric key, column header, format)
FULL_TABLE_METRICS = (
    ("psnr", "PSNR", "{:.2f}"), ("ssim", "SSIM", "{:.4f}"), ("lpips", "LPIPS", "{:.4f}"),
    ("image_reward", "ImageReward", "{:.3f}"), ("clip", "CLIP", "{:.2f}"),
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})


# ----- data ------------------------------------------------------------------


METRIC_ORDER = ("psnr", "ssim", "lpips", "image_reward", "clip")


def manifest_rows(known: bool = False) -> dict[tuple[str, int], list[str]]:
    """(model, K) -> schedule rows the frozen manifests declare.

    Two different questions need two different sets. `expected` (default) is
    what must be present, and excludes `gpf_manifest.tsv`: that file enumerates
    a whole constructed family of which only a screened handful was ever meant
    to run, so requiring all of it would report ten phantom absences per budget.
    `known=True` adds the family, so the handful that DID run is not reported as
    an unexplained extra.
    """
    names = ["sp_cross_schedules/manifest.tsv",
             "spx_supplement_schedules/manifest.tsv"]
    if known:
        names.append("sp_cross_schedules/gpf_manifest.tsv")
    rows: dict[tuple[str, int], list[str]] = {}
    for name in names:
        path = Path("resources") / name
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                key = (row["model"], int(row["target_k"]))
                rows.setdefault(key, [])
                if row["name"] not in rows[key]:
                    rows[key].append(row["name"])
    return rows


def load_cells(paths: Sequence[Path]) -> dict:
    """Read the staged per-image tables into

    {"cells": {(model, K, schedule, payload, seed): (5, n) array},
     "metrics": [...], "raw_schedule": {(model, K, schedule): on-disk name}}.
    """
    buckets: dict[tuple, dict[str, dict[int, float]]] = {}
    for path in paths:
        model = path.stem.replace("perprompt_spx_", "").replace(".tsv", "")
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                raw = row["schedule"]
                key = (model, int(row["k"]), GPF_ALIAS.get(raw, raw), row["payload"],
                       int(row["seed"]), raw)
                bucket = buckets.setdefault(key, {metric: {} for metric in METRIC_ORDER})
                prompt = int(row["prompt_idx"])
                for metric in METRIC_ORDER:
                    text = row.get(metric)
                    if text:
                        bucket[metric][prompt] = float(text)
    cells, raw_schedule = {}, {}
    prompt_sets: set[tuple[int, ...]] = set()
    for key, bucket in buckets.items():
        model, budget_k, schedule, payload, seed, raw = key
        prompts = sorted(bucket["psnr"])
        prompt_sets.add(tuple(prompts))
        array = np.array(
            [[bucket[metric].get(prompt, np.nan) for prompt in prompts]
             for metric in METRIC_ORDER],
            dtype=float,
        )
        cell_key = (model, budget_k, schedule, payload, seed)
        assert cell_key not in cells, f"duplicate {cell_key}"
        cells[cell_key] = array
        assert raw_schedule.setdefault(cell_key[:3], raw) == raw, (cell_key, raw)
    assert len(prompt_sets) == 1, (
        f"cells disagree about which prompts they cover: {len(prompt_sets)} distinct sets"
    )
    return {"cells": cells, "metrics": list(METRIC_ORDER), "raw_schedule": raw_schedule,
            "n_prompts": len(next(iter(prompt_sets)))}


def dense_rows() -> frozenset[str]:
    """Row names the frozen manifest marks as densification rows."""

    path = _ROOT / "resources/spx_supplement_schedules/manifest.tsv"
    if not path.exists():
        return frozenset()
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    name, group = header.index("name"), header.index("group")
    return frozenset(
        parts[name] for parts in (line.split("\t") for line in lines[1:])
        if parts[group] == "dense"
    )


def sanity(store: dict) -> None:
    """Census by name. A missing cell is named; an extra one is named too."""
    metrics = store["metrics"]
    ip = metrics.index("psnr")
    for model in MODELS:
        vals = [c[ip].mean() for (m, *_), c in store["cells"].items() if m == model]
        lo, hi = min(vals), max(vals)
        print(f"[sanity] {model}: {len(vals)} runs, per-run mean PSNR range {lo:.2f}..{hi:.2f} dB")
        if lo < 15:
            low = sorted(
                ((c[ip].mean(), k) for k, c in store["cells"].items() if k[0] == model)
            )[:6]
            print("[sanity]   NOTE runs below 15 dB (higher-order payloads at K41 collapse):")
            for v, k in low:
                print(f"[sanity]     {k[1:]} -> {v:.2f} dB")
    counts: dict[tuple, int] = {}
    for (model, k, s, p, _seed) in store["cells"]:
        counts[(model, k, s, p)] = counts.get((model, k, s, p), 0) + 1
    # Three seed streams everywhere EXCEPT the densification rows, which run
    # one by design (plan section 12). Naming the exception here rather than
    # loosening the check keeps a genuinely half-run cell an assertion.
    dense = dense_rows()
    expected = {key: (1 if key[2] in dense else 3) for key in counts}
    bad = {k: v for k, v in counts.items() if v != expected[k]}
    assert not bad, (
        "cells whose seed count is neither the frozen 3 nor the dense 1: "
        f"{ {k: (v, expected[k]) for k, v in bad.items()} }"
    )
    single = sorted({k for k, v in counts.items() if v == 1})
    if single:
        print(f"[sanity] {len(single)} single-seed cells (densification rows, seed 42)")
    declared = manifest_rows()
    known = manifest_rows(known=True)
    observed: dict[tuple[str, int], set[str]] = {}
    for (model, k, _s, _p) in counts:
        observed.setdefault((model, k), set()).add(store["raw_schedule"][(model, k, _s)])
    missing = {
        key: sorted(set(rows) - observed.get(key, set()))
        for key, rows in declared.items()
        if set(rows) - observed.get(key, set())
    }
    if missing:
        print(f"[sanity] declared but not staged: {missing}")
    extra = {
        key: sorted(rows - set(known.get(key, ())))
        for key, rows in observed.items()
        if rows - set(known.get(key, ()))
    }
    if extra:
        print(f"[sanity] staged but not in any manifest: {extra}")
    n_three = sum(1 for k, v in counts.items() if v == 3)
    print(f"[sanity] {len(counts)} (model,K,schedule,payload) cells "
          f"({n_three} x 3 seeds + {len(single)} x 1 seed) "
          f"= {len(store['cells'])} runs over {store['n_prompts']} prompts")


def panel_entries(store: dict, model: str, k: int, metric: str, *, oriented: bool = True) -> dict:
    """(schedule, payload) -> {value (seed-mean, oriented), seeds, seed_sd, n_seeds}."""
    im = store["metrics"].index(metric)
    out: dict = {}
    for (m, kk, s, p, seed), arr in store["cells"].items():
        if m != model or kk != k:
            continue
        v = float(arr[im].mean())
        out.setdefault((s, p), {"seeds": {}})["seeds"][seed] = orient(v, metric) if oriented else v
    for entry in out.values():
        vals = list(entry["seeds"].values())
        entry["value"] = float(statistics.fmean(vals))
        entry["n_seeds"] = len(vals)
        entry["seed_sd"] = float(statistics.stdev(vals)) if len(vals) >= 2 else None
    return out


def w1_decomposition(entries: dict) -> dict:
    matrix = {(s, p): entries[(s, p)]["value"] for s in W1 for p in PAYLOADS if (s, p) in entries}
    missing = [f"{s}x{p}" for s in W1 for p in PAYLOADS if (s, p) not in entries]
    assert not missing, f"W1 grid incomplete, missing: {', '.join(missing)}"
    dec = decompose(matrix)
    n_s, n_p = len(W1), len(PAYLOADS)
    ss_s = n_p * sum(a * a for a in dec["alpha"].values())
    ss_p = n_s * sum(b * b for b in dec["beta"].values())
    ss_g = sum(g * g for g in dec["gamma_pairs"].values())
    ss_tot = sum((v - dec["mu"]) ** 2 for v in matrix.values())
    dec["ss"] = {"schedule": ss_s, "payload": ss_p, "interaction": ss_g, "total": ss_tot}
    # A metric that is constant across the grid has nothing to share out; that
    # is a degenerate panel, not a crash.
    dec["share"] = {
        key: (ss / ss_tot if ss_tot > 0 else None)
        for key, ss in dec["ss"].items() if key != "total"
    }
    dec["matrix"] = matrix
    return dec


def diagonal_cells(schedules=W1) -> list[tuple[str, str]]:
    return [(s, p) for s in schedules for p in DISPLAY_HOMOLOGOUS.get(s, ()) if p in PAYLOADS]


# ----- helpers ---------------------------------------------------------------


def savefig(fig, out: Path, stem: str) -> list[Path]:
    paths = []
    for ext in ("png", "pdf"):
        path = out / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def draw_heatmap(ax, grid, row_keys, col_keys, *, norm, cmap, fmt="{:.1f}",
                 bold_cells=(), hatch_missing=True):
    """grid: 2D array with NaN for missing. Returns the image."""
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    n_r, n_c = grid.shape
    for r in range(n_r):
        for c in range(n_c):
            v = grid[r, c]
            if np.isnan(v):
                if hatch_missing:
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="///",
                                           edgecolor="#b0b0b0", linewidth=0))
                continue
            rgba = cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if lum < 0.5 else "black"
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=7.5, color=color)
    for (r, c) in bold_cells:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="black",
                               linewidth=2.2, zorder=5))
    ax.set_xticks(range(n_c))
    ax.set_xticklabels([PAYLOAD_LABEL[p] for p in col_keys], rotation=35, ha="right")
    ax.set_yticks(range(n_r))
    ax.set_yticklabels([SCHEDULE_LABEL[s] for s in row_keys])
    ax.set_xticks(np.arange(-0.5, n_c, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_r, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


# ----- figures ---------------------------------------------------------------


def fig1_cross(store, out, tables):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.4), constrained_layout=True)
    for r, model in enumerate(MODELS):
        rows = [s for s in SCHEDULE_ORDER
                if any(k[0] == model and k[2] == s for k in store["cells"])]
        grids = {}
        for k in KS:
            entries = panel_entries(store, model, k, "psnr")
            grid = np.full((len(rows), len(PAYLOADS)), np.nan)
            for i, s in enumerate(rows):
                for j, p in enumerate(PAYLOADS):
                    if (s, p) in entries:
                        grid[i, j] = entries[(s, p)]["value"]
            grids[k] = grid
        vmin = np.nanmin([g for g in grids.values()])
        vmax = np.nanmax([g for g in grids.values()])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        bold = [(rows.index(s), PAYLOADS.index(p)) for (s, p) in diagonal_cells(rows)]
        for c, k in enumerate(KS):
            ax = axes[r, c]
            im = draw_heatmap(ax, grids[k], rows, PAYLOADS, norm=norm, cmap="viridis",
                              bold_cells=bold)
            ax.set_title(f"{MODEL_LABEL[model]}, K = {k} cached steps")
            if c == 0:
                ax.set_ylabel("schedule")
            if r == 1:
                ax.set_xlabel("payload")
            tables.append((f"Fig 1 cell means, PSNR dB, seed mean (n=3 seeds x 1632 prompts): {model} K{k}",
                           ["schedule"] + [PAYLOAD_LABEL[p] for p in PAYLOADS],
                           [[SCHEDULE_LABEL[s]] + [("--" if np.isnan(grids[k][i, j]) else f"{grids[k][i, j]:.2f}")
                                                   for j in range(len(PAYLOADS))]
                            for i, s in enumerate(rows)]))
        cbar = fig.colorbar(im, ax=axes[r, :].tolist(), shrink=0.9, pad=0.01)
        cbar.set_label(f"PSNR vs full-step reference (dB), {MODEL_LABEL[model]}")
    fig.suptitle("SPX cross: seed-averaged PSNR for every schedule x payload cell "
                 "(bold border = pre-registered homologous pair; hatched = not run)", fontsize=10.5)
    return savefig(fig, out, "fig1_cross_psnr")


def fig2_gamma(store, out, tables, decs):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), constrained_layout=True)
    grids = {}
    for model in MODELS:
        for k in KS:
            dec = decs[(model, k)]
            grid = np.array([[dec["gamma_pairs"][(s, p)] for p in PAYLOADS] for s in W1])
            grids[(model, k)] = grid
    lim = max(np.abs(g).max() for g in grids.values())
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    bold = [(W1.index(s), PAYLOADS.index(p)) for (s, p) in diagonal_cells()]
    diag_means = {}
    for r, model in enumerate(MODELS):
        for c, k in enumerate(KS):
            ax = axes[r, c]
            grid = grids[(model, k)]
            im = draw_heatmap(ax, grid, W1, PAYLOADS, norm=norm, cmap="RdBu", fmt="{:+.2f}",
                              bold_cells=bold)
            dvals = [grid[i, j] for (i, j) in bold]
            n_pos = sum(1 for v in dvals if v > 1e-9)
            diag_means[(model, k)] = (float(np.mean(dvals)), n_pos, len(dvals))
            ax.set_title(f"{MODEL_LABEL[model]}, K = {k}\n"
                         f"diag mean γ = {np.mean(dvals):+.2f} dB, {n_pos}/{len(dvals)} positive",
                         fontsize=9.5)
            if c == 0:
                ax.set_ylabel("schedule")
            if r == 1:
                ax.set_xlabel("payload")
            tables.append((f"Fig 2 interaction gamma_sp, PSNR dB (W1 grid): {model} K{k}",
                           ["schedule"] + [PAYLOAD_LABEL[p] for p in PAYLOADS],
                           [[SCHEDULE_LABEL[s]] + [f"{grid[i, j]:+.3f}" for j in range(len(PAYLOADS))]
                            for i, s in enumerate(W1)]))
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.01)
    cbar.set_label("interaction γ_sp = y_sp − (μ + α_s + β_p)  [dB PSNR]")
    fig.suptitle("P1 evidence: interaction residual on the balanced W1 grid. Homologous "
                 "(bold) cells are not systematically positive.", fontsize=10.5)
    return savefig(fig, out, "fig2_gamma_psnr"), diag_means


def fig3_payload_vs_k(store, out, tables):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), constrained_layout=True)
    rows = []
    for c, model in enumerate(MODELS):
        ax = axes[c]
        for p in PAYLOADS:
            means, spreads = [], []
            for k in KS:
                entries = panel_entries(store, model, k, "psnr")
                vals = [entries[(s, p)]["value"] for s in W1]
                means.append(float(np.mean(vals)))
                spreads.append((min(vals), max(vals)))
                # faint per-schedule markers, dodged by payload so columns do not overlap
                x_off = 0.28 * (PAYLOADS.index(p) - 2)
                for s in W1:
                    ax.plot(k + x_off, entries[(s, p)]["value"], marker=PAYLOAD_MARKER[p],
                            color=PAYLOAD_COLOR[p], ms=3.2, alpha=0.35, lw=0)
            ax.plot(KS, means, marker=PAYLOAD_MARKER[p], color=PAYLOAD_COLOR[p],
                    label=PAYLOAD_LABEL[p], lw=2.2, ms=7)
            rows.append([model, PAYLOAD_LABEL[p]] + [f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
                                                    for m, (lo, hi) in zip(means, spreads)])
        ax.set_xticks(KS)
        ax.set_xlabel("K (cached steps of 50)")
        ax.set_ylabel("PSNR (dB), mean over 4 W1 schedules")
        ax.set_title(f"{MODEL_LABEL[model]}")
        if c == 1:
            ax.legend(frameon=False, loc="lower left", title="payload")
    fig.suptitle("P3: payload marginal means vs compression (faint markers = the 4 individual W1 schedules)",
                 fontsize=10.5)
    tables.append(("Fig 3 payload marginal mean PSNR over W1 schedules (dB) [min, max across schedules]",
                   ["model", "payload"] + [f"K{k}" for k in KS], rows))
    return savefig(fig, out, "fig3_payload_vs_k")


# ----- supplement figures (need `--supplement`) ------------------------------


ROW_LABEL_EXTRA = {
    "rand_1": "random 1", "rand_2": "random 2", "ham2f": "MeanCache d=2",
    "ham4f": "MeanCache d=4", "ham8f": "MeanCache d=8", "dp_rho2": "ρ₂-DP",
    "sencache_top1_off": "SenCache path (28 steps)",
    "dicache_top1_r2": "DiCache rank-2", "teacache_top1_r2": "TeaCache rank-2",
    "gpf_reuse_e05_1": "GPF (reuse)", "gpf_o1_e15_1": "GPF (O1)",
    "gpf_o1_e20_1": "GPF (O1)", "meancache": "MeanCache",
}


def row_label(row: str) -> str:
    return ROW_LABEL_EXTRA.get(row) or SCHEDULE_LABEL.get(row, row)


def panels_of(supplement: dict, split: str = "all") -> list[dict]:
    return supplement["splits"][split]["panels"]


def fig4_p4_offmodal(supplement, out, tables):
    """P4 the way the pairing actually works: modal pairs are a deterministic
    zero, so the readable number is the off-modal paired difference."""
    panels = panels_of(supplement)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4), constrained_layout=True, sharey="row")
    rows = []
    for r, model in enumerate(MODELS):
        for c, k in enumerate(KS):
            ax = axes[r, c]
            panel = next(p for p in panels if p["model"] == model and p["budget_k"] == k)
            gates = panel["P4"]["psnr"]["gates"]
            ys = np.arange(len(gates))
            for i, entry in enumerate(gates):
                off = entry.get("offmodal")
                if off is None:
                    continue
                ax.errorbar(off["mean"], i,
                            xerr=[[off["mean"] - off["ci_low"]], [off["ci_high"] - off["mean"]]],
                            fmt="o", ms=5, color=OI["blue"], capsize=3, lw=1.6)
                allp = entry.get("all")
                if allp is not None:
                    ax.plot(allp["mean"], i, marker="|", ms=11, color=OI["orange"], lw=0)
                rows.append([model, str(k), GATE_LABEL.get(entry["gate"] + "_top1", entry["gate"]),
                             f"{entry['modal_mass']*100:.1f}" if entry["modal_mass"] is not None else "--",
                             str(entry["n_offmodal"]),
                             f"{off['mean']:+.3f}", f"±{2*off['se']:.3f}",
                             f"{allp['mean']:+.3f}" if allp else "--",
                             ("0.0000" if entry.get("modal_max_abs") is not None
                              else "--") if entry["n_modal"] else "--"])
            ax.axvline(0.0, color="0.35", lw=1.0, ls="--")
            ax.set_yticks(ys)
            ax.set_yticklabels([
                f"{GATE_LABEL.get(e['gate'] + '_top1', e['gate'])}\n"
                f"modal {e['modal_mass']*100:.0f}%" if e["modal_mass"] is not None else e["gate"]
                for e in gates], fontsize=8)
            ax.set_title(f"{MODEL_LABEL[model]}, K = {k}", fontsize=9.5)
            if r == 1:
                ax.set_xlabel("fixed modal path − gate's own run (dB PSNR)")
            ax.invert_yaxis()
    fig.suptitle("P4: on off-modal prompts only (blue, ±2 paired SE); the orange tick is the "
                 "all-prompt mean the modal pairs dilute toward zero", fontsize=10.5)
    tables.append(("P4 per-prompt paired difference, fixed modal path − native gate run, PSNR dB",
                   ["model", "K", "gate", "modal mass %", "n off-modal", "off-modal mean",
                    "±2 paired SE", "all-prompt mean", "modal |max diff|"], rows))
    return savefig(fig, out, "fig4_p4_offmodal")


def fig5_ps1_rows(supplement, out, tables):
    """P-S1: every reuse-column row against MeanCache, with the random band."""
    panels = panels_of(supplement)
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.2), constrained_layout=True)
    rows_out = []
    for r, model in enumerate(MODELS):
        for c, k in enumerate(KS):
            ax = axes[r, c]
            panel = next(p for p in panels if p["model"] == model and p["budget_k"] == k)
            block = panel["PS1"]["psnr"]
            entries = block["rows"]
            labels, means, los, his, colors = [], [], [], [], []
            for entry in entries:
                diff = entry["paired_vs_meancache"]
                labels.append(row_label(entry["row"]))
                means.append(diff["mean"])
                los.append(diff["mean"] - diff["ci_low"])
                his.append(diff["ci_high"] - diff["mean"])
                colors.append(OI["orange"] if entry["row"] in ("rand_1", "rand_2")
                              else OI["green"] if entry["row"].startswith("ham")
                              else OI["purple"] if entry["row"] == "dp_rho2"
                              else OI["blue"])
                rows_out.append([model, str(k), row_label(entry["row"]), entry.get("group", ""),
                                 f"{entry['mean_value']:.2f}", f"{diff['mean']:+.3f}",
                                 f"±{2*diff['se']:.3f}", str(diff["n_pairs"])])
            ys = np.arange(len(labels))
            ax.errorbar(means, ys, xerr=[los, his], fmt="none", ecolor="0.5", lw=1.2, capsize=2)
            ax.scatter(means, ys, c=colors, s=26, zorder=3)
            if block.get("random_rows"):
                ax.axvspan(min(block["random_rows"]), max(block["random_rows"]),
                           color=OI["orange"], alpha=0.13, lw=0)
            ax.axvline(0.0, color="0.25", lw=1.1, ls="--")
            ax.set_yticks(ys)
            ax.set_yticklabels(labels, fontsize=7.5)
            ax.invert_yaxis()
            ax.set_title(f"{MODEL_LABEL[model]}, K = {k}\nMeanCache = {block['anchor_value']:.2f} dB",
                         fontsize=9.3)
            if r == 1:
                ax.set_xlabel("row − MeanCache, paired per prompt (dB PSNR)")
    fig.suptitle("P-S1: how many schedules reach the best one. Shaded band = the two random "
                 "exact-K rows, i.e. the level a path with no design reaches.", fontsize=10.5)
    tables.append(("P-S1 reuse column, paired difference against the MeanCache row, PSNR dB",
                   ["model", "K", "row", "group", "mean PSNR", "vs MeanCache", "±2 paired SE", "n pairs"],
                   rows_out))
    return savefig(fig, out, "fig5_ps1_rows")


def fig6_ps2_dose(supplement, out, tables):
    """P-S2: the Hamming dose curve away from MeanCache."""
    panels = panels_of(supplement)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True, sharey=True)
    rows_out = []
    for c, model in enumerate(MODELS):
        ax = axes[c]
        for k, colour in zip(KS, (OI["blue"], OI["green"], OI["vermil"])):
            panel = next(p for p in panels if p["model"] == model and p["budget_k"] == k)
            block_ps2 = panel["PS2"]["psnr"]
            for family, style in (("ladder", "-"), ("ladder_free", ":")):
                ladder = block_ps2.get(family) or []
                if not ladder:
                    continue
                xs = [entry["hamming"] for entry in ladder]
                ys = [entry["delta"] for entry in ladder]
                # The error bar is the DRAW RANGE wherever a rung has more than
                # one draw, and the paired band only where it has one. Mixing
                # them on one curve would be wrong; the range dominates and is
                # the uncertainty this figure is about.
                lo, hi = [], []
                for entry in ladder:
                    if entry.get("n_draws", 1) > 1:
                        lo.append(entry["delta"] - entry["delta_min"])
                        hi.append(entry["delta_max"] - entry["delta"])
                    else:
                        lo.append(entry["delta"] - entry["ci_low"])
                        hi.append(entry["ci_high"] - entry["delta"])
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="o" if style == "-" else "s",
                            ms=5, lw=1.8, color=colour, ls=style, capsize=3,
                            label=(f"K = {k}" if style == "-" else None))
                for entry in ladder:
                    rows_out.append([
                        model, str(k),
                        "preserving" if family == "ladder" else "free",
                        str(entry["hamming"]), entry["row"], str(entry.get("n_draws", 1)),
                        f"{entry['delta']:+.3f}",
                        (f"[{entry['delta_min']:+.3f}, {entry['delta_max']:+.3f}]"
                         if entry.get("n_draws", 1) > 1 else "—"),
                        f"[{entry['ci_low']:+.3f}, {entry['ci_high']:+.3f}]",
                    ])
            # the random rows as the far end of the same axis
            block = panel["PS1"]["psnr"]
            by_row = {e["row"]: e for e in block["rows"]}
            for name in RANDOM_ROWS:
                if name in by_row:
                    ax.plot(by_row[name].get("hamming_to_meancache", 20),
                            by_row[name]["paired_vs_meancache"]["mean"],
                            marker="x", ms=6, color=colour, lw=0, alpha=0.8)
        ax.axhline(0.0, color="0.3", lw=1.0, ls="--")
        # The window differs per curve, so the axis names the distance only.
        ax.set_xlabel("Hamming distance from the MeanCache path")
        if c == 0:
            ax.set_ylabel("paired difference vs MeanCache (dB PSNR)")
        ax.set_title(MODEL_LABEL[model], fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("P-S2: dose. Moving a fixed number of cached steps off the best path "
                 "at fixed budget; solid = first cache step held, dotted = free "
                 "(× = the random rows). Bars are the draw range where a rung has "
                 "more than one draw.", fontsize=10.5)
    tables.append(("P-S2 Hamming ladder, paired difference vs MeanCache, PSNR dB",
                   ["model", "K", "ladder", "Hamming", "median row", "n draws",
                    "median delta", "draw range", "95% band (±2 paired SE)"], rows_out))
    return savefig(fig, out, "fig6_ps2_dose")


def fig7_ps4_geometry(supplement, out, tables):
    """P-S4: what geometry explains about which schedules are good."""
    pooled = supplement["splits"]["all"]["PS4_pooled"]
    partitions = supplement["splits"]["all"]["PS4_partitions"]
    names = [name for name in pooled if pooled[name] is not None]
    names.sort(key=lambda name: -abs(pooled[name]["rho"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    ys = np.arange(len(names))
    rhos = [pooled[name]["rho"] for name in names]
    ax.barh(ys, rhos, color=[OI["blue"] if r > 0 else OI["vermil"] for r in rhos], height=0.68)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0.0, color="0.25", lw=1.0)
    ax.set_xlabel("pooled Spearman ρ (within-partition ranks, 6 partitions)")
    ax.set_title("geometry vs row quality", fontsize=10)
    rows_out = [[name, f"{pooled[name]['rho']:+.3f}", f"{pooled[name]['p']:.3g}",
                 str(pooled[name]["n"]), str(pooled[name].get("n_partitions", ""))]
                for name in names]
    best = names[0] if names else None
    ax2 = axes[1]
    if best:
        for block, colour in zip(partitions, plt.get_cmap("viridis")(np.linspace(0, 0.85, len(partitions)))):
            xs = [row.get(best) for row in block["rows"]]
            qs = [row["quality"] for row in block["rows"]]
            keep = [(x, q) for x, q in zip(xs, qs) if x is not None]
            if len(keep) < 2:
                continue
            ax2.plot([x for x, _ in keep], [q for _, q in keep], "o", ms=4.5, color=colour,
                     label=f"{block['model']} K{block['budget_k']}", lw=0)
        ax2.set_xlabel(best)
        ax2.set_ylabel("row mean PSNR, reuse column (dB)")
        ax2.set_title(f"strongest pooled predictor: {best}", fontsize=10)
        ax2.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("P-S4: geometry of a schedule against the quality it delivers, over every "
                 "on-budget row of the reuse column", fontsize=10.5)
    tables.append(("P-S4 pooled Spearman of row quality (reuse column, PSNR) on geometry",
                   ["predictor", "rho", "p", "n rows pooled", "n partitions"], rows_out))
    per_partition = []
    for block in partitions:
        for name in names:
            entry = block["spearman"].get(name)
            if entry is None:
                continue
            per_partition.append([block["model"], str(block["budget_k"]), name,
                                  f"{entry['rho']:+.3f}", f"{entry['p']:.3g}", str(entry["n"])])
    tables.append(("P-S4 per-partition Spearman", ["model", "K", "predictor", "rho", "p", "n"],
                   per_partition))
    return savefig(fig, out, "fig7_ps4_geometry")


# ----- table-only computations (no figure) ------------------------------------


def variance_table(tables, decs):
    rows = []
    for model in MODELS:
        for k in KS:
            share = decs[(model, k)]["share"]
            rows.append([model, str(k)] + [
                "--" if share[c] is None else f"{share[c] * 100:.1f}"
                for c in ("schedule", "payload", "interaction")]
                        + [f"{decs[(model, k)]['ss']['total']:.2f}"])
    tables.append(("W1 variance shares (%, PSNR, W1 grid; last col = total SS in dB^2)",
                   ["model", "K", "schedule (alpha)", "payload (beta)", "interaction (gamma)", "total SS"],
                   rows))


def main_effects_table(tables, decs):
    alpha_rows, beta_rows = [], []
    for model in MODELS:
        for s in W1:
            ys = [decs[(model, k)]["alpha"][s] for k in KS]
            alpha_rows.append([model, SCHEDULE_LABEL[s]] + [f"{y:+.2f}" for y in ys])
        for p in PAYLOADS:
            ys = [decs[(model, k)]["beta"][p] for k in KS]
            beta_rows.append([model, PAYLOAD_LABEL[p]] + [f"{y:+.2f}" for y in ys])
    tables.append(("W1 schedule main effect alpha_s (dB PSNR)", ["model", "schedule"] + [f"K{k}" for k in KS], alpha_rows))
    tables.append(("W1 payload main effect beta_p (dB PSNR)", ["model", "payload"] + [f"K{k}" for k in KS], beta_rows))


def p4_table(store, tables, reference):
    rows = []
    p4_deltas = {}
    for model in MODELS:
        for k in KS:
            entries = panel_entries(store, model, k, "psnr")
            band = pooled_seed_band(entries)
            for sched, method in P4_GATES.items():
                native = HOMOLOGOUS[sched][0]
                ref = reference.get((model, method, k, "psnr"))
                if ref is None:
                    continue
                fixed = entries[(sched, native)]
                delta = fixed["value"] - ref
                p4_deltas[(model, k, method)] = (delta, band)
                rows.append([model, str(k), GATE_LABEL[sched], PAYLOAD_LABEL[native],
                             f"{fixed['value']:.2f}", f"{ref:.2f}", f"{delta:+.2f}", f"±{band:.2f}",
                             "yes" if abs(delta) <= band else "no"])
                if sched == "dicache_top1":
                    mis = entries[(sched, "reuse")]
                    dmis = mis["value"] - ref
                    rows.append([model, str(k), GATE_LABEL[sched] + " (mispaired)", "reuse",
                                 f"{mis['value']:.2f}", f"{ref:.2f}", f"{dmis:+.2f}", f"±{band:.2f}",
                                 "yes" if abs(dmis) <= band else "no"])
    tables.append(("P4: fixed modal path (native payload, seed mean) vs native gated reference, PSNR dB",
                   ["model", "K", "method", "payload", "fixed", "native ref", "delta", "band (2 pooled seed SD)", "within band"],
                   rows))
    return p4_deltas


def diag_metrics_table(store, tables):
    rows = []
    diag = diagonal_cells()
    for metric in store["metrics"]:
        for model in MODELS:
            for k in KS:
                dec = w1_decomposition(panel_entries(store, model, k, metric))
                g = [dec["gamma_pairs"][(s, p)] for (s, p) in diag]
                scale = max((abs(v) for v in dec["gamma_pairs"].values()), default=0.0)
                tol = max(scale, 1.0) * 1e-9
                n_pos = sum(1 for v in g if v > tol)
                n_nz = sum(1 for v in g if abs(v) > tol)
                rows.append([metric, model, str(k), f"{np.mean(g):+.4g}", f"{n_pos}/{n_nz}"])
    tables.append(("Diagonal gamma per metric (W1 grid, 5 homologous cells; oriented higher-is-better)",
                   ["metric", "model", "K", "mean gamma", "positive/nonzero"], rows))


# ----- tables ----------------------------------------------------------------


def write_tables(path: Path, tables) -> None:
    lines = ["# SPX figure tables", "",
             "Generated by `analysis/plot_sp_cross.py`. PSNR in dB, seed-averaged over 3 seeds, "
             "each seed = mean over 1632 PartiPrompts. LPIPS is negated where oriented.", ""]
    for title, header, rows in tables:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_full_results_table(path: Path, store: dict) -> int:
    """All 180 cells, un-oriented raw metric values, seed mean +- seed SD. Returns row count."""
    per_metric = {
        (m, k, key): panel_entries(store, m, k, key, oriented=False)
        for m in MODELS for k in KS for key, _, _ in FULL_TABLE_METRICS
    }
    lines = [
        "# SPX 全量结果表", "",
        "每行一个 (模型, K, schedule, payload) cell；数值 = 3 seed 各自对 1632 条 Parti prompt 取均值后的 "
        "seed 均值 ± seed 标准差。PSNR dB；LPIPS 越小越好，其余越大越好。"
        "共 180 个 cell 定义（× 3 seeds = 540 个运行）。由 `analysis/plot_sp_cross.py` 生成。", "",
        "| model | K | schedule | payload | n_seeds | " + " | ".join(h for _, h, _ in FULL_TABLE_METRICS) + " |",
        "|---|---:|---|---|---:|" + "|".join(["---"] * len(FULL_TABLE_METRICS)) + "|",
    ]
    n_rows = 0
    for model in MODELS:
        for k in KS:
            present = {key[2] for key in store["cells"] if key[0] == model and key[1] == k}
            order = [s for s in SCHEDULE_ORDER if s in present]
            order += sorted(present - set(order))
            for s in order:
                raw = store["raw_schedule"].get((model, k, s))
                if raw is None:
                    continue
                for p in PAYLOADS:
                    if (s, p) not in per_metric[(model, k, "psnr")]:
                        continue
                    cols = []
                    n_seeds = None
                    for key, _, fmt in FULL_TABLE_METRICS:
                        e = per_metric[(model, k, key)][(s, p)]
                        n_seeds = e["n_seeds"]
                        # A single-seed cell has no across-seed sd to quote.
                        cols.append(
                            f"{fmt.format(e['value'])} ± {fmt.format(e['seed_sd'])}"
                            if e["seed_sd"] is not None
                            else f"{fmt.format(e['value'])} ± —")
                    lines.append(f"| {model} | {k} | {raw} | {p} | {n_seeds} | " + " | ".join(cols) + " |")
                    n_rows += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_rows


# ----- main ------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perprompt", type=Path, nargs="+",
                    default=[Path(f"resources/spx/perprompt_spx_{m}.tsv.gz") for m in MODELS])
    ap.add_argument("--supplement", type=Path,
                    default=Path("resources/sp_cross_supplement/sp_cross_results.json"))
    ap.add_argument("--reference", type=Path,
                    default=Path("resources/sp_cross_schedules/native_reference_parti.tsv"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/spx"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    store = load_cells(args.perprompt)
    sanity(store)
    reference = read_reference(args.reference)
    tables: list = []
    decs = {(m, k): w1_decomposition(panel_entries(store, m, k, "psnr")) for m in MODELS for k in KS}

    written = []
    written += fig1_cross(store, args.out, tables)
    paths, diag_means = fig2_gamma(store, args.out, tables, decs)
    written += paths
    variance_table(tables, decs)
    main_effects_table(tables, decs)
    written += fig3_payload_vs_k(store, args.out, tables)
    p4 = p4_table(store, tables, reference)
    diag_metrics_table(store, tables)
    if args.supplement.is_file():
        supplement = json.loads(args.supplement.read_text(encoding="utf-8"))
        written += fig4_p4_offmodal(supplement, args.out, tables)
        written += fig5_ps1_rows(supplement, args.out, tables)
        written += fig6_ps2_dose(supplement, args.out, tables)
        written += fig7_ps4_geometry(supplement, args.out, tables)
    else:
        print(f"[plot] no supplement JSON at {args.supplement}; figures 4-7 skipped")
    tpath = args.out / "tables.md"
    write_tables(tpath, tables)
    written.append(tpath)
    fpath = args.out / "full_results_table.md"
    n_rows = write_full_results_table(fpath, store)
    expected_rows = len({key[:4] for key in store["cells"]})
    assert n_rows == expected_rows, (n_rows, expected_rows)
    written.append(fpath)

    print("\n[key numbers] variance shares (schedule/payload/interaction, %):")
    for (m, k), d in decs.items():
        sh = d["share"]
        text = " / ".join("--" if sh[c] is None else f"{sh[c]*100:.1f}"
                          for c in ("schedule", "payload", "interaction"))
        print(f"  {m} K{k}: {text}")
    print("[key numbers] P1 diagonal gamma mean (dB) and #positive:")
    for (m, k), (mean, npos, n) in diag_means.items():
        print(f"  {m} K{k}: {mean:+.3f}  {npos}/{n}")
    print("[key numbers] P4 delta (fixed - native, dB) [band]:")
    for (m, k, method), (delta, band) in sorted(p4.items()):
        print(f"  {m} K{k} {method}: {delta:+.2f} [±{band:.2f}]")
    print("\n[files]")
    for p in written:
        print(f"  {p}  {p.stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
