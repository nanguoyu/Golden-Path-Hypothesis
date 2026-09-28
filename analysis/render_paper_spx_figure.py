#!/usr/bin/env python3
"""Render the paper figures of the schedule--payload cross."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis._palette_check import check_cross, check_group   # noqa: E402
from analysis.palette import COMPONENT, METHOD_NAMED, MODEL_LABEL   # noqa: E402

OUT = ROOT / "paper" / "figs" / "schedule_approximation_separation.pdf"
OUT_SCHEDULE = ROOT / "paper" / "figs" / "schedule_control_residual_reuse.pdf"
OUT_DECOMP = ROOT / "paper" / "figs" / "spx_variance_decomposition.pdf"
SPX_JSON = ROOT / "resources" / "sp_cross_supplement" / "sp_cross_results.json"
VIDEO_SPX_JSON = ROOT / "resources" / "video_spx" / "video_spx_results.json"
RATIOS = (0.58, 0.74, 0.82)
KS = (29, 37, 41)
MODEL_KEYS = ("flux", "qwen", "hunyuan_video", "wan21")
MODELS = tuple(MODEL_LABEL[key] for key in MODEL_KEYS)
PAYLOADS = (
    ("reuse", "Residual reuse", "o", "-", "0.10"),
    ("taylor_o1", "Taylor O1", "s", "--", "0.38"),
    ("hermite_o2", "Hermite O2", "^", "-.", "0.22"),
    ("mean_avg_vel", "Interval-average velocity", "D", ":", "0.48"),
    ("di_two_anchor", "Two-anchor", "P", "-", "0.62"),
)
SCHEDULES = (
    ("budcache", "BudCache (fixed)", "o", "-"),
    ("seacache", "SeaCache (adaptive)", "s", "--"),
)
CONTROL_COLOR = "0.12"
# Final width 5.5 in = the \linewidth the paper includes the figure at, so the
# sizes below are the sizes the reader sees.  They match Figure 1.
FIG_W, FIG_H = 5.5, 2.0
FS_LAB = 7.6
FS_TICK = 7.2

# The decomposition figure.  Its three segments are variance shares, not cache
# methods, models, or schedule families, so they stay clear of the four method
# hues and of the grey that marks random schedules.  They come from the
# COMPONENT role of `analysis/palette.py`.  The three segments sit on one bar,
# so a greyscale print has to
# separate them by lightness alone: they are spread 20 L* or more apart, with
# the interaction share the darkest of the three.  `check_palette` measures it.
COMPONENTS = ("schedule", "payload", "interaction")
COMP_COLORS = COMPONENT
COMP_TEXT = dict(zip(COMPONENTS, ("white", "0.12", "white")))
COMP_LABELS = dict(zip(COMPONENTS, COMPONENTS))
DECOMP_MODELS = {key: MODEL_LABEL[key] for key in ("flux", "qwen")}
FIG_DECOMP_H = 2.05
METHOD_COLORS = METHOD_NAMED
# Colours on different figures are compared from memory, not side by side, so
# they are held to a weaker limit than colours that share one panel.
CROSS_FIGURE_DISTANCE = 18.0


def check_palette() -> None:
    """Measure the separation both figures of this file are claimed to have."""
    check_group("spx_variance_decomposition", "variance shares",
                {name: COMP_COLORS[name] for name in COMPONENTS},
                min_distance=25.0, min_lightness_gap=20.0, min_chroma=20.0)
    # No method hue appears on either figure of this file, so the two checks
    # below are the weaker cross-figure ones.  Both print the closest pair, a
    # green against SenCache's green in each case; neither pair is ever read
    # side by side.  The in-figure limits above are the ones that matter.
    check_cross("spx_variance_decomposition", "variance shares",
                {name: COMP_COLORS[name] for name in COMPONENTS},
                "cache methods", METHOD_COLORS,
                min_distance=CROSS_FIGURE_DISTANCE, min_cvd_distance=10.0)


def fixed_payload_values() -> dict:
    """PSNR under the BudCache schedule and five payloads at each ratio.

    FLUX: 1,632 PartiPrompts x three seeds. HunyuanVideo: 150 videos from
    each of Penguin599 and VBench944, with equal dataset weights.
    """
    image_data = json.loads(SPX_JSON.read_text("utf-8"))
    video_data = json.loads(VIDEO_SPX_JSON.read_text("utf-8"))
    values = {}
    for k in KS:
        panel = next(p for p in image_data["splits"]["all"]["panels"]
                     if p["model"] == "flux" and int(p["budget_k"]) == k)
        video_panel = video_data["backbones"]["hunyuan_video"]["partitions"][f"K{k}"]
        for payload, *_ in PAYLOADS:
            row = panel["cells"]["psnr"][f"budcachex{payload}"]
            assert row["n_pairs"] == 4896 and row["n_seeds"] == 3
            values[(MODEL_LABEL["flux"], k, payload)] = float(row["value"])
            video_payload = "mean_vel" if payload == "mean_avg_vel" else payload
            row = video_panel["cross"]["psnr"]["budcache"][video_payload]
            assert row["n_videos"] == 300 and not row["impure"]
            assert set(row["per_dataset"]) == {"penguin599", "vbench944"}
            values[(MODEL_LABEL["hunyuan_video"], k, payload)] = float(row["mean"])
    return values


def image_schedule_values() -> dict:
    path = ROOT / "resources" / "full_results_local_analysis" / "tables" / "m0_env240.tsv"
    with path.open(encoding="utf-8") as handle:
        lines = (line for line in handle if not line.startswith("#"))
        rows = list(csv.DictReader(lines, delimiter="\t"))
    by = defaultdict(list)
    for row in rows:
        by[(row["model"], int(row["target_k"]), row["method"])].append(float(row["psnr_mean"]))
    values = {}
    labels = {"flux": "FLUX.1-dev", "qwen": "Qwen-Image"}
    for model, label in labels.items():
        for k in KS:
            for method in [x[0] for x in SCHEDULES]:
                cells = by[(model, k, method)]
                values[(label, k, method)] = sum(cells) / len(cells)
    return values


def video_values() -> dict:
    data = json.loads(
        (ROOT / "resources" / "video_full_results" / "report_numbers.json").read_text("utf-8")
    )
    labels = {"hunyuan_video": "HunyuanVideo", "wan21": "Wan2.1"}
    out = {}
    for model, label in labels.items():
        quality = data["backbones"][model]["quality"]
        for k in KS:
            for method in [x[0] for x in SCHEDULES]:
                values = [quality[f"{dataset}/K{k}"]["methods"][method]["psnr_mean"]["mean"]
                          for dataset in ("penguin599", "vbench944")]
                out[(label, k, method)] = sum(values) / len(values)
    return out


def decomposition_values() -> list[dict]:
    """Read the image variance shares that also fill the appendix table.

    Source: `splits.all.panels[*].P1.psnr.w1`, the least-squares fit with
    sum-to-zero coding on the complete four-schedule by five-payload grid of
    PSNR configuration means.  One entry per image model and cache ratio.
    """
    data = json.loads(SPX_JSON.read_text("utf-8"))
    rows = []
    for panel in data["splits"]["all"]["panels"]:
        fit = panel["P1"]["psnr"]["w1"]
        share = fit["share"]
        total = sum(share[name] for name in COMPONENTS)
        assert abs(total - 1.0) < 1e-9, f"shares do not sum to one: {total}"
        rows.append({
            "model": DECOMP_MODELS[panel["model"]],
            "k": int(panel["budget_k"]),
            "mean_psnr": fit["mu"],
            "n_cells": fit["n_obs"],
            **{name: share[name] for name in COMPONENTS},
        })
    rows.sort(key=lambda row: (list(DECOMP_MODELS.values()).index(row["model"]), row["k"]))
    return rows


def draw_decomposition(rows: list[dict]) -> None:
    """Horizontal stacked bars: one row per model and cache ratio."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_DECOMP_H))
    # Two model groups of three ratios, separated by one blank slot.
    positions = [0.0, 1.0, 2.0, 3.6, 4.6, 5.6]
    for position, row in zip(positions, rows):
        left = 0.0
        for name in COMPONENTS:
            value = row[name]
            ax.barh(position, value, left=left, height=0.62,
                    color=COMP_COLORS[name], edgecolor="white", linewidth=0.6,
                    label=COMP_LABELS[name] if position == positions[0] else None,
                    zorder=3)
            if value >= 0.075:            # a narrower segment has no room
                # one convention in the figure: the numbers on the bars are
                # written the way the axis writes them, leading zero included
                ax.text(left + value / 2, position, f"{value:.3f}",
                        ha="center", va="center", fontsize=FS_TICK - 0.6,
                        color=COMP_TEXT[name], zorder=4)
            left += value
    ax.set_yticks(positions)
    ax.set_yticklabels([f"$K$ = {row['k']}   ratio {ratio:.2f}"
                        for row, ratio in zip(rows, RATIOS * 2)])
    ax.invert_yaxis()
    ax.set_ylim(positions[-1] + 0.75, positions[0] - 1.15)
    for position, model in zip((positions[0], positions[3]), DECOMP_MODELS.values()):
        ax.text(0.006, position - 0.62, model, ha="left", va="center",
                fontsize=FS_LAB, transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("share of the variance in mean PSNR across the grid", labelpad=1.5)
    ax.tick_params(labelsize=FS_TICK, pad=1.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=FS_LAB, ncol=3, handletextpad=0.4,
              columnspacing=1.2, handlelength=1.4, borderpad=0.0,
              loc="lower left", bbox_to_anchor=(0.0, 1.005))
    fig.subplots_adjust(left=0.225, right=0.982, bottom=0.20, top=0.885)
    OUT_DECOMP.parent.mkdir(parents=True, exist_ok=True)
    # no creation date, so the same data always gives the same bytes
    fig.savefig(OUT_DECOMP, metadata={"CreationDate": None})
    print(f"  decomposition figure {FIG_W} x {FIG_DECOMP_H} in, "
          f"{len(rows)} rows x 3 components")
    for row in rows:
        print("   {model:10s} K{k:<3d} schedule {schedule:.3f}  payload "
              "{payload:.3f}  interaction {interaction:.3f}  "
              "mean PSNR {mean_psnr:.2f}".format(**row))
    plt.close(fig)


def draw_controls(payload_values: dict, schedule_values: dict) -> None:
    """One row: payload and schedule controls for each of two models."""
    fig, axes = plt.subplots(1, 4, figsize=(FIG_W, FIG_H))
    for col, model in zip((0, 2), (MODEL_LABEL["flux"], MODEL_LABEL["hunyuan_video"])):
        ax = axes[col]
        ax.set_title(f"({chr(97 + col)}) Fixed\nschedule", fontsize=FS_TICK, pad=4)
        ax.axhline(0, color="0.30", linewidth=0.8, zorder=1)
        for index, (payload, label, _, _, color) in enumerate(PAYLOADS[1:]):
            deltas = [payload_values[(model, k, payload)] - payload_values[(model, k, "reuse")]
                      for k in KS]
            ax.bar(
                np.arange(len(KS)) + (index - 1.5) * 0.19, deltas,
                width=0.17, color=color, edgecolor="0.15", linewidth=0.5,
                hatch=("", "//", "..", "xx")[index], label=label, zorder=3)
            print(f"  {model} {payload}: PSNR minus reuse {deltas}")
        ax.set_xticks(np.arange(len(KS)), [str(x) for x in RATIOS])
        ax.set_xlim(-0.55, len(KS) - 0.45)
        ax.set_xlabel("cache ratio", fontsize=FS_LAB, labelpad=2)
        ax.set_ylabel("PSNR change (dB)", fontsize=FS_TICK, labelpad=1)
        ax = axes[col + 1]
        ax.set_title(f"({chr(98 + col)}) Fixed\nresidual reuse", fontsize=FS_TICK, pad=4)
        for method, label, marker, linestyle in SCHEDULES:
            ax.plot(RATIOS, [schedule_values[(model, k, method)] for k in KS],
                    color=CONTROL_COLOR, marker=marker, linestyle=linestyle,
                    linewidth=1.0, markersize=2.8, markerfacecolor="white",
                    markeredgewidth=0.7, label=label, zorder=3)
        ax.set_xticks(RATIOS)
        ax.set_xlim(RATIOS[0] - 0.025, RATIOS[-1] + 0.025)
        ax.set_xlabel("target cache ratio", fontsize=FS_TICK, labelpad=2)
        ax.set_ylabel("PSNR (dB)", fontsize=FS_TICK, labelpad=1)
    for ax in axes:
        ax.tick_params(labelsize=FS_TICK, pad=1.5)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # Separate scales keep the smaller FLUX differences legible.
    axes[0].set_ylim(-1.08, 1.05)
    axes[0].set_yticks([-1, 0, 1])
    axes[1].set_yticks([20, 22, 24, 26])
    axes[2].set_ylim(-5.2, 3.15)
    axes[2].set_yticks([-4, -2, 0, 2])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False,
               fontsize=FS_TICK, ncol=4, handletextpad=0.3,
               columnspacing=0.75, handlelength=1.35, borderpad=0.0,
               loc="lower center", bbox_to_anchor=(0.52, 0.095))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False,
               fontsize=FS_TICK, ncol=2, handletextpad=0.35,
               columnspacing=1.2, handlelength=1.5, borderpad=0.0,
               loc="lower center", bbox_to_anchor=(0.52, 0.0))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.36, top=0.77, wspace=0.55)
    for col, model in zip((0, 2), (MODEL_LABEL["flux"], MODEL_LABEL["hunyuan_video"])):
        center = (axes[col].get_position().x0 + axes[col + 1].get_position().x1) / 2
        fig.text(center, 0.965, model, ha="center", va="top",
                 fontsize=FS_LAB, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, metadata={"CreationDate": None})
    fig.savefig(OUT.with_suffix(".png"), dpi=220)
    plt.close(fig)


def draw_schedule_control(values: dict) -> None:
    """The other two models, without duplicating the main figure."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 1.75))
    for ax, model in zip(axes, (MODEL_LABEL["qwen"], MODEL_LABEL["wan21"])):
        ax.set_title(model, fontsize=FS_LAB, fontweight="bold", pad=2)
        for method, label, marker, linestyle in SCHEDULES:
            ax.plot(
                RATIOS, [values[(model, k, method)] for k in KS],
                color=CONTROL_COLOR, marker=marker, linestyle=linestyle,
                linewidth=1.0, markersize=2.8, markerfacecolor="white",
                markeredgewidth=0.7, label=label, zorder=3)
        ax.set_xticks(RATIOS)
        ax.set_xlim(RATIOS[0] - 0.025, RATIOS[-1] + 0.025)
        ax.set_xlabel("target cache ratio", fontsize=FS_TICK, labelpad=2)
        ax.tick_params(labelsize=FS_TICK, pad=1.2)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        lo, hi = ax.get_ylim()
        pad = max(0.08 * (hi - lo), 0.15)
        ax.set_ylim(lo - pad, hi + pad)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("PSNR (dB)", labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False,
               title="Same approximation policy: residual reuse", title_fontsize=FS_LAB,
               fontsize=FS_TICK, ncol=2, handletextpad=0.35,
               columnspacing=1.2, handlelength=1.5, borderpad=0.0,
               loc="upper center", bbox_to_anchor=(0.54, 1.015))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.24, top=0.64, wspace=0.38)
    fig.savefig(OUT_SCHEDULE, metadata={"CreationDate": None})
    plt.close(fig)


def main() -> int:
    check_palette()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FS_LAB,
        "axes.labelsize": FS_LAB, "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    })
    draw_decomposition(decomposition_values())
    schedule_values = {**image_schedule_values(), **video_values()}
    draw_controls(fixed_payload_values(), schedule_values)
    draw_schedule_control(schedule_values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
