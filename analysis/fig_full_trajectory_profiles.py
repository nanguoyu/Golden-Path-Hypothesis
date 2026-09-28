#!/usr/bin/env python3
"""Plot full-compute trajectory profiles and their cross-run variation.

Run from any directory:
    python analysis/fig_full_trajectory_profiles.py

The image summaries contain all medians and coefficients of variation, but
only the distance-from-line quartiles. Regenerate the missing quartiles from
the existing per-run tables with:
    python analysis/fig_full_trajectory_profiles.py --refresh-image-quantiles

The figures compare datasets within each model. Each model block pairs median
and interquartile profiles with shorter CV strips for three measurements.
The main figure shows FLUX.1-dev and HunyuanVideo; the companion appendix
figure shows Qwen-Image and Wan2.1. All four image
datasets and both video datasets are included, with the same dataset colors
in every figure.
Every curve uses one model and one dataset, with three seeds per prompt.
Datasets are never pooled. All distances are Euclidean latent-space distances.
State n is the state after n updates, for n=0,...,50. Step n produces the
displacement from state n to state n+1, for n=0,...,49. Distance from the line
joining the initial and final states has zero endpoints, where its CV is
undefined. These two CV values are omitted, not replaced with zero.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.palette import (  # noqa: E402
    DATASET, DATASET_IMAGE, DATASET_LABEL, DATASET_VIDEO, MODEL_LABEL,
)

IMAGE_SOURCE = ROOT / "resources/full_trajectory_analysis/step_profiles.json"
QUANTILES = ROOT / "resources/full_trajectory_analysis/main_profile_quantiles.json"
RAW_IMAGE = ROOT / "resources/full_trajectory/tables_jsonl"
OUT = ROOT / "paper/figs"
MODELS = ("flux", "qwen", "hunyuan_video", "wan21")
MAIN_MODELS = ("flux", "hunyuan_video")
OTHER_MODELS = ("qwen", "wan21")
METRICS = ("dev_over_chord", "spacing_over_chord", "magnitude_over_sqrt_d")
TITLES = ("Distance from line", "Step displacement", "State norm")
UNITS = ("/ chord length L", "/ chord length L", "/ √d")
DATASET_STYLE = {
    "drawbench_full": "-", "geneval_style": (0, (4, 2)),
    "parti_full": (0, (1, 1.3)), "diffusiondb_clean10k": (0, (5, 1.5, 1, 1.5)),
    "penguin599": "-", "vbench944": (0, (4, 2)),
}


def image_quartiles() -> None:
    """Store only the missing quantiles, checking the existing statistics."""
    source = json.loads(IMAGE_SOURCE.read_text())["cells"]
    summary = {
        "source": "resources/full_trajectory/tables_jsonl/full_traj_<model>_<dataset>.jsonl",
        "produced_by": "analysis/fig_full_trajectory_profiles.py --refresh-image-quantiles",
        "quantile_method": "numpy.quantile, linear interpolation, across prompt-seed runs",
        "cells": {},
    }
    for key, reference in source.items():
        model, dataset = key.split("/")
        path = RAW_IMAGE / f"full_traj_{model}_{dataset}.jsonl"
        values = {"spacing_over_chord": [], "magnitude_over_sqrt_d": []}
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                values["spacing_over_chord"].append(
                    np.asarray(row["spacing"], dtype=float) / row["chord_len"])
                values["magnitude_over_sqrt_d"].append(
                    np.asarray(row["magnitude"], dtype=float) / math.sqrt(reference["dim"]))
        profiles = {}
        for metric, rows in values.items():
            arr = np.stack(rows)
            assert arr.shape == (reference["n"], 50 if metric.startswith("spacing") else 51)
            # This verifies that the new bands describe the same runs as the
            # archived median and CV, not a different subset or normalization.
            np.testing.assert_allclose(np.median(arr, axis=0), reference["profiles"][metric],
                                       rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(arr.std(axis=0) / arr.mean(axis=0),
                                       reference["profiles"][metric + "_cv"],
                                       rtol=1e-11, atol=1e-14)
            for q, suffix in ((0.25, "q25"), (0.75, "q75")):
                profiles[f"{metric}_{suffix}"] = np.quantile(arr, q, axis=0).tolist()
        summary["cells"][key] = {"n": reference["n"], "profiles": profiles}
        print(f"Computed image quartiles: {key}, n={reference['n']}")
    QUANTILES.write_text(json.dumps(summary, indent=1) + "\n")


def load() -> dict:
    images = json.loads(IMAGE_SOURCE.read_text())["cells"]
    bands = json.loads(QUANTILES.read_text())["cells"]
    result = {}
    for model in MODELS[:2]:
        result[model] = {}
        for dataset in DATASET_IMAGE:
            key = f"{model}/{dataset}"
            item = images[key]
            assert item["n"] == bands[key]["n"]
            profiles = {**item["profiles"], **bands[key]["profiles"]}
            result[model][dataset] = {
                "n": item["n"], "n_prompts": item["n_prompts"],
                "n_seeds": item["n_seeds"], "profiles": profiles,
                "cv": {m: profiles[m + "_cv"] for m in METRICS},
            }
    for model in MODELS[2:]:
        path = ROOT / f"resources/video_full_trajectory/{model}/step_profiles_{model}.json"
        video = json.loads(path.read_text())
        result[model] = {}
        for dataset in DATASET_VIDEO:
            item = video["datasets"][dataset]
            result[model][dataset] = {
                "n": item["n"], "n_prompts": item["n_prompts"],
                "n_seeds": item["n_seeds"], "profiles": item["profiles"],
                "cv": {m: item["cv"][m]["cv_profile"] for m in METRICS},
            }
    for model, datasets in result.items():
        for dataset, item in datasets.items():
            assert item["n"] == item["n_prompts"] * item["n_seeds"]
            assert item["n_seeds"] == 3
            for metric in METRICS:
                p = item["profiles"]
                n = 50 if metric == "spacing_over_chord" else 51
                med, lo, hi = [np.asarray(p[metric + suffix], dtype=float)
                               for suffix in ("", "_q25", "_q75")]
                assert med.shape == lo.shape == hi.shape == (n,)
                assert np.isfinite(med).all() and np.isfinite(lo).all() and np.isfinite(hi).all()
                assert np.all(lo <= med) and np.all(med <= hi)
                cv = np.asarray(item["cv"][metric], dtype=float)
                assert cv.shape == (n,)
                if metric == "dev_over_chord":
                    assert np.isnan(cv[[0, -1]]).all()
                    assert np.isfinite(cv[1:-1]).all()
                else:
                    assert np.isfinite(cv).all() and np.all(cv > 0)
    return result


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.4,
        "axes.labelsize": 7.4, "axes.titlesize": 8.0,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "xtick.major.width": 0.55, "ytick.major.width": 0.55,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "savefig.dpi": 220,
    })


def style_axis(ax, metric_index: int):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.90", lw=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 49 if metric_index == 1 else 50)
    ax.set_xticks([0, 25, 49] if metric_index == 1 else [0, 25, 50])
    ax.tick_params(pad=2)


def combined_layout(models):
    set_style()
    compact = tuple(models) == MAIN_MODELS
    if compact:
        plt.rcParams.update({
            "font.size": 7.0, "axes.labelsize": 7.0,
            "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        })
    fig = plt.figure(figsize=(5.5, 2.70 if compact else 3.8))
    blocks = fig.add_gridspec(2, 3, left=0.14, right=0.987,
                              bottom=0.13 if compact else 0.095,
                              top=0.79 if compact else 0.80,
                              wspace=0.35, hspace=0.44 if compact else 0.29)
    axes = np.empty((2, 3, 2), dtype=object)
    handles = [Line2D([0], [0], color=DATASET[d], linestyle=DATASET_STYLE[d], lw=1.2,
                      label=DATASET_LABEL[d]) for d in (*DATASET_IMAGE, *DATASET_VIDEO)]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5 if compact else 0.55, 0.995),
               ncol=6 if compact else 3, frameon=False,
               fontsize=7.0 if compact else 7.2,
               handlelength=1.5 if compact else 2.6,
               columnspacing=0.75 if compact else 1.5,
               handletextpad=0.4 if compact else 0.6, labelspacing=0.65)
    for row, model in enumerate(models):
        for col in range(3):
            pair = blocks[row, col].subgridspec(2, 1,
                                               height_ratios=(1.1, 1) if compact else (1.8, 1),
                                               hspace=0.14)
            profile = fig.add_subplot(pair[0])
            cv = fig.add_subplot(pair[1], sharex=profile)
            axes[row, col] = (profile, cv)
            for ax in (profile, cv):
                style_axis(ax, col)
            profile.tick_params(axis="x", bottom=False, labelbottom=False)
            cv.set_xlabel("Step index" if col == 1 else "State index",
                          labelpad=2 if compact else 3)
            if col == 0:
                profile.set_ylabel("Median", labelpad=4)
                cv.set_ylabel("CV (%)", labelpad=4)
            if row == 0:
                profile.set_title(f"{TITLES[col]}\n{UNITS[col]}", pad=6,
                                  fontsize=7.2 if compact else 7.8)
        bounds = blocks[row, 0].get_position(fig)
        fig.text(0.025, (bounds.y0 + bounds.y1) / 2, MODEL_LABEL[model],
                 ha="center", va="center", rotation=90, weight="bold",
                 fontsize=7.4 if compact else 8)
    return fig, axes


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    # At the natural 5.5-inch paper width these are the final printed sizes.
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            assert text.get_fontsize() >= 7.0, (text.get_text(), text.get_fontsize())
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def draw_profile(ax, item, metric, *, color, style):
    p = item["profiles"]
    med, lo, hi = [np.asarray(p[metric + suffix])
                   for suffix in ("", "_q25", "_q75")]
    x = np.arange(len(med))
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0, zorder=2)
    ax.plot(x, med, color=color, ls=style, lw=1.1, zorder=3)
    return float(hi.max())


def profile_axis(ax, high):
    ax.set_ylim(0, high * 1.06)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))


def variation_axis(ax):
    # All panels share a scale. Zero line-distance endpoints have NaN
    # CV in the input and remain absent from the plots.
    ax.set_yscale("log")
    ax.set_ylim(0.04, 40)
    ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
    ax.minorticks_off()


def plot_models(data, models, suffix=""):
    fig, axes = combined_layout(models)
    for row, model in enumerate(models):
        for col, metric in enumerate(METRICS):
            profile, variation = axes[row, col]
            high = 0.0
            for dataset, item in data[model].items():
                high = max(high, draw_profile(profile, item, metric,
                           color=DATASET[dataset], style=DATASET_STYLE[dataset]))
                cv = np.asarray(item["cv"][metric], dtype=float) * 100
                variation.plot(np.arange(len(cv)), cv, color=DATASET[dataset],
                               ls=DATASET_STYLE[dataset], lw=1.1)
            profile_axis(profile, high)
            variation_axis(variation)
    save(fig, f"full_trajectory_profiles_variation{suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-image-quantiles", action="store_true")
    args = parser.parse_args()
    if args.refresh_image_quantiles:
        image_quartiles()
    data = load()
    plot_models(data, MAIN_MODELS)
    plot_models(data, OTHER_MODELS, "_other_models")
    print("All curves retain their own dataset. CV uses population standard deviation.")
    for model, datasets in data.items():
        count = sum(item["n"] for item in datasets.values())
        print(f"{MODEL_LABEL[model]}: {count:,} prompt-seed runs")
        print("  Datasets: " + ", ".join(
            f"{name}: {item['n']:,} runs" for name, item in datasets.items()))
        for metric in METRICS:
            cvs = [np.asarray(item["cv"][metric], dtype=float) * 100 for item in datasets.values()]
            medians = [float(np.nanmedian(cv)) for cv in cvs]
            print(f"  {metric}: dataset medians of per-step CV {min(medians):.6f} to "
                  f"{max(medians):.6f}%; all-step range "
                  f"{min(np.nanmin(cv) for cv in cvs):.6f} to "
                  f"{max(np.nanmax(cv) for cv in cvs):.6f}%")


if __name__ == "__main__":
    main()
