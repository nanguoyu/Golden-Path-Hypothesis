#!/usr/bin/env python3
"""Plot the measured schedule-by-approximation-policy comparison at K=41."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "resources/sp_cross_supplement/sp_cross_results.json"
OUTPUT = ROOT / "paper/figs/schedule_policy_interaction"
POLICIES = (
    ("reuse", "Reuse"),
    ("taylor_o1", "Taylor\norder 1"),
    ("hermite_o2", "Hermite\norder 2"),
    ("mean_avg_vel", "Average\nvelocity"),
    ("di_two_anchor", "Two-\nanchor"),
)
SCHEDULES = (
    ("budcache", "BudCache", "o", "-"),
    ("dpcache", "DPCache", "s", "--"),
    ("uniform", "Uniform", "^", "-."),
    ("dicache_top1", "DiCache most frequent", "D", ":"),
)
MODELS = (("flux", "FLUX.1-dev"), ("qwen", "Qwen-Image"))


def load_means():
    data = json.loads(SOURCE.read_text())
    values = {}
    for model, _ in MODELS:
        panel = next(p for p in data["splits"]["all"]["panels"]
                     if p["model"] == model and int(p["budget_k"]) == 41)
        matrix = []
        for schedule, *_ in SCHEDULES:
            row = []
            for policy, _ in POLICIES:
                cell = panel["cells"]["psnr"][f"{schedule}x{policy}"]
                assert cell["n_pairs"] == 4896 and cell["n_seeds"] == 3
                row.append(float(cell["value"]))
            matrix.append(row)
        values[model] = np.asarray(matrix)
        fitted = (values[model].mean(axis=1, keepdims=True)
                  + values[model].mean(axis=0, keepdims=True)
                  - values[model].mean())
        residual = values[model] - fitted
        share = np.sum(residual ** 2) / np.sum(
            (values[model] - values[model].mean()) ** 2)
        recorded = panel["P1"]["psnr"]["w1"]["share"]["interaction"]
        assert abs(share - recorded) < 1e-10
        print(f"{model}: interaction share {share:.6f}; 20 measured means")
    return values


def main():
    values = load_means()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), sharey=True)
    x = np.arange(len(POLICIES))
    for ax, (model, title) in zip(axes, MODELS):
        for row, (_, label, marker, style) in zip(values[model], SCHEDULES):
            ax.plot(x, row, color="0.12", linestyle=style, marker=marker,
                    markersize=3.6, markerfacecolor="white", markeredgewidth=0.8,
                    linewidth=1.05, label=label)
        ax.set_title(title, fontweight="bold", pad=5)
        ax.set_xticks(x, [label for _, label in POLICIES])
        ax.set_xlim(-0.2, 4.25)
        ax.set_ylim(7.5, 22)
        ax.set_yticks([8, 12, 16, 20])
        ax.set_xlabel("Approximation policy", labelpad=4)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.grid(axis="y", color="0.86", linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Mean PSNR (dB)", labelpad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1),
               frameon=False, ncol=4, fontsize=7.5, handlelength=2.3,
               columnspacing=1.0, handletextpad=0.45)
    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.32,
                        top=0.77, wspace=0.20)
    fig.savefig(OUTPUT.with_suffix(".pdf"),
                metadata={"CreationDate": None,
                          "Creator": "render_paper_spx_interaction.py"})
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
