#!/usr/bin/env python3
"""Render the main and appendix figures for the exhaustive K=41 experiment.

The appendix figure summarizes the near-optimal tail after every schedule in the
constrained 1,370,754-schedule space was scored on four calibration examples.
The main figure measures how the selected schedule transfers to unseen prompts
relative to the best result in the 337-schedule held-out pool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis._palette_check import check_cross  # noqa: E402
from analysis.palette import FAMILY, METHOD_NAMED  # noqa: E402


SUMMARY = ROOT / "resources" / "exhaustive_k41" / "formal_results" / "summary.json"
MERGED = ROOT / "resources" / "exhaustive_k41" / "merged.tsv.gz"
COVERAGE = ROOT / "resources" / "exhaustive_k41" / "deep_pool_check.json"
OUT_PDF = ROOT / "paper" / "figs" / "exhaustive_existence.pdf"
OUT_PNG = ROOT / "paper" / "figs" / "exhaustive_existence.png"
COUNTS_PDF = ROOT / "paper" / "figs" / "exhaustive_selection_counts.pdf"
COUNTS_PNG = ROOT / "paper" / "figs" / "exhaustive_selection_counts.png"

FIG_W, FIG_H = 2.6, 1.85
MAIN_FIG_H = 1.25
FS_LAB = 7.6
FS_TICK = 7.1
COLOR = FAMILY["search"]
INK = "0.15"


def check_palette() -> None:
    """Keep the selected-path hue distinct from the cache-method palette."""
    check_cross(
        "exhaustive_existence",
        "selected path",
        {"selected path": COLOR},
        "cache methods",
        METHOD_NAMED,
        min_distance=18.0,
        min_cvd_distance=10.0,
    )


def load_values() -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    summary = json.loads(SUMMARY.read_text("utf-8"))
    coverage = json.loads(COVERAGE.read_text("utf-8"))

    assert summary["space"]["total"] == 1_370_754
    assert summary["n_rows"] == summary["space"]["total"]
    scores = pd.read_csv(MERGED, sep="\t", usecols=["mean_psnr_db"])[
        "mean_psnr_db"
    ].to_numpy()
    assert len(scores) == summary["space"]["total"]
    best = float(summary["best_mean"]["psnr_db"])
    gaps = np.sort(best - scores)
    tail = gaps[gaps <= 1.0]
    # Preserve the exact endpoints and sample the dense middle by log rank.
    ranks = np.unique(np.rint(np.geomspace(1, len(tail), 700)).astype(int))
    exhaustive_gaps = tail[ranks - 1]
    exhaustive_counts = ranks

    assert coverage["pool_size"] == 337
    pooled = next(row for row in coverage["partitions"] if row["partition"] == "pooled")
    assert pooled["n_prompts"] == 2_381
    selected = next(
        row for row in pooled["members"] if row["rank"] == coverage["selected_rank"]
    )
    heldout_margins = [float(value) for value in coverage["margins_db"]]
    heldout_shares = [100.0 * float(value) for value in selected["share"]]

    for margin, expected in ((0.10, 76), (0.25, 381), (0.50, 2044)):
        assert int(np.count_nonzero(gaps <= margin)) == expected
    assert all(abs(a - b) < 0.01 for a, b in zip(heldout_shares, [28.81, 57.08, 78.87]))
    return exhaustive_gaps, exhaustive_counts, heldout_margins, heldout_shares


def style_axis(ax) -> None:
    ax.grid(axis="y", color="0.87", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS_TICK, pad=1.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def render_counts(gaps: np.ndarray, counts: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(gaps, counts, color=COLOR, linewidth=1.35, zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(1, 2.1e4)
    ax.set_yticks([1, 10, 100, 1_000, 10_000], ["1", "10", "100", "1,000", "10,000"])
    ax.set_ylabel("Schedules within gap", labelpad=2)
    ax.set_xlabel("PSNR gap from the highest\nmean PSNR (dB)", labelpad=2)
    ax.set_title("Scores on four examples", fontweight="bold", pad=6)
    for margin, count, offset in ((0.10, 76, (6, 7)), (0.50, 2044, (6, 7))):
        ax.scatter([margin], [count], s=18, facecolor="white", edgecolor=COLOR,
                   linewidth=1.0, zorder=4)
        ax.annotate(
            f"{count:,}", (margin, count), xytext=offset, textcoords="offset points",
            fontsize=FS_TICK, ha="left", va="bottom", color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7},
        )

    style_axis(ax)
    fig.subplots_adjust(left=0.245, right=0.96, bottom=0.29, top=0.82)
    save_figure(fig, COUNTS_PDF, COUNTS_PNG)


def render_coverage(margins: list[float], shares: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, MAIN_FIG_H))
    for margin, share in zip(margins, shares):
        ax.vlines(margin, 0, share, color=COLOR, linewidth=0.8, alpha=0.45,
                  zorder=1)
    ax.scatter(margins, shares, s=22, facecolor="white", edgecolor=COLOR,
               linewidth=1.1, zorder=3)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Prompts within\ngap (%)", labelpad=2)
    ax.set_xlabel("PSNR gap from each prompt's\nbest evaluated schedule (dB)", labelpad=2)
    ax.set_title("Quality of one shared schedule", fontweight="bold", pad=4)
    for margin, share, align in zip(margins, shares, ("left", "left", "right")):
        offset = (5, 2) if align == "left" else (-5, 2)
        ax.annotate(
            f"{share:.0f}%", (margin, share), xytext=offset,
            textcoords="offset points", fontsize=FS_TICK, ha=align, va="bottom",
            color=INK, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7},
        )

    style_axis(ax)
    fig.subplots_adjust(left=0.21, right=0.96, bottom=0.405, top=0.82)
    save_figure(fig, OUT_PDF, OUT_PNG)


def save_figure(fig, pdf_path: Path, png_path: Path) -> None:
    metadata = {
        "CreationDate": None,
        "Creator": "render_paper_exhaustive_existence.py",
        "Producer": "matplotlib",
    }
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, metadata=metadata)
    fig.savefig(png_path, dpi=300, metadata={"Software": metadata["Creator"]})
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def main() -> int:
    check_palette()
    exhaustive_gaps, exhaustive_counts, margins, shares = load_values()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FS_LAB,
        "axes.labelsize": FS_LAB,
        "axes.titlesize": FS_LAB,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })
    render_coverage(margins, shares)
    render_counts(exhaustive_gaps, exhaustive_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
