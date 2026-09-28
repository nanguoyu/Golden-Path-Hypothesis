#!/usr/bin/env python3
"""Render the paper figure for the action-state split of the velocity drift.

Natural width 2.4 in supports a narrow figure beside the main text. Labels
are 7.0-7.2 pt, with 10 pt velocity symbols and 7 pt subscripts. All velocity
annotations stay within the figure rather than extending past the curves.

Nothing on this figure is measured.  Two schematic trajectories share an exact
prefix and then separate, and at one later step the three velocities of the
action-state identity are drawn where they live: the velocity the cached run
uses, the counterfactual velocity a full pass would return at the cached
state, and the full-compute velocity at the reference state.

Colours: because no mark here carries a measurement, the figure must not read
as any measured role.  The full model takes a near-black ink and the cached
run takes a brown, both from `analysis/palette.py`.  Neither is a cache-method
hue, neither is a model hue, and neither sits in the tones the paper reserves
for random controls.  The counterfactual velocity comes from the full model,
so it wears the full model's ink and is dashed, because it is never fed back
into generation.  `check_palette` measures all of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis._palette_check import check_cross, check_group   # noqa: E402
from analysis.palette import (METHOD_NAMED, MODEL_NAMED, RANDOM_GREY,   # noqa: E402
                              SCHEMATIC, SCHEMATIC_NAMED)

OUT_PDF = ROOT / "paper" / "figs" / "action_state_schematic.pdf"
OUT_PNG = ROOT / "paper" / "figs" / "action_state_schematic.png"

FIG_W, FIG_H = 2.4, 1.58
FS_LAB = 7.2            # the run names
FS_SMALL = 7.0          # the velocity descriptions and first-step note
FS_SYMBOL = 10.0        # math subscripts render at 0.7 of the symbol size
FULL = SCHEMATIC["full"]
CACHED = SCHEMATIC["cached"]
PDF_META = {"CreationDate": None,
            "Creator": "render_paper_action_state_figure.py",
            "Producer": "matplotlib"}


def check_palette() -> None:
    """Measure the separation this figure's two colours are claimed to have."""
    check_group("action_state_schematic", "schematic sides", SCHEMATIC_NAMED,
                min_distance=25.0, min_lightness_gap=20.0)
    check_cross("action_state_schematic", "schematic sides", SCHEMATIC_NAMED,
                "cache methods", METHOD_NAMED,
                min_distance=25.0, min_cvd_distance=20.0)
    check_cross("action_state_schematic", "schematic sides", SCHEMATIC_NAMED,
                "models", MODEL_NAMED,
                min_distance=20.0, min_cvd_distance=15.0)
    check_cross("action_state_schematic", "schematic sides", SCHEMATIC_NAMED,
                "random control", RANDOM_GREY,
                min_distance=25.0, min_cvd_distance=25.0)


def bezier(control: np.ndarray, t: np.ndarray) -> np.ndarray:
    """A cubic curve through four control points."""
    p0, p1, p2, p3 = control
    t = t[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def arrow(ax, tail, step, color, dashed=False) -> None:
    head = (tail[0] + step[0], tail[1] + step[1])
    ax.add_patch(FancyArrowPatch(
        tail, head, arrowstyle="-|>", mutation_scale=6.0, linewidth=1.2,
        color=color, linestyle=(0, (3.2, 1.8)) if dashed else "-",
        shrinkA=0.0, shrinkB=0.0, zorder=6))


def velocity_label(ax, x, y, symbol, description, color) -> None:
    """Stack each readable symbol above its description inside the figure."""
    content = VPacker(children=[
        TextArea(symbol, textprops={"fontsize": FS_SYMBOL, "color": color}),
        TextArea(description, textprops={"fontsize": FS_SMALL, "color": color,
                                        "multialignment": "center",
                                        "linespacing": 1.1}),
    ], align="center", pad=0, sep=0.5)
    ax.add_artist(AnnotationBbox(content, (x, y), xycoords="data",
                                 box_alignment=(0.5, 0.5), frameon=False, pad=0))


def tangent(curve, index, length, aspect) -> np.ndarray:
    """A step of the given on-page length along a curve, in data units."""
    step = curve[index + 6] - curve[index]
    step = step * np.array([aspect, 1.0])
    step /= np.linalg.norm(step)
    return step * length * np.array([1.0 / aspect, 1.0])


def rotate(vector, degrees, aspect) -> np.ndarray:
    """Turn a vector by an on-page angle rather than a data-space one."""
    angle = np.deg2rad(degrees)
    turn = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
    page = turn @ (vector * np.array([aspect, 1.0]))
    return page * np.array([1.0 / aspect, 1.0])


def main() -> int:
    check_palette()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FS_LAB,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    aspect = FIG_W / FIG_H      # data units per page unit, x against y

    # Both runs end at the step the identity is written for, so the three
    # velocity arrows leave the curves and read against clear paper.
    steps = np.linspace(0.0, 1.0, 400)
    reference = bezier(np.array([[0.025, 0.500], [0.14, 0.620],
                                 [0.34, 0.620], [0.55, 0.730]]), steps)
    split = 0.34
    drop = np.where(steps < split, 0.0,
                    0.420 * ((steps - split) / (1 - split)) ** 2)
    cached = reference.copy()
    cached[:, 1] -= drop
    after = steps >= split

    # the prefix is one shared run, so it is drawn once, in the full model's
    # ink; the cached run appears where it starts to differ
    ax.plot(reference[:, 0], reference[:, 1], color=FULL, linewidth=1.0,
            zorder=3)
    ax.plot(cached[after, 0], cached[after, 1], color=CACHED, linewidth=1.0,
            zorder=3)
    for value in np.linspace(0.03, 1.0, 12):
        index = int(value * 399)
        ax.scatter(*reference[index], s=4.0, color=FULL, zorder=4)
        if steps[index] >= split:
            ax.scatter(*cached[index], s=4.0, color=CACHED, zorder=4)

    at_split = int(split * 399)
    ax.scatter(*reference[at_split], s=15, facecolor="white", edgecolor=FULL,
               linewidth=0.9, zorder=5)
    ax.annotate("first cached\nstep", reference[at_split],
                xytext=(0.020, 0.200), fontsize=FS_SMALL, color=FULL,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", linewidth=0.7, color=FULL,
                                alpha=0.55,
                                shrinkA=1.5, shrinkB=2.5))

    ax.text(0.020, 0.900, "full-compute\ntrajectory", fontsize=FS_LAB, color=FULL,
            ha="left", va="center", linespacing=1.1)
    ax.text(0.300, 0.205, "cached\ntrajectory", fontsize=FS_LAB, color=CACHED,
            ha="left", va="center", linespacing=1.1)

    at_n = 399
    on_full = reference[at_n]
    on_cached = cached[at_n]
    ax.plot([on_full[0], on_cached[0]], [on_full[1], on_cached[1]],
            linestyle=(0, (1.2, 1.6)), color=FULL, alpha=0.55,
            linewidth=0.9, zorder=2)
    length = 0.200
    v_full = tangent(reference, at_n - 8, length, aspect)
    v_bar = tangent(cached, at_n - 8, length, aspect)
    v_cf = rotate(v_full, -8.0, aspect)

    arrow(ax, on_full, v_full, FULL)
    velocity_label(ax, 0.825, 0.875, "$v_n^F$", "full-compute", FULL)

    arrow(ax, on_cached, v_cf, FULL, dashed=True)
    velocity_label(ax, 0.805, 0.510,
                   "$v_n^{CF}$", "full model at\ncached state", FULL)

    arrow(ax, on_cached, v_bar, CACHED)
    velocity_label(ax, 0.790, 0.140, r"$\bar v_n$", "cached velocity", CACHED)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, metadata=PDF_META)
    fig.savefig(OUT_PNG, format="png", dpi=400)
    plt.close(fig)
    print("saved", OUT_PDF)
    print("saved", OUT_PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
