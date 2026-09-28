#!/usr/bin/env python3
"""Render the paper figure for trajectory-specific bending planes.

Final width 5.5 in = the \\linewidth the paper includes this figure at, so the
font sizes below are the sizes the reader sees: 7.6 pt labels, 7.2 pt ticks,
2.4 pt ticks with a 1.6 pt pad, as in Figure 1 and the clock figure.

Panel (a) draws two measured FLUX.1-dev trajectories, each the 51 latent
states of a full-compute 50-step generation.  The frame is the one the paper
defines, fitted to the first trajectory: the chord from its first state to its
last, and the top two principal directions of the motion left after that chord
is removed.  The second trajectory is the same prompt at a different initial
noise, and it is drawn in that same frame, so it runs almost flat.  Every
fraction the caption quotes is recomputed here and asserted, so a change of
data or of frame fails the render.  Panel (b) is the measured
plane-orientation comparison.

Each trajectory runs along its own chord, and both bending coordinates are the
first trajectory's two bending directions.  Without that the second curve
would only report how far apart two chords sit in a million-dimensional space,
which is not the claim the panel makes.

Colours: the step index of the first trajectory is an ordered quantity, so it
takes the one olive ramp the paper keeps for ordered sets, read light to dark
from the first state to the last.  The second trajectory carries no order, so
it takes the near-black ink that marks a full-compute run in the drawn
schematic; using the FLUX.1-dev indigo here would collide with panel (b),
where that indigo names the model against Qwen-Image.
Every colour comes from `analysis/palette.py`.  None is a cache-method hue,
none is the brown of the drawn schematic, and none is grey, which the paper
reserves for random controls.  `check_palette` measures all of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis._palette_check import check_cross, check_group   # noqa: E402
from analysis.palette import (METHOD_NAMED, MODEL, MODEL_LABEL, SCHEMATIC,   # noqa: E402
                              OLIVE_RAMP, RANDOM_GREY)

OUT_PDF = ROOT / "paper" / "figs" / "trajectory_specific_planes.pdf"
OUT_PNG = ROOT / "paper" / "figs" / "trajectory_specific_planes.png"
LATENTS = ROOT / "resources" / "slide_traj" / "pomeranian_latents.npz"
# the second generation: same prompt, seed 43 instead of 42
LATENTS_SECOND = (ROOT / "resources" / "slide_traj"
                  / "pomeranian_seed43_latents.npz")

FIG_W, FIG_H = 5.5, 2.45
FS_LAB = 7.6            # axis labels and panel titles
FS_TICK = 7.2           # tick labels
FS_SMALL = 7.0          # in-panel annotations
# the ordered olive ramp, light at the first state and dark at the last, so a
# greyscale print still reads the direction of travel
STEP_RAMP = OLIVE_RAMP[::-1]
LINE_COLOR = OLIVE_RAMP[1]
EDGE_COLOR = OLIVE_RAMP[0]
FIRST_COLOR = OLIVE_RAMP[1]
# the same two model colours the appendix separation figure uses
MODEL_KEYS = ("flux", "qwen")
# the second trajectory takes the schematic's full-compute ink; the model
# indigo would clash with panel (b)'s legend
SECOND_COLOR = SCHEMATIC["full"]
FIRST_LABEL = "trajectory 1"
SECOND_LABEL = "trajectory 2"
METHOD_COLORS = METHOD_NAMED
PDF_META = {"CreationDate": None, "Creator": "render_paper_plane_figure.py",
            "Producer": "matplotlib"}

# what the caption claims about the drawn trajectory, to three decimals
TOTAL_FRACTION = 0.999
OFF_CHORD_FRACTION = 0.954
# and about the second one: the share of its off-chord motion that lies in the
# first trajectory's bending plane, and its own peak bend for comparison
IN_PLANE_FRACTION = 0.034
SECOND_OWN_PEAK = 0.080


def check_palette() -> None:
    """Measure the separation this figure's two colour groups are claimed to have."""
    lines = {"trajectory 1": FIRST_COLOR, "trajectory 2": SECOND_COLOR}
    check_group("trajectory_specific_planes", "the two trajectories", lines,
                min_distance=25.0)
    check_cross("trajectory_specific_planes", "the two trajectories", lines,
                "cache methods", METHOD_COLORS,
                min_distance=25.0, min_cvd_distance=15.0)
    # trajectory 2 wears the schematic's achromatic ink, so its visibility
    # rests on lightness distance, not chroma
    check_cross("trajectory_specific_planes", "the two trajectories", lines,
                "the random control", RANDOM_GREY, min_distance=25.0,
                min_cvd_distance=15.0)
    models = {MODEL_LABEL[key]: MODEL[key] for key in MODEL_KEYS}
    check_group("trajectory_specific_planes", "models", models,
                min_distance=25.0, min_chroma=20.0)
    # No method hue appears on this figure, so this is the weaker cross-figure
    # limit; the closest pair it prints is Qwen's teal against SenCache's green.
    check_cross("trajectory_specific_planes", "models", models,
                "cache methods", METHOD_COLORS,
                min_distance=18.0, min_cvd_distance=10.0)


def _off_chord(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The stored states as travel along their own chord and motion off it.

    Returns the step indices, the position along the unit chord in units of
    the chord length, and the off-chord part of the same scaled positions.
    """
    with np.load(path, allow_pickle=False) as store:
        states = store["latents"].reshape(store["latents"].shape[0], -1)
        steps = store["steps"].astype(int)
    states = states.astype(np.float64)

    origin = states[0]
    chord = states[-1] - origin
    length = float(np.linalg.norm(chord))
    unit = chord / length

    scaled = (states - origin) / length
    along = scaled @ unit
    return steps, along, scaled - np.outer(along, unit)


def trajectory_frame(
        path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """One stored trajectory in its own frame.

    Returns the 51 coordinates, the step indices, the two bending directions,
    the share of the motion the three plotted directions carry, and the share
    of the off-chord motion the two bending directions carry.  Axis 1 is the
    unit chord from the first state to the last; axes 2 and 3 are the top two
    principal directions of what the chord leaves.  All three are in units of
    the chord length.
    """
    with np.load(path, allow_pickle=False) as store:
        states = store["latents"].reshape(store["latents"].shape[0], -1)
        steps = store["steps"].astype(int)
    states = states.astype(np.float64)

    origin = states[0]
    chord = states[-1] - origin
    length = float(np.linalg.norm(chord))
    unit = chord / length

    scaled = (states - origin) / length
    along = scaled @ unit
    off = scaled - np.outer(along, unit)

    centered_off = off - off.mean(axis=0, keepdims=True)
    gram = centered_off @ centered_off.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    off_chord_fraction = float(values[order[:2]].sum()
                               / values[order].clip(min=0.0).sum())

    directions = []
    for index in order[:2]:
        raw = centered_off.T @ vectors[:, index]
        directions.append(raw / float(np.linalg.norm(raw)))
    plane = np.stack(directions, axis=1)
    bending = off @ plane

    centered = scaled - scaled.mean(axis=0, keepdims=True)
    basis = np.concatenate([unit[:, None], plane], axis=1)
    total_fraction = float(((centered @ basis) ** 2).sum()
                           / (centered ** 2).sum())

    coords = np.column_stack([along, bending[:, 0], bending[:, 1]])
    return coords, steps, plane, total_fraction, off_chord_fraction


def second_trajectory(path: Path,
                      plane: np.ndarray) -> tuple[np.ndarray, float, float]:
    """A second trajectory, read in the first one's bending plane.

    It travels along its own chord, so the first coordinate is its own
    progress from 0 to 1.  The other two are its off-chord motion read in the
    first trajectory's two bending directions, in units of its own chord
    length, which is within 7 percent of the first one's.  Returns those
    coordinates, the share of its off-chord motion that lies in that plane,
    and the largest bend it makes in its own plane.
    """
    _steps, along, off = _off_chord(path)
    in_plane = off @ plane

    centered = off - off.mean(axis=0, keepdims=True)
    values, vectors = np.linalg.eigh(centered @ centered.T)
    order = np.argsort(values)[::-1]
    own = []
    for index in order[:2]:
        raw = centered.T @ vectors[:, index]
        own.append(raw / float(np.linalg.norm(raw)))
    own_plane = np.stack(own, axis=1)

    fraction = float(np.linalg.norm(in_plane) / np.linalg.norm(off))
    own_peak = float(np.abs(off @ own_plane).max())
    coords = np.column_stack([along, in_plane[:, 0], in_plane[:, 1]])
    return coords, fraction, own_peak


def draw_trajectory(ax, coords, steps, second) -> None:
    """Panel (a): the 51 states of two generations, one flat colour each.
    The panel's claim is shape, not time, so there is no step ramp and no
    legend apparatus: each line carries its own name, and start and end
    give the direction."""
    # Screen axes: the chord runs across, the stronger bending direction runs
    # up, the weaker one into the page.  The labels follow the coordinates.
    x, y, z = coords[:, 0], coords[:, 2], coords[:, 1]

    points = np.stack([x, y, z], axis=1)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    ax.add_collection3d(Line3DCollection(segments, colors=LINE_COLOR,
                                         linewidths=1.0, zorder=1))
    ax.scatter(x, y, z, color=FIRST_COLOR, s=9,
               edgecolors=EDGE_COLOR, linewidths=0.25, depthshade=False,
               zorder=3)
    # the second generation, one flat ink and a thinner line: it is not an
    # ordered set here, only a shape to compare against the arch above it
    sx, sy, sz = second[:, 0], second[:, 2], second[:, 1]
    second_points = np.stack([sx, sy, sz], axis=1)
    ax.add_collection3d(Line3DCollection(
        np.stack([second_points[:-1], second_points[1:]], axis=1),
        colors=SECOND_COLOR, linewidths=0.7, zorder=2))
    # its own 51 states, small and unramped, so the line reads as a measured
    # trajectory and not as an axis of the frame
    ax.scatter(sx, sy, sz, s=2.0, c=SECOND_COLOR, depthshade=False, zorder=2)

    for index, label, dy in ((0, "start", -0.9), (-1, "end", 0.9)):
        ax.scatter([x[index]], [y[index]], [z[index]], s=34,
                   facecolors="none", edgecolors=EDGE_COLOR, linewidths=1.0,
                   depthshade=False, zorder=4)
        ax.text(x[index], y[index], z[index] + dy * 0.022, label,
                fontsize=FS_SMALL, color=EDGE_COLOR,
                ha="right" if index == 0 else "left",
                va="top" if index == 0 else "bottom", zorder=6)

    apex = int(np.argmax(z))
    ax.text(x[apex], y[apex], z[apex] + 0.012, FIRST_LABEL,
            fontsize=FS_SMALL, color=EDGE_COLOR, ha="center", va="bottom",
            linespacing=1.05, zorder=6)
    ax.text(0.38, float(sy[24]), float(sz[24]) - 0.014, SECOND_LABEL,
            fontsize=FS_SMALL, color=SECOND_COLOR, ha="center", va="top",
            linespacing=1.05, zorder=6)

    ax.set_xlabel("along the chord", labelpad=-5, fontsize=FS_LAB)
    ax.set_ylabel("bending direction 2", labelpad=-13, fontsize=FS_LAB)
    # the vertical axis name is placed by hand: centred on its axis it runs
    # into the tail of the depth axis name, and 3D axes offer no way to slide
    # one label along its own axis
    ax.set_zlabel("")
    ax.get_figure().text(0.478, 0.615, "bending direction 1", rotation=90,
                         fontsize=FS_LAB, ha="center", va="center")
    ax.set_xlim(-0.05, 1.05)
    # the two bending axes share one scale, so the shape of the bend is honest
    # between them; the chord axis is an order of magnitude longer and is
    # scaled on its own
    low = float(min(y.min(), z.min()))
    high = float(max(y.max(), z.max()))
    pad = 0.12 * (high - low)
    ax.set_ylim(low - pad, high + pad)
    ax.set_zlim(low - pad, high + pad)
    ax.set_xticks((0.0, 0.5, 1.0))
    ax.set_yticks((0.0, 0.08))
    # the two bending axes share one scale, so one set of numbers serves both
    ax.set_yticklabels(("", ""))
    ax.set_zticks((0.0, 0.08))
    ax.tick_params(labelsize=FS_TICK, pad=-2.5)
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((2.05, 0.90, 1.05), zoom=0.95)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#d0d0d0")
        axis.line.set_linewidth(0.7)
        axis._axinfo["grid"]["color"] = "#e6e6e6"
        axis._axinfo["grid"]["linewidth"] = 0.5



def draw_angles(ax) -> None:
    """Panel (b): the measured plane-orientation comparison, unchanged."""
    categories = ("same\nnoise", "same\nprompt", "neither", "random\nplanes")
    flux = (60.9, 85.5, 87.5, 89.8)
    qwen = (70.9, 79.8, 87.8, 89.9)
    x = np.arange(len(categories))
    # nominal categories carrying a magnitude with a true zero: grouped bars,
    # not a line; the random-plane pair wears the reserved grey of random
    # controls, with the model told by its position in the pair
    w = 0.32
    for i, (vals, color, label) in enumerate(
            ((flux, MODEL[MODEL_KEYS[0]], MODEL_LABEL[MODEL_KEYS[0]]),
             (qwen, MODEL[MODEL_KEYS[1]], MODEL_LABEL[MODEL_KEYS[1]]))):
        xs = x + (i - 0.5) * w
        measured = xs[:-1]
        ax.bar(measured, vals[:-1], width=w, color=color, label=label,
               zorder=3)
        ax.bar(xs[-1:], vals[-1:], width=w, facecolor=RANDOM_GREY["fill"],
               edgecolor=RANDOM_GREY["dot"], linewidth=0.7, zorder=3)
    ax.axhline(90, color="0.55", linestyle=(0, (1.6, 1.6)), linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, linespacing=1.05)
    ax.set_xlim(-0.35, len(categories) - 0.65)
    ax.set_ylabel("median principal angle (degrees)", labelpad=2)
    # the axis starts at zero: an angle is a magnitude, and a truncated axis
    # would make the gap between the measured curves and the random line look
    # larger than it is
    ax.set_ylim(0, 99)
    ax.set_yticks((0, 20, 40, 60, 80))
    ax.set_title("(b) measured plane orientation", pad=3)
    ax.tick_params(labelsize=FS_TICK, pad=1.6)
    ax.legend(frameon=False, fontsize=FS_LAB - 0.8, loc="upper center",
              bbox_to_anchor=(0.5, 1.015), ncols=2, columnspacing=1.0,
              handlelength=1.4, handletextpad=0.4, borderpad=0.0)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    check_palette()
    coords, steps, plane, total, off_chord = trajectory_frame(LATENTS)
    print(f"[plane] drawn trajectory: {coords.shape[0]} states, "
          f"three directions carry {100 * total:.2f} % of the motion, "
          f"two bending directions carry {100 * off_chord:.2f} % of the "
          f"motion off the chord")
    assert round(total, 3) == TOTAL_FRACTION, round(total, 4)
    assert round(off_chord, 3) == OFF_CHORD_FRACTION, round(off_chord, 4)

    second, in_plane, own_peak = second_trajectory(LATENTS_SECOND, plane)
    print(f"[plane] second trajectory: {second.shape[0]} states, "
          f"{100 * in_plane:.2f} % of its off-chord motion lies in the drawn "
          f"plane, where its largest bend is "
          f"{np.abs(second[:, 1:]).max():.4f} of its chord length against "
          f"{own_peak:.4f} in its own plane")
    assert round(in_plane, 3) == IN_PLANE_FRACTION, round(in_plane, 4)
    assert round(own_peak, 3) == SECOND_OWN_PEAK, round(own_peak, 4)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FS_LAB,
        "axes.titlesize": FS_LAB, "axes.labelsize": FS_LAB,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
    })
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # panel (b) keeps the box the previous gridspec gave it, 2.23 in wide and
    # 1.89 in tall.  The step colourbar stands at the left edge with its name
    # and numbers outside it, and panel (a) takes the rest of the left half,
    # wider than it was when the bar sat between the two panels
    ax0 = fig.add_axes((0.030, 0.030, 0.420, 0.900), projection="3d")
    ax1 = fig.add_axes((0.575, 0.135, 0.405, 0.770))

    draw_trajectory(ax0, coords, steps, second)
    ax0.set_title("(a) two measured trajectories", pad=-2, x=0.46)
    draw_angles(ax1)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, metadata=PDF_META)
    fig.savefig(OUT_PNG, format="png", dpi=400)
    plt.close(fig)
    print("saved", OUT_PDF)
    print("saved", OUT_PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
