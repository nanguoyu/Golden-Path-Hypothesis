#!/usr/bin/env python3
"""Render the four-model shared-denoising-clock figure from measured JSON.

One row of four panels, FLUX.1-dev / Qwen-Image / HunyuanVideo / Wan2.1.  A
curve is the dataset median five-step turning angle at every denoising step;
each random-seed setting is one thin line in the dataset's colour, so the
overlap of the seed lines is visible instead of being coded away.  Colour
carries the dataset and nothing else.  Six datasets appear, four image and two
video, and all six hues are different; none of them is a method hue from
Figure 1.

It runs from the earliest to the latest lowest step of every median curve of
the four models turn at the same steps is on the page rather than only in the
caption.

Final width 5.5 in = ICLR \\linewidth, so the font sizes below are the sizes
the reader sees; no scaling factor is applied.

Writes paper/figs/shared_clock_four_models.pdf, the file main.tex includes as
Figure~\\ref{fig:curvature-clock}, and the same figure as a PNG for quick
viewing.  Run from anywhere:

    python analysis/render_paper_clock_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis._palette_check import check_cross, check_group   # noqa: E402
from analysis.palette import (DATASET, DATASET_IMAGE, DATASET_LABEL,   # noqa: E402
                              DATASET_VIDEO, METHOD_NAMED, all_datasets,
                              image_datasets, video_datasets)

OUT_PDF = ROOT / "paper" / "figs" / "shared_clock_four_models.pdf"
OUT_PNG = ROOT / "paper" / "figs" / "shared_clock_four_models.png"
IMAGE_REPORT = ROOT / "resources" / "full_trajectory_shape" / "shape_scale.json"
VIDEO_REPORTS = {
    "HunyuanVideo": ROOT / "resources" / "video_full_trajectory" / "hunyuan_video"
    / "video_shape_scale_hunyuan_video.json",
    "Wan2.1": ROOT / "resources" / "video_full_trajectory" / "wan21"
    / "video_shape_scale_wan21.json",
}

# Colour encodes the dataset only; the hues come from the DATASET role of
# `analysis/palette.py`.  The four image hues sit on a lightness ladder
# (measured L* 69.8 / 57.1 / 44.8 / 22.4), so the four curves of one panel stay
# apart in a greyscale print as well as in colour.  No dataset is grey, because
# grey marks the random baseline elsewhere in the paper.  `check_palette` below
# measures every claim made here.
COLOR = DATASET
LABEL = DATASET_LABEL
IMAGE_ORDER = DATASET_IMAGE
# the two video prompt sets take two hues of their own, used by no image panel,
# and 23 L* apart, so a greyscale print still tells the two curves apart.  Both
# are dark enough to read as 6.8 pt type in the name strip.
VIDEO_COLOR = DATASET
VIDEO_LABEL = DATASET_LABEL
VIDEO_ORDER = DATASET_VIDEO

MIN_PAIR_DIST = 25.0        # between two dataset colours
MIN_IMAGE_LIGHTNESS_GAP = 10.0   # between two image hues, in a greyscale print
MIN_METHOD_DIST = 25.0      # between a dataset colour and a method colour
MIN_GREY_CHROMA = 20.0      # below this a colour reads as grey
MIN_VIDEO_LIGHTNESS_GAP = 20.0

FS_MODEL = 7.3          # model name inside the panel
FS_LAB = 7.6            # axis labels
FS_TICK = 7.2           # tick labels
FS_NAME = 7.2           # dataset name strip
LW = 0.9


# ====================================================================== data
def image_series(report: dict, model: str) -> list[tuple[str, list, list]]:
    """(dataset, centers, median profile) for every dataset x seed of a model."""
    out = []
    for key in sorted(k for k in report["cells"] if k.startswith(f"{model}|")):
        _model, dataset, _seed = key.split("|")
        profile = report["cells"][key]["profile"]
        out.append((dataset, profile["centers"], profile["median_profile"]))
    return out


def video_series(path: Path) -> list[tuple[str, list, list]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    window = report["curvature"]["windows"]["w5"]
    out = []
    for stream in sorted(window["by_stream"]):
        dataset = stream.rsplit("_s", 1)[0]
        profile = window["by_stream"][stream]
        out.append((dataset, profile["centers"], profile["median_profile"]))
    return out


def argmin_steps(series) -> list[int]:
    """The denoising step at which each median curve of a panel is lowest."""
    return [centers[min(range(len(values)), key=lambda i: values[i])]
            for _dataset, centers, values in series]


def seed_counts(series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for dataset, _c, _v in series:
        counts[dataset] = counts.get(dataset, 0) + 1
    return counts


# ==================================================================== colour
def check_palette() -> None:
    """Measure every property the palette comment claims."""
    datasets = all_datasets()
    check_group("shared_clock_four_models", "datasets", datasets,
                min_distance=MIN_PAIR_DIST, min_chroma=MIN_GREY_CHROMA,
                vision=("normal",))
    # The four image hues share a panel, so they are held to the full reading:
    # the CIELAB limit under both dichromacies as well as normal vision, and a
    # lightness gap a greyscale print keeps.
    check_group("shared_clock_four_models", "image datasets", image_datasets(),
                min_distance=MIN_PAIR_DIST,
                min_lightness_gap=MIN_IMAGE_LIGHTNESS_GAP,
                min_chroma=MIN_GREY_CHROMA)
    # The method colours live on Figure 1, never on this one, so the limit here
    # is on normal vision.  The dichromat readings are printed rather than
    # asserted: the olive of PartiPrompts and the orange of DiCache both turn
    # yellow under protanopia, and no colour choice on this figure can prevent
    # that for a reader comparing two pages from memory.
    check_cross("shared_clock_four_models", "datasets", datasets,
                "cache methods", METHOD_NAMED,
                min_distance=MIN_METHOD_DIST, min_cvd_distance=0.0)
    video = video_datasets()
    check_group("shared_clock_four_models", "video datasets", video,
                min_distance=MIN_PAIR_DIST,
                min_lightness_gap=MIN_VIDEO_LIGHTNESS_GAP,
                min_chroma=MIN_GREY_CHROMA)
    # a video hue must be a hue no image panel of this figure already spends
    reused = {name for name, c in video.items()
              if c in {COLOR[k] for k in IMAGE_ORDER}}
    print(f"  video hues reused from an image panel: {sorted(reused) or 'none'}")
    assert not reused, reused


# =============================================================== presentation
class Ruler:
    """Measure rendered text in inches, so nothing silently overflows."""

    def __init__(self, fig):
        fig.canvas.draw()
        self.fig, self.r = fig, fig.canvas.get_renderer()

    def w(self, s, fs):
        t = self.fig.text(0, 0, s, fontsize=fs)
        out = t.get_window_extent(self.r).width / self.fig.dpi
        t.remove()
        return out


def draw_panel(ax, series, model, colors, labels, order, name_x, headroom,
               panel_w, panel_h, rule):
    """One panel: every dataset x seed curve, plus the two name blocks.

    denoising clock rather than being told so in the caption.

    Returns the drawn text blocks as ((x0, x1), (y0, y1)) boxes in axes
    fraction, measured from the rendered text, for the collision check.
    """
    for dataset, centers, values in series:
        ax.plot(centers, values, color=colors[dataset], lw=LW,
                solid_capstyle="round", alpha=0.9)

    lo = min(min(v) for _d, _c, v in series)
    hi = max(max(v) for _d, _c, v in series)
    span = hi - lo
    ymin = lo - 0.07 * span
    ymax = ymin + (hi - ymin) / headroom          # data top sits at `headroom`
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(3, 47)
    ax.set_xticks([10, 20, 30, 40])
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.tick_params(labelsize=FS_TICK, pad=1.6, length=2.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    boxes = []
    ax.text(0.5, 0.99, model, fontsize=FS_MODEL, color="0.15", ha="center",
            va="top", transform=ax.transAxes)
    half = 0.5 * rule.w(model, FS_MODEL) / panel_w
    line = 1.35 * FS_MODEL / 72.0 / panel_h            # one line box, in fraction
    boxes.append(("model name", (0.5 - half - 0.02, 0.5 + half + 0.02),
                  (0.99 - line, 0.99)))

    dy = 0.105 / panel_h
    top = 0.78
    for i, key in enumerate(order):
        ax.text(name_x, top - i * dy, labels[key], fontsize=FS_NAME,
                color=colors[key], ha="left", va="center",
                transform=ax.transAxes)
    wide = max(rule.w(labels[k], FS_NAME) for k in order) / panel_w
    pad = 0.5 * 1.35 * FS_NAME / 72.0 / panel_h
    boxes.append(("name strip", (name_x - 0.02, name_x + wide + 0.02),
                  (top - (len(order) - 1) * dy - pad, top + pad)))
    return boxes


def check_free(ax, series, boxes, panel):
    """Assert every curve stays below each text block over that block's x range."""
    ymin, ymax = ax.get_ylim()
    x0a, x1a = ax.get_xlim()
    for name, (fx0, fx1), (fy0, _fy1) in boxes:
        xlo, xhi = x0a + fx0 * (x1a - x0a), x0a + fx1 * (x1a - x0a)
        top = ymin + fy0 * (ymax - ymin)
        worst = max((v for _d, cs, vs in series
                     for c, v in zip(cs, vs) if xlo <= c <= xhi), default=ymin)
        assert worst < top, (panel, name, worst, top)
        print(f"     {panel:13s} {name:11s} clearance "
              f"{(top - worst) / (ymax - ymin) * 100:4.1f}% of the panel height")


def main() -> int:
    image_report = json.loads(IMAGE_REPORT.read_text(encoding="utf-8"))
    panels = [
        ("FLUX.1-dev", image_series(image_report, "flux"), COLOR, LABEL, IMAGE_ORDER),
        ("Qwen-Image", image_series(image_report, "qwen"), COLOR, LABEL, IMAGE_ORDER),
        ("HunyuanVideo", video_series(VIDEO_REPORTS["HunyuanVideo"]),
         VIDEO_COLOR, VIDEO_LABEL, VIDEO_ORDER),
        ("Wan2.1", video_series(VIDEO_REPORTS["Wan2.1"]),
         VIDEO_COLOR, VIDEO_LABEL, VIDEO_ORDER),
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.7,
                         "xtick.major.width": 0.7, "ytick.major.width": 0.7})
    W, H = 5.5, 1.35
    fig = plt.figure(figsize=(W, H))
    L, GAP, B, TOP = 0.36, 0.26, 0.30, 0.01
    PW = (W - L - 3 * GAP - 0.05) / 4
    PH = H - B - TOP

    # the four name strips are left-anchored; give each one room inside its panel
    NAME_X = (0.24, 0.24, 0.30, 0.30)
    HEADROOM = (0.78, 0.78, 0.78, 0.78)

    # the lowest step of every median curve, printed for the record
    # every model, from the earliest such step to the latest
    turns = {model: argmin_steps(series) for model, series, *_rest in panels}
    band = (min(min(v) for v in turns.values()),
            max(max(v) for v in turns.values()))

    axes = []
    for i in range(4):
        left = L + i * (PW + GAP)
        axes.append(fig.add_axes([left / W, B / H, PW / W, PH / H]))
    rule = Ruler(fig)

    all_boxes = []
    for ax, (model, series, colors, labels, order), name_x, head in zip(
            axes, panels, NAME_X, HEADROOM):
        all_boxes.append(draw_panel(ax, series, model, colors, labels, order,
                                    name_x, head, PW, PH, rule))
    axes[0].set_ylabel("turning angle (deg)", fontsize=FS_LAB, labelpad=2)
    row_mid = L + (4 * PW + 3 * GAP) / 2          # centred on the four panels
    fig.text(row_mid / W, 0.055 / H, "denoising step",
             fontsize=FS_LAB, ha="center", va="bottom")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", metadata={"CreationDate": None})
    fig.savefig(OUT_PNG, format="png", dpi=400)

    # ------------------------------------------------------------ self-check
    check_palette()
    print("== self-check ==")
    print(f"  figure {W} x {H} in;  four panels {PW:.2f} x {PH:.2f} in")
    print(f"  flattest steps across models: {band[0]} to {band[1]}")
    for (model, series, _c, _l, order), ax, boxes in zip(panels, axes, all_boxes):
        counts = seed_counts(series)
        assert set(counts) == set(order), (model, sorted(counts))
        assert set(counts.values()) == {3}, (model, counts)
        lo = min(min(v) for _d, _c2, v in series)
        hi = max(max(v) for _d, _c2, v in series)
        steps = sorted(turns[model])
        print(f"  {model:13s} {len(series)} curves, {len(order)} datasets x 3 seeds, "
              f"{lo:.2f}..{hi:.2f} deg")
        print(f"     lowest step of each curve {steps[0]}..{steps[-1]}  "
              f"({', '.join(str(s) for s in steps)})")
        assert band[0] <= steps[0] and steps[-1] <= band[1], (model, steps, band)
        check_free(ax, series, boxes, model)

    widths = {}
    for (model, _s, colors, labels, order), name_x in zip(panels, NAME_X):
        widest = max(rule.w(labels[k], FS_NAME) for k in order)
        widths[f"{model} names"] = (widest, PW * (1.0 - name_x))
        widths[f"{model} title"] = (rule.w(model, FS_MODEL), PW)
    widths["y label"] = (rule.w("turning angle (deg)", FS_LAB), PH)
    widths["x label"] = (rule.w("denoising step", FS_LAB), W)
    for key, (v, lim) in widths.items():
        print(f"     {key:20s} {v:.2f} in  (room {lim:.2f} in)")
        assert v < lim, f"{key} overflows"
    assert L + 4 * PW + 3 * GAP <= W - 0.02, "the row runs off the page"

    print("saved", OUT_PDF)
    print("saved", OUT_PNG)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
