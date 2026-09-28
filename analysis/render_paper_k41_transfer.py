#!/usr/bin/env python3
"""Render the appendix figure for the exhaustive high-ratio schedule search.

One panel, 337 dots.  Each dot is one schedule of the fixed evaluation panel.
Its horizontal position is the score the search itself saw: the mean PSNR over
the four PartiPrompts prompt--noise pairs that selection used.  Its vertical
position is what that schedule later reached on data the search never touched:
the mean PSNR over the nine held-out dataset--seed settings, each setting
weighted equally.  The figure therefore shows how far a four-pair score carries
to 7,151 new pairs, and where the schedule the search picked lands once it gets
there.

Final width 5.5 in = the \\linewidth the paper includes this figure at, so the
font sizes below are the sizes the reader sees: 7.6 pt labels, 7.2 pt ticks,
2.4 pt ticks with a 1.6 pt pad, as in Figure 1 and the clock figure.

Colours: the panel is a set of fixed schedules split by the family each was
drawn from, which is exactly the role `FAMILY` names in `analysis/palette.py`.
Search-derived schedules use indigo circles, the eleven existing or synthetic
controls use dark olive triangles, and the 128 uniformly sampled schedules use
open grey squares. The selected schedule is a larger indigo diamond.
`check_palette` measures all of it.

Writes paper/figs/k41_transfer.pdf and .png.  Run from anywhere:

    python analysis/render_paper_k41_transfer.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis._palette_check import check_cross, check_group   # noqa: E402
from analysis.palette import FAMILY, METHOD_NAMED, RANDOM_GREY, OLIVE_RAMP   # noqa: E402

MANIFEST = ROOT / "resources/exhaustive_k41/formal_results/candidate_manifest.tsv"
HELDOUT = ROOT / "resources/exhaustive_k41/all_dataset_results/candidate_statistics.tsv"
OUT_PDF = ROOT / "paper" / "figs" / "k41_transfer.pdf"
OUT_PNG = ROOT / "paper" / "figs" / "k41_transfer.png"

# ------------------------------------------------------------------ geometry
W, H = 5.5, 2.65         # inches; the figure is included at \linewidth
L, B = 0.52, 0.36        # axes origin, inches from the lower-left corner
PW, PH = 4.88, 2.16      # axes size, inches

FS_LAB = 7.6             # axis labels
FS_TICK = 7.2            # tick labels
FS_NAME = 7.0            # names written next to a mark
INK = "0.15"             # near-black: every word in the figure is set in it
GUIDE = "0.72"           # the two limit lines, structure rather than data

# the four PartiPrompts selection pairs, one column each in the manifest
SELECTION_COLUMNS = ("psnr_p5", "psnr_p8", "psnr_p9", "psnr_p15")
# the reason string the panel builder wrote for its 128 uniform draws
RANDOM_REASON = "uniform_random_128_seed20270826"
# the schedule the search selected: full compute at 0, 1, 2, 4, 6, 11, 24, 41, 49
SELECTED_RANK = 164762
SELECTED_X = 22.746226765313466      # its four-pair mean, from the manifest
SELECTED_Y = 20.944417556823502      # its equal-weight held-out mean
SELECTED_HELDOUT_RANK = 36
# the two offline baseline schedules the body compares the selection against
BASELINES = {"included:meancache": "MeanCache", "included:budcache": "BudCache"}

FAMILY_KEYS = ("search", "other", "random")
FAMILY_LABEL = {"search": "from the search (198)",
                "other": "existing or synthetic control (11)",
                "random": "uniformly sampled (128)"}
FAMILY_COLORS = {**FAMILY, "other": OLIVE_RAMP[0]}
FAMILY_MARKERS = {"search": "o", "other": "^", "random": "s"}
FAMILY_SIZES = {"search": 7.0, "other": 24.0, "random": 9.0}

PDF_META = {"CreationDate": None,
            "Creator": "render_paper_k41_transfer.py",
            "Producer": "matplotlib"}


def check_palette() -> None:
    """Measure the separation this figure's one colour group is claimed to have."""
    colors = {FAMILY_LABEL[key]: FAMILY_COLORS[key] for key in FAMILY_KEYS}
    # Shape and fill distinguish the families in monochrome; the control
    # triangles use a darker olive so their small number remains visible.
    check_group("k41_transfer", "schedule families", colors,
                min_distance=25.0)
    coloured = {FAMILY_LABEL[key]: FAMILY_COLORS[key] for key in ("search", "other")}
    # Neither coloured family may read as the grey the random family wears.
    check_group("k41_transfer", "coloured families", coloured, min_chroma=20.0,
                min_distance=25.0, verbose=False)
    # No cache method appears on this figure; this is the weaker cross-figure
    # limit, and the closest pair it prints is the indigo against SeaCache.
    check_cross("k41_transfer", "schedule families", coloured,
                "cache methods", METHOD_NAMED,
                min_distance=18.0, min_cvd_distance=10.0)
    assert FAMILY["random"] == RANDOM_GREY["mark"], FAMILY["random"]


# =================================================================== the data
def family_of(reasons: str) -> str:
    """Which family the panel builder drew this schedule from."""
    if reasons == RANDOM_REASON:
        return "random"
    if reasons.startswith("included:"):
        return "other"
    return "search"


def load_selection() -> dict[int, dict]:
    """rank -> manifest row, the four-pair scores the search itself saw."""
    with open(MANIFEST, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {int(row["rank"]): row for row in rows}


def load_heldout() -> dict[int, dict[str, float]]:
    """rank -> environment -> that environment's mean PSNR over its pairs."""
    out: dict[int, dict[str, float]] = defaultdict(dict)
    with open(HELDOUT, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            out[int(row["rank"])][row["environment"]] = float(row["mean_psnr_db"])
    return dict(out)


def rankdata(values: list[float]) -> list[float]:
    """Ascending ranks, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(one: list[float], other: list[float]) -> float:
    a, b = rankdata(one), rankdata(other)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den


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


def main() -> int:
    check_palette()

    selection = load_selection()
    heldout = load_heldout()
    ranks = sorted(selection)
    assert set(ranks) == set(heldout), "manifest and held-out table disagree"

    x = {}
    y = {}
    for rank in ranks:
        row = selection[rank]
        pairs = [float(row[column]) for column in SELECTION_COLUMNS]
        mean_pairs = statistics.fmean(pairs)
        assert abs(mean_pairs - float(row["mean_psnr_db"])) < 1e-9, rank
        x[rank] = mean_pairs
        environments = heldout[rank]
        assert len(environments) == 9, (rank, sorted(environments))
        y[rank] = statistics.fmean(environments.values())

    families = {rank: family_of(selection[rank]["reasons"]) for rank in ranks}
    members = {key: [r for r in ranks if families[r] == key] for key in FAMILY_KEYS}

    xs = [x[r] for r in ranks]
    ys = [y[r] for r in ranks]
    pooled = spearman(xs, ys)
    environments = sorted({e for v in heldout.values() for e in v})
    per_setting = {e: spearman(xs, [heldout[r][e] for r in ranks])
                   for e in environments}

    by_x = sorted(ranks, key=lambda r: -x[r])
    by_y = sorted(ranks, key=lambda r: -y[r])
    top_x, top_y = set(by_x[:64]), set(by_y[:64])
    kept = len(top_x & top_y)
    x_limit, y_limit = x[by_x[63]], y[by_y[63]]
    heldout_rank = by_y.index(SELECTED_RANK) + 1
    setting_ranks = {e: sorted(ranks, key=lambda r: -heldout[r][e]).index(SELECTED_RANK) + 1
                     for e in environments}

    # ------------------------------------------------------------ the figure
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FS_LAB,
        "axes.titlesize": FS_LAB, "axes.labelsize": FS_LAB,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
    })
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L / W, B / H, PW / W, PH / H])

    # the two limits that cut out the top 64 of each axis; drawn first and in
    # a light grey so they read as structure and never as a data point
    ax.axvline(x_limit, color=GUIDE, lw=0.7, ls=(0, (2.2, 2.0)), zorder=1)
    ax.axhline(y_limit, color=GUIDE, lw=0.7, ls=(0, (2.2, 2.0)), zorder=1)

    for key in ("random", "search", "other"):
        group = members[key]
        ax.scatter([x[r] for r in group], [y[r] for r in group],
                   s=FAMILY_SIZES[key], marker=FAMILY_MARKERS[key],
                   facecolor="none" if key == "random" else FAMILY_COLORS[key],
                   edgecolor=FAMILY_COLORS[key],
                   alpha=0.7 if key == "search" else 1.0,
                   linewidths=0.55 if key == "random" else 0.25,
                   zorder={"random": 2, "search": 3, "other": 4}[key])

    ax.scatter([SELECTED_X], [SELECTED_Y], s=42, marker="D",
               facecolor=FAMILY["search"], edgecolor="white", linewidth=0.8,
               zorder=5)

    named = {}
    for reason, label in BASELINES.items():
        rank = next(r for r in ranks if selection[r]["reasons"] == reason)
        named[label] = (x[rank], y[rank])
        ax.scatter([x[rank]], [y[rank]], s=30, marker="^",
                   facecolor=FAMILY_COLORS["other"], edgecolor="white", linewidth=0.65,
                   zorder=4)

    ax.set_xlim(15.7, 23.05)
    ax.set_ylim(15.5, 21.70)
    ax.set_xticks([16, 17, 18, 19, 20, 21, 22, 23])
    ax.set_yticks([16, 17, 18, 19, 20, 21])
    ax.tick_params(labelsize=FS_TICK, pad=1.6)
    ax.set_xlabel("mean PSNR on the four scoring runs (dB)", labelpad=1.6)
    ax.set_ylabel("held-out mean PSNR (dB)", labelpad=2.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # names written next to their mark, in ink, the way the other panels do it
    ax.annotate("selected schedule", xy=(SELECTED_X, SELECTED_Y),
                xytext=(SELECTED_X - 0.24, 21.53),
                fontsize=FS_NAME, color=INK, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=INK,
                                shrinkA=1.0, shrinkB=2.6))
    ax.annotate("MeanCache", xy=named["MeanCache"],
                xytext=(named["MeanCache"][0] - 0.07, 20.16),
                fontsize=FS_NAME, color=INK, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=INK,
                                shrinkA=1.0, shrinkB=2.2))
    ax.annotate("BudCache", xy=named["BudCache"],
                xytext=(named["BudCache"][0] + 0.33, 19.56),
                fontsize=FS_NAME, color=INK, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=INK,
                                shrinkA=1.0, shrinkB=2.2))

    handles = []
    for key in FAMILY_KEYS:
        handles.append(ax.scatter(
            [], [], s=FAMILY_SIZES[key], marker=FAMILY_MARKERS[key],
            facecolor="none" if key == "random" else FAMILY_COLORS[key],
            edgecolor=FAMILY_COLORS[key], linewidths=0.55,
            label=FAMILY_LABEL[key]))
    legend = ax.legend(handles=handles, frameon=False, fontsize=FS_NAME,
                       loc="lower right", bbox_to_anchor=(0.935, -0.014),
                       handlelength=0.8, handletextpad=0.35,
                       labelspacing=0.30, borderpad=0.0)
    for text in legend.get_texts():
        text.set_color(INK)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", metadata=PDF_META)
    fig.savefig(OUT_PNG, format="png", dpi=400)

    # ------------------------------------------------------------ self-check
    print("== self-check ==")
    print(f"  figure {W} x {H} in;  one panel {PW:.2f} x {PH:.2f} in")
    assert len(ranks) == 337, len(ranks)
    assert len(members["random"]) == 128, len(members["random"])
    assert len(members["other"]) == 11, len(members["other"])
    assert len(members["search"]) == 198, len(members["search"])
    print(f"  {len(ranks)} schedules: "
          + ", ".join(f"{len(members[k])} {k}" for k in FAMILY_KEYS))
    assert abs(x[SELECTED_RANK] - SELECTED_X) < 1e-12, x[SELECTED_RANK]
    assert abs(y[SELECTED_RANK] - SELECTED_Y) < 1e-12, y[SELECTED_RANK]
    assert heldout_rank == SELECTED_HELDOUT_RANK, heldout_rank
    print(f"  selected schedule  four-pair {SELECTED_X:.3f} dB, "
          f"held-out {SELECTED_Y:.3f} dB, held-out rank {heldout_rank} of 337")
    print(f"  its rank in the nine settings: "
          f"{min(setting_ranks.values())} to {max(setting_ranks.values())}")
    for label, point in named.items():
        print(f"  {label:10s} four-pair {point[0]:.3f} dB, held-out {point[1]:.3f} dB, "
              f"selected leads by {SELECTED_Y - point[1]:.3f} dB")
    print(f"  pooled Spearman (four-pair vs equal-weight held-out): {pooled:.3f}")
    lo = min(per_setting.values())
    hi = max(per_setting.values())
    print(f"  per-setting Spearman: {lo:.3f} to {hi:.3f}")
    for e in environments:
        print(f"     {e:14s} {per_setting[e]:.4f}")
    assert 0.895 <= lo < 0.897, lo          # the 0.896 the appendix states
    assert 0.945 <= hi < 0.947, hi          # the 0.946 the appendix states
    assert 34 <= kept <= 42, kept
    print(f"  top-64 overlap under the equal-weight average: {kept} of 64")
    print(f"  limits drawn: x {x_limit:.3f} dB, y {y_limit:.3f} dB")

    rule = Ruler(fig)
    widths = {
        "x label": (rule.w("mean PSNR on the four scoring runs (dB)", FS_LAB), PW),
        "y label": (rule.w("held-out mean PSNR (dB)", FS_LAB), PH),
        "legend": (max(rule.w(FAMILY_LABEL[k], FS_NAME) for k in FAMILY_KEYS) + 0.14,
                   PW * 0.42),
        "selected name": (rule.w("selected schedule", FS_NAME), PW * 0.40),
        "baseline names": (rule.w("MeanCache", FS_NAME), PW * 0.30),
    }
    for key, (value, room) in widths.items():
        print(f"  {key:16s} {value:.2f} in  (room {room:.2f} in)")
        assert value < room, f"{key} overflows"
    assert L + PW <= W - 0.02, "the panel runs off the page"
    assert B + PH <= H - 0.02, "the panel runs off the top"

    print("saved", OUT_PDF)
    print("saved", OUT_PNG)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
