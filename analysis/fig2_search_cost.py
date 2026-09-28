"""Figure 2 of the paper -- how many schedule evaluations a search needs.

One panel.  x  oracle calls (schedule evaluations, log axis),
             y  four-pair mean PSNR of the best schedule found so far
                (linear axis), so higher is better.

Every curve is one searcher on the 1,370,754-schedule FLUX.1-dev K41 truth
table, 50 repetitions, read from `resources/search_bench/results.json` as
written by `analysis/golden_path_search_bench.py`.  The four searchers that
were later run on the real models are drawn in black with distinct dashes and
named; the six the benchmark also measured are drawn in light grey without
names, so the reader sees the whole field the four were chosen from.

A dotted horizontal line marks the best schedule in the table, 22.746 dB,
which is what every curve is climbing towards.

Black, white and grey only; final width 5.5 in = ICLR \\linewidth, so the font
sizes below are the sizes the reader sees.

Writes paper/figs/fig2_search_cost.pdf and .png.  Run from anywhere:

    python analysis/fig2_search_cost.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis._palette_check import check_group          # noqa: E402

OUT = REPO / "paper" / "figs"
STEM = "fig2_search_cost"
RESULTS = REPO / "resources/search_bench/results.json"

INK = "0.10"                  # near-black: every word and every named curve
GREY = "0.78"                 # the unnamed searchers, one tone, no words
GUIDE = "0.45"                # the table-best guide line and its word

# The four searchers that were run on the real models, in the benchmark's own
# names.  The hill climb and the annealer were warm-started from the setting's
# baseline schedules in the real-model runs, so the seeded benchmark rows are
# the matching ones; greedy and random have no seeded variant.
DRAWN = [
    ("random", "random", (1.4, 1.4)),
    ("hill_first_seeded", "hill climb", (None, None)),
    ("budcache_sa_seeded", "annealing", (3.2, 1.3)),
    ("greedy_coordinate", "greedy", (0.9, 0.9, 3.0, 0.9)),
]
THRESHOLD = 0.05              # the benchmark's own calls-to-threshold limit
BEST = 22.746226765313466     # the best of the 1,370,754 schedules, in dB
REPORT_AT = (100, 171, 300, 1000)


def check_palette():
    """The figure is greyscale, so the only claim to measure is lightness.

    Three tones carry meaning here: the near-black of the four named curves,
    the light grey of the unnamed field behind them, and the mid grey of the
    table-best guide.  A greyscale print keeps CIELAB lightness and nothing else,
    so each pair is asked to stay far apart on that axis alone.
    """
    check_group("fig2_search_cost", "greyscale tones",
                {"named searchers": INK, "unnamed field": GREY,
                 "table-best guide": GUIDE},
                min_distance=18.0, min_lightness_gap=18.0)


def load():
    payload = json.load(open(RESULTS))
    calls = np.array(payload["curve_calls"], dtype=float)
    return payload, calls


def curve(method, best=None):
    """The quality curve of one searcher: the PSNR of its best schedule so far.

    `results.json` stores the distance to the table best, averaged over the 50
    repetitions, so the quality itself is that distance taken off the table
    best -- the same numbers, read as PSNR instead of as a shortfall.
    """
    best = BEST if best is None else best
    return best - np.array(method["curve_gap_mean"], dtype=float)


def main():
    payload, calls = load()
    methods = payload["methods"]
    named = {name for name, _label, _dash in DRAWN}

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.7,
                         "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                         "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
                         "xtick.major.size": 2.4, "ytick.major.size": 2.4,
                         "xtick.minor.size": 1.3, "ytick.minor.size": 1.3})
    W, H = 5.5, 2.05
    fig = plt.figure(figsize=(W, H))
    # the key stands in its own column at the right, so no word is ever laid
    # over a curve and no curve has to end where its name would fit
    L, B, TOP, AW = 0.58, 0.40, 0.10, 4.72
    KEY_L = L + AW - 1.55                 # the key sits inside the axes, lower right
    ax = fig.add_axes([L / W, B / H, AW / W, (H - B - TOP) / H])
    fs_lab, fs_tick, fs_name = 8.0, 7.2, 7.4

    # the field first, so the four named curves sit on top of it
    for name, method in methods.items():
        if name in named:
            continue
        ax.plot(calls, curve(method), color=GREY, lw=0.8, zorder=2,
                solid_capstyle="round")

    ax.axhline(BEST, color=GUIDE, lw=0.7, ls=(0, (1.2, 1.6)), zorder=3)

    ends = {}
    for name, label, dash in DRAWN:
        y = curve(methods[name])
        ax.plot(calls, y, color=INK, lw=1.15, zorder=5,
                dashes=dash if dash[0] else (1, 0), solid_capstyle="round",
                dash_capstyle="round")
        ends[name] = (label, y[-1])

    ax.set_xscale("log")
    ax.set_xlim(1, 5000)
    ax.set_ylim(18.0, 23.0)
    ax.set_xticks([1, 10, 100, 1000, 5000])
    ax.set_xticklabels(["1", "10", "100", "1,000", "5,000"])
    ax.set_yticks([18, 19, 20, 21, 22, 23])
    ax.tick_params(labelsize=fs_tick, pad=1.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("schedule evaluations", fontsize=fs_lab, labelpad=1.5)
    ax.set_ylabel("mean PSNR of the best\nschedule found (dB)",
                  fontsize=fs_lab, labelpad=2)

    # the guide's own word, under the line at the left, where no curve
    # reaches that height until several hundred evaluations
    ax.text(1.25, BEST - 0.13, "best of all 1,370,754 schedules",
            fontsize=fs_name - 0.4, color=GUIDE, ha="left", va="top", zorder=6)

    # ------------------------------------------------------------------ key
    # one column of its own at the right: a dash sample and the searcher's
    # name, in the order the curves leave the top of the panel
    key = [(label, INK, dash) for _name, label, dash in DRAWN]
    key.append(("six other searchers", GREY, (None, None)))
    y0, dy = (B + 0.88) / H, 0.085
    x_line, x_word = KEY_L / W, (KEY_L + 0.24) / W
    for i, (label, color, dash) in enumerate(key):
        y = y0 - i * dy
        fig.add_artist(plt.Line2D([x_line, x_line + 0.17 / W], [y, y],
                                  color=color, lw=1.15 if color == INK else 0.8,
                                  dashes=dash if dash[0] else (1, 0),
                                  solid_capstyle="round",
                                  dash_capstyle="round"))
        fig.text(x_word, y, label, fontsize=fs_name,
                 color=INK if color == INK else "0.42",
                 ha="left", va="center")
    fig.text(x_line, y0 + 0.1, "50 repetitions each", fontsize=fs_name - 0.5,
             color="0.42", ha="left", va="center")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{STEM}.png", dpi=600)
    fig.savefig(OUT / f"{STEM}.pdf",
                metadata={"CreationDate": None, "Creator": f"{STEM}.py",
                          "Producer": "matplotlib"})

    # ------------------------------------------------------------ self-check
    check_palette()
    print("== self-check ==")
    print(f"  figure {W} x {H} in, one panel, log-log")
    print(f"  table {payload['table']}, space {payload['space_size']:,}, "
          f"{payload['repetitions']} repetitions, "
          f"best {payload['global_max_mean_psnr_db']:.3f} dB")
    print(f"  {len(methods)} searchers drawn, {len(DRAWN)} of them named")
    print("  median schedule evaluations to come within 0.05 dB of the best:")
    for name, method in methods.items():
        row = method["calls_to_threshold"][str(THRESHOLD)]
        tag = "named " if name in named else "      "
        med = row["median_calls"]
        med_s = "not reached" if med is None else f"{med:.1f}"
        print(f"    {tag}{name:20s} {med_s:>12s}   "
              f"reached in {row['reached_share'] * 100:.0f}% of repetitions")
    print(f"  table best {BEST:.3f} dB; mean PSNR of the best schedule found "
          f"so far, at the grid point nearest each of {REPORT_AT}:")
    print(f"    {'':11s} " + "  ".join(f"{t:>6d}" for t in REPORT_AT))
    for name, label, _dash in DRAWN:
        y = curve(methods[name])
        cells = []
        for t in REPORT_AT:
            i = int(np.argmin(np.abs(calls - t)))
            cells.append(f"{y[i]:6.3f}")
        print(f"    {label:11s} " + "  ".join(cells))
    print("    grid points used: " + ", ".join(
        f"{t} -> {int(calls[int(np.argmin(np.abs(calls - t)))])}"
        for t in REPORT_AT))
    print(f"    at 5,000 evaluations: " + ", ".join(
        f"{label} {curve(methods[name])[-1]:.3f}" for name, label, _d in DRAWN))

    # the drawn curves are the stored means over repetitions, and every drawn
    # point is inside the axes
    for name, method in methods.items():
        y = curve(method)
        assert y.size == calls.size
        assert float(y.max()) <= 23.0 and float(y.min()) >= 18.0
    assert calls[0] == 1 and calls[-1] == payload["max_budget"]

    # --- nothing in the key column runs off the page
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def width(text, size):
        handle = fig.text(0, 0, text, fontsize=size)
        out = handle.get_window_extent(renderer).width / fig.dpi
        handle.remove()
        return out

    room = L + AW - (KEY_L + 0.24) - 0.02
    for label, _color, _dash in key:
        got = width(label, fs_name)
        print(f"     key word {label:22s} {got:.2f} in  (room {room:.2f} in)")
        assert got < room, f"{label} runs off the page"
    assert width("50 repetitions each", fs_name - 0.5) < L + AW - KEY_L - 0.02
    print("saved", OUT / f"{STEM}.pdf", "and", OUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
