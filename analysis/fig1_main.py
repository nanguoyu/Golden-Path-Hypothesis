"""Figure 1 of the paper -- recurring schedules and fixed replay.

A  the schedules four adaptive methods actually realise, FLUX.1-dev at 37 of
   50 steps cached: for each method its five most frequent 50-step schedules,
   one row of 50 cells each, black cell = full step, white cell = cached
   step, ranked top 1..5 at the left, with the percentage of that method's
   generations that ran it at the right, and the method name and count of
   unique schedules under its block. Mean PSNR and LPIPS are computed over
   all 4,896 runs of each method, with one metric pair beside its five rows.
   A separated final row shows the validation-selected ss_hill schedule,
   used for all 1,632 PartiPrompts captions x 3 seeds = 4,896 runs.
B  x  PSNR of the adaptive method,  y  PSNR of one fixed most-frequent
   schedule, on the same off-modal held-out pairing:
     large marks   12 FLUX.1-dev settings (method x cache ratio), held-out
                   prompts, per-setting means, shape = method, colour = ratio
     small points  the prompt--seed pairs behind those means, subsampled to
                   <= 300 per setting, coloured by ratio

Black, white, grey, plus three cache-ratio colours in panel B.  Final width 5.5 in = ICLR \\linewidth, so all
font sizes below are the sizes the reader sees; no scaling factor is applied.

Writes paper/figs/fig1_main.pdf and .png, the files main.tex includes as
Figure~\\ref{fig:golden-path}.  Run from anywhere:

    python analysis/fig1_main.py
"""
import csv
import gzip
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.transforms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis._palette_check import check_cross, check_group   # noqa: E402
from analysis.palette import METHOD_LABEL                 # noqa: E402
from analysis.golden_path_search_bench import (           # noqa: E402
    STEP_OFFSET, SPACE_SIZE, build_all_combos, load_table, ranks_of)

OUT = REPO / "paper" / "figs"
STEM = "fig1_main"

GATES = ["seacache", "teacache", "sencache", "dicache"]
LABEL = METHOD_LABEL
# homologous forecast payload per gate -- DO NOT CHANGE
HOM = {"seacache": "reuse", "teacache": "reuse",
       "sencache": "reuse", "dicache": "di_two_anchor"}
KS = ("29", "37", "41")
STEPS = 50
BARCODE_K = "37"               # panel A is read at one cache ratio
TOP_PATHS = 5                  # rows drawn per method in panel A
SEARCHED_SCHEDULE = "ss_hill"  # selected on 50 validation prompts, not this test set

INK = "0.10"                   # near-black: every word, and every full step
CLOUD = "0.72"                 # the per-prompt cloud of panel B
CELL_EDGE = "0.72"             # the grid between two cells of a barcode row
GUIDE = "0.55"                 # the identity line, the leaders, the axis words
# one shape per method in panel B; the shapes are the only thing that tells
# the four apart, so no two of them share a silhouette at 3 pt
SHAPE = {"seacache": "o", "teacache": "s", "sencache": "^", "dicache": "D"}
# panel B draws one model, FLUX.1-dev, so the only thing left to separate is
# the cache ratio: a dark red, an amber and a dark blue, the widest-separated
# triple we could find around the amber the panel was asked for.  These mean
# a cache ratio in this panel and nothing else; the methods are shapes here,
# so no method hue of analysis/palette.py is used or implied.
RATIO = {"29": "#B2182B", "37": "#E0A800", "41": "#004488"}
RATIO_LABEL = {"29": "target cache ratio 0.58", "37": "target cache ratio 0.74", "41": "target cache ratio 0.82"}
PANEL_B_MODEL = "flux"

# panel C: the exhaustive K41 table, and the schedules marked on it
K41_TABLE = REPO / "resources/exhaustive_k41/merged.tsv.gz"
DELIVERY = REPO / "resources/schedule_search"
MEDIAN_RANK = 685_377          # the middle schedule of the sorted 1,370,754
# the names too long for one line beside the curve
# with three marks there is room for every name on one line
WRAP = {}
NOTE = "black = full step"      # the barcode's own encoding, said once
FS_SHARE = 7.0                 # schedule percentages at final paper width


def check_palette():
    """The figure is greyscale, so the only claim to measure is lightness.

    Nothing in this figure is told apart by hue: a method is named beneath
    its own barcode rows in panel A and carries its own marker shape in
    panel B.  Four tones do carry meaning -- the near-black of the words, the
    marks and the full-step cells, the white of a cached cell, the light grey
    of the per-prompt cloud and the cell grid, and the mid grey of the guide
    lines -- so each pair is asked to stay far apart on CIELAB lightness,
    which is all a greyscale print keeps.
    """
    check_group("fig1_main", "greyscale tones",
                {"ink": INK, "cached cell": "1.0", "cloud and cell grid": CLOUD,
                 "guides": GUIDE},
                min_distance=15.0, min_lightness_gap=15.0)
    # panel B's three cache ratios: each mark is a black-edged shape filled
    # with one of these, and the key names all three, so the hues only have to
    # stay apart from one another and from the grey of the per-prompt cloud.
    ratios = {RATIO_LABEL[k]: RATIO[k] for k in KS}
    check_group("fig1_main", "cache ratios", ratios,
                min_distance=25.0, min_lightness_gap=8.0, min_chroma=20.0)
    check_cross("fig1_main", "cache ratios", ratios,
                "greyscale tones", {"cloud": CLOUD, "guides": GUIDE},
                min_distance=25.0, min_cvd_distance=20.0)


# ====================================================================== data
def load_paths(model="flux", k=BARCODE_K):
    """Native schedule counts and all-run means, plus the searched schedule.

    The quality columns pool every prompt-seed run for a method, regardless
    of which path that run realised. Search uses the same PartiPrompts keys.
    """
    counts = defaultdict(Counter)
    native = {g: {} for g in GATES}
    with gzip.open(REPO / f"resources/spx/perprompt_native_{model}.tsv.gz",
                   "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["method"] in GATES and r["k"] == k:
                g = r["method"]
                key = (int(r["seed"]), int(r["prompt_idx"]))
                assert key not in native[g], (g, key)
                native[g][key] = (float(r["psnr"]), float(r["lpips"]))
                counts[g][r["path"]] += 1
    keys = set(native[GATES[0]])
    assert len(keys) == 4896
    assert Counter(seed for seed, _ in keys) == {41: 1632, 42: 1632, 43: 1632}
    assert all(set(native[g]) == keys for g in GATES)
    means = {g: tuple(statistics.fmean(v[i] for v in native[g].values())
                      for i in range(2)) for g in GATES}

    searched = {}
    for path in sorted(DELIVERY.glob(f"perprompt_search_{model}_*.tsv.gz")):
        with gzip.open(path, "rt") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if (r["schedule"], r["payload"], r["k"], r["dataset"]) != (
                        SEARCHED_SCHEDULE, "reuse", k, "parti_full"):
                    continue
                key = (int(r["seed"]), int(r["prompt_idx"]))
                assert key not in searched, (path, key)
                searched[key] = (float(r["psnr"]), float(r["lpips"]))
    assert set(searched) == keys, "searched and native evaluation keys differ"
    bits = read_delivery(DELIVERY / "delivery.txt")[(model, k, SEARCHED_SCHEDULE)]
    assert len(bits) == STEPS and set(bits) <= {"0", "1"}
    assert bits.count("1") == int(k) and bits[0] == "0"
    search_means = tuple(statistics.fmean(v[i] for v in searched.values())
                         for i in range(2))
    return ({g: (counts[g], len(native[g])) for g in GATES}, means,
            dict(path=bits, total=len(searched), means=search_means))


def load_image():
    """Per-setting aggregates, pooled per-prompt paired diffs, random control.

    A setting is (gate, model, K).  Pairing: held-out prompts whose realised
    gate path differs from that (model, K, gate) fixed most-frequent schedule.
    The random control reuses exactly that prompt/seed pairing.
    """
    splits = json.load(open(REPO / "resources/sp_cross_schedules/parti_spx_splits.v1.json"))
    disc = set(splits["roles"]["discovery"])
    settings, pair_pool, rand_pts = [], defaultdict(list), defaultdict(list)
    rand_pool = defaultdict(list)          # v9 addition: per-prompt random deltas

    for model in ("flux", "qwen"):
        spx, nat = defaultdict(dict), defaultdict(dict)
        with gzip.open(REPO / f"resources/spx/perprompt_spx_{model}.tsv.gz", "rt") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                sc = r["schedule"]
                if sc.endswith("_top1"):
                    g = sc[:-5]
                    if g in HOM and r["payload"] == HOM[g]:
                        spx[(g, r["k"], r["seed"])][int(r["prompt_idx"])] = float(r["psnr"])
                elif sc.startswith("rand_"):
                    spx[(sc, r["payload"], r["k"], r["seed"])][int(r["prompt_idx"])] = float(r["psnr"])
        with gzip.open(REPO / f"resources/spx/perprompt_native_{model}.tsv.gz", "rt") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["method"] in GATES:
                    nat[(r["method"], r["k"], r["seed"])][int(r["prompt_idx"])] = (float(r["psnr"]), r["path"])

        for g in GATES:
            for k in KS:
                sched = open(REPO / f"resources/sp_cross_schedules/{model}_k{k}_{g}_top1.txt").read().strip()
                off = {}                                   # seed -> {pid: native psnr}
                for (gg, kk, s), pp in nat.items():
                    if (gg, kk) != (g, k):
                        continue
                    sel = {pid: v[0] for pid, v in pp.items()
                           if pid not in disc and v[1] != sched}
                    if sel:
                        off[s] = sel
                xs, ys, ds = [], [], []
                for s, sel in off.items():
                    fx = spx.get((g, k, s), {})
                    for pid, npsnr in sel.items():
                        if pid in fx:
                            xs.append(npsnr)
                            ys.append(fx[pid])
                            ds.append(fx[pid] - npsnr)
                            pair_pool[g].append(fx[pid] - npsnr)
                if not xs:
                    continue
                sd = sorted(ds)                      # per-setting spread, same pairing
                settings.append(dict(gate=g, model=model, k=k, kind="image",
                                     nat=statistics.fmean(xs), fix=statistics.fmean(ys),
                                     delta=statistics.fmean(ys) - statistics.fmean(xs),
                                     n=len(xs), pairs=list(zip(xs, ys)),
                                     p5=sd[int(.05 * len(sd))],
                                     p95=sd[int(.95 * len(sd))]))
                for j in range(1, 6):
                    row = f"rand_{j}"
                    ds = []
                    for s, sel in off.items():
                        rx = spx.get((row, HOM[g], k, s), {})
                        ds += [rx[pid] - npsnr for pid, npsnr in sel.items() if pid in rx]
                    if ds:
                        rand_pts[g].append(dict(model=model, k=k, row=row,
                                                delta=statistics.fmean(ds), n=len(ds)))
                        rand_pool[g] += ds           # v9 addition
    return settings, pair_pool, rand_pts, rand_pool


def read_delivery(path):
    """(model, K, name) -> 50-character schedule string."""
    out = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        model, k, name, bits = line.split()
        out[(model, k, name)] = bits
    return out


def table_rank(bits):
    """The exhaustive table's rank of a 50-step schedule, or None.

    The table enumerates the K41 schedules that keep steps 0, 1, 2 and 49
    full and choose the remaining five full steps freely among 3..48, so a
    schedule that caches one of the four forced steps is simply not in it.
    """
    full = [i for i, c in enumerate(bits) if c == "0"]
    if len(full) != 9 or full[:3] != [0, 1, 2] or full[-1] != STEPS - 1:
        return None
    return int(ranks_of(np.array([[f - STEP_OFFSET for f in full[3:8]]]))[0])


def load_landscape():
    """The sorted K41 curve, and the six schedules marked on it.

    `load_table` is the exhaustive benchmark's own reader, so the PSNR
    convention here is the one `analysis/golden_path_search_bench.py` runs on:
    the table's `mean_psnr_db`, the mean over the four prompt-noise pairs of
    20 log10(255) - 10 log10(mse).
    """
    if not K41_TABLE.exists():
        raise SystemExit(
            f"{K41_TABLE} is missing.  It is the untracked local copy of the "
            "exhaustive K41 table; fetch it from the cluster before rendering.")
    combos = build_all_combos()
    scores, info = load_table(K41_TABLE, combos)
    order = np.argsort(-scores)
    curve = scores[order]                       # descending, rank r at index r-1
    place = np.empty(SPACE_SIZE, dtype=np.int64)
    place[order] = np.arange(SPACE_SIZE)        # table rank -> 0-based position

    delivered = read_delivery(DELIVERY / "delivery.txt")

    # three marks, no comparison: the two ends of what the table contains and
    # the one schedule the search delivered from eight unrelated pairs
    marks = [
        ("table best", int(np.argmax(scores))),
        ("searched on 8 other pairs",
         table_rank(delivered[("flux", "41", "ss_hill")])),
        ("median of all schedules", int(order[MEDIAN_RANK - 1])),
    ]
    out = [dict(name=name, table_rank=rank, rank=int(place[rank]) + 1,
                psnr=float(scores[rank]))
           for name, rank in marks]
    return curve, out, info


# ================================================================ self-check
def spot_check_random(gate, model, k, row):
    """Independent recomputation of one random-control mean, written against the
    raw files without reusing load_image()'s indices."""
    disc = set(json.load(open(REPO / "resources/sp_cross_schedules/parti_spx_splits.v1.json"))["roles"]["discovery"])
    sched = open(REPO / f"resources/sp_cross_schedules/{model}_k{k}_{gate}_top1.txt").read().strip()
    native, rnd = {}, {}
    with gzip.open(REPO / f"resources/spx/perprompt_native_{model}.tsv.gz", "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["method"] == gate and r["k"] == k and r["path"] != sched \
                    and int(r["prompt_idx"]) not in disc:
                native[(r["seed"], int(r["prompt_idx"]))] = float(r["psnr"])
    with gzip.open(REPO / f"resources/spx/perprompt_spx_{model}.tsv.gz", "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["schedule"] == row and r["payload"] == HOM[gate] and r["k"] == k:
                rnd[(r["seed"], int(r["prompt_idx"]))] = float(r["psnr"])
    d = [rnd[key] - v for key, v in native.items() if key in rnd]
    return statistics.fmean(d), len(d)


def spot_check_psnr(n_rows=200):
    """The table's mean_psnr_db really is the four-pair mean of the MSE columns.

    Panel C's y axis is that column, so the convention behind it is checked
    against the raw per-pair MSEs the same file carries.
    """
    worst = 0.0
    with gzip.open(K41_TABLE, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_mse = [header.index(f"mse_p{p}") for p in (5, 8, 9, 15)]
        i_mean = header.index("mean_psnr_db")
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            fields = line.rstrip("\n").split("\t")
            got = statistics.fmean(
                20 * math.log10(255.0) - 10 * math.log10(float(fields[j]))
                for j in i_mse)
            worst = max(worst, abs(got - float(fields[i_mean])))
    return worst


# =============================================================== presentation
NAME_ORDER = ("seacache", "sencache", "dicache", "teacache")
LO, HI = 13.2, 33.2           # equal limits: the identity line is the diagonal
CLOUD_CAP = 300               # per-prompt points drawn per setting
CLOUD_DPI = 600               # resolution the rasterised cloud is written at


def cloud_points(img, cap=CLOUD_CAP, seed=17):
    """Per-setting subsample of the per-prompt (native, fixed) pairs."""
    rng = np.random.default_rng(seed)
    out = []
    for s in sorted(img, key=lambda s: (s["gate"], s["model"], s["k"])):
        p = s["pairs"]
        if len(p) > cap:
            p = [p[i] for i in sorted(rng.choice(len(p), cap, replace=False))]
        out.append((s["gate"], s["k"], np.array([q[0] for q in p]),
                    np.array([q[1] for q in p])))
    return out


def panel_letter(fig, x, y, letter, fs_lab):
    """The panel's letter above its top-left corner, the order the caption reads."""
    fig.text(x, y, letter, fontsize=fs_lab, color=INK, ha="left", va="bottom")


def draw_barcode(fig, geom, counts, means, searched, fs_lab, fs_tick, fs_share):
    """Panel A: frequent native paths and one fixed searched schedule.

    Each native quality pair spans all five rows: it is one method-level
    mean over all runs, not the conditional quality of the adjacent barcode.
    """
    left, right, metric_left, metric_right, top, bottom = geom
    width = right - left
    metric_width = (metric_right - metric_left) / 2
    share_x = metric_left - 0.055
    foot, gap = 0.112, 0.010
    search_band = 0.225
    pitch = (top - bottom - len(GATES) * foot
             - (len(GATES) - 1) * gap - search_band) / (len(GATES) * TOP_PATHS)
    row_h = 0.052
    W, H = fig.get_size_inches()

    def barcode(path, center, share, rank=None):
        assert len(path) == STEPS and set(path) <= {"0", "1"}
        if rank is not None:
            fig.text((left - 0.03) / W, center / H, str(rank),
                     fontsize=fs_share, color="0.35", ha="right", va="center",
                     gid="barcode-row-label")
        ax = fig.add_axes([left / W, (center - row_h / 2) / H,
                           width / W, row_h / H])
        for step, ch in enumerate(path):
            ax.add_patch(Rectangle((step, 0), 1, 1,
                                   facecolor=INK if ch == "0" else "white",
                                   edgecolor=CELL_EDGE, lw=0.16))
        ax.set_xlim(0, STEPS)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        fig.text(share_x / W, center / H,
                 f"{100 * share:.1f}" if share < 0.01 else f"{100 * share:.0f}",
                 fontsize=fs_share, color="0.30", ha="right", va="center",
                 gid="barcode-row-label")

    def draw_metrics(values, block_top, block_bottom, *, bold=False):
        divider_x = (metric_left + metric_width) / W
        fig.add_artist(plt.Line2D(
            [divider_x, divider_x], [block_bottom / H, block_top / H],
            transform=fig.transFigure, color="0.82", lw=0.30,
            gid="barcode-metric-divider"))
        for col, (value, digits) in enumerate(zip(values, (2, 3))):
            x = metric_left + col * metric_width
            # The bold searched values need extra space around the divider.
            offset = (-0.025 if col == 0 else 0.025) if bold else 0
            fig.text((x + metric_width / 2 + offset) / W,
                     (block_top + block_bottom) / 2 / H, f"{value:.{digits}f}",
                     fontsize=fs_share, color=INK, ha="center", va="center",
                     weight="bold" if bold else "normal",
                     gid="barcode-metric")

    drawn = []
    y = top
    for i, g in enumerate(GATES):
        c, total = counts[g]
        distinct, top_paths = len(c), c.most_common(TOP_PATHS)
        block_top = y
        for rank, (path, n) in enumerate(top_paths, 1):
            barcode(path, y - pitch / 2, n / total, rank)
            y -= pitch
        draw_metrics(means[g], block_top, y)
        fig.text(left / W, (y - foot * 0.52) / H,
                 f"{LABEL[g]}: {distinct} unique schedules",
                 fontsize=fs_share, color=INK, ha="left", va="center",
                 gid="barcode-footer")
        y -= foot + (gap if i < len(GATES) - 1 else 0)
        drawn.append((g, total, distinct,
                      [(p, n / total) for p, n in top_paths]))

    separator_y = y - 0.021
    fig.add_artist(plt.Line2D([left / W, metric_right / W],
                              [separator_y / H, separator_y / H],
                              color=CELL_EDGE, lw=0.45))
    search_center = y - 0.087
    barcode(searched["path"], search_center, 1.0)
    draw_metrics(searched["means"], search_center + 0.051,
                 search_center - 0.051, bold=True)
    fig.text(left / W, (bottom + 0.040) / H,
             "Searched: 1 schedule",
             fontsize=fs_share, color=INK, ha="left", va="center",
             gid="barcode-footer")

    # the step axis, once, under the bottom row
    for step in (0, 10, 20, 30, 40, 49):
        x = left + width * (step + 0.5) / STEPS
        fig.add_artist(plt.Line2D([x / W, x / W],
                                  [(bottom - 0.0333) / H, bottom / H],
                                  color="0.35", lw=0.6))
        fig.text(x / W, (bottom - 0.080) / H, str(step), fontsize=fs_tick,
                 color=INK, ha="center", va="center", gid="barcode-tick")
    fig.text((left + width / 2) / W, (bottom - 0.195) / H, "step",
             fontsize=fs_lab, color=INK, ha="center", va="center",
             gid="barcode-xlabel")
    return drawn


def draw_main(ax, img, cloud, fs_lab, fs_tick):
    """The whole message: everything measured sits on the identity line."""
    for g, k, xs, ys in cloud:
        ax.scatter(xs, ys, s=9.5, marker="o", facecolor=RATIO[k], alpha=0.32,
                   edgecolor=INK, lw=0.2, zorder=2, rasterized=True)
    ax.plot([LO, HI], [LO, HI], color=GUIDE, lw=0.8, zorder=3)
    # the two axes carry the same data range, so the identity line's screen
    # slope is the height/width ratio of the axes box; read it off the figure
    fw, fh = ax.figure.get_size_inches()
    box = ax.get_position()
    tilt = float(np.degrees(np.arctan2(box.height * fh, box.width * fw)))
    ax.text(26.0, 27.4, "y = x", fontsize=fs_tick, color="0.42",
            ha="left", va="bottom", rotation=tilt, rotation_mode="anchor",
            zorder=5, path_effects=[matplotlib.patheffects.withStroke(
                linewidth=1.6, foreground="white")])
    # a method is a shape, a cache ratio is a fill colour, so each method
    # shows exactly three marks: K29, K37 and K41 of the same model
    for g in GATES:
        for k in KS:
            ss = [s for s in img if s["gate"] == g and s["k"] == k]
            ax.scatter([s["nat"] for s in ss], [s["fix"] for s in ss], s=28,
                       marker=SHAPE[g], facecolor=RATIO[k], edgecolor=INK,
                       lw=0.45, zorder=6.8 if g == "teacache" else 6)

    ax.set_xlim(LO, HI)
    ax.set_ylim(LO, HI)
    ax.set_xticks([15, 20, 25, 30])
    ax.set_yticks([15, 20, 25, 30])
    ax.tick_params(labelsize=fs_tick, pad=1.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("adaptive PSNR (dB)", fontsize=fs_lab, labelpad=1.5)
    ax.set_ylabel("fixed schedule PSNR (dB)", fontsize=fs_lab, labelpad=2)

    # Put method labels in the lower-right corner, away from the diagonal.
    # No background patch masks the measured points.
    kx, ky, dky = 27.0, 19.6, 1.30
    for i, g in enumerate(NAME_ORDER):
        ax.scatter([kx], [ky - i * dky], s=28, marker=SHAPE[g],
                   facecolor="white", edgecolor=INK, lw=0.45, zorder=8)
        ax.text(kx + 0.85, ky - i * dky, LABEL[g], fontsize=fs_tick,
                color=INK, ha="left", va="center", zorder=8)
    # the three ratio colours go in the other empty corner, above the
    # identity line, as wide flat swatches no one can read as a method shape
    rx, ry, dry = 14.0, 32.3, 1.45
    for i, k in enumerate(KS):
        y = ry - i * dry
        ax.add_patch(matplotlib.patches.Rectangle(
            (rx, y - 0.32), 1.5, 0.64, facecolor=RATIO[k], edgecolor=INK,
            lw=0.35, zorder=8))
        ax.text(rx + 2.0, y, RATIO_LABEL[k], fontsize=fs_tick,
                color=INK, ha="left", va="center", zorder=8)
    ax.text(HI - 0.2, LO + 0.35, "one prompt-seed run", fontsize=fs_tick,
            color="0.35", ha="right", va="bottom", zorder=8)
    return kx, rx


def draw_landscape(ax, curve, marks, fs_lab, fs_tick):
    """Panel C: the share of all K41 schedules per 0.25 dB of four-pair mean
    PSNR, with the median, the searched schedule and the best marked."""
    lo = math.floor(curve.min() * 4) / 4
    hi = math.ceil(curve.max() * 4) / 4
    edges = np.arange(lo, hi + 0.2501, 0.25)
    counts, _ = np.histogram(curve, bins=edges)
    share = 100.0 * counts / curve.size
    ax.bar(edges[:-1], share, width=0.25, align="edge", color=CLOUD,
           edgecolor="white", lw=0.3, zorder=2)
    ax.set_xlim(lo - 0.15, hi + 0.75)
    ax.set_ylim(0, 11.8)
    ax.set_xticks([16, 18, 20, 22])
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.tick_params(labelsize=fs_tick, pad=1.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("schedule PSNR (dB)", fontsize=fs_lab, labelpad=1.5)
    ax.set_ylabel("share of schedules (%)", fontsize=fs_lab, labelpad=2)
    ax.text(lo - 0.05, 11.5, "all 1,370,754 schedules", fontsize=fs_lab - 1.6,
            color="0.35", ha="left", va="top", zorder=7)

    # three vertical lines; the label of each sits above its line, and the
    # heights are staggered so no two labels share a band
    label = {"table best": ("best\n22.7 dB", "#004488", 9.9, "right"),
             "searched on 8 other pairs":
                 ("searched\n21.8 dB", "#B2182B", 8.3, "right"),
             "median of all schedules": ("median\n18.7 dB", INK, 8.3, "right")}
    for m in marks:
        text, colour, y, ha = label[m["name"]]
        ax.plot([m["psnr"], m["psnr"]], [0, y - 0.25], color=colour, lw=1.0,
                zorder=5)
        x = {"searched on 8 other pairs": m["psnr"] - 0.1, "table best": hi + 0.7,
             "median of all schedules": m["psnr"] - 0.1}[m["name"]]
        ax.text(x, y, text, fontsize=fs_lab - 1.2, color=colour, ha=ha,
                va="bottom", zorder=7, linespacing=1.05)
    return {m["name"]: float(share[min(int((m["psnr"] - lo) / 0.25),
                                       share.size - 1)]) for m in marks}


class Ruler:
    """Measure rendered text width in inches, so nothing silently overflows."""

    def __init__(self, fig):
        fig.canvas.draw()
        self.fig, self.r = fig, fig.canvas.get_renderer()

    def w(self, s, fs, weight="normal"):
        t = self.fig.text(0, 0, s, fontsize=fs, weight=weight)
        out = t.get_window_extent(self.r).width / self.fig.dpi
        t.remove()
        return out


def main():
    counts, means, searched = load_paths()
    img, pool, rand, rand_pool = load_image()
    # Panel B shows the same model as panel A, so the 24 settings returned by
    # the loader are filtered to the 12 FLUX.1-dev settings before drawing.
    shown = [s for s in img if s["model"] == PANEL_B_MODEL]
    cloud = cloud_points(shown)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.7,
                         "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                         "xtick.major.size": 2.4, "ytick.major.size": 2.4,
                         "xtick.major.pad": 1.6, "ytick.major.pad": 1.6})
    W, H = 5.5, 2.98
    fig = plt.figure(figsize=(W, H))
    AT, AB = 2.615, 0.315
    AL, AR = 0.12, 2.02                       # 50 cells, 2.74 pt each
    ML, MR = 2.335, 2.955                     # two compact, method-wide columns
    BL, BW = 3.36, 2.06
    B = (AT + AB - BW) / 2                    # centre B against all of A's rows
    ax = fig.add_axes([BL / W, B / H, BW / W, BW / H])
    ax.set_aspect("equal", adjustable="box")
    fs_lab, fs_tick, fs_share = 7.4, 7.0, 7.4

    bars = draw_barcode(fig, (AL, AR, ML, MR, AT, AB), counts, means, searched,
                        fs_share, fs_tick, FS_SHARE)
    key_x, ratio_x = draw_main(ax, shown, cloud, fs_lab, fs_tick)

    panel_letter(fig, 0.008, 2.875 / H, "(a)", fs_lab)
    panel_letter(fig, (BL - 0.36) / W, 2.875 / H, "(b)", fs_lab)
    fig.text((AL + AR) / 2 / W, 2.835 / H, NOTE,
             fontsize=FS_SHARE, color="0.35", ha="center", va="top")
    fig.text((AR + ML) / 2 / W, 2.935 / H,
             f"% of\n{bars[0][1]:,}\nruns",
             fontsize=FS_SHARE, color=INK, ha="center", va="top",
             linespacing=1.03, gid="barcode-header")
    fig.text((ML + MR) / 2 / W, 2.905 / H, "Mean",
             fontsize=FS_SHARE, color=INK, ha="center", va="center",
             gid="barcode-header")
    for col, title in enumerate(("PSNR\ndB ↑", "LPIPS\n↓")):
        fig.text((ML + (col + 0.5) * (MR - ML) / 2) / W, 2.835 / H, title,
                 fontsize=FS_SHARE, color=INK, ha="center", va="top",
                 linespacing=1.03, gid="barcode-header")

    OUT.mkdir(parents=True, exist_ok=True)
    # 600 dpi: the per-prompt cloud in panel B is the one rasterised object in
    # the figure, and this is the resolution it is written at inside the PDF
    fig.savefig(OUT / f"{STEM}.png", dpi=CLOUD_DPI)
    # deterministic PDF: no timestamp, no tool version string, so a re-run
    # reproduces the checksum in paper/figs/SHA256SUMS
    fig.savefig(OUT / f"{STEM}.pdf", dpi=CLOUD_DPI,
                metadata={"CreationDate": None, "Creator": "fig1_main.py",
                          "Producer": "matplotlib"})

    # ------------------------------------------------------------ self-check
    rule = Ruler(fig)
    for s in img:                       # the cloud and the dots are one pairing
        assert len(s["pairs"]) == s["n"]
        assert abs(statistics.fmean(a for a, b in s["pairs"]) - s["nat"]) < 1e-12
        assert abs(statistics.fmean(b for a, b in s["pairs"]) - s["fix"]) < 1e-12
    n_cloud = sum(len(xs) for _, _, xs, _ in cloud)
    cx = np.concatenate([xs for _, _, xs, _ in cloud])
    cy = np.concatenate([ys for _, _, _, ys in cloud])
    outside = int((~((cx >= LO) & (cx <= HI) & (cy >= LO) & (cy <= HI))).sum())

    check_palette()
    print("== self-check ==")
    print(f"  figure {W} x {H} in;  panels  A {MR - AL:.2f} x {AT - AB:.2f} in, "
          f"B {BW:.2f} x {BW:.2f} ({LO}..{HI} dB both)")

    # --- A
    print(f"  A: FLUX.1-dev, K{BARCODE_K}, PartiPrompts, "
          f"{bars[0][1]} prompt-seed runs per method "
          f"({bars[0][1] // 3} prompts x 3 seeds)")
    for g, total, distinct, top in bars:
        print(f"     {LABEL[g]:9s} {distinct:3d} distinct paths, "
              f"top {TOP_PATHS} together {sum(s for _p, s in top):.4f}; "
              f"all-run mean PSNR {means[g][0]:.6f}, LPIPS {means[g][1]:.6f}")
        for path, share in top:
            print(f"        {share:.4f}  {path}")
        assert total == bars[0][1]
    print(f"     searched {SEARCHED_SCHEDULE}: {searched['total']} runs, "
          f"100% use frequency; mean PSNR {searched['means'][0]:.6f}, "
          f"LPIPS {searched['means'][1]:.6f}; {searched['path']}")
    assert sum(t.get_gid() == "barcode-metric" for t in fig.texts) == 10
    assert sum(a.get_gid() == "barcode-metric-divider" for a in fig.artists) == 5
    assert sum(t.get_gid() == "barcode-metric" and t.get_weight() == "bold"
               for t in fig.texts) == 2
    assert len(fig.axes) == len(GATES) * TOP_PATHS + 2

    # --- B
    print(f"  B: FLUX.1-dev only, {len(shown)} setting means "
          f"(4 methods x 3 cache ratios), |dy| max "
          f"{max(abs(s['delta']) for s in shown):.2f} dB, "
          f"{sum(s['delta'] >= -0.25 for s in shown)}/{len(shown)} "
          f"not below -0.25 dB")
    print(f"     cloud {n_cloud} of {sum(s['n'] for s in shown)} per-prompt pairs "
          f"drawn (cap {CLOUD_CAP}/setting), {outside} drawn points fall outside "
          f"the axes ({outside / n_cloud * 100:.1f}%)")
    for k in KS:
        print(f"       K{k}: {sum(len(xs) for _, kk, xs, _ in cloud if kk == k)} points drawn")
    drawn_pairs = np.array([b - a for s in shown for a, b in s["pairs"]])
    print(f"     drawn pool {drawn_pairs.size} pairs: median "
          f"{np.median(drawn_pairs):+.4f} dB, 5-95% "
          f"{np.quantile(drawn_pairs, .05):+.3f}.."
          f"{np.quantile(drawn_pairs, .95):+.3f}")
    for s in sorted(shown, key=lambda s: (NAME_ORDER.index(s["gate"]), s["k"])):
        print(f"     {LABEL[s['gate']]:9s} {s['model']:5s} K{s['k']}  "
              f"adaptive {s['nat']:6.3f}  fixed {s['fix']:6.3f}  "
              f"delta {s['delta']:+.3f} dB  n = {s['n']}")
    # the loader itself is unchanged, so its 24-setting pool is checked too
    allp = np.array([d for g in GATES for d in pool[g]], dtype=float)
    fmed, fp5, fp95 = np.median(allp), np.quantile(allp, .05), np.quantile(allp, .95)
    print(f"     loader check, all 24 settings: {allp.size} pairs, median "
          f"{fmed:+.4f} dB, 5-95% {fp5:+.3f}..{fp95:+.3f}")
    assert allp.size == sum(s["n"] for s in img)
    assert abs(fmed - (-0.0096)) < 0.02, fmed
    assert abs(fp5 - (-0.849)) < 0.02 and abs(fp95 - 1.318) < 0.02, (fp5, fp95)

    # --- Ruler: nothing overflows its panel
    widths = {
        "A rank label": (rule.w("5", FS_SHARE), 0.15),
        "A method and count": (max(rule.w(
            f"{LABEL[g]}: {unique} unique schedules", FS_SHARE)
            for g, total, unique, _ in bars),
                               MR - AL),
        "A searched count": (rule.w(
            "Searched: 1 schedule", FS_SHARE), MR - AL),
        "A share": (rule.w("100", FS_SHARE), ML - AR - 0.075),
        "A frequency header": (max(rule.w(s, FS_SHARE)
                                    for s in ("% of", f"{bars[0][1]:,}", "runs")),
                                ML - AR),
        "A PSNR header": (max(rule.w(s, FS_SHARE) for s in ("PSNR", "dB ↑")),
                          (MR - ML) / 2),
        "A LPIPS header": (rule.w("LPIPS", FS_SHARE), (MR - ML) / 2),
        "A metric value": (max(rule.w(f"{v:.{digits}f}", FS_SHARE)
                                for pair in means.values()
                                for v, digits in zip(pair, (2, 3))), (MR - ML) / 2),
        "A searched value": (max(rule.w(f"{v:.{digits}f}", FS_SHARE, weight="bold")
                                  for v, digits in zip(searched["means"], (2, 3))),
                              (MR - ML) / 2 + 0.06),
        "A encoding note": (rule.w(NOTE, FS_SHARE), AR - AL),
        "B xlabel": (rule.w("adaptive PSNR (dB)", fs_lab), BW),
        "B ylabel": (rule.w("fixed schedule PSNR (dB)", fs_lab), BW),
        "B widest key word": (max(rule.w(LABEL[g], fs_tick) for g in GATES),
                              BW * (HI - (key_x + 0.85)) / (HI - LO)),
        # a ratio row: swatch, gap, then the word
        "B ratio row": (BW * 2.0 / (HI - LO) + rule.w(RATIO_LABEL["41"], fs_tick),
                        BW * (HI - ratio_x) / (HI - LO)),
        "B point note": (rule.w("one prompt-seed run", fs_tick), BW - 0.02),
    }
    for k, (v, lim) in widths.items():
        print(f"     {k:20s} {v:.2f} in  (room {lim:.2f} in)")
        assert v < lim, f"{k} overflows its panel"
    assert BL - 0.36 > MR + 0.025, "the metric columns would touch panel B"
    barcode_text = [t for t in fig.texts
                    if t.get_gid() in ("barcode-row-label", "barcode-footer",
                                       "barcode-metric")]
    extents = [t.get_window_extent(rule.r) for t in barcode_text]
    a_lo = min(bb.y0 for bb in extents) / fig.dpi
    a_hi = max(bb.y1 for bb in extents) / fig.dpi
    print(f"     A rendered content y = {a_lo:.4f}..{a_hi:.4f} in; "
          f"B frame y = {B:.4f}..{B + BW:.4f} in")
    assert a_lo >= AB - 0.02 and a_hi <= AT + 0.02
    assert abs((AT + AB) / 2 - (B + BW / 2)) < 1e-12
    assert all(t.get_fontsize() >= 7 for t in fig.texts)
    for t in fig.texts:
        bb = t.get_window_extent(rule.r)
        assert bb.x0 >= 0 and bb.x1 <= fig.bbox.width, t.get_text()
        assert bb.y0 >= 0 and bb.y1 <= fig.bbox.height, t.get_text()

    # --- independent recomputation of one random-control cell
    chk, nchk = spot_check_random("seacache", "flux", "29", "rand_1")
    ref = next(r for r in rand["seacache"] if (r["model"], r["k"], r["row"])
               == ("flux", "29", "rand_1"))
    assert abs(chk - ref["delta"]) < 1e-9 and nchk == ref["n"]
    print(f"  spot check seacache/flux/K29/rand_1 reproduced: {chk:+.6f} (n={nchk})")
    print("saved", OUT / f"{STEM}.pdf", "and", OUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
