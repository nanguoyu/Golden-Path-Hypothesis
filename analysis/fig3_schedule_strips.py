"""Figure 3 of the paper -- the schedules themselves, six settings side by side.

Each row is one 50-step schedule drawn as 50 cells, the same encoding as
Figure 1(a): a black cell is a full step, a white cell is a cached step, and
every cell carries a thin grey border so an isolated full step is still a
cell and not a hairline.

Five rows per block, in this order:

    searched (PSNR)    the delivered schedule of the search, arbitration-best
                       of the four searchers      (delivery_best.txt)
    searched (LPIPS)   the schedule the same algorithm delivers when the
                       search objective is LPIPS  (delivery_lpips.txt)
    MeanCache          the incumbent's schedule   (delivery_incumbents.txt)
    BudCache           the incumbent's schedule   (delivery_incumbents.txt)
    random             the random control arm     (delivery.txt, ss_random)

Six blocks in a 2 x 3 grid: the models down, the cache ratios across.

Black, white and grey only; final width 5.5 in = ICLR \\linewidth, so the font
sizes below are the sizes the reader sees.

Writes paper/figs/fig3_schedule_strips.pdf and .png.  Run from anywhere:

    python analysis/fig3_schedule_strips.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis._palette_check import check_group          # noqa: E402

OUT = REPO / "paper" / "figs"
STEM = "fig3_schedule_strips"
SRC = REPO / "resources/schedule_search"

STEPS = 50
INK = "0.10"                  # near-black: every word, and every full-step cell
CELL_EDGE = "0.72"            # the grid between cells
MODELS = (("flux", "FLUX.1-dev"), ("qwen", "Qwen-Image"))
KS = ("29", "37", "41")
ROWS = ("searched (PSNR)", "searched (LPIPS)", "MeanCache", "BudCache",
        "random")


def check_palette():
    """The figure is greyscale; the reading is cell fill against cell border.

    A full step is a filled near-black cell, a cached step is white, and the
    border that separates them is a light grey.  All three have to stay apart
    on CIELAB lightness alone, which is all a greyscale print keeps.
    """
    check_group("fig3_schedule_strips", "greyscale tones",
                {"full step": INK, "cached step": "1.0",
                 "cell border": CELL_EDGE},
                min_distance=18.0, min_lightness_gap=18.0)


# ====================================================================== data
def read_delivery(path):
    """(model, K, name) -> 50-character schedule string."""
    out = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        model, k, name, bits = line.split()
        assert len(bits) == STEPS and set(bits) <= {"0", "1"}, (path, name)
        assert bits.count("1") == int(k), (path, name, bits.count("1"))
        out[(model, k, name)] = bits
    return out


def load():
    """Per setting, the five schedules in row order, plus the algorithm names.

    The LPIPS row is the same algorithm as the PSNR row of that setting: a
    setting delivered by `ss_hill` is paired with `ss_lpips_hill`, one
    delivered by `ss_anneal` with `ss_lpips_anneal`.
    """
    best = read_delivery(SRC / "delivery_best.txt")
    incumbents = read_delivery(SRC / "delivery_incumbents.txt")
    delivery = read_delivery(SRC / "delivery.txt")
    lpips = read_delivery(SRC / "delivery_lpips.txt")

    best_name = {}
    for line in open(SRC / "delivery_best.txt"):
        if line.startswith("#") or not line.strip():
            continue
        model, k, name, _bits = line.split()
        best_name[(model, k)] = name

    blocks = {}
    for model, _label in MODELS:
        for k in KS:
            won = best_name[(model, k)]
            twin = "ss_lpips_hill" if "hill" in won else "ss_lpips_anneal"
            blocks[(model, k)] = {
                "algorithm": won,
                "lpips_algorithm": twin,
                "rows": [best[(model, k, won)],
                         lpips[(model, k, twin)],
                         incumbents[(model, k, "meancache")],
                         incumbents[(model, k, "budcache")],
                         delivery[(model, k, "ss_random")]],
            }
    return blocks


def full_steps(bits):
    return [i for i, c in enumerate(bits) if c == "0"]


def last_free_full_step(bits):
    """The largest full step below 49 -- how late the schedule spends its
    last free evaluation, which is the one reading these strips are drawn for."""
    return max(s for s in full_steps(bits) if s < STEPS - 1)


# =============================================================== presentation
def draw_block(ax, rows, fs_small):
    """Five schedules, one above the other, 50 cells each."""
    for r, bits in enumerate(rows):
        y = len(rows) - 1 - r
        for step, c in enumerate(bits):
            ax.add_patch(Rectangle((step, y), 1, 1,
                                   facecolor=INK if c == "0" else "white",
                                   edgecolor=CELL_EDGE, lw=0.16))
    ax.set_xlim(0, STEPS)
    ax.set_ylim(0, len(rows))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def main():
    blocks = load()

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    W, H = 5.5, 1.76
    fig = plt.figure(figsize=(W, H))
    fs_row, fs_head, fs_tick = 6.6, 7.6, 6.8

    MODEL_X = 0.045            # the rotated model name
    LAB_R = 0.96               # right edge of the row-name gutter
    BW, GAP = 1.38, 0.14       # a block, and the gap that keeps the 49 of
                               # one column clear of the 0 of the next
    ROW_H = 0.104              # one schedule row
    BH = ROW_H * len(ROWS)
    TOP = 0.19                 # room above the top block for the K headings
    VGAP = 0.24                # between the two model rows of blocks
    BOTTOM = 0.34              # room under the bottom blocks for the step axis

    tops = [H - TOP, H - TOP - BH - VGAP]
    lefts = [LAB_R + 0.04 + i * (BW + GAP) for i in range(3)]

    for mi, (model, model_label) in enumerate(MODELS):
        y_top = tops[mi]
        for ki, k in enumerate(KS):
            ax = fig.add_axes([lefts[ki] / W, (y_top - BH) / H, BW / W, BH / H])
            draw_block(ax, blocks[(model, k)]["rows"], fs_row)
            if mi == 0:
                fig.text((lefts[ki] + BW / 2) / W, (y_top + 0.055) / H,
                         f"cache ratio {int(k) / 50:.2f}", fontsize=fs_head,
                         color=INK, ha="center", va="bottom")
            if mi == len(MODELS) - 1:          # the step axis, once, at the foot
                for step in (0, 10, 20, 30, 40, 49):
                    x = lefts[ki] + BW * (step + 0.5) / STEPS
                    fig.add_artist(plt.Line2D([x / W, x / W],
                                              [(y_top - BH - 0.035) / H,
                                               (y_top - BH - 0.005) / H],
                                              color="0.35", lw=0.6))
                    fig.text(x / W, (y_top - BH - 0.045) / H, str(step),
                             fontsize=fs_tick, color=INK, ha="center", va="top")
        # the model's name, upright words turned on their side at the far left
        fig.text(MODEL_X / W, (y_top - BH / 2) / H, model_label,
                 fontsize=fs_head, color=INK, ha="center", va="center",
                 rotation=90)
        # the five row names, once per model row of blocks
        for r, name in enumerate(ROWS):
            y = y_top - (r + 0.5) * ROW_H
            fig.text(LAB_R / W, y / H, name, fontsize=fs_row, color=INK,
                     ha="right", va="center")

    fig.text((lefts[1] + BW / 2) / W, 0.045 / H, "step", fontsize=fs_head,
             color=INK, ha="center", va="bottom")
    fig.text(LAB_R / W, 0.045 / H, "black: full step", fontsize=fs_row,
             color="0.35", ha="right", va="bottom")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{STEM}.png", dpi=600)
    fig.savefig(OUT / f"{STEM}.pdf",
                metadata={"CreationDate": None, "Creator": f"{STEM}.py",
                          "Producer": "matplotlib"})

    # ------------------------------------------------------------ self-check
    check_palette()
    print("== self-check ==")
    print(f"  figure {W} x {H} in; block {BW:.2f} x {BH:.2f} in, "
          f"cell {BW / STEPS * 72:.2f} x {ROW_H * 72:.2f} pt")
    print("  last free full step (largest full step below 49) per row:")
    for model, model_label in MODELS:
        for k in KS:
            block = blocks[(model, k)]
            print(f"    {model_label} K{k}  (search delivered by "
                  f"{block['algorithm']}, LPIPS twin {block['lpips_algorithm']})")
            for name, bits in zip(ROWS, block["rows"]):
                steps = full_steps(bits)
                assert len(steps) == STEPS - int(k)
                print(f"       {name:17s} last free full step "
                      f"{last_free_full_step(bits):2d}   "
                      f"{len(steps)} full steps: {steps}")

    # --- nothing in the gutters overflows
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def width(text, size):
        handle = fig.text(0, 0, text, fontsize=size)
        out = handle.get_window_extent(renderer).width / fig.dpi
        handle.remove()
        return out

    gutter = LAB_R - (MODEL_X + 0.06)
    for name in ROWS:
        got = width(name, fs_row)
        print(f"     row name {name:17s} {got:.2f} in  (gutter {gutter:.2f} in)")
        assert got < gutter, f"{name} overflows the row-name gutter"
    head = width("cache ratio 0.58", fs_head)
    print(f"     column heading      {head:.2f} in  (block {BW:.2f} in)")
    assert head < BW + GAP
    assert lefts[-1] + BW <= W - 0.02, "the right block runs off the page"
    print("saved", OUT / f"{STEM}.pdf", "and", OUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
