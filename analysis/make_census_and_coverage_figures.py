"""Two slide assets from staged data.

1. slides/assets/census_paths.png: every distinct compute path one adaptive
   method realized in one setting, drawn as bit strips ranked by share.
2. slides/assets/coverage_bars.png: the pool-best coverage share per
   partition at the 0.5 dB margin, with its exact binomial lower bound.
"""

import collections
import gzip
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BLUE = "#0072B2"
FULL = "#2b2b2b"
CACHED = "#d9d9d9"
INK = "#1a1a1a"


def census():
    rows = collections.Counter()
    n = 0
    with gzip.open("resources/spx/perprompt_native_flux.tsv.gz", "rt") as f:
        f.readline()
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[0] == "seacache" and c[1] == "29":
                rows[c[9]] += 1
                n += 1
    top = rows.most_common(len(rows))

    W, H = 12.6, 4.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.02, 0.10, 0.96, 0.86])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_show = 8
    seg = 0.60 / 50
    y0, dy = 0.88, 0.115
    for i, (path, cnt) in enumerate(top[:n_show]):
        y = y0 - i * dy
        for j, b in enumerate(path):
            ax.add_patch(Rectangle((0.03 + j * seg, y), seg * 0.8, 0.075,
                                   facecolor=CACHED if b == "1" else FULL,
                                   edgecolor="none"))
        share = cnt / n
        ax.add_patch(Rectangle((0.66, y), 0.30 * share / 0.86, 0.075,
                               facecolor=BLUE, edgecolor="none"))
        ax.text(0.665 + 0.30 * share / 0.86, y + 0.0375,
                f"{share * 100:.1f}%", va="center", ha="left",
                fontsize=10.5, color=INK)
    rest = 1 - sum(c for _, c in top[:n_show]) / n
    ax.text(0.03, y0 - n_show * dy + 0.02,
            f"and {len(top) - n_show} more paths, {rest * 100:.1f}% together",
            fontsize=10.5, color="#666", va="center")
    fig.text(0.03, 0.015,
             "dark = full transformer pass, light = cached step; "
             "bar = share of the 4,896 generations (PartiPrompts, three seeds)",
             fontsize=10, color="#666")
    fig.savefig("slides/assets/census_paths.png", dpi=170, facecolor="white")
    print("census written,", len(top), "paths")


def coverage():
    d = json.load(open("resources/spx_coverage_deep/results.json"))
    parts = d["partitions"]
    label = {"flux": "FLUX", "qwen": "Qwen", "hunyuan_video": "HYV",
             "wan21": "Wan"}
    order = ["flux", "qwen", "hunyuan_video", "wan21"]
    parts = sorted(parts, key=lambda p: (order.index(p["model"]), p["k"]))

    fig, ax = plt.subplots(figsize=(12.6, 4.8))
    xs = range(12)
    def half(p):
        return next(m for m in p["old"]["margins"] if m["eps"] == 0.5)
    shares = [half(p)["share"] for p in parts]
    lbs = [p["verdict"]["0.5"]["old_lcb95"] for p in parts]
    ax.bar(xs, shares, width=0.62, color=BLUE, alpha=0.85,
           label="share of held-out prompts within 0.5 dB of the best evaluated schedule")
    ax.scatter(xs, lbs, marker="_", s=420, color=INK, lw=2.2,
               label="lower limit after sampling error: the true share is at least this (95% confidence)")
    ax.axhline(0.5, color="#444444", lw=1.5, ls="--",
               label="one half: the lower limit must stay above this line")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{label[p['model']]}\nK{p['k']}" for p in parts],
                       fontsize=10.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("share of prompts", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="lower left",
              bbox_to_anchor=(0.0, 1.01), borderaxespad=0)
    fig.tight_layout()
    fig.savefig("slides/assets/coverage_bars.png", dpi=170,
                facecolor="white")
    print("coverage written")


if __name__ == "__main__":
    census()
    coverage()
