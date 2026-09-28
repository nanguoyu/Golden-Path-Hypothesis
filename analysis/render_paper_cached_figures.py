#!/usr/bin/env python3
"""Render paper-sized cached-trajectory figures from the staged results."""

from __future__ import annotations

from pathlib import Path
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis._palette_check import check_cross, check_group
from analysis.palette import FAMILY, METHOD_NAMED, OLIVE_RAMP
from analysis.render_image_cached_trajectory_report import (
    Data,
    EARLY_N,
    KS,
    MODELS,
    PAYLOADS,
    SHORT,
    _row_quality,
    fmt,
    schedule_family,
)


OUT = ROOT / "paper" / "figs"

# Final width 5.5 in = the \linewidth the paper includes these figures at, so
# the font sizes below are the sizes the reader sees.  They match Figure 1:
# labels 7.6 pt, tick labels 7.2 pt.
FIG_W = 5.5
FS_LAB = 7.6
FS_TICK = 7.2

# Schedule families are not cache methods, so they stay clear of the four
# method hues of Figure 1.  Random schedules are grey, the paper's colour for a
# random control everywhere else.  Both registries live in
# `analysis/palette.py`, under the FAMILY and RANDOM_GREY roles.
#
# Three families are rare: nine geometry-derived rows, six from the second
# principal direction, and two second-most-frequent gate paths.  Seventeen
# points spread over six panels cannot support three separate readings, so
# they are drawn as one category with one hue.  What they share is what the
# reader needs: each is a fixed schedule that no cache method proposes and no
# random draw produced.  `check_palette` measures the separation, including
# the one that matters most here, the merged olive against the random grey
# under protanopia.
GROUP_OF_FAMILY = {"search": "search", "ladder": "ladder",
                   "geometry": "other", "rho2": "other",
                   "gate variant": "other", "random": "random"}
FAMILY_STYLE = (
    # group,   marker, colour,               label
    ("search",  "o", FAMILY["search"], "method-derived"),
    ("ladder",  "s", FAMILY["ladder"], "schedule perturbation"),
    ("other",   "D", FAMILY["other"],  "other fixed controls"),
    ("random",  "^", FAMILY["random"], "random"),
)
# 1.3 mm is 3.69 pt across, and matplotlib's scatter size is the square of the
# marker's width in points, so nothing here goes below 13.6.
MARKER_SIZE = {"D": 15.0}
MARKER_SIZE_DEFAULT = 17.0

# the three stacked pieces of the companion figure, dark to light, on the olive
# ramp the paper keeps for an ordered set of three
SHARE_LABELS = ("reference chord", "reference bend plane",
                "outside reference frame")


def check_palette() -> None:
    """Measure the separation the schedule-family palette is claimed to have."""
    named = {label: colour for _g, _m, colour, label in FAMILY_STYLE}
    check_group("early_drift_vs_quality", "schedule families", named,
                min_distance=25.0)
    check_cross("early_drift_vs_quality", "schedule families",
                {k: v for k, v in named.items() if k != "random"},
                "cache methods", METHOD_NAMED,
                min_distance=25.0, min_cvd_distance=10.0)
    # The stacked shares of the companion figure are one ordered decomposition,
    # so they wear one olive hue at three lightnesses and are read by lightness
    # rather than by hue.
    check_group("cache_offset_direction_shares", "energy shares",
                dict(zip(SHARE_LABELS, OLIVE_RAMP)),
                min_distance=25.0, min_lightness_gap=20.0)

# Deterministic PDF: no timestamp, no tool version string.
PDF_META = {"CreationDate": None, "Creator": "render_paper_cached_figures.py",
            "Producer": "matplotlib"}


class Ruler:
    """Measure rendered text width in inches, so nothing silently overflows."""

    def __init__(self, fig) -> None:
        fig.canvas.draw()
        self.fig, self.renderer = fig, fig.canvas.get_renderer()

    def width(self, text: str, size: float) -> float:
        item = self.fig.text(0, 0, text, fontsize=size)
        out = item.get_window_extent(self.renderer).width / self.fig.dpi
        item.remove()
        return out


def bend_rows(data: Data, *, held_out_only: bool = False
              ) -> dict[tuple[str, str, str, int], dict[str, float | int | None]]:
    """(model, schedule, payload, k) -> median early gap and first cached step.

    Read straight from the staged per-prompt table, so the figure never has to
    trust a stored summary.
    """
    groups: dict[tuple[str, str, str, int], list[dict]] = {}
    for row in data.rows:
        if held_out_only and row["role"] == "discovery":
            continue
        key = (row["model"], row["schedule"], row["payload"], int(row["k"]))
        groups.setdefault(key, []).append(row)
    out: dict[tuple[str, str, str, int], dict[str, float | int | None]] = {}
    for key, rows in groups.items():
        values = [float(r["D10_over_chord_ref"]) for r in rows
                  if r["D10_over_chord_ref"] not in ("", None)]
        starts = {int(r["k0"]) for r in rows}
        out[key] = {
            "D10": statistics.median(values) if values else None,
            "k0": starts.pop() if len(starts) == 1 else None,
            "n_prompts": len(values),
        }
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation of the values handed in, and nothing else."""
    if len(xs) < 4 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(stats.spearmanr(xs, ys)[0])


def zscore(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    spread = float(arr.std())
    return ((arr - arr.mean()) / spread if spread > 0 else arr * 0.0).tolist()


def r_squared(y: list[float], columns: list[list[float]]) -> float:
    """R^2 of an ordinary least-squares fit with an intercept."""
    target = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(target.size)]
                             + [np.asarray(c, dtype=np.float64) for c in columns])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    total = float(((target - target.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / total


def panel_points(data: Data, quality: dict, medians: dict, model: str, k: int
                 ) -> tuple[list[dict], list[tuple[str, str]]]:
    """The points one panel draws, plus every row it left out and why."""
    kept: list[dict] = []
    dropped: list[tuple[str, str]] = []
    for cell in sorted(data.cell_list(model, k=k), key=lambda c: c["cell_id"]):
        schedule, payload = cell["schedule"], cell["payload"]
        x = cell["median"]["D10_over_chord_ref"]
        y = quality.get((model, schedule, payload, k))
        if x is None or y is None:
            dropped.append((cell["cell_id"], "no quality reading for the row"))
            continue
        # A panel is one budget.  A gate row that caches a different number of
        # steps sits on a different point of the quality-budget curve, so its
        # vertical position is not comparable with the rest of the panel and
        # the row is left out; the drop list below names it and its count.
        if cell["k_realized"] != k:
            dropped.append((cell["cell_id"],
                            f"realized K {cell['k_realized']} is not the "
                            f"panel budget {k}"))
            continue
        staged = medians[(model, schedule, payload, k)]
        assert staged["D10"] is not None and abs(staged["D10"] - x) <= 1e-12, (
            f"{cell['cell_id']}: stored median {x} differs from the median "
            f"recomputed from the per-prompt table {staged['D10']}")
        structural = x == 0.0
        if structural:
            assert staged["k0"] is not None and staged["k0"] >= EARLY_N, (
                f"{cell['cell_id']}: early gap is 0 but the first cached step "
                f"is {staged['k0']}")
        family = schedule_family(schedule)
        if family == "gate variant":
            assert schedule.endswith("_r2"), (
                f"{cell['cell_id']}: the second-most-frequent-path family "
                f"holds a schedule that is not a rank-2 row")
        kept.append({"cell_id": cell["cell_id"], "x": x * 1e3, "y": y,
                     "family": family, "group": GROUP_OF_FAMILY[family],
                     "structural": structural, "k0": staged["k0"]})
    assert len(kept) + len(dropped) == len(data.cell_list(model, k=k))
    return kept, dropped


def early_drift(data: Data) -> None:
    """Median early gap against final quality, six panels, one family per mark."""
    quality = _row_quality(data)
    medians = bend_rows(data)
    style = {
        "font.family": "DejaVu Sans", "font.size": FS_LAB,
        "axes.titlesize": FS_LAB, "axes.labelsize": FS_LAB,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
    }
    restore = {key: plt.rcParams[key] for key in style}
    plt.rcParams.update(style)
    height = 3.35
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, height), sharex=True)
    # the legend is two rows tall now, so the strip above the panels is deeper
    # than it was and the legend sits inside the canvas rather than over its rim
    fig.subplots_adjust(left=0.095, right=0.972, bottom=0.135, top=0.785,
                        wspace=0.32, hspace=0.62)

    panels, drops = {}, []
    for model, _long in MODELS:
        for k in KS:
            kept, dropped = panel_points(data, quality, medians, model, k)
            panels[(model, k)] = kept
            drops += [(model, k, cell_id, why) for cell_id, why in dropped]
    present = {point["group"] for kept in panels.values() for point in kept}
    open_groups = {point["group"] for kept in panels.values() for point in kept
                   if point["structural"]}
    x_all = [point["x"] for kept in panels.values() for point in kept]
    x_lo, x_hi = min(x_all), max(x_all)
    pad = 0.04 * (x_hi - x_lo)

    title_lines, counts, filled_rho = [], [], []
    for i, (model, _long) in enumerate(MODELS):
        for j, k in enumerate(KS):
            ax = axes[i][j]
            kept = panels[(model, k)]
            counts.append(len(kept))
            for group, marker, colour, _label in FAMILY_STYLE:
                for structural in (False, True):
                    chosen = [p for p in kept if p["group"] == group
                              and p["structural"] is structural]
                    if not chosen:
                        continue
                    if structural:
                        # An open mark: the row's first cached step is at or
                        # after state 10, so its early gap is zero by
                        # construction rather than by measurement.  Every
                        # family can draw one, because every marker here has
                        # an interior.
                        extra = {"facecolors": "none", "edgecolors": colour,
                                 "linewidths": 0.7}
                    else:
                        extra = {"linewidths": 0.25, "edgecolors": "white",
                                 "color": colour}
                    ax.scatter(
                        [p["x"] for p in chosen], [p["y"] for p in chosen],
                        marker=marker,
                        s=MARKER_SIZE.get(marker, MARKER_SIZE_DEFAULT),
                        alpha=0.9,
                        zorder=4 if group == "search" else 5, **extra)
            rho = spearman([p["x"] for p in kept], [p["y"] for p in kept])
            solid = [p for p in kept if not p["structural"]]
            rho_solid = spearman([p["x"] for p in solid], [p["y"] for p in solid])
            filled_rho.append((model, k, len(kept), rho, len(solid), rho_solid))
            # The panel prints the correlation over the rows that carry a
            # measurement.  A row whose first cache action is at or after state
            # 10 has an early gap of exactly zero by construction, so it is
            # drawn as an open mark and left out of the number.
            lines = [f"{SHORT[model]}, ratio {k / 50:.2f}",
                     f"$\\rho_s={fmt(rho_solid, 2)}$, $n={len(solid)}$"]
            title_lines += lines
            ax.set_title("\n".join(lines), pad=3.0, linespacing=1.25)
            ax.set_xlim(x_lo - pad, x_hi + pad)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
            ax.tick_params(labelsize=FS_TICK, pad=1.6, labelbottom=True)
            ax.grid(axis="y", alpha=0.25, lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.supxlabel(r"median early gap $D_{10}$  ($\times 10^{-3}$)",
                  fontsize=FS_LAB, y=0.018)
    fig.supylabel("mean PSNR (dB)", fontsize=FS_LAB, x=0.010)

    handles = [
        matplotlib.lines.Line2D(
            [], [], linestyle="none", marker=marker, color=colour,
            markersize=MARKER_SIZE.get(marker, MARKER_SIZE_DEFAULT) ** 0.5,
            markeredgecolor="white", markeredgewidth=0.25, label=label)
        for group, marker, colour, label in FAMILY_STYLE if group in present
    ]
    # what an open mark means, said once, in the shape the panels use most
    handles.append(matplotlib.lines.Line2D(
        [], [], linestyle="none", marker="o", markerfacecolor="none",
        markeredgecolor="0.30", markeredgewidth=0.7,
        markersize=MARKER_SIZE_DEFAULT ** 0.5,
        label="first cache at or after state 10"))
    legend = fig.legend(handles=handles, labels=[h.get_label() for h in handles],
                        loc="upper center", bbox_to_anchor=(0.5, 0.998), ncol=3,
                        frameon=False, fontsize=FS_LAB - 0.4, handletextpad=0.3,
                        columnspacing=1.4, borderpad=0.0, handlelength=1.0,
                        labelspacing=0.35)

    fig.savefig(OUT / "early_drift_vs_quality.png", dpi=400)
    fig.savefig(OUT / "early_drift_vs_quality.pdf", metadata=PDF_META)

    # ------------------------------------------------------------ self-check
    rule = Ruler(fig)
    # the legend must sit inside the canvas: anchored above the rim, the tops
    # of its tallest letters are cut off by the figure's own bounding box
    box = legend.get_window_extent(fig.canvas.get_renderer())
    print(f"     legend box top {box.y1 / fig.dpi:.2f} in, figure "
          f"{height:.2f} in, panel titles start "
          f"{height * 0.785:.2f} in")
    assert box.y1 <= fig.bbox.y1, "the legend is drawn past the top of the figure"
    assert box.y0 / fig.dpi > height * 0.785 - 0.32, "the legend touches the titles"
    panel_w = FIG_W * (0.972 - 0.095) / (3 + 2 * 0.32)
    widest = max(title_lines, key=lambda t: rule.width(t, FS_LAB))
    print(f"  early_drift: {FIG_W} x {height} in, panel {panel_w:.2f} in wide, "
          f"{sum(counts)} points over 6 panels {counts}")
    print(f"     widest title line {rule.width(widest, FS_LAB):.2f} in "
          f"(panel {panel_w:.2f} in): {widest}")
    assert rule.width(widest, FS_LAB) < panel_w + 0.28, widest
    ncol = 3
    rows = [handles[n:n + ncol] for n in range(0, len(handles), ncol)]
    legend_w = max(sum(rule.width(h.get_label(), FS_LAB - 0.4) + 0.32
                       for h in row) for row in rows)
    print(f"     widest legend row {legend_w:.2f} in of {len(rows)} "
          f"(figure {FIG_W} in)")
    assert legend_w < FIG_W, "legend row is wider than the figure"
    print(f"     shared x axis {x_lo:.2f} to {x_hi:.2f} "
          f"(x 10^-3), families drawn: {sorted(present)}")
    counts_by_group = {group: sum(p["group"] == group for kept in panels.values()
                                  for p in kept)
                       for group, _m, _c, _l in FAMILY_STYLE}
    print(f"     points per family {counts_by_group}")
    # Every marker in the palette has an interior, so any family can draw an
    # open mark.  Two of them have no row whose first cache action lands at or
    # after state 10, which is a property of the schedules, not of the drawing.
    print(f"     families with an open mark: {sorted(open_groups)}; "
          f"without one: {sorted(set(counts_by_group) - open_groups)}")
    print(f"     dropped {len(drops)} rows:")
    for model, k, cell_id, why in drops:
        print(f"       {model} k{k}  {cell_id}  --  {why}")
    print("     panel rank correlations, printed subset then all points:")
    link = data.link["slices"]["all"]["q2_row_level"]["partitions"]
    for model, k, n_all, rho, n_solid, rho_solid in filled_rho:
        staged = link[f"{model}_k{k}"]["k0_lt_early_subset"]["D10_over_chord_ref"]
        print(f"       {SHORT[model]} k{k}: printed n={n_solid} "
              f"rho={fmt(rho_solid, 4)}  |  with the open marks n={n_all} "
              f"rho={fmt(rho, 4)}  |  staged n={staged['n']} "
              f"rho={staged['rho']:.4f}")
        assert n_solid == staged["n"], (model, k, n_solid, staged["n"])
        assert abs(rho_solid - staged["rho"]) < 1e-9, (model, k, rho_solid)
    plt.close(fig)
    plt.rcParams.update(restore)     # leave the other figures untouched


def section_numbers(data: Data) -> None:
    """Print the configuration-level readings section 4.3 of the paper quotes.

    The population is the 256 rows of the staged link file, every row with a
    quality reading and a first cached step, the off-budget row included.  The
    figure leaves that row out because a panel holds one budget, but a pooled
    statistic does not, and the paper should quote one number with one
    provenance.  The assertions below tie every printed value to
    `resources/image_trajectory/early_quality_link.json`.

    Same definitions as the staged analysis: the pooled correlation is taken
    over quality and early-gap values standardised inside each model-and-ratio
    group, and the early-cache subset reuses those values rather than
    standardising itself.  The explained fractions come from that subset
    standardised on its own, because a row whose first cached step is at or
    after state 10 has an early gap of zero by construction and would otherwise
    stand in for the timestep variable.
    """
    quality = _row_quality(data)
    for slice_name, key in (("all 49 prompts", "all"),
                            ("32 held-out prompts", "held_out")):
        staged = data.link["slices"][key]
        medians = bend_rows(data, held_out_only=key == "held_out")
        pooled_q, pooled_e, early_mask = [], [], []
        sub_q, sub_k0, sub_e = [], [], []
        for model, _long in MODELS:
            for k in KS:
                qs, es, mask = [], [], []
                fit_q, fit_k0, fit_e = [], [], []
                for point in sorted(data.cell_list(model, k=k),
                                    key=lambda c: c["cell_id"]):
                    schedule, payload = point["schedule"], point["payload"]
                    row = medians[(model, schedule, payload, k)]
                    y = quality.get((model, schedule, payload, k))
                    if y is None or row["D10"] is None or row["k0"] is None:
                        continue
                    qs.append(y)
                    es.append(row["D10"])
                    mask.append(row["k0"] < EARLY_N)
                    if row["k0"] < EARLY_N:
                        fit_q.append(y)
                        fit_k0.append(float(row["k0"]))
                        fit_e.append(row["D10"])
                zq, ze = zscore(qs), zscore(es)
                pooled_q += zq
                pooled_e += ze
                early_mask += mask
                sub_q += zscore(fit_q)
                sub_k0 += zscore(fit_k0)
                sub_e += zscore(fit_e)
        rho_all = spearman(pooled_e, pooled_q)
        keep = [n for n, flag in enumerate(early_mask) if flag]
        rho_early = spearman([pooled_e[n] for n in keep],
                             [pooled_q[n] for n in keep])
        r2_k0 = r_squared(sub_q, [sub_k0])
        r2_early = r_squared(sub_q, [sub_e])
        r2_both = r_squared(sub_q, [sub_k0, sub_e])
        pooled = staged["q2_row_level"]["pooled"]["D10_over_chord_ref"]
        subset = staged["q2_row_level"]["pooled_k0_lt_early"]["D10_over_chord_ref"]
        fits = staged["q2_mediation_D10_k0_lt_early"]
        print(f"  section 4.3 ({slice_name}):")
        print(f"     all configurations           n={len(pooled_q)} "
              f"pooled rho={fmt(rho_all, 4)}  (staged n={pooled['n']} "
              f"rho={pooled['rho']:.4f})")
        print(f"     first cache before state 10  n={len(keep)} "
              f"pooled rho={fmt(rho_early, 4)}  (staged n={subset['n']} "
              f"rho={subset['rho']:.4f})")
        print(f"     R^2 first cached timestep only {100 * r2_k0:.1f}%  "
              f"(staged {100 * fits['quality_on_k0']['r2']:.1f}%)")
        print(f"     R^2 early gap only             {100 * r2_early:.1f}%  "
              f"(staged {100 * fits['quality_on_early']['r2']:.1f}%)")
        print(f"     R^2 both                       {100 * r2_both:.1f}%  "
              f"(staged {100 * fits['quality_on_both']['r2']:.1f}%)")
        print(f"     early gap added after the timestep "
              f"{100 * (r2_both - r2_k0):.1f} points")
        print(f"     timestep added after the early gap "
              f"{100 * (r2_both - r2_early):.1f} points")
        assert len(pooled_q) == pooled["n"] == staged["n_rows"], len(pooled_q)
        assert abs(rho_all - pooled["rho"]) < 1e-9, (rho_all, pooled["rho"])
        assert len(keep) == subset["n"], (len(keep), subset["n"])
        assert abs(rho_early - subset["rho"]) < 1e-9, (rho_early, subset["rho"])
        assert abs(r2_k0 - fits["quality_on_k0"]["r2"]) < 1e-9, r2_k0
        assert abs(r2_early - fits["quality_on_early"]["r2"]) < 1e-9, r2_early
        assert abs(r2_both - fits["quality_on_both"]["r2"]) < 1e-9, r2_both


def direction_shares(data: Data) -> None:
    """Stacked energy shares of the final cached-trajectory gap, two models.

    The three stacked pieces are one ordered decomposition, so they wear one
    hue at three lightnesses rather than three separate colours, and the reader
    tells them apart by lightness alone.  `check_palette` measures the three
    against each other, including the greyscale gap a print keeps.
    """
    style = {
        "font.family": "DejaVu Sans", "font.size": FS_LAB,
        "axes.titlesize": FS_LAB, "axes.labelsize": FS_LAB,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
    }
    restore = {key: plt.rcParams[key] for key in style}
    plt.rcParams.update(style)

    ramp = list(zip(SHARE_LABELS, OLIVE_RAMP))
    height = 2.35
    left, right, bottom, top = 0.088, 0.986, 0.155, 0.845
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, height), sharey=True)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=0.10)

    values = data.link["slices"]["all"]["q3_direction"]["by_payload"]
    # two short lines, so the labels stay upright at 7.2 pt in a 0.47 in slot
    short_labels = {
        "reuse": "reuse",
        "taylor_o1": "1st\norder",
        "hermite_o2": "2nd\norder",
        "di_two_anchor": "two\nanchor",
        "mean_avg_vel": "interval\naverage",
    }
    drawn: list[tuple[str, str, float, float, float]] = []
    for ax, (model, long_name) in zip(axes, MODELS):
        labels, chord, plane, outside = [], [], [], []
        for payload, _zh, _label in PAYLOADS:
            entry = values.get(model, {}).get(payload)
            if not entry:
                continue
            labels.append(short_labels[payload])
            chord.append(entry["share_chord_50"]["median"] or 0.0)
            plane.append(entry["share_in_plane_50"]["median"] or 0.0)
            outside.append(entry["share_off_plane_50"]["median"] or 0.0)
            drawn.append((model, labels[-1], chord[-1], plane[-1], outside[-1]))
        x = np.arange(len(labels))
        base = np.asarray(chord)
        ax.bar(x, chord, width=0.68, color=ramp[0][1], label=ramp[0][0],
               linewidth=0.0)
        ax.bar(x, plane, width=0.68, bottom=base, color=ramp[1][1],
               label=ramp[1][0], linewidth=0.0)
        ax.bar(x, outside, width=0.68, bottom=base + np.asarray(plane),
               color=ramp[2][1], label=ramp[2][0], linewidth=0.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, linespacing=1.05)
        ax.set_xlim(-0.62, len(labels) - 0.38)
        ax.set_ylim(0.0, 1.06)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(long_name, pad=3.0)
        ax.tick_params(labelsize=FS_TICK, pad=1.6)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("energy share of the final gap", labelpad=2.0)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=FS_LAB - 0.4, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.998), ncol=3,
               handletextpad=0.4, columnspacing=1.6, borderpad=0.0,
               handlelength=1.0)

    fig.savefig(OUT / "cache_offset_direction_shares.png", dpi=400)
    fig.savefig(OUT / "cache_offset_direction_shares.pdf", metadata=PDF_META)

    # ------------------------------------------------------------ self-check
    rule = Ruler(fig)
    panel_w = FIG_W * (right - left) / (2 + 0.10)
    slot = panel_w / 5
    lines = [line for _m, label, *_ in drawn for line in label.split("\n")]
    widest = max(set(lines), key=lambda t: rule.width(t, FS_TICK))
    ylab = rule.width("energy share of the final gap", FS_LAB)
    legend_w = sum(rule.width(text, FS_LAB - 0.4) + 0.34 for text, _c in ramp)
    print(f"  direction_shares: {FIG_W} x {height} in, panel {panel_w:.2f} in, "
          f"{len(drawn)} bars over 2 panels")
    print(f"     widest x tick line {rule.width(widest, FS_TICK):.2f} in "
          f"(slot {slot:.2f} in): {widest}")
    assert rule.width(widest, FS_TICK) < slot, widest
    print(f"     y label {ylab:.2f} in (axis {height * (top - bottom):.2f} in)")
    assert ylab < height * (top - bottom), "y label is taller than the axes"
    print(f"     legend row {legend_w:.2f} in (figure {FIG_W} in)")
    assert legend_w < FIG_W, "legend row is wider than the figure"
    for model, label, chord, plane, outside in drawn:
        total = chord + plane + outside
        label = label.replace("\n", " ")
        print(f"     {model:5s} {label:16s} chord {chord:.3f}  plane {plane:.3f}"
              f"  outside {outside:.3f}  sum {total:.3f}")
        assert abs(total - 1.0) < 0.06, (model, label, total)
    plt.close(fig)
    plt.rcParams.update(restore)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = Data()
    check_palette()
    early_drift(data)
    section_numbers(data)
    direction_shares(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
