#!/usr/bin/env python3
"""Figures for the video schedule x payload cross, from the analysis JSON.

Everything is read from `resources/video_spx/video_spx_results.json`
(`analysis/video_spx.py`), so a figure can never disagree with the numbers in
the results document. One backbone per figure row; a backbone whose cells have
not landed yet is simply absent.

Labels are Chinese when a CJK-capable font is installed, English otherwise; the
results document carries the concordance either way, so both renderings read
against the same prose.

    python analysis/plot_video_spx.py

Writes into `docs/figures/video_spx/`, numbered in the reading order of the
results document:

    fig1_cross_psnr        cross heatmap, every schedule x payload cell
    fig2_gamma_psnr        interaction residuals on the 7-row balanced subgrid
    fig3_p2_band           rows against the MeanCache row, +- 2 paired SE
    fig4_p3_gate           fixed modal path minus native gate, with the seed band
    fig5_geometry          row quality against two geometry predictors
    fig6_dose              quality against transposition distance from MeanCache
    fig7_payload_vs_k      payload marginal mean against the budget
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_JSON = REPO / "resources/video_spx/video_spx_results.json"
DEFAULT_OUT = REPO / "docs/figures/video_spx"
METRIC = "psnr"

#: Tried in order; the first one actually installed carries the Chinese labels.
CJK_FONTS = ("Heiti TC", "Hiragino Sans GB", "Songti SC", "Arial Unicode MS",
             "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
             "Microsoft YaHei", "SimHei")


def pick_cjk_font() -> str | None:
    installed = {font.name for font in fm.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in installed:
            return name
    return None


CJK = pick_cjk_font()
if CJK is not None:
    plt.rcParams["font.sans-serif"] = [CJK] + list(plt.rcParams["font.sans-serif"])
    plt.rcParams["axes.unicode_minus"] = False


def L(zh: str, en: str) -> str:
    """The Chinese label when the figure can draw it, the English one otherwise."""

    return zh if CJK is not None else en


BACKBONE_LABEL = {"hunyuan_video": "HunyuanVideo", "wan21": "Wan2.1"}
PAYLOAD_LABEL = {
    "reuse": L("零阶沿用", "reuse (0th)"),
    "taylor_o1": L("一阶外推", "Taylor 1st"),
    "hermite_o2": L("二阶外推", "Hermite 2nd"),
    "mean_vel": L("区间均速", "mean velocity"),
    "di_two_anchor": L("两锚外推", "2-anchor"),
}
ROW_LABEL = {
    "shared": L("共享表", "shared table"),
    "budcache": L("BudCache 表", "BudCache"),
    "meancache": L("MeanCache 表", "MeanCache"),
    "sea_top1": L("SeaCache 模态路径", "SeaCache top-1"),
    "tea_top1": L("TeaCache 模态路径", "TeaCache top-1"),
    "sen_top1": L("SenCache 模态路径", "SenCache top-1"),
    "di_top1": L("DiCache 模态路径", "DiCache top-1"),
    "sea_top1_off": L("SeaCache 模态路径（越预算）", "SeaCache top-1 (off-budget)"),
    "tea_top1_off": L("TeaCache 模态路径（越预算）", "TeaCache top-1 (off-budget)"),
    "sen_top1_off": L("SenCache 模态路径（越预算）", "SenCache top-1 (off-budget)"),
    "uniform": L("均匀表", "uniform"),
    "dp_rho2": L("弯折最优表", "DP on rho2"),
    "rand_1": L("随机表一", "random 1"),
    "rand_2": L("随机表二", "random 2"),
    "ham2": L("换位 2 位", "transpose 2"),
    "ham4": L("换位 4 位", "transpose 4"),
    "ham8": L("换位 8 位", "transpose 8"),
    "ham2f": L("保首跳换位 2 位", "transpose 2 (first kept)"),
    "ham4f": L("保首跳换位 4 位", "transpose 4 (first kept)"),
    "ham8f": L("保首跳换位 8 位", "transpose 8 (first kept)"),
}
CONTROL_ROWS = ("uniform", "rand_1", "rand_2")
GEOMETRY_PANELS = (
    ("rho2_cost_cv", L("缺口代价的变异系数（二阶弯折量）",
                       "gap-cost coefficient of variation (rho2)")),
    ("first_cache_step", L("首个跳步的位置（第几步）", "first cached step")),
)
LADDER_LABEL = {
    "free": L("自由换位（首个跳步同时提前）", "free swaps (first cached step also moves)"),
    "first_preserving": L("保住首个跳步的换位", "first cached step held fixed"),
}


def savefig(fig, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "pdf"):
        path = out / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", dpi=170 if extension == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def budgets_of(block: Mapping[str, Any]) -> list[str]:
    order = {"K29": 0, "K37": 1, "K41": 2}
    return sorted(block["partitions"], key=lambda name: order.get(name, 99))


def budget_title(backbone: str, budget: str) -> str:
    return L(f"{BACKBONE_LABEL.get(backbone, backbone)}，{budget[1:]} 步跳过",
             f"{BACKBONE_LABEL.get(backbone, backbone)}, {budget[1:]} cached steps")


def draw_heatmap(ax, grid, row_keys, col_keys, *, norm, cmap, fmt="{:.1f}"):
    masked = np.ma.masked_invalid(grid)
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad("#e8e8e8")
    image = ax.imshow(masked, cmap=palette, norm=norm, aspect="auto")
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = grid[r, c]
            if np.isnan(value):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="///",
                                       edgecolor="#b8b8b8", linewidth=0))
                continue
            rgba = palette(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(c, r, fmt.format(value), ha="center", va="center", fontsize=6.6,
                    color="white" if luminance < 0.5 else "black")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([PAYLOAD_LABEL.get(key, key) for key in col_keys], rotation=35, ha="right",
                       fontsize=7.5)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([ROW_LABEL.get(key, key) for key in row_keys], fontsize=7.5)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def fig1_cross(report, out: Path) -> None:
    backbones = sorted(report["backbones"])
    payloads = report["payloads"]
    fig, axes = plt.subplots(len(backbones), 3, figsize=(12.6, 5.0 + 4.4 * (len(backbones) - 1)),
                             constrained_layout=True, squeeze=False)
    for r, backbone in enumerate(backbones):
        block = report["backbones"][backbone]
        budgets = budgets_of(block)
        rows = [row for row in report["row_order"]
                if any(row in block["partitions"][k]["cross"][METRIC] for k in budgets)]
        grids = {}
        for budget in budgets:
            table = block["partitions"][budget]["cross"][METRIC]
            grid = np.full((len(rows), len(payloads)), np.nan)
            for i, row in enumerate(rows):
                for j, payload in enumerate(payloads):
                    cell = table.get(row, {}).get(payload)
                    if cell is not None:
                        grid[i, j] = cell["mean"]
            grids[budget] = grid
        norm = mcolors.Normalize(vmin=float(np.nanmin(list(grids.values()))),
                                 vmax=float(np.nanmax(list(grids.values()))))
        image = None
        for c, budget in enumerate(budgets):
            ax = axes[r, c]
            image = draw_heatmap(ax, grids[budget], rows, payloads, norm=norm, cmap="viridis")
            ax.set_title(budget_title(backbone, budget), fontsize=9)
            if c == 0:
                ax.set_ylabel(L("调度", "schedule"))
            ax.set_xlabel(L("载荷", "payload"))
        bar = fig.colorbar(image, ax=axes[r, :].tolist(), shrink=0.9, pad=0.01)
        bar.set_label(L("相对全算参照的 PSNR（dB）",
                        "PSNR vs the full-step reference (dB)"), fontsize=8)
    fig.suptitle(L("调度 × 载荷交叉：每格的 PSNR（每格 300 条视频；斜纹格未运行）",
                   "Schedule x payload cross: PSNR of every cell "
                   "(300 videos per cell; hatched = not run)"), fontsize=10.5)
    savefig(fig, out, "fig1_cross_psnr")


def fig2_gamma(report, out: Path) -> None:
    backbones = sorted(report["backbones"])
    payloads = report["payloads"]
    fig, axes = plt.subplots(len(backbones), 3, figsize=(12.0, 3.6 + 3.2 * (len(backbones) - 1)),
                             constrained_layout=True, squeeze=False)
    for r, backbone in enumerate(backbones):
        block = report["backbones"][backbone]
        budgets = budgets_of(block)
        grids, rows_by_budget = {}, {}
        for budget in budgets:
            fit = block["partitions"][budget]["P1"]["grids"]["w1"]["metrics"].get(METRIC, {})
            rows = fit.get("schedules", [])
            grid = np.full((len(rows), len(payloads)), np.nan)
            for i, row in enumerate(rows):
                for j, payload in enumerate(payloads):
                    value = fit.get("gamma", {}).get(f"{row}x{payload}")
                    if value is not None:
                        grid[i, j] = value
            grids[budget], rows_by_budget[budget] = grid, rows
        limit = float(np.nanmax([np.nanmax(np.abs(grid)) for grid in grids.values()]))
        norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = None
        for c, budget in enumerate(budgets):
            image = draw_heatmap(axes[r, c], grids[budget], rows_by_budget[budget], payloads,
                                 norm=norm, cmap="RdBu_r", fmt="{:+.1f}")
            axes[r, c].set_title(budget_title(backbone, budget), fontsize=9)
        bar = fig.colorbar(image, ax=axes[r, :].tolist(), shrink=0.9, pad=0.01)
        bar.set_label(L("交互残差（dB）", "interaction residual (dB)"), fontsize=8)
    fig.suptitle(L("七行满格子网格上的交互残差：两个主效应解释不了的部分",
                   "Interaction residuals on the balanced 7-row subgrid: what the two "
                   "main effects do not explain"), fontsize=10.5)
    savefig(fig, out, "fig2_gamma_psnr")


def fig3_p2_band(report, out: Path) -> None:
    backbones = sorted(report["backbones"])
    fig, axes = plt.subplots(len(backbones), 3, figsize=(12.4, 4.4 + 4.0 * (len(backbones) - 1)),
                             constrained_layout=True, squeeze=False)
    for r, backbone in enumerate(backbones):
        block = report["backbones"][backbone]
        for c, budget in enumerate(budgets_of(block)):
            ax = axes[r, c]
            band = block["partitions"][budget]["P2"]["pooled"][METRIC]
            if band is None:
                ax.axis("off")
                continue
            entries = sorted(band["rows"].items(), key=lambda item: item[1]["mean"])
            names = [name for name, _ in entries] + [band["anchor"]]
            means = [entry["mean"] for _, entry in entries] + [0.0]
            errors = [2.0 * entry["se"] for _, entry in entries] + [0.0]
            colors = ["#c0392b" if name in CONTROL_ROWS else
                      ("#2c7fb8" if name != band["anchor"] else "#1a1a1a") for name in names]
            ax.errorbar(means, range(len(names)), xerr=errors, fmt="o", markersize=4,
                        ecolor="#888888", elinewidth=1.1, capsize=2.4, linestyle="none",
                        color="none", zorder=2)
            ax.scatter(means, range(len(names)), c=colors, s=26, zorder=3)
            stream = band.get("stream_band")
            if stream is not None:
                ax.axvspan(-stream, stream, color="#f0c419", alpha=0.28, zorder=0,
                           label=L("矩阵种子带", "matrix seed band"))
            ax.axvline(0.0, color="#1a1a1a", linewidth=0.9, zorder=1)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([ROW_LABEL.get(name, name) for name in names], fontsize=7.5)
            ax.set_xlabel(L("相对 MeanCache 表的配对差（dB）",
                            "paired difference vs the MeanCache row (dB)"))
            ax.set_title(budget_title(backbone, budget), fontsize=9)
            ax.grid(axis="x", alpha=0.3)
            if c == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(L("零阶沿用列内每一行相对 MeanCache 表的配对差"
                   "（300 对 prompt，横条 = ± 2 倍配对标准误；红点 = 设计对照）",
                   "Every row against the MeanCache row inside the reuse column "
                   "(300 paired prompts, bars = +- 2 paired standard errors; "
                   "red = design control)"), fontsize=10.5)
    savefig(fig, out, "fig3_p2_band")


def fig4_p3(report, out: Path) -> None:
    """Two panels per backbone: all prompts, and the off-modal prompts only."""

    backbones = sorted(report["backbones"])
    fig, axes = plt.subplots(len(backbones), 2,
                             figsize=(12.0, 4.0 * len(backbones)),
                             constrained_layout=True, squeeze=False)
    panels = (("native_payload_cell", L("全部 prompt（含逐位相同的那些）",
                                        "all prompts (including the identical runs)")),
              ("off_modal_cell", L("门没有走模态路径的 prompt",
                                   "prompts where the gate took another path")))
    for r, backbone in enumerate(backbones):
        block = report["backbones"][backbone]
        budgets = budgets_of(block)
        rows: list[str] = []
        for budget in budgets:
            for row in block["partitions"][budget]["P3"]["pooled"][METRIC]["rows"]:
                if row not in rows:
                    rows.append(row)
        rows.sort(key=lambda name: report["row_order"].index(name))
        for c, (key, title) in enumerate(panels):
            ax = axes[r, c]
            width = 0.8 / max(len(budgets), 1)
            for b, budget in enumerate(budgets):
                panel = block["partitions"][budget]["P3"]["pooled"][METRIC]
                values, errors, positions = [], [], []
                for i, row in enumerate(rows):
                    cell = (panel["rows"].get(row) or {}).get(key)
                    if cell is None:
                        continue
                    positions.append(i + (b - (len(budgets) - 1) / 2) * width)
                    values.append(cell["mean"])
                    errors.append(2.0 * cell["se"])
                bars = ax.bar(positions, values, width=width * 0.92, yerr=errors, capsize=2.2,
                              label=L(f"{budget[1:]} 步跳过", f"{budget[1:]} cached steps"))
                band = panel.get("band_2sd")
                if band is not None:
                    color = bars[0].get_facecolor() if len(bars) else "#888888"
                    for sign in (-1.0, 1.0):
                        ax.axhline(sign * band, color=color, linestyle="--", linewidth=1.0,
                                   alpha=0.8, zorder=0)
            ax.axhline(0.0, color="#1a1a1a", linewidth=0.9)
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels([ROW_LABEL.get(row, row) for row in rows], rotation=20,
                               ha="right", fontsize=8)
            ax.set_ylabel(L(f"{BACKBONE_LABEL.get(backbone, backbone)}\n"
                            "模态路径固定执行 − 原生门控（dB）",
                            f"{BACKBONE_LABEL.get(backbone, backbone)}\n"
                            "fixed modal path minus native gate (dB)"))
            ax.set_title(title, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(L("门最常走的那条路固定执行，减同一条 prompt 上门自己的运行"
                   "（虚线 = 该预算档的矩阵种子带）",
                   "Each gate's most frequent path, run as a fixed schedule, against the "
                   "gate's own per-prompt run (dashed = the matrix seed band)"), fontsize=10.5)
    savefig(fig, out, "fig4_p3_gate")


def fig5_geometry(report, out: Path) -> None:
    backbones = sorted(report["backbones"])
    fig, axes = plt.subplots(len(backbones), len(GEOMETRY_PANELS),
                             figsize=(5.6 * len(GEOMETRY_PANELS), 4.0 * len(backbones)),
                             constrained_layout=True, squeeze=False)
    markers = {"K29": "o", "K37": "s", "K41": "^"}
    for r, backbone in enumerate(backbones):
        block = report["backbones"][backbone]
        for c, (predictor, label) in enumerate(GEOMETRY_PANELS):
            ax = axes[r, c]
            for budget in budgets_of(block):
                panel = block["partitions"][budget]
                quality = panel["P4"]["quality"][METRIC]["reuse_column_mean"]
                predictors = panel["P4"]["predictors"]
                xs, ys = [], []
                for row, value in quality.items():
                    x = predictors.get(row, {}).get(predictor)
                    if x is None or row in panel["rows_off_budget"]:
                        continue
                    xs.append(x)
                    ys.append(value)
                entry = (panel["P4"]["spearman"][METRIC]["reuse_column_mean"]["spearman"]
                         .get(predictor))
                tag = "" if entry is None else L(f"，秩相关 {entry['rho']:+.2f}",
                                                 f", rank corr. {entry['rho']:+.2f}")
                ax.scatter(xs, ys, s=30, marker=markers.get(budget, "o"),
                           label=L(f"{budget[1:]} 步跳过{tag}",
                                   f"{budget[1:]} steps{tag}"), alpha=0.85)
            ax.set_xlabel(label)
            if c == 0:
                ax.set_ylabel(L(f"{BACKBONE_LABEL.get(backbone, backbone)}\n"
                                "零阶沿用列的行质量（PSNR，dB）",
                                f"{BACKBONE_LABEL.get(backbone, backbone)}\n"
                                "row PSNR in the reuse column (dB)"))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7.5)
    fig.suptitle(L("行质量对调度的两个几何读数", "Row quality against two geometry readings "
                                                 "of the schedule"), fontsize=10.5)
    savefig(fig, out, "fig5_geometry")


def fig6_dose(report, out: Path) -> None:
    backbones = sorted(report["backbones"])
    ladders = [name for name in ("free", "first_preserving")
               if any(name in block["partitions"][budget]["P4"].get("dose_ladders", {})
                      for block in report["backbones"].values()
                      for budget in budgets_of(block))]
    fig, axes = plt.subplots(len(backbones), len(ladders),
                             figsize=(5.8 * len(ladders), 4.0 * len(backbones)),
                             constrained_layout=True, squeeze=False)
    for r, backbone in enumerate(backbones):
        block = report["backbones"][backbone]
        for c, ladder in enumerate(ladders):
            ax = axes[r, c]
            drawn = False
            for budget in budgets_of(block):
                entry = block["partitions"][budget]["P4"].get("dose_ladders", {}).get(ladder)
                if not entry:
                    continue
                points = entry["metrics"][METRIC]
                xs = [point["hamming_actual"] // 2 for point in points]
                ys = [point["paired_diff"]["mean"] if point["paired_diff"] else 0.0
                      for point in points]
                errors = [2.0 * point["paired_diff"]["se"] if point["paired_diff"] else 0.0
                          for point in points]
                ax.errorbar(xs, ys, yerr=errors, marker="o", capsize=2.6,
                            label=L(f"{budget[1:]} 步跳过", f"{budget[1:]} cached steps"))
                drawn = True
            if not drawn:
                ax.axis("off")
                continue
            ax.axhline(0.0, color="#1a1a1a", linewidth=0.9)
            ax.set_xlabel(L("离 MeanCache 表的换位数",
                            "transpositions away from the MeanCache table"))
            if c == 0:
                ax.set_ylabel(L(f"{BACKBONE_LABEL.get(backbone, backbone)}\n"
                                "配对 PSNR 差（dB）",
                                f"{BACKBONE_LABEL.get(backbone, backbone)}\n"
                                "paired PSNR difference (dB)"))
            ax.set_title(LADDER_LABEL[ladder], fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(L("剂量曲线：把跳步与真算步互换着离开最优表"
                   "（零阶沿用列，300 对 prompt）",
                   "Dose curve: swapping cached and full steps away from the best table "
                   "(reuse column, 300 paired prompts)"), fontsize=10.5)
    savefig(fig, out, "fig6_dose")


def fig7_payload_vs_k(report, out: Path) -> None:
    backbones = sorted(report["backbones"])
    payloads = report["payloads"]
    fig, axes = plt.subplots(1, len(backbones), figsize=(5.6 * len(backbones), 3.9),
                             constrained_layout=True, squeeze=False)
    for c, backbone in enumerate(backbones):
        ax = axes[0, c]
        block = report["backbones"][backbone]
        budgets = budgets_of(block)
        xs = [int(budget[1:]) for budget in budgets]
        for payload in payloads:
            ys = []
            for budget in budgets:
                fit = block["partitions"][budget]["P1"]["grids"]["w1"]["metrics"].get(METRIC, {})
                matrix = fit.get("matrix", {})
                values = [value for key, value in matrix.items() if key.endswith(f"x{payload}")]
                ys.append(float(np.mean(values)) if values else np.nan)
            ax.plot(xs, ys, marker="o", label=PAYLOAD_LABEL.get(payload, payload))
        ax.set_title(BACKBONE_LABEL.get(backbone, backbone), fontsize=10)
        ax.set_xlabel(L("50 步里跳过的步数", "cached steps out of 50"))
        ax.set_xticks(xs)
        if c == 0:
            ax.set_ylabel(L("七行均值 PSNR（dB）",
                            "PSNR, mean over the 7 balanced rows (dB)"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.suptitle(L("载荷的边际均值随预算档的变化（七行满格子网格）",
                   "Payload marginal against the budget (balanced subgrid)"), fontsize=10.5)
    savefig(fig, out, "fig7_payload_vs_k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.results.read_text(encoding="utf-8"))
    if not report.get("backbones"):
        raise SystemExit(f"{args.results}: no backbone has cells yet")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[video-spx-plot] label language: {'zh (' + CJK + ')' if CJK else 'en (no CJK font)'}")
    for builder in (fig1_cross, fig2_gamma, fig3_p2_band, fig4_p3,
                    fig5_geometry, fig6_dose, fig7_payload_vs_k):
        builder(report, args.out)
        print(f"[video-spx-plot] {builder.__name__}")
    # Figures were renumbered into reading order; drop the old stems so the
    # directory never carries two numbers for the same picture.
    for stale in ("fig3_payload_vs_k", "fig4_p2_band", "fig5_dose",
                  "fig6_geometry", "fig7_p3_gate"):
        for extension in ("png", "pdf"):
            path = args.out / f"{stale}.{extension}"
            if path.exists():
                path.unlink()
    print(f"[video-spx-plot] wrote figures into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
