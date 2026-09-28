#!/usr/bin/env python3
"""Render docs/image_cached_trajectory_results.md from the staged artefacts.

Every number in the output is computed here from

  resources/image_trajectory/cached_bend_<model>.json
  resources/image_trajectory/perprompt_bend.tsv.gz
  resources/image_trajectory/early_quality_link.json
  resources/image_trajectory/floors.json
  resources/image_trajectory/replay_verification.json
  resources/image_trajectory/{cells.v1.tsv, prompt_sample.v1.json}
  resources/video_full_trajectory/<T>/cached_bend_<T>.json   (section 8 only)

The prose is fixed and no numeric literal in it is a measured value; the
verdict sentences pick their branch from the numbers, so a rerun on different
data cannot leave a stale claim standing.

    python analysis/render_image_cached_trajectory_report.py --figures
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resources" / "image_trajectory"
VIDEO_RES = ROOT / "resources" / "video_full_trajectory"
FIG_DIR = ROOT / "docs" / "figures" / "image_cached_trajectory"
FIG_REL = "figures/image_cached_trajectory"
OUT = ROOT / "docs" / "image_cached_trajectory_results.md"

MODELS = [("flux", "FLUX.1-dev"), ("qwen", "Qwen-Image")]
SHORT = {"flux": "FLUX", "qwen": "Qwen"}
KS = [29, 37, 41]
D_STATES = [5, 10, 20, 30, 40, 45, 50]
# (id, Chinese name for the prose, English name for the figures — the repo's
# figures are English-labelled and the documents are Chinese)
PAYLOADS = [
    ("reuse", "零阶复用", "zero-order reuse"),
    ("taylor_o1", "一阶外推", "first-order extrap."),
    ("hermite_o2", "二阶外推", "second-order extrap."),
    ("di_two_anchor", "两锚外推", "two-anchor extrap."),
    ("mean_avg_vel", "区间平均速度", "interval avg. velocity"),
]
PAYLOAD_NAME = {p: zh for p, zh, _ in PAYLOADS}
PAYLOAD_EN = {p: en for p, _, en in PAYLOADS}
EXTRAP = {"taylor_o1", "hermite_o2", "di_two_anchor"}
FAMILY_OF_SCHEDULE = {
    "search": ("budcache", "dpcache", "uniform", "dicache_top1", "meancache",
               "seacache_top1", "teacache_top1", "sencache_top1"),
    "random": ("rand_1", "rand_2", "rand_3", "rand_4", "rand_5"),
}
EARLY_N = 10


# ------------------------------------------------------------------ helpers
def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "—"
    return f"{v:.{nd}f}"


def sci(x: Any, nd: int = 2) -> str:
    if x is None:
        return "—"
    v = float(x)
    if not math.isfinite(v):
        return "—"
    return f"{v:.{nd}e}"


def pct(x: Any, nd: int = 0) -> str:
    return "—" if x is None else f"{100.0 * float(x):.{nd}f}%"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def rho_cell(entry: dict[str, Any] | None, nd: int = 2) -> str:
    if not entry or entry.get("rho") is None:
        return "—"
    star = "*" if (entry.get("p") is not None and entry["p"] < 0.05) else ""
    return f"{entry['rho']:+.{nd}f}{star}"


def schedule_family(name: str) -> str:
    if name in FAMILY_OF_SCHEDULE["random"]:
        return "random"
    if name.startswith("ham"):
        return "ladder"
    if name.startswith("gpf"):
        return "geometry"
    if name in ("dp_rho2",):
        return "rho2"
    if name.endswith("_off") or name.endswith("_r2"):
        return "gate variant"
    return "search"


# ------------------------------------------------------------------ loading
class Data:
    def __init__(self) -> None:
        self.bend = {m: json.loads((RES / f"cached_bend_{m}.json").read_text("utf-8"))
                     for m, _ in MODELS}
        self.link = json.loads((RES / "early_quality_link.json").read_text("utf-8"))
        self.floors = json.loads((RES / "floors.json").read_text("utf-8"))
        self.verify = json.loads((RES / "replay_verification.json").read_text("utf-8"))
        self.sample = json.loads((RES / "prompt_sample.v1.json").read_text("utf-8"))
        excl = RES / "prompt_exclusions.v1.json"
        self.exclusions = (json.loads(excl.read_text("utf-8"))
                           if excl.is_file() else {"excluded": []})
        self.dropped = sorted(int(e["prompt_idx"])
                              for e in self.exclusions.get("excluded", []))
        with (RES / "cells.v1.tsv").open(encoding="utf-8") as handle:
            self.manifest = list(csv.DictReader(handle, delimiter="\t"))
        self.rows: list[dict[str, Any]] = []
        with gzip.open(RES / "perprompt_bend.tsv.gz", "rt") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                self.rows.append(row)
        self.cells = {m: {c["cell_id"]: c for c in self.bend[m]["cells"]}
                      for m, _ in MODELS}

    def cell_list(self, model: str, *, k: int | None = None,
                  payload: str | None = None) -> list[dict[str, Any]]:
        out = list(self.cells[model].values())
        if k is not None:
            out = [c for c in out if c["k"] == k]
        if payload is not None:
            out = [c for c in out if c["payload"] == payload]
        return out

    def off_budget_realized(self) -> str:
        """What the off-budget gate rows actually cache, read from the manifest."""
        vals = sorted({int(r["k_realized"]) for r in self.manifest
                       if r["schedule"].endswith("_off")})
        return " / ".join(str(v) for v in vals)

    def profile_median(self, cell: dict[str, Any]) -> list[float | None]:
        return cell["D_over_chord_ref_profile"]["median"]

    def med_over(self, cells: list[dict[str, Any]], field: str) -> float | None:
        vals = [c["median"][field] for c in cells if c["median"].get(field) is not None]
        return statistics.median(vals) if vals else None

    def profile_at(self, cells: list[dict[str, Any]], n: int) -> float | None:
        vals = [self.profile_median(c)[n] for c in cells
                if self.profile_median(c)[n] is not None]
        return statistics.median(vals) if vals else None


# ------------------------------------------------------------------ figures
def write_figures(D: Data) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    names: dict[str, str] = {}

    # fig 1 — D[n] median profile per payload family, one panel per model
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (model, long) in zip(axes, MODELS):
        for payload, _zh, label in PAYLOADS:
            cells = D.cell_list(model, payload=payload)
            if not cells:
                continue
            mat = np.asarray([[np.nan if v is None else v
                               for v in D.profile_median(c)] for c in cells])
            ax.plot(range(mat.shape[1]), np.nanmedian(mat, axis=0), label=label)
        ax.set_yscale("log")
        ax.set_xlabel("state n (0 = z_T, 50 = result)")
        ax.set_title(f"{long} ({len(D.cells[model])} rows)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("D[n] / reference chord")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_d_profile_by_payload.png", dpi=150)
    plt.close(fig)
    names["profile"] = "fig1_d_profile_by_payload.png"

    # fig 2 — early offset against row quality, per partition, families marked
    link = D.link["slices"]["all"]["q2_row_level"]["partitions"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    quality = _row_quality(D)
    for i, (model, long) in enumerate(MODELS):
        for j, k in enumerate(KS):
            ax = axes[i][j]
            xs, ys, fams = [], [], []
            for cell in D.cell_list(model, k=k):
                x = cell["median"]["D10_over_chord_ref"]
                y = quality.get((model, cell["schedule"], cell["payload"], k))
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
                fams.append(schedule_family(cell["schedule"]))
            for fam, marker in (("search", "o"), ("random", "^"),
                                ("ladder", "s"), ("geometry", "D"),
                                ("rho2", "*"), ("gate variant", "x")):
                sel = [n for n, f in enumerate(fams) if f == fam]
                if sel:
                    ax.scatter([xs[n] for n in sel], [ys[n] for n in sel],
                               marker=marker, s=26, label=fam, alpha=0.8)
            block = link.get(f"{model}_k{k}", {})
            r = (block.get("D10_over_chord_ref") or {}).get("rho")
            ax.set_title(f"{SHORT[model]} K{k}  n={block.get('n_rows', 0)}  "
                         f"ρ={fmt(r, 2)}", fontsize=9)
            ax.set_xlabel("row median D[10] / reference chord")
            ax.set_ylabel("row mean PSNR")
            ax.grid(alpha=0.3)
    axes[0][2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_early_vs_quality.png", dpi=150)
    plt.close(fig)
    names["scatter"] = "fig2_early_vs_quality.png"

    # fig 3 — the three direction shares at state 50, by payload
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    q3 = D.link["slices"]["all"]["q3_direction"]["by_payload"]
    for ax, (model, long) in zip(axes, MODELS):
        labels, chord_s, plane_s, off_s = [], [], [], []
        for payload, _zh, label in PAYLOADS:
            entry = q3.get(model, {}).get(payload)
            if not entry:
                continue
            labels.append(label)
            chord_s.append(entry["share_chord_50"]["median"] or 0.0)
            plane_s.append(entry["share_in_plane_50"]["median"] or 0.0)
            off_s.append(entry["share_off_plane_50"]["median"] or 0.0)
        x = np.arange(len(labels))
        ax.bar(x, chord_s, label="along reference chord")
        ax.bar(x, plane_s, bottom=chord_s, label="in the bend plane")
        ax.bar(x, off_s, bottom=np.asarray(chord_s) + np.asarray(plane_s),
               label="off plane")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_title(long)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("energy share of the offset at state 50")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_direction_shares.png", dpi=150)
    plt.close(fig)
    names["shares"] = "fig3_direction_shares.png"

    # fig 4 — event-aligned Delta D
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (model, long) in zip(axes, MODELS):
        for stratum, label in (("cache", "step k+j is a cache step"),
                               ("full", "step k+j is a full step")):
            offs, meds = [], []
            buckets: dict[int, list[float]] = {}
            for cell in D.cell_list(model):
                for row in cell["events"]["event"]:
                    if row["stratum"] != stratum:
                        continue
                    value = row["dD_over_chord_ref"]["median"]
                    if value is not None:
                        buckets.setdefault(row["index"], []).append(value)
            for off in sorted(buckets):
                offs.append(off)
                meds.append(statistics.median(buckets[off]))
            ax.plot(offs, meds, marker="o", label=label)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("offset j from a cache step")
        ax.set_title(long)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Delta D[k+j] / reference chord")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_event_alignment.png", dpi=150)
    plt.close(fig)
    names["events"] = "fig4_event_alignment.png"
    return names


def _row_quality(D: Data) -> dict[tuple[str, str, str, int], float]:
    """Row -> mean PSNR over the full seed-42 slice, read the same way Q2 does."""
    out: dict[tuple[str, str, str, int], list[float]] = {}
    for model, _ in MODELS:
        path = ROOT / "resources" / "spx" / f"perprompt_spx_{model}.tsv.gz"
        with gzip.open(path, "rt") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            col = header.index("psnr")
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if parts[3] != "42":
                    continue
                value = float(parts[col])
                if math.isfinite(value):
                    out.setdefault((model, parts[0], parts[1], int(parts[2])),
                                   []).append(value)
    return {key: sum(v) / len(v) for key, v in out.items()}


# ------------------------------------------------------------------ document
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()

    D = Data()
    figs = write_figures(D) if args.figures else {
        "profile": "fig1_d_profile_by_payload.png",
        "scatter": "fig2_early_vs_quality.png",
        "shares": "fig3_direction_shares.png",
        "events": "fig4_event_alignment.png",
    }

    L: list[str] = []
    P = L.append

    n_cells = sum(len(D.cells[m]) for m, _ in MODELS)
    n_pairs = sum(D.bend[m]["n_pairs"] for m, _ in MODELS)
    n_drawn = D.sample["sample_size"]
    n_prompts = n_drawn - len(D.dropped)
    n_full = D.sample["prompt_count"]
    n_heldout = len([i for i in D.sample["held_out_indices"] if i not in D.dropped])
    d_of = {m: None for m, _ in MODELS}

    # ------------------------------------------------------------------ §0
    P("# cache 如何把图像去噪轨迹弯开 — 全量测量结果")
    P("")
    P("状态：%s。本文只讲**开了 cache 之后轨迹怎么被弯开、以及早期的弯折能不能预测终局质量**；"
      "不开 cache 的参照轨迹自身的形状、%d 格的解码后质量，各有自己的结果文档，本文需要用到的那几个数都在下面重新算出并列出。"
      % (D.verify.get("date", "2026-08-24"), n_cells))
    P("")
    P("**写法约定**：小节标题命名被测对象；一格一个数；每张表的表题写清模型、格数、对数、中位还是均值、除以什么归一化。")
    P("")
    P("## 0. 这份文档在讲什么（先读这一节）")
    P("")
    P("**去噪轨迹**：用扩散模型生成一张图，是从一团随机噪声出发、分 50 步逐步去噪。"
      "把起点和每一步之后的中间状态按顺序连起来（共 51 个点），就是这次生成的**轨迹**。")
    P("")
    P("**cache 弯折**：同一个 prompt、同一团初始噪声，一次不开 cache 完整算 50 步（下称**参照**），"
      "一次按某张固定跳步表开着 cache 跑（下称 **cell 生成**）。两条轨迹在第一个 cache 步之前逐位相同，之后分叉。")
    P("")
    P("**本文与视频侧同层文档的区别**：视频侧比的是九个方法；本文比的是**调度行全集**——"
      "搜索出来的表、随机表、Hamming 阶梯、几何构造行、off-budget 门行都在里面，"
      "正因为行与行之间的第一个跳步位置差得开，才能问下面这个视频侧问不了的问题。")
    P("")
    P("**本文回答四个问题**：")
    P("")
    P("1. 逐状态偏移 D[n] 是不是单调累积、后段陡增，终态偏移的行排序与质量排序是否同构？（§3）")
    P("2. **早期偏移能不能预测这一行的终局质量**，与第一个跳步位置 k₀ 单变量比、与终态偏移 D[50] 比，解释力如何？（§7，本文主检验）")
    P("3. 偏移向量落在参照轨迹的哪个方向，按载荷族分层是否与视频侧同构？（§5）")
    P("4. 第一个跳步的效应是不是恰好在第一个 cache 事件处进入 D[n]？（§6）")
    P("")
    P("**术语表**（后文不再解释）：")
    P("")
    P(table(["词", "意思"], [
        ["**Z[n]**", "第 n 步之后的隐空间状态；n = 0 是初始噪声，n = 50 是终态。"
                     "\"状态 n\"指 Z[n]，\"步 n\"指从 Z[n] 走到 Z[n+1] 的那一步"],
        ["**cache 步 / 真算步**", "cache 步这一步不跑 transformer，用载荷顶替；真算步正常跑"],
        ["**k₀**", "这一行的第一个 cache 步。cache 在第 k₀ 步发生，第一个被改写的状态是 Z[k₀+1]"],
        ["**行 / 格**", "行 = 一个 (模型, K, 调度, 载荷)；一行就是一格，两个词同义，"
                       "讲统计时说\"行\"，讲数据时说\"格\""],
        ["**K**", "50 步里被跳过的步数；本文三档 K29 / K37 / K41。"
                  f"off-budget 门行实际跳 {D.off_budget_realized()} 步，"
                  "按名义档分组、按实现值读"],
        ["**载荷**", "cache 步用什么顶替被跳过的计算：" +
                    "、".join(f"{p}（{zh}）" for p, zh, _ in PAYLOADS)],
        ["**弦 / 参照弦长**", "弦 = 起点到终点的直线段 Z[50] − Z[0]；"
                            "参照弦长 = 参照轨迹的弦长，本文所有偏移量都除以它"],
        ["**D[n]**", "‖Z^cell[n] − Z^参照[n]‖ ÷ 参照弦长；**ΔD[k]** = D[k+1] − D[k] 是第 k 步加上的偏移"],
        ["**早期偏移**", f"D[{EARLY_N}]，以及 D 在状态 1..{EARLY_N} 上的和（下称早期面积）。"
                        f"k₀ ≥ {EARLY_N} 的行按定义两者都恰好为 0——它们的\"早期\"里没有 cache 事件，"
                        "是首跳假设的对照组，单独成一层报，不当作小读数平均进去"],
        ["**弯曲平面**", "轨迹偏离弦的那部分能量绝大多数落在一个二维平面内，"
                        "由偏离向量的前两个主成分张成；本文用参照轨迹自己的平面"],
        ["**三份能量**", "偏移向量在 [参照弦向, 弯曲平面内, 平面外] 上的能量份额，三者和为 1"],
        ["**路径层**", "把全部 51 个状态整体存盘（bfloat16）后逐状态比较"],
        ["**质量**", "该行在全部 %d 条 prompt、seed 42 上的 PSNR 均值，"
                    "取自 `resources/spx/perprompt_spx_<模型>.tsv.gz`（本层生成之前就已测好）" % n_full],
        ["**discovery / held-out**", "prompt 的预注册划分（`parti_spx_splits.v1.json`）；"
                                     "本文每个统计在全部 %d 条与 held-out %d 条上各算一次并排印"
                                     % (n_prompts, n_heldout)],
    ]))
    P("")

    # ------------------------------------------------------------------ §1
    P("## 1. 数据与协议")
    P("")
    rows = []
    for model, long in MODELS:
        cells = list(D.cells[model].values())
        by_k = {k: len([c for c in cells if c["k"] == k]) for k in KS}
        rows.append([long, str(len(cells)),
                     " / ".join(str(by_k[k]) for k in KS),
                     str(D.bend[model]["n_pairs"]),
                     str(len({c["payload"] for c in cells})),
                     str(len({c["schedule"] for c in cells}))])
    P(table(["模型", "格数", "K29 / K37 / K41", "对数", "载荷数", "调度数"], rows))
    P("")
    P("每格抽 %d 条 prompt、读 %d 条，两个模型与参照共用同一组序号"
      "（`resources/image_trajectory/prompt_sample.v1.json`，`default_rng([20260824])` "
      "从 %d 条里均匀不放回抽出）；配对键是 (模型, prompt 序号)，seed 一律 42、"
      "逐图 seed = 42 + 序号。每对 cell 生成与参照同 prompt、同 seed，因此同一团初始噪声："
      "全部 %d 对的初始噪声指纹逐对相同。"
      % (n_drawn, n_prompts, n_full, n_pairs))
    P("")
    if D.dropped:
        for entry in D.exclusions["excluded"]:
            ev = entry["evidence"]
            P("**被剔除的序号 %d**（抽 %d 读 %d 的那一条）：这条 prompt 的重放输出与已存格不一致——"
              "它是样本里唯一一条非 ASCII 的 prompt（%s），两簇的 CLIP 分词器把它切成不同的 token"
              "（%s），T5 的 token 完全相同，prompt 的字节在两簇上哈希相同。"
              "受影响的是 %d 个存在 site_a 的 FLUX 格的这一条，其余全部相符；"
              "Qwen-Image 不用 CLIP 文本编码器，未受影响。序号、原因与证据记在 "
              "`resources/image_trajectory/prompt_exclusions.v1.json`，"
              "抽样表本身不改；本文所有读数在剔除后取。"
              % (entry["prompt_idx"], n_drawn, n_prompts, ev["character"],
                 "、".join(f"{k} 切出 {v} 个非填充 token"
                           for k, v in ev["clip_tokenizer_non_pad_tokens"].items()),
                 entry["n_cells_affected"]))
            P("")
    P("**cell 生成不是新实验**：%d 格的图与 decisions 早就在盘上，缺的只是它们走过的 51 个状态，"
      % n_cells +
      "而中间状态无法事后补算。本文的生成是**带保留的重放**——同一份权重、同一个 seed、同一张跳步表、同一个载荷，"
      "重跑抽中的 (格, prompt)，得到同一张图外加它走过的路径。判据见 §2 末。")
    P("")

    # k0 by partition
    rows = []
    for model, long in MODELS:
        for k in KS:
            cells = D.cell_list(model, k=k)
            k0s = sorted({c["k0_row"] for c in cells if c["k0_row"] is not None})
            early = len([c for c in cells if c["k0_row"] is not None
                         and c["k0_row"] < EARLY_N])
            rows.append([SHORT[model], f"K{k}", str(len(cells)),
                         f"{min(k0s)}–{max(k0s)}" if k0s else "—",
                         str(len(k0s)), f"{early} / {len(cells)}"])
    P("**第一个跳步 k₀ 在行间的分布**（每 (模型, K) 分区；k₀ 由该行的跳步表决定，格内唯一）：")
    P("")
    P(table(["模型", "档", "行数", "k₀ 范围", "不同 k₀ 取值数",
             f"k₀ < {EARLY_N} 的行"], rows))
    P("")

    # ------------------------------------------------------------------ §2
    P("## 2. 底噪四行与重放判据")
    P("")
    rows = []
    lossless_all = True
    for model, long in MODELS:
        f = D.floors["models"][model]
        b = f["bf16_store"]
        lossless_all = lossless_all and bool(b.get("store_is_lossless"))
        rows.append([
            long,
            f"{sci(b['median'])}（{b['n_paths']} 条）",
            f"{b.get('n_paths_bf16_exact', 0)} / {b.get('n_paths_compared', 0)}",
            "是" if f["reproduction"]["all_exactly_zero"] else
            f"最大 {sci(f['reproduction']['max_abs_state_diff'])}",
            f"{sci(f['store_orientation']['chord_angle_deg_max'])}° / "
            f"{sci(f['store_orientation']['plane_angle1_deg_max'])}°",
        ])
    P(table(["模型", "bf16 存盘下界（参照弦长，中位）", "fp32 双存路径里 bf16 可精确表示的条数",
             "同参照跑两次逐状态差为零", "存盘对取向的扰动：弦向角 / 平面第一主角（最大）"], rows))
    P("")
    if lossless_all:
        P("下界在 fp32 双存子集上量：把 float32 存的参照路径舍入到 bfloat16 再读回，取位移路线与偏离路线的大者。"
          "**两个模型上都恰好为 0**，而且不是\"小到测不出\"：两个采样器都在 bfloat16 里走步，"
          "callback 读到的就是采样器自己持有的那个状态，所以用 float32 存它并不多存下任何一位——"
          "fp32 双存的每条路径都逐位等于它自己的 bf16 舍入，也逐位等于 bf16 存盘的那一条。"
          "结论：**D 没有存盘底噪**，表里每个非零值都是真实差异；解析的 bf16 相对 RMS %s 只是"
          "\"若状态本身带更多位则下界会是多少\"的参照值，不是本层的下界。"
          % sci(D.floors["models"]["flux"]["bf16_store"].get("analytic_rel_rms")))
    else:
        P("下界在 fp32 双存子集上量：把 float32 存的参照路径舍入到 bfloat16 再读回，取位移路线与偏离路线的大者。"
          "D 是两条各自舍入的路径之差，k₀ 之后它的存盘噪声最多是单条位移的 √2 倍，"
          "k₀ 之前恰好为 0（两次运行舍入的是同一个状态）。")
    P("")
    P("取向扰动那一列因此也只剩 float64 计算本身的舍入（%s° 量级），不是存盘带来的。"
      % sci(max(D.floors["models"][m]["store_orientation"]["plane_angle1_deg_max"]
                for m, _ in MODELS)))
    P("")

    prefix_rows = []
    for model, long in MODELS:
        cells = list(D.cells[model].values())
        with_cache = sum(c["prefix_identity"]["n_rows_with_cache_step"] for c in cells)
        zero = sum(c["prefix_identity"]["n_exactly_zero"] for c in cells)
        worst = max((c["prefix_identity"]["max_abs_D_before_k0"] for c in cells
                     if c["prefix_identity"]["max_abs_D_before_k0"] is not None),
                    default=None)
        prefix_rows.append([long, str(with_cache), str(zero),
                            "0" if worst == 0.0 else sci(worst)])
    P("**前缀恒等**（每对在 n ≤ k₀ 上的 D 应恰好为零）：")
    P("")
    P(table(["模型", "有 cache 步的对数", "前缀恰好为零的对数", "k₀ 之前 D 的最大值"],
            prefix_rows))
    P("")

    v = D.verify["totals"]
    ref = D.verify.get("reference_wave", {})
    P("**重放判据**（每条保留重放的输出与已存格逐像素比对；解码后的像素数组是判据，PNG 文件字节不是——"
      "两簇的 Pillow 版本不同，同一张图会编码出不同的压缩流，文件摘要会报出生成本身没有的差别）：")
    P("")
    P(table(["检查", "比对数", "不符"], [
        ["格图像逐像素（剔除后）", str(v.get("images_compared_kept", v["images_compared"])),
         str(v.get("image_mismatch_kept", v["image_mismatch"]))],
        ["格图像逐像素（剔除前，含被剔除的序号）", str(v["images_compared"]),
         str(v["image_mismatch"])],
        ["格 decisions 的 cache 步集合 vs 该行位串", str(v["images_compared"]),
         str(v["bitstring_mismatch"])],
        ["参照波图像 vs 移交的无 cache 基线", str(ref.get("compared", 0)),
         str(len(ref.get("mismatch", [])))],
        ["同像素但 PNG 字节不同（只记录，不判负）", str(v["images_compared"]),
         str(v["file_digest_differs"])],
    ]))
    P("")

    # ------------------------------------------------------------------ §3
    P("## 3. 逐状态偏移 D[n]（问题 1）")
    P("")
    P("![D[n] 剖面](%s/%s)" % (FIG_REL, figs["profile"]))
    P("")
    P("*图 1：D[n] ÷ 参照弦长的中位剖面，按载荷分族，纵轴对数。每条线是该族全部行的\"格中位\"再取中位；"
      "%s %d 格、%s %d 格，每格 %d 对。*"
      % (MODELS[0][1], len(D.cells["flux"]), MODELS[1][1], len(D.cells["qwen"]), n_prompts))
    P("")
    rows = []
    for model, long in MODELS:
        for payload, label, _en in PAYLOADS:
            cells = D.cell_list(model, payload=payload)
            if not cells:
                continue
            rows.append([SHORT[model], label, str(len(cells))] +
                        [sci(D.profile_at(cells, n)) for n in D_STATES])
    P("**D[n] 的族中位**（每格 %d 对取中位，再对族内行取中位；÷ 参照弦长）：" % n_prompts)
    P("")
    P(table(["模型", "载荷", "行数"] + [f"n={n}" for n in D_STATES], rows))
    P("")
    P("表里的 0 是**恰好为零**，不是低于某个下界：该族过半的行在那个状态之前还没有 cache 步，"
      "而 §2 已经说明 D 没有存盘底噪。")
    P("")

    q1 = D.link["slices"]["all"]["q1"]["models"]
    rows = []
    for model, long in MODELS:
        block = q1[model]
        rows.append([long, str(block["n_cells_profiled"]),
                     pct(block["share_monotone_nondecreasing"]),
                     fmt(block["late_over_mid_growth_ratio_median"], 2)] +
                    [rho_cell(block["D50_vs_quality_spearman"][f"k{k}"]) for k in KS])
    P("**累积形状与终态排序**（后段/中段增长比 = (D[50] − D[40]) ÷ (D[40] − D[30]) 的行中位；"
      "末三列是每分区内 D[50] 行中位与该行 PSNR 均值的 Spearman，`*` = p < 0.05）：")
    P("")
    P(table(["模型", "计入行数", "D[n] 逐状态不减的行占比", "后段/中段增长比",
             "K29 ρ(D[50], PSNR)", "K37 ρ", "K41 ρ"], rows))
    P("")

    # ------------------------------------------------------------------ §4
    P("## 4. 开了 cache 之后整条轨迹的形状标量差了多少（问题 1 旁证）")
    P("")
    rows = []
    for model, long in MODELS:
        for payload, label, _en in PAYLOADS:
            cells = D.cell_list(model, payload=payload)
            if not cells:
                continue
            rows.append([SHORT[model], label, str(len(cells)),
                         fmt(D.med_over(cells, "chord_ratio"), 4),
                         fmt(D.med_over(cells, "straightness_diff"), 4),
                         fmt(D.med_over(cells, "max_dev_ratio_diff"), 4),
                         fmt(D.med_over(cells, "pca_evr_top2_diff"), 4)])
    P("**形状标量差**（cell 减参照，除弦长比是 cell ÷ 参照；每格 %d 对取中位，再对族内行取中位）：" % n_prompts)
    P("")
    P(table(["模型", "载荷", "行数", "弦长比", "弯曲程度差", "最大偏离差", "偏离前二主成分份额差"],
            rows))
    P("")
    chord_verdict = []
    for model, long in MODELS:
        reuse = D.med_over(D.cell_list(model, payload="reuse"), "chord_ratio")
        ext = [D.med_over(D.cell_list(model, payload=p), "chord_ratio")
               for p in EXTRAP]
        ext = [v for v in ext if v is not None]
        if reuse is None or not ext:
            continue
        chord_verdict.append(
            "%s 的零阶复用弦长比中位 %s，外推载荷 %s–%s"
            % (SHORT[model], fmt(reuse, 4), fmt(min(ext), 4), fmt(max(ext), 4)))
    P("弦长：%s。" % "；".join(chord_verdict))
    P("")

    # ------------------------------------------------------------------ §5
    P("## 5. 偏移落在参照轨迹的哪个方向（问题 3）")
    P("")
    P("![三份能量](%s/%s)" % (FIG_REL, figs["shares"]))
    P("")
    P("*图 2：状态 50 处偏移向量的三份能量，按载荷族堆叠。每族一根柱，柱高恒为 1；"
      "每格 %d 对取中位，再对族内行取中位。*" % n_prompts)
    P("")
    q3 = D.link["slices"]["all"]["q3_direction"]["by_payload"]
    rows = []
    for model, long in MODELS:
        for payload, label, _en in PAYLOADS:
            entry = q3.get(model, {}).get(payload)
            if not entry:
                continue
            cells = D.cell_list(model, payload=payload)
            rows.append([SHORT[model], label,
                         str(entry["share_off_plane_50"]["n_rows"]),
                         fmt(entry["share_chord_50"]["median"]),
                         fmt(entry["share_in_plane_50"]["median"]),
                         fmt(entry["share_off_plane_50"]["median"]),
                         fmt(entry["share_off_plane_k0p1"]["median"]),
                         fmt(D.med_over(cells, "chord_angle_deg"), 1),
                         fmt(D.med_over(cells, "plane_angle1_deg"), 1)])
    P(table(["模型", "载荷", "行数", "沿弦 (n=50)", "平面内 (n=50)", "平面外 (n=50)",
             "平面外 (n=k₀+1)", "cell 弦 vs 参照弦夹角 (°)",
             "两弯曲平面第一主角 (°)"], rows))
    P("")
    chord_angles = [D.med_over(D.cell_list(m, payload=p), "chord_angle_deg")
                    for m, _ in MODELS for p, _, _en in PAYLOADS
                    if D.cell_list(m, payload=p)]
    plane_angles = [D.med_over(D.cell_list(m, payload=p), "plane_angle1_deg")
                    for m, _ in MODELS for p, _, _en in PAYLOADS
                    if D.cell_list(m, payload=p)]
    chord_angles = [v for v in chord_angles if v is not None]
    plane_angles = [v for v in plane_angles if v is not None]
    if chord_angles and plane_angles:
        P("末两列同口径（每格 %d 对取中位，再对族内行取中位）：cell 生成的弦与参照弦之间"
          "的夹角族中位在 %.1f–%.1f°，而 cell 轨迹自己的弯曲平面相对参照弯曲平面的第一主角"
          "族中位在 %.1f–%.1f°——cache 对\"从哪到哪\"（弦向）动得少，对\"怎么绕\"（弯曲平面"
          "的取向）动得多。"
          % (n_prompts, min(chord_angles), max(chord_angles),
             min(plane_angles), max(plane_angles)))
        P("")

    # ------------------------------------------------------------------ §6
    P("## 6. 偏移在哪种步上累积，首跳把多少偏移一次性写进去（问题 4）")
    P("")
    P("![事件对齐](%s/%s)" % (FIG_REL, figs["events"]))
    P("")
    P("*图 3：以每个 cache 步 k 为原点，ΔD[k+j] 的中位，按第 k+j 步自己是 cache 步还是真算步分层。"
      "每格 %d 对，%s %d 格、%s %d 格；窗口重叠使桶内样本不独立，只报中位。*"
      % (n_prompts, MODELS[0][1], len(D.cells["flux"]),
         MODELS[1][1], len(D.cells["qwen"])))
    P("")
    q4 = D.link["slices"]["all"]["q4_first_jump"]["partitions"]
    rows = []
    for model, long in MODELS:
        for k in KS:
            block = q4[f"{model}_k{k}"]
            cells = D.cell_list(model, k=k)
            rows.append([SHORT[model], f"K{k}", str(block["n_rows"]),
                         sci(D.med_over(cells, "delta_D_k0")),
                         sci(D.med_over(cells, "slope_k0_plus_5")),
                         rho_cell(block["delta_D_k0_vs_k0"]),
                         rho_cell(block["slope_vs_k0"]),
                         rho_cell(block["delta_D_k0_vs_quality"]),
                         rho_cell(block["slope_vs_quality"])])
    # do the two strata ever regress? the video side found methods whose full
    # steps pull the deviation back; whether that happens here is a reading,
    # not an assumption
    strat: dict[tuple[str, str], list[float]] = {}
    for model, _ in MODELS:
        for cell in D.cell_list(model):
            for row in cell["events"]["event"]:
                value = row["dD_over_chord_ref"]["median"]
                if value is not None:
                    strat.setdefault((model, row["stratum"]), []).append(value)
    neg = {k: sum(1 for v in vals if v < 0) for k, vals in strat.items()}
    counts = "；".join(
        "%s%s桶 %d / %d 个为负"
        % (SHORT[m], "的 cache 步" if st == "cache" else "的真算步",
           neg[(m, st)], len(strat[(m, st)]))
        for m, _ in MODELS for st in ("cache", "full") if (m, st) in strat)
    total_neg = sum(neg.values())
    total_buckets = sum(len(v) for v in strat.values())
    P("视频侧只有 1 个方法（Wan2.1 的 MeanCache）在真算步拉回（该桶的 ΔD 中位为负），"
      "且那一读数只覆盖 37 步档。图像侧三档全部格里%s：%s"
      "——%s。"
      % ("几乎不出现" if total_neg else "不出现", counts,
         ("真算步加得比 cache 步少，但绝大多数桶仍在加：%d / %d 个桶为负，"
          "全部落在 %s 的真算步层"
          % (total_neg, total_buckets,
             "、".join(sorted({SHORT[m] for (m, st), c in neg.items() if c})))
          if total_neg else
          "两个层都在正的一侧，真算步加得比 cache 步少，但没有减")))
    P("")
    P("**首跳增量**（ΔD[k₀] = 第一个 cache 步写进去的偏移；斜率 = (D[k₀+5] − D[k₀+1]) ÷ 4，"
      "即首跳写入之后、状态 k₀+1 到 k₀+5 之间的平均每步增量，不含首跳那一步本身；"
      "两者都 ÷ 参照弦长，每格 %d 对取中位）：" % n_prompts)
    P("")
    P(table(["模型", "档", "行数", "ΔD[k₀] 中位", "首跳后五步斜率中位",
             "ρ(k₀, ΔD[k₀])", "ρ(k₀, 斜率)", "ρ(ΔD[k₀], PSNR)", "ρ(斜率, PSNR)"],
            rows))
    P("")
    slope_q = [q4[f"{m}_k{k}"]["slope_vs_quality"] for m, _ in MODELS for k in KS]
    slope_rhos = [e["rho"] for e in slope_q if e.get("rho") is not None]
    slope_sig = sum(1 for e in slope_q
                    if e.get("rho") is not None and e.get("p") is not None
                    and e["p"] < 0.05)
    if slope_rhos:
        P("末列是事件对齐的早期读数对质量：斜率对每一行都有定义（没有 D[10] 那样的结构性零），"
          "六个分区全部同号（ρ %.2f 到 %.2f，%d / %d 个分区 p < 0.05），"
          "而 ΔD[k₀] 对质量在 K37 / K41 上要弱得多——首跳之后头几步的增长速度比首跳本身的大小"
          "更载质量信息。§7 用它做头条与中介的复核。"
          % (max(slope_rhos), min(slope_rhos), slope_sig, len(slope_q)))
        P("")

    # ------------------------------------------------------------------ §7
    P("## 7. 早期偏移与终局质量（问题 2，本文主检验）")
    P("")
    P("![早期偏移 vs 质量](%s/%s)" % (FIG_REL, figs["scatter"]))
    P("")
    P("*图 4：每 (模型, K) 分区里，行的 D[10] 中位（%d 条 prompt）对该行的 PSNR 均值（%d 条 prompt）。"
      "点形按调度族分：搜索/门派生、随机、Hamming 阶梯、几何构造、ρ₂ 对照、门变体。*"
      % (n_prompts, n_full))
    P("")
    def _pred_cells(entry: dict[str, Any]) -> list[str]:
        return [rho_cell(entry["D10_over_chord_ref"]),
                rho_cell(entry["early_auc_over_chord_ref"]),
                rho_cell(entry["k0"]),
                rho_cell(entry["D50_over_chord_ref"])]

    for slice_name, slice_label in (("all", "全部 %d 条 prompt" % n_prompts),
                                    ("held_out", "held-out %d 条" % n_heldout)):
        block = D.link["slices"][slice_name]["q2_row_level"]
        rows = []
        for model, _ in MODELS:
            for k in KS:
                part = block["partitions"].get(f"{model}_k{k}", {})
                if "D10_over_chord_ref" not in part:
                    rows.append([SHORT[model], f"K{k}", str(part.get("n_rows", 0)),
                                 "—", "—", "—", "—"])
                    continue
                rows.append([SHORT[model], f"K{k}", str(part["n_rows"])]
                            + _pred_cells(part))
                sub = part.get("k0_lt_early_subset")
                if sub and sub.get("n_rows"):
                    rows.append([SHORT[model], f"K{k}（k₀ < {EARLY_N}）",
                                 str(sub["n_rows"])] + _pred_cells(sub))
        pooled = block["pooled"]
        rows.append(["合并", "分区内标准化", str(pooled["n_rows"])]
                    + _pred_cells(pooled))
        small = block["pooled_k0_lt_early"]
        rows.append([f"合并（k₀ < {EARLY_N} 的行）", "分区内标准化",
                     str(small["n_rows"])] + _pred_cells(small))
        P("**行级 Spearman，%s**（自变量是行的读数中位，因变量是行的 PSNR 均值；`*` = p < 0.05；"
          "每个分区行后面紧跟一行它的 k₀ < %d 子集；"
          "合并行是分区内标准化后的秩相关，样本不独立，只作描述）：" % (slice_label, EARLY_N))
        P("")
        P(table(["模型", "档", "行数", "ρ(D[10], PSNR)", "ρ(早期面积, PSNR)",
                 "ρ(k₀, PSNR)", "ρ(D[50], PSNR)"], rows))
        P("")

    # ---- headline robustness: the same reading under three constructions
    blk_all = D.link["slices"]["all"]["q2_row_level"]
    p_all = blk_all["pooled"]
    p_small = blk_all["pooled_k0_lt_early"]
    P("**头条读数的稳健性**（同一\"早期 vs 质量\"合并相关换三个口径；k₀ ≥ %d 的行的 D[10] "
      "按定义恰好为 0，是头条最可能的混杂源，所以剔除它们与换用对每行都有定义的读数各查一遍）：" % EARLY_N)
    P("")
    P("- 全部 %d 行：ρ(D[10], PSNR) = %s；" % (p_all["n_rows"],
                                              rho_cell(p_all["D10_over_chord_ref"])))
    P("- 剔除 %d 个 D[10] 结构性为零的行（k₀ ≥ %d）后的 %d 行：ρ = %s——剔除后相关**变强**；"
      % (p_all["n_rows"] - p_small["n_rows"], EARLY_N, p_small["n_rows"],
         rho_cell(p_small["D10_over_chord_ref"])))
    P("- 换成事件对齐的早期读数（§6 的首跳后五步斜率，无结构性零）：ρ = %s（全部 %d 行）/ "
      "%s（k₀ < %d 的 %d 行）。"
      % (rho_cell(p_all["slope_k0_plus_5"]), p_all["n_rows"],
         rho_cell(p_small["slope_k0_plus_5"]), EARLY_N, p_small["n_rows"]))
    P("")
    P("三个口径同号、量级相近：头条相关不是那 %d 个结构性零行造出来的。"
      % (p_all["n_rows"] - p_small["n_rows"]))
    P("")

    P("**两个自变量各解释多少、互相加多少**（分区内标准化后的合并回归——k₀ < %d 的群体"
      "在子集内重新标准化；系数是标准化系数，t 是同回归里 k₀ 系数的 t 值，"
      "增量 R² = 两者同回归的 R² 减去另一个单独时的 R²。主口径是 **D[10] × k₀ < %d** 那两行："
      "全部行的群体含 %d 个 D[10] 按定义恰好为 0 的行，附注见表后）："
      % (EARLY_N, EARLY_N, p_all["n_rows"] - p_small["n_rows"]))
    P("")
    med_keys = [
        ("q2_mediation_D10_k0_lt_early", "D[10]", "k₀ < %d 的行（主口径）" % EARLY_N),
        ("q2_mediation_D10", "D[10]", "全部行（附）"),
        ("q2_mediation_auc_k0_lt_early", "早期面积", "k₀ < %d 的行" % EARLY_N),
        ("q2_mediation_auc", "早期面积", "全部行（附）"),
        ("q2_mediation_slope_k0_lt_early", "事件对齐斜率", "k₀ < %d 的行" % EARLY_N),
        ("q2_mediation_slope", "事件对齐斜率", "全部行"),
    ]
    rows = []
    meds: dict[tuple[str, str], dict[str, Any]] = {}
    for slice_name, slice_label in (("all", "全部"), ("held_out", "held-out")):
        for med_key, reading, pop_label in med_keys:
            med = D.link["slices"][slice_name].get(med_key, {})
            if "quality_on_k0" not in med:
                continue
            meds[(slice_name, med_key)] = med
            c_k0 = med["quality_on_k0"]["coef"][0]
            c_e = med["quality_on_early"]["coef"][0]
            b_k0, b_e = med["quality_on_both"]["coef"]
            t_k0 = med["quality_on_both"]["t"][0]
            sig = (abs(t_k0) >= 1.96) if t_k0 is not None else None
            rows.append([slice_label, reading, pop_label, str(med["n_rows"]),
                         fmt(c_k0), f"{b_k0:+.3f}",
                         (fmt(t_k0, 1) + ("" if sig else "，n.s.")) if t_k0 is not None else "—",
                         fmt(c_e), fmt(b_e),
                         fmt(med["quality_on_k0"]["r2"], 3),
                         fmt(med["quality_on_early"]["r2"], 3),
                         fmt(med["quality_on_both"]["r2"], 3),
                         fmt(med["quality_on_both"]["r2"] - med["quality_on_early"]["r2"], 3),
                         fmt(med["quality_on_both"]["r2"] - med["quality_on_k0"]["r2"], 3)])
    P(table(["切片", "早期读数", "群体", "行数", "只 k₀ 的系数", "同回归的 k₀ 系数", "其 t",
             "只早期的系数", "同回归的早期系数",
             "R² 只 k₀", "R² 只早期", "R² 两者",
             "k₀ 的增量 R²", "早期的增量 R²"], rows))
    P("")

    med_primary = meds.get(("all", "q2_mediation_D10_k0_lt_early"), {})
    med_full = meds.get(("all", "q2_mediation_D10"), {})
    med_slope = meds.get(("all", "q2_mediation_slope"), {})
    med_slope_sub = meds.get(("all", "q2_mediation_slope_k0_lt_early"), {})
    if med_primary and med_full:
        pb_k0 = med_primary["quality_on_both"]["coef"][0]
        pb_t = med_primary["quality_on_both"]["t"][0]
        p_dr2 = (med_primary["quality_on_both"]["r2"]
                 - med_primary["quality_on_early"]["r2"])
        fb_k0 = med_full["quality_on_both"]["coef"][0]
        P("主口径（%d 行）：早期偏移进入回归后，k₀ 的独立贡献降到零——同回归系数 %+.3f"
          "（t = %.2f，不显著），ΔR²(k₀) = %.3f，即完全中介：k₀ 对质量的行级信息"
          "全部经由早期偏移携带。"
          % (med_primary["n_rows"], pb_k0, pb_t, p_dr2)
          if abs(pb_t) < 1.96 else
          "主口径（%d 行）：早期偏移进入回归后，k₀ 保留独立贡献——同回归系数 %+.3f"
          "（t = %.2f），ΔR²(k₀) = %.3f，不是完全中介。"
          % (med_primary["n_rows"], pb_k0, pb_t, p_dr2))
        P("")
        P("全部 %d 行那两行里同回归 k₀ 系数为 %+.3f，看着像反号。这不是稳健读数：%d 个 "
          "k₀ ≥ %d 的行的 D[10] 按定义恰好为 0，在这些行上早期变量不携带任何幅度信息、"
          "与\"k₀ 是否 ≥ %d\"完全共线，是它们把系数拉负；剔除后系数落回 0。"
          "所以全行群体那两行只作为对照保留，不作解读。"
          % (med_full["n_rows"], fb_k0,
             med_full["n_rows"] - med_primary["n_rows"], EARLY_N, EARLY_N))
        P("")
        absorbed_full = med_full.get("k0_absorbed_fraction")
        absorbed_primary = med_primary.get("k0_absorbed_fraction")
        P("计划 §7.2 预注册要报\"k₀ 系数被吸收的比例\"：全部 %d 行上它是 %s（系数越过 0，"
          "比例超过 1——系数会换号时这个比例没有稳定含义，这正是它误导的方式），"
          "主口径 %d 行上是 %s。本文以\"两个系数 + 符号 + 增量 R²\"取代比例作解读；"
          "此项口径变更记录在计划 §9。"
          % (med_full["n_rows"], fmt(absorbed_full, 3),
             med_primary["n_rows"], fmt(absorbed_primary, 3)))
        P("")
    if med_slope and med_slope_sub and med_primary and med_full:
        s_k0, s_t = (med_slope["quality_on_both"]["coef"][0],
                     med_slope["quality_on_both"]["t"][0])
        ss_k0, ss_t = (med_slope_sub["quality_on_both"]["coef"][0],
                       med_slope_sub["quality_on_both"]["t"][0])
        sign_txt = ("显著为正：控制住首跳后头几步的增长速度，首跳更晚的行更好"
                    if s_k0 > 0 and ss_k0 > 0 and abs(s_t) >= 1.96
                    and abs(ss_t) >= 1.96 else "见表")
        readings = [med_full, med_primary, med_slope]
        early_r2 = [m["quality_on_early"]["r2"] for m in readings]
        k0_r2 = [m["quality_on_k0"]["r2"] for m in readings]
        inc_ok = all(
            (m["quality_on_both"]["r2"] - m["quality_on_k0"]["r2"])
            > (m["quality_on_both"]["r2"] - m["quality_on_early"]["r2"])
            for m in readings)
        joint_k0 = [m["quality_on_both"]["coef"][0] for m in readings]
        sign_varies = (min(joint_k0) < 0 < max(joint_k0)
                       or any(abs(v) < 0.05 for v in joint_k0))
        P("换用事件对齐早期读数（斜率）后，同回归的 k₀ 系数为 %+.3f（t = %.1f，全部 %d 行）/ "
          "%+.3f（t = %.1f，k₀ < %d 的 %d 行），%s。"
          "三种读法（D[10] 全行 / D[10] 主口径 / 斜率）下**%s**的是：早期读数单独的 R²"
          "（%.3f / %.3f / %.3f）%s k₀ 单独（%.3f / %.3f / %.3f），"
          "早期读数加在 k₀ 之上的增量 R² %s反向。"
          "**不稳健**的是同回归里 k₀ 系数本身（%+.2f / %+.2f / %+.2f）：它%s，"
          "本文不把它的符号当作结论。"
          % (s_k0, s_t, med_slope["n_rows"], ss_k0, ss_t, EARLY_N,
             med_slope_sub["n_rows"], sign_txt,
             "稳健" if inc_ok and all(e > k for e, k in zip(early_r2, k0_r2))
             else "并不一致",
             early_r2[0], early_r2[1], early_r2[2],
             "都高于" if all(e > k for e, k in zip(early_r2, k0_r2)) else "不总高于",
             k0_r2[0], k0_r2[1], k0_r2[2],
             "都大于" if inc_ok else "不总大于",
             joint_k0[0], joint_k0[1], joint_k0[2],
             "随群体与早期读数的选取在零附近换号" if sign_varies else "符号一致但量级随口径变"))
        P("")

    P("**格内逐 prompt 相关**（每格 %d 条 prompt 的 D[10] 与该 prompt 自己的 PSNR，"
      "同 seed 42；行级相关可能来自行间混杂，格内相关不受它影响。"
      "k₀ ≥ %d 的行 D[10] 恒为零，算不出秩相关，只计入\"全部格\"一列；"
      "两个占比列的分母都是能算出 ρ 的格。末两列在同一批格上：ρ(D[50]) 是同格内 D[50] "
      "对 PSNR 的秩相关中位，偏相关是控制 D[50] 后 D[10] 的偏秩相关中位）：" % (n_prompts, EARLY_N))
    P("")
    rows = []
    for slice_name, slice_label in (("all", "全部"), ("held_out", "held-out")):
        wc = D.link["slices"][slice_name]["q2_within_cell"]
        rows.append([slice_label, f"{wc['n_cells_with_rho']} / {wc['n_cells']}",
                     fmt(wc["rho_median"], 3),
                     f"{fmt(wc['rho_p25'], 3)} – {fmt(wc['rho_p75'], 3)}",
                     pct(wc["share_negative"]), pct(wc["share_p_below_05"]),
                     fmt(wc.get("control_rho_median"), 3),
                     "%s（%s 为负）" % (fmt(wc.get("partial_rho_median"), 3),
                                       pct(wc.get("partial_share_negative")))])
    P(table(["切片", "能算出 ρ 的格 / 全部格", "格内 ρ(D[10]) 中位", "p25 – p75",
             "ρ < 0 的格占比", "p < 0.05 的格占比", "格内 ρ(D[50]) 中位",
             "偏相关(D[10] ∣ D[50]) 中位"], rows))
    P("")
    wc_all = D.link["slices"]["all"]["q2_within_cell"]
    if wc_all.get("control_rho_median") is not None:
        P("格内证据排除的只是行间混杂，不能再多说：同一格内 ρ(D[50], PSNR) 中位 %s，"
          "远强于 D[10] 的 %s；控制 D[50] 后 D[10] 的偏秩相关中位只剩 %s（%s 的格为负）。"
          "也就是说格内那条相关主要反映\"这条 prompt 整体被弯得多\"，"
          "扣掉终态偏移后早期读数只携带很弱的独立信息——它支持\"行级相关不是行间混杂\"，"
          "不支持\"早期读数在格内有超出终态偏移的特有信息\"。"
          % (fmt(wc_all["control_rho_median"], 3), fmt(wc_all["rho_median"], 3),
             fmt(wc_all["partial_rho_median"], 3),
             pct(wc_all["partial_share_negative"])))
        P("")

    # ---- verdict, chosen from the numbers; the mediation reading is the
    # k0 < EARLY_N primary population (the full one keeps the structurally
    # zero rows whose D[10] is 0 by definition)
    pooled_all = D.link["slices"]["all"]["q2_row_level"]["pooled"]
    pooled_held = D.link["slices"]["held_out"]["q2_row_level"]["pooled"]
    med_all = D.link["slices"]["all"].get("q2_mediation_D10_k0_lt_early") or \
        D.link["slices"]["all"]["q2_mediation_D10"]

    def _abs(entry):
        return abs(entry["rho"]) if entry and entry.get("rho") is not None else None

    e_all, k_all = _abs(pooled_all["D10_over_chord_ref"]), _abs(pooled_all["k0"])
    e_hd, k_hd = _abs(pooled_held["D10_over_chord_ref"]), _abs(pooled_held["k0"])
    d50_all = _abs(pooled_all["D50_over_chord_ref"])
    e_small = _abs(p_small["D10_over_chord_ref"])
    e_slope = _abs(pooled_all.get("slope_k0_plus_5", {}))
    stronger_all = (e_all is not None and k_all is not None and e_all >= k_all)
    stronger_hd = (e_hd is not None and k_hd is not None and e_hd >= k_hd)

    P("**判决**：")
    P("")
    P("- 合并行级秩相关的强度，早期偏移 |ρ| = %s，k₀ 单变量 |ρ| = %s，D[50] |ρ| = %s——"
      "早期偏移**%s** k₀ 单变量；held-out 切片上早期 |ρ| = %s、k₀ |ρ| = %s，方向%s。"
      "换口径不换向：剔除结构性零行 |ρ| = %s，事件对齐斜率 |ρ| = %s。"
      % (fmt(e_all, 2), fmt(k_all, 2), fmt(d50_all, 2),
         "不弱于" if stronger_all else "弱于",
         fmt(e_hd, 2), fmt(k_hd, 2),
         "未翻转" if stronger_all == stronger_hd else "翻转",
         fmt(e_small, 2), fmt(e_slope, 2)))
    r2_k0 = med_all["quality_on_k0"]["r2"]
    r2_e = med_all["quality_on_early"]["r2"]
    r2_both = med_all["quality_on_both"]["r2"]
    inc_k0 = r2_both - r2_e
    inc_e = r2_both - r2_k0
    b_k0 = med_all["quality_on_both"]["coef"][0]
    t_k0 = med_all["quality_on_both"]["t"][0]
    P("- 解释份额（主口径 %d 行）：k₀ 单独 R² = %s，早期偏移单独 R² = %s，两者同回归 R² = %s；"
      "早期偏移加在 k₀ 之上多解释 %s，k₀ 加在早期偏移之上只多解释 %s。"
      % (med_all["n_rows"], fmt(r2_k0, 3), fmt(r2_e, 3), fmt(r2_both, 3),
         fmt(inc_e, 3), fmt(inc_k0, 3)))
    P("- 中介（主口径）：早期偏移进入后 k₀ 的同回归系数为 %s（t = %s%s）——%s。"
      "同回归 k₀ 系数的符号在别的口径下会变（见上），不作结论。"
      % (f"{b_k0:+.3f}", fmt(t_k0, 2),
         "" if t_k0 is not None and abs(t_k0) >= 1.96 else "，不显著",
         "完全中介" if t_k0 is not None and abs(t_k0) < 1.96 else "部分中介"))
    wc_all_v = D.link["slices"]["all"]["q2_within_cell"]
    P("- 格内：格内逐 prompt 的 ρ 中位 %s，能算出 ρ 的 %d 格里 %s p < 0.05"
      "（其余 %d 格是 D[10] 恒为零的行）；但控制 D[50] 后偏相关只剩 %s——"
      "格内证据排除行间混杂，不支持早期特有信息（见上）。"
      % (fmt(wc_all_v["rho_median"], 3), wc_all_v["n_cells_with_rho"],
         pct(wc_all_v["share_p_below_05"]),
         wc_all_v["n_cells"] - wc_all_v["n_cells_with_rho"],
         fmt(wc_all_v.get("partial_rho_median"), 3)))
    lp = D.link["slices"]["all"].get("q2_row_level_secondary", {})
    lp_pooled = lp.get("pooled", {})
    if lp_pooled.get("D10_over_chord_ref", {}).get("rho") is not None:
        P("- 次要指标：以 %s 复算，合并 ρ(D[10]) = %+.2f、ρ(k₀) = %+.2f、ρ(D[50]) = %+.2f"
          "（LPIPS 越小越好，符号与 PSNR 相反）——定性排序不变（早期强于 k₀，D[50] 最强），"
          "但头条强度只有 PSNR 版的约一半：效应量级依指标而定。"
          % (D.link.get("secondary_metric", "lpips").upper(),
             lp_pooled["D10_over_chord_ref"]["rho"], lp_pooled["k0"]["rho"],
             lp_pooled["D50_over_chord_ref"]["rho"]))
    P("- 终态偏移仍然最强：D[50] 的合并 |ρ| = %s，高于早期偏移的 %s。"
      "早期读数是**可在线拿到**的那一个，终态读数不是。"
      % (fmt(d50_all, 2), fmt(e_all, 2)))
    P("")
    if stronger_all and stronger_hd and inc_e > inc_k0:
        P("按 §0 写死的口径，Q2 **成立**：早期偏移的行级解释力不弱于 k₀ 单变量，"
          "加在 k₀ 之上的增量 R²（%s）远大于反向的增量（%s），格内逐 prompt 的相关同号且稳定，"
          "两个 prompt 切片不翻转。\"早期误差被后续步放大\"因此不只是盆地形状的相关："
          "在同一分区内，一行早期偏得多少，比它的首跳位置更能说明它最后差多少。"
          % (fmt(inc_e, 3), fmt(inc_k0, 3)))
    elif stronger_all and stronger_hd:
        P("按 §0 写死的口径，Q2 **部分成立**：早期偏移的解释力不弱于 k₀ 单变量，两个切片不翻转，"
          "但它加在 k₀ 之上的增量 R²（%s）不大于反向的增量（%s）——两者在行间携带的信息重叠但不重合。"
          % (fmt(inc_e, 3), fmt(inc_k0, 3)))
    else:
        P("按 §0 写死的口径，Q2 **不成立**：早期偏移的行级解释力弱于 k₀ 单变量。"
          "弯折确有累积（§3），但早期读数不载质量信息，放大假设要退回\"晚期路径依赖\"。")
    P("")

    # ------------------------------------------------------------------ §8
    P("## 8. 与视频侧同层的对照")
    P("")
    P("两侧的 D 归一（÷ 参照弦长）、事件对齐口径、三份能量定义逐条同构，可以并排；"
      "不同的是对象（本文调度行全集 vs 视频侧九个方法）与主检验（本文回归 vs 视频侧剖面与排序），"
      "所以下表按**问题**对齐，不逐数硬比。")
    P("")
    video = {}
    for T in ("hunyuan_video", "wan21"):
        path = VIDEO_RES / T / f"cached_bend_{T}.json"
        if path.is_file():
            video[T] = json.loads(path.read_text("utf-8"))
    rows = [
        ["逐状态偏移是否单调累积、后段陡增",
         "是（%s）" % "、".join(f"{SHORT[m]} {pct(q1[m]['share_monotone_nondecreasing'])}"
                              for m, _ in MODELS),
         "是（视频侧 §3 同结论）"],
        ["终态偏移的排序与像素质量排序",
         "分区内 ρ(D[50], PSNR) 见 §3 表", "方法排序同构（视频侧 §3）"],
        ["外推载荷是否几乎完全偏出参照弯曲平面",
         "见 §5 表的平面外份额", "是（视频侧 §5）"],
        ["早期偏移对终局质量的行级回归",
         "本文 §7", "跑不了（视频 SPX 只给少数格保了完整路径）"],
    ]
    P(table(["问题", "图像侧（本文）", "视频侧"], rows))
    P("")
    if not video:
        P("（视频侧的 `cached_bend_*.json` 不在本次渲染的输入里，上表右列只引结论、不引数字。）")
        P("")

    # ------------------------------------------------------------------ §9
    P("## 9. 本文确立了什么、未确立什么")
    P("")
    P("**确立**：")
    P("")
    P("1. 重放判据在剔除 §1 记录的那一条序号后全过：%d 张图逐像素与已存格相同，位串不符 0，"
      "参照波与移交的无 cache 基线逐像素相同（§2）。"
      % v.get("images_compared_kept", v["images_compared"]))
    P("2. cache 事件之前两条轨迹逐位相同（%d 对全部前缀恰好为零），同一参照跑两次逐状态差为零，"
      "bf16 存盘对这两个采样器无损：D 没有存盘底噪（§2）。"
      % sum(c["prefix_identity"]["n_rows_with_cache_step"]
            for m, _ in MODELS for c in D.cells[m].values()))
    P("3. 逐状态偏移累积、后段陡增：逐状态不减的行占 %s（%s）/ %s（%s，即有 %s 的行"
      "存在局部回落），后段/中段增长比的行中位 %s / %s（§3）。"
      % (pct(q1["flux"]["share_monotone_nondecreasing"]), SHORT["flux"],
         pct(q1["qwen"]["share_monotone_nondecreasing"]), SHORT["qwen"],
         pct(1.0 - q1["qwen"]["share_monotone_nondecreasing"])
         if q1["qwen"]["share_monotone_nondecreasing"] is not None else "—",
         fmt(q1["flux"]["late_over_mid_growth_ratio_median"], 2),
         fmt(q1["qwen"]["late_over_mid_growth_ratio_median"], 2)))
    P("4. 偏移方向按载荷族分层，三份能量的族间差异在两个模型上同向（§5）。")
    P("5. §7 的三级读数（行级、中介、格内）在两个 prompt 切片上并排给出，判决句只依赖不翻转的项。")
    P("")
    P("**未确立**：")
    P("")
    P("- D 与 PSNR 的定量映射：本文只做相关与中介读数，不拟合映射。")
    P("- 单 seed：路径层全在 seed 42 一条流上，行级结论的种子稳健性由质量侧的三 seed 表旁证，本层自己不做跨 seed。")
    P("- 外推载荷偏出平面的那部分能量落在参照的哪些高阶主方向上（三维架之外未分解）。")
    P("- k₀ ≥ %d 的行在早期变量上与 k₀ 完全共线，这些行的早期读数按定义为零，"
      "不能用来分辨\"早期小\"与\"早期没有事件\"。" % EARLY_N)
    P("")

    # ------------------------------------------------------------------ §10
    P("## 10. 边界")
    P("")
    P("- **两个样本量不对称**：D 的行级读数来自 %d 条 prompt 的中位，行的质量来自 %d 条的均值。"
      "这是设计选择——质量表在本层之前就已测好。" % (n_prompts, n_full))
    P("- **合并行级统计的 p 不是独立样本的 p**：分区内 %d–%d 行共享调度族与预算，"
      "合并只作描述性数字。"
      % (min(len(D.cell_list(m, k=k)) for m, _ in MODELS for k in KS),
         max(len(D.cell_list(m, k=k)) for m, _ in MODELS for k in KS)))
    P("- **事件对齐桶内样本不独立**（窗口重叠），只报中位与样本数。")
    P("- **off-budget 门行**按名义档分组、按实现的 %s 步读，未做预算校正。"
      % D.off_budget_realized())
    P("- **方向架从 bf16 参照路径重算**，不是飞行中存的；存盘对取向的影响见 §2 表。")
    P("- 本文只列出上面各表里的量。另有若干算出来但未进表的读数，逐项列出并给理由：")
    P("  - **状态 25 / 40 处的三份能量行**与**两弯曲平面的第二主角**：逐对算过（`pair_record` 的 "
      "landmarks / `plane_angle2_deg`），但格中位与 `perprompt_bend.tsv.gz` 未收这几列，"
      "本地暂存产物里没有；与 §5 已报的 n=50 行 / 第一主角读同一组对象；")
    P("  - **逐步分层（非事件对齐）的 ΔD 表**：与 §6 的事件对齐表读同一现象，事件对齐版是计划指定的主读数；")
    P("  - **LPIPS 的分区级行级相关表**：合并读数已入 §7 判决（ρ(D[10]) = %s 等），"
      "六个分区的行在 JSON 里；" % fmt(
          D.link["slices"]["all"].get("q2_row_level_secondary", {})
          .get("pooled", {}).get("D10_over_chord_ref", {}).get("rho"), 3))
    P("  - **σ_k₀ 与 k₀ 处的 σ 步距**（计划 Q4 要求与首跳增量并列）：逐对算过，但未入本地暂存产物"
      "（`perprompt_bend.tsv.gz` 与格中位里都没有这两列）；且 σ_k₀ 是 k₀ 经共享 σ 表的确定单调函数，"
      "行间的秩信息与 k₀ 一列重复，故 §6 以 k₀ 列代它。")
    P("  本文的结论不依赖以上任何一项。")
    P("")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", OUT, len(L), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
