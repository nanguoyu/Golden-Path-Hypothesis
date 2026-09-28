#!/usr/bin/env python3
"""Extract and compare native-gate schedule paths from Stage E decisions."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METHODS = ("seacache", "teacache", "sencache", "dicache")
MODELS = ("flux", "qwen")
DATASETS = ("drawbench_full", "parti_full", "geneval_style")
SEED_STREAMS = ("S0", "S1", "S2")
TARGET_K = (29, 37, 41)
METHOD_LABELS = {
    "seacache": "SeaCache",
    "teacache": "TeaCache",
    "sencache": "SenCache",
    "dicache": "DiCache",
}
MODEL_LABELS = {"flux": "FLUX.1-dev", "qwen": "Qwen-Image"}
DATASET_LABELS = {
    "drawbench_full": "DrawBench",
    "parti_full": "PartiPrompts full",
    "geneval_style": "GenEval-style",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def step_action(step: Mapping[str, Any]) -> str:
    action = step.get("action")
    if action in {"full", "cache"}:
        return str(action)
    return "cache" if int(step.get("u", 0)) else "full"


def schedule_from_payload(payload: Mapping[str, Any], source: Path) -> str:
    steps = payload.get("per_step")
    if not isinstance(steps, list):
        steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) != 50:
        raise ValueError(f"{source}: expected 50 per-step decisions")
    ordered = sorted(steps, key=lambda item: int(item["step"]))
    if [int(item["step"]) for item in ordered] != list(range(50)):
        raise ValueError(f"{source}: decision steps are not exactly 0..49")
    return "".join("1" if step_action(step) == "cache" else "0" for step in ordered)


def cache_steps(schedule: str) -> list[int]:
    return [index for index, bit in enumerate(schedule) if bit == "1"]


def full_steps(schedule: str) -> list[int]:
    return [index for index, bit in enumerate(schedule) if bit == "0"]


def format_steps(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in values)


def hamming(left: str, right: str) -> int:
    if len(left) != 50 or len(right) != 50:
        raise ValueError("schedule distance requires two 50-bit schedules")
    return sum(a != b for a, b in zip(left, right))


def cache_jaccard_distance(left: str, right: str) -> float:
    left_set = set(cache_steps(left))
    right_set = set(cache_steps(right))
    union = left_set | right_set
    return 0.0 if not union else 1.0 - len(left_set & right_set) / len(union)


def ranked_counter(counter: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def collect(args: argparse.Namespace) -> int:
    rows = [
        row
        for row in read_tsv(args.run_stats)
        if row["method"] in METHODS
        and row["method_kind"] == "dynamic_native"
        and row["eval_cluster"] == args.cluster
    ]
    if not rows:
        raise ValueError(f"no native dynamic rows assigned to {args.cluster}")

    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        run_dir = Path(os.path.expandvars(row["run_dir"]))
        paths = sorted(run_dir.glob("decisions_*.json"))
        expected = int(row["decision_count"])
        if len(paths) != expected:
            raise ValueError(
                f"{row['cell_id']}: found {len(paths)} decisions, expected {expected}"
            )

        schedules: Counter[str] = Counter()
        prompt_indices: set[int] = set()
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            prompt_idx = int(payload.get("prompt_idx", -1))
            if prompt_idx < 0 or prompt_idx in prompt_indices:
                raise ValueError(f"{path}: invalid or duplicate prompt_idx={prompt_idx}")
            prompt_indices.add(prompt_idx)
            schedule = schedule_from_payload(payload, path)
            summary = payload.get("summary")
            if not isinstance(summary, Mapping):
                summary = payload
            reported = int(summary.get("n_cached", -1))
            observed = schedule.count("1")
            if reported != observed:
                raise ValueError(
                    f"{path}: reported n_cached={reported}, observed {observed}"
                )
            schedules[schedule] += 1

        total = sum(schedules.values())
        for rank, (schedule, count) in enumerate(ranked_counter(schedules), start=1):
            output.append(
                {
                    "cell_id": row["cell_id"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "seed_stream": row["seed_stream"],
                    "base_seed": row["base_seed"],
                    "method": row["method"],
                    "target_k": row["target_k"],
                    "prompt_count": total,
                    "unique_schedules": len(schedules),
                    "rank": rank,
                    "schedule": schedule,
                    "cache_count": schedule.count("1"),
                    "count": count,
                    "mass": count / total,
                    "cache_steps": format_steps(cache_steps(schedule)),
                    "full_steps": format_steps(full_steps(schedule)),
                    "run_dir": str(run_dir),
                }
            )
        print(
            f"[{index:03d}/{len(rows):03d}] {row['cell_id']}: "
            f"{total} decisions, {len(schedules)} paths",
            flush=True,
        )

    fields = (
        "cell_id",
        "model",
        "dataset",
        "seed_stream",
        "base_seed",
        "method",
        "target_k",
        "prompt_count",
        "unique_schedules",
        "rank",
        "schedule",
        "cache_count",
        "count",
        "mass",
        "cache_steps",
        "full_steps",
        "run_dir",
    )
    write_tsv(args.output, output, fields)
    print(f"[OK] wrote {args.output}: {len(rows)} cells, {len(output)} path rows")
    return 0


def path_record(
    *,
    key: tuple[str, str, int, str],
    rank: int,
    schedule: str,
    count: int,
    total: int,
) -> dict[str, Any]:
    model, dataset, target_k, method = key
    return {
        "model": model,
        "dataset": dataset,
        "target_k": target_k,
        "method": method,
        "rank": rank,
        "schedule": schedule,
        "cache_count": schedule.count("1"),
        "count": count,
        "mass": count / total,
        "cache_steps": format_steps(cache_steps(schedule)),
        "full_steps": format_steps(full_steps(schedule)),
    }


def top_distance_rows(
    *,
    scope: str,
    model: str,
    dataset: str,
    target_k: int,
    method_a: str,
    method_b: str,
    top_a: Sequence[Mapping[str, Any]],
    top_b: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left in top_a:
        for right in top_b:
            schedule_a = str(left["schedule"])
            schedule_b = str(right["schedule"])
            distance = hamming(schedule_a, schedule_b)
            k_a = schedule_a.count("1")
            k_b = schedule_b.count("1")
            count_gap = abs(k_a - k_b)
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "dataset": dataset,
                    "target_k": target_k,
                    "method_a": method_a,
                    "method_b": method_b,
                    "rank_a": int(left["rank"]),
                    "rank_b": int(right["rank"]),
                    "schedule_a": schedule_a,
                    "schedule_b": schedule_b,
                    "mass_a": float(left["mass"]),
                    "mass_b": float(right["mass"]),
                    "cache_count_a": k_a,
                    "cache_count_b": k_b,
                    "hamming": distance,
                    "normalized_hamming": distance / 50.0,
                    "cache_count_gap": count_gap,
                    "paired_swaps": (distance - count_gap) / 2.0,
                    "cache_jaccard_distance": cache_jaccard_distance(
                        schedule_a, schedule_b
                    ),
                }
            )
    return rows


def distance_summary(
    detail: Sequence[Mapping[str, Any]],
    top_a: Sequence[Mapping[str, Any]],
    top_b: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not detail or not top_a or not top_b:
        raise ValueError("cannot summarize an empty top-path comparison")
    lookup = {
        (int(row["rank_a"]), int(row["rank_b"])): int(row["hamming"])
        for row in detail
    }
    size = min(len(top_a), len(top_b), 3)
    left = list(top_a[:size])
    right = list(top_b[:size])
    matching_costs = []
    for permutation in itertools.permutations(right):
        matching_costs.append(
            sum(
                lookup[(int(a["rank"]), int(b["rank"]))]
                for a, b in zip(left, permutation)
            )
            / size
        )

    mass_a_total = sum(float(row["mass"]) for row in top_a)
    mass_b_total = sum(float(row["mass"]) for row in top_b)
    weighted = 0.0
    for row in detail:
        weight_a = float(row["mass_a"]) / mass_a_total
        weight_b = float(row["mass_b"]) / mass_b_total
        weighted += weight_a * weight_b * int(row["hamming"])

    schedules_a = {str(row["schedule"]) for row in top_a}
    schedules_b = {str(row["schedule"]) for row in top_b}
    top1 = next(
        row
        for row in detail
        if int(row["rank_a"]) == 1 and int(row["rank_b"]) == 1
    )
    return {
        "top1_hamming": int(top1["hamming"]),
        "top1_normalized_hamming": float(top1["normalized_hamming"]),
        "top1_cache_count_a": int(top1["cache_count_a"]),
        "top1_cache_count_b": int(top1["cache_count_b"]),
        "top1_cache_count_gap": int(top1["cache_count_gap"]),
        "top1_paired_swaps": float(top1["paired_swaps"]),
        "top1_cache_jaccard_distance": float(top1["cache_jaccard_distance"]),
        "min_top3_hamming": min(int(row["hamming"]) for row in detail),
        "optimal_matching_hamming_mean": min(matching_costs),
        "top3_conditional_weighted_hamming": weighted,
        "exact_shared_top3_paths": len(schedules_a & schedules_b),
    }


def build_distance_tables(
    top_groups: Mapping[tuple[str, str, int, str], Sequence[Mapping[str, Any]]],
    *,
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    settings = sorted({key[:3] for key in top_groups})
    for model, dataset, target_k in settings:
        for method_a, method_b in itertools.combinations(METHODS, 2):
            top_a = top_groups[(model, dataset, target_k, method_a)][:3]
            top_b = top_groups[(model, dataset, target_k, method_b)][:3]
            detail = top_distance_rows(
                scope=scope,
                model=model,
                dataset=dataset,
                target_k=target_k,
                method_a=method_a,
                method_b=method_b,
                top_a=top_a,
                top_b=top_b,
            )
            summary = {
                "scope": scope,
                "model": model,
                "dataset": dataset,
                "target_k": target_k,
                "method_a": method_a,
                "method_b": method_b,
                **distance_summary(detail, top_a, top_b),
            }
            detail_rows.extend(detail)
            summary_rows.append(summary)
    return detail_rows, summary_rows


def render_top_paths(rows: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for row in rows[:3]:
        parts.append(
            f"#{int(row['rank'])} {100 * float(row['mass']):.1f}% "
            f"`{row['schedule']}`; F={{{row['full_steps']}}}"
        )
    return "<br>".join(parts)


def render_report(
    *,
    dataset_summaries: Sequence[Mapping[str, Any]],
    dataset_paths: Sequence[Mapping[str, Any]],
    dataset_distances: Sequence[Mapping[str, Any]],
    equal_paths: Sequence[Mapping[str, Any]],
    equal_distances: Sequence[Mapping[str, Any]],
    cross_dataset_distances: Sequence[Mapping[str, Any]],
) -> str:
    summary_map = {
        (
            row["model"],
            row["dataset"],
            int(row["target_k"]),
            row["method"],
        ): row
        for row in dataset_summaries
    }
    path_map: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in dataset_paths:
        path_map[
            (
                str(row["model"]),
                str(row["dataset"]),
                int(row["target_k"]),
                str(row["method"]),
            )
        ].append(row)
    equal_map: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in equal_paths:
        equal_map[
            (str(row["model"]), int(row["target_k"]), str(row["method"]))
        ].append(row)

    lines = [
        "# Native Dynamic Gates 的完整 Path 分布与距离",
        "",
        "日期：2026-07-29",
        "",
        "本文只分析 SeaCache、TeaCache、SenCache、DiCache 的 native gate decisions。"
        "每条 schedule 是 50 位 bitstring：`1=cache`，`0=full`。",
        "",
        "## 统计定义",
        "",
        "- `U(S0/S1/S2)`：三条 generation-seed streams 各自出现的 unique paths 数；",
        "- `U pooled`：一个数据集中所有 prompts × 三条 seed streams 合并后的并集大小；",
        "- Top-1/Top-3 mass：pooled 样本中最高频一条/三条 paths 的累计比例；",
        "- `F={...}`：该 schedule 执行 full transformer 的 step indices；",
        "- Hamming：两个 50-bit schedules 不同的 steps 数；",
        "- paired swaps：扣除 cache-count 差后，需要成对交换 full/cache 的数量；",
        "- best-match：两个 Top-3 集合做一一匹配时的最小平均 Hamming；",
        "- 数据集等权结果先在每个数据集内部归一化 path mass，再对三个数据集取平均，"
        "避免 PartiPrompts 因 prompts 更多而支配结果。",
        "",
        "## 数据覆盖与完整性",
        "",
        "本报告覆盖：",
        "",
        "- 两个模型：FLUX.1-dev、Qwen-Image；",
        "- 三个完整数据集：DrawBench 200、PartiPrompts full 1632、"
        "GenEval-style 553；",
        "- 三档目标 budget：\\(K=29/37/41\\)，即 cache ratio "
        "\\(0.58/0.74/0.82\\)；",
        "- 四个 native dynamic gates：SeaCache、TeaCache、SenCache、DiCache；",
        "- 每个 setting 三条 generation-seed streams。",
        "",
        "因此共核对 \\(2\\times3\\times3\\times4\\times3=216\\) 个 cell、"
        "171,720 份 `decisions_*.json`。所有文件均成功解析；每条记录都核对了 50 个 "
        "step decisions，并确认从 decisions 重建的 cache count 与原始 summary 一致。"
        "这里没有使用 exact-budget closure；表中的路径是各方法 native gate 实际产生的"
        "路径。native gate 偶尔会落在目标 \\(K\\) 的相邻值，因此所有距离表同时保留 "
        "cache-count gap 和 `paired swaps` 口径。",
        "",
        "本文列出每个 setting 的 Top-3 bitstrings 和 full-step 集合。全部 1,023 条 "
        "pooled unique paths（包括低频路径）、频数、mass、cache/full steps 位于 "
        "[`dataset_path_counts.tsv`](../resources/"
        "cross_model_multiseed_stage_e_native_paths/dataset_path_counts.tsv)；逐 seed "
        "stream 的 2,569 条 path-frequency rows 位于 "
        "[`per_seed_path_counts.tsv`](../resources/"
        "cross_model_multiseed_stage_e_native_paths/per_seed_path_counts.tsv)。",
        "",
        "## 核心发现",
        "",
        "1. **同一 gate 的 schedule family 跨数据集高度稳定。**在 72 组"
        "“同模型、同 \\(K\\)、同 gate、两个不同数据集”的比较中，60 组（83.3%）"
        "Top-1 bitstring 完全相同；69 组（95.8%）至少共享两条 Top-3 paths；41 组"
        "（56.9%）三条 Top-3 paths 全部相同。换数据集通常改变的是 family 成员的频率"
        "和排序，而不是重新产生完全不同的主路径。",
        "2. **不同 gates 通常形成不同的稳定 families。**在 36 组"
        "“同模型、同 \\(K\\)、不同 gate”的三数据集等权比较中，没有一组共享完全相同的"
        " Top-3 bitstring。Top-1 Hamming 为 8--27 steps。即使 \\(K=41\\) 时绝对 "
        "Hamming 降到 8--14，折算为 full-anchor swaps 后仍相当于替换 9 个 full "
        "steps 中的 4--7 个；这不是“所有方法在高 cache ratio 自动收敛到同一路径”。",
        "3. **SeaCache 最常呈现高度集中的 native family，但并非唯一能集中。**FLUX "
        "\\(K=29\\) 的 SeaCache Top-1 在三个数据集占 84.7%--89.0%，Top-3 占 "
        "97.0%--99.5%；Qwen \\(K=37\\) 更接近单一路径，Top-1 为 "
        "98.0%--98.9%。另一方面，Qwen \\(K=29\\) 的 TeaCache、FLUX \\(K=37\\) "
        "的 DiCache 也形成高度集中的、但与 SeaCache 明显不同的 families。",
        "4. **unique-path 数必须和 path mass 一起解释。**PartiPrompts prompts 更多，"
        "因而通常观察到更多低频 unique paths；这不必然表示主 family 更弱。例如 Qwen "
        "\\(K=41\\) SeaCache 在三个数据集 pooled \\(U=9/11/3\\)，但 Top-1 都约为 "
        "77%，Top-3 为 94.6%--100%。相反，FLUX \\(K=29\\) DiCache 的 Top-3 "
        "identities 跨数据集稳定，但 Top-3 mass 只有 40.7%--51.2%，大量 prompts "
        "仍散布在长尾路径中。",
        "5. **这些统计支持“存在 model- and budget-specific Golden Path Family”的"
        "结构证据，但频率本身不是 terminal quality。**它表明 native gates 在大量 "
        "prompts 和不同数据集上反复落入少数稳定 families；它不证明最高频 family "
        "必然终端最优，也不证明四种 gates 找到的是同一个 family。是否是“好的” "
        "Golden Path Family，必须再与同 setting 的 PSNR、SSIM、LPIPS、CLIP、"
        "ImageReward 以及 fixed-path replay/search 结果联合判断。",
        "",
        "固定路径方法与上述 native families 的统一距离、有效距离和质量对照见 "
        "[`cross_model_all_schedule_family_distances_zh.md`]"
        "(cross_model_all_schedule_family_distances_zh.md)。",
        "",
    ]

    for model in MODELS:
        lines.extend([f"## {MODEL_LABELS[model]}", ""])
        for target_k in TARGET_K:
            lines.extend(
                [
                    f"### K={target_k}/50，cache ratio={target_k / 50:.2f}",
                    "",
                ]
            )
            for dataset in DATASETS:
                lines.extend(
                    [
                        f"#### {DATASET_LABELS[dataset]}",
                        "",
                        "| Gate | U(S0/S1/S2) | U pooled | Top-1 | Top-3 | "
                        "三条主路径 |",
                        "|---|---:|---:|---:|---:|---|",
                    ]
                )
                for method in METHODS:
                    key = (model, dataset, target_k, method)
                    summary = summary_map[key]
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                METHOD_LABELS[method],
                                f"{summary['unique_s0']}/"
                                f"{summary['unique_s1']}/"
                                f"{summary['unique_s2']}",
                                str(summary["unique_pooled"]),
                                f"{100 * float(summary['top1_mass']):.1f}%",
                                f"{100 * float(summary['top3_mass']):.1f}%",
                                render_top_paths(path_map[key]),
                            ]
                        )
                        + " |"
                    )
                lines.extend(
                    [
                        "",
                        "| Gate pair | Top1 Hamming | Min Top3 Hamming | "
                        "Best-match mean | Weighted Top3 mean | Shared exact paths |",
                        "|---|---:|---:|---:|---:|---:|",
                    ]
                )
                relevant = [
                    row
                    for row in dataset_distances
                    if row["model"] == model
                    and row["dataset"] == dataset
                    and int(row["target_k"]) == target_k
                ]
                for row in relevant:
                    lines.append(
                        f"| {METHOD_LABELS[str(row['method_a'])]} / "
                        f"{METHOD_LABELS[str(row['method_b'])]} | "
                        f"{int(row['top1_hamming'])} | "
                        f"{int(row['min_top3_hamming'])} | "
                        f"{float(row['optimal_matching_hamming_mean']):.2f} | "
                        f"{float(row['top3_conditional_weighted_hamming']):.2f} | "
                        f"{int(row['exact_shared_top3_paths'])} |"
                    )
                lines.append("")

            lines.extend(
                [
                    "#### 三数据集等权主路径",
                    "",
                    "| Gate | Top-1 mass | Top-3 mass | 三条主路径 |",
                    "|---|---:|---:|---|",
                ]
            )
            for method in METHODS:
                top = equal_map[(model, target_k, method)]
                lines.append(
                    f"| {METHOD_LABELS[method]} | "
                    f"{100 * float(top[0]['mass']):.1f}% | "
                    f"{100 * sum(float(row['mass']) for row in top[:3]):.1f}% | "
                    f"{render_top_paths(top)} |"
                )
            lines.extend(
                [
                    "",
                    "| Gate pair | Top1 Hamming | Min Top3 Hamming | "
                    "Best-match mean | Weighted Top3 mean | Shared exact paths |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            relevant_equal = [
                row
                for row in equal_distances
                if row["model"] == model and int(row["target_k"]) == target_k
            ]
            for row in relevant_equal:
                lines.append(
                    f"| {METHOD_LABELS[str(row['method_a'])]} / "
                    f"{METHOD_LABELS[str(row['method_b'])]} | "
                    f"{int(row['top1_hamming'])} | "
                    f"{int(row['min_top3_hamming'])} | "
                    f"{float(row['optimal_matching_hamming_mean']):.2f} | "
                    f"{float(row['top3_conditional_weighted_hamming']):.2f} | "
                    f"{int(row['exact_shared_top3_paths'])} |"
                )
            lines.extend(
                [
                    "",
                    "#### 同一 gate 的跨数据集 Top-3 距离",
                    "",
                    "| Gate | Dataset pair | Top1 Hamming | Min Top3 Hamming | "
                    "Best-match mean | Shared exact paths |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            relevant_cross = [
                row
                for row in cross_dataset_distances
                if row["model"] == model and int(row["target_k"]) == target_k
            ]
            for row in relevant_cross:
                lines.append(
                    f"| {METHOD_LABELS[str(row['method'])]} | "
                    f"{DATASET_LABELS[str(row['dataset_a'])]} / "
                    f"{DATASET_LABELS[str(row['dataset_b'])]} | "
                    f"{int(row['top1_hamming'])} | "
                    f"{int(row['min_top3_hamming'])} | "
                    f"{float(row['optimal_matching_hamming_mean']):.2f} | "
                    f"{int(row['exact_shared_top3_paths'])} |"
                )
            lines.append("")

    return "\n".join(lines)


def aggregate(args: argparse.Namespace) -> int:
    raw_rows = [row for path in args.inputs for row in read_tsv(path)]
    cells: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        cells[row["cell_id"]].append(row)
    expected_cells = len(MODELS) * len(DATASETS) * len(SEED_STREAMS) * len(TARGET_K) * len(METHODS)
    if len(cells) != expected_cells:
        raise ValueError(f"expected {expected_cells} cells, found {len(cells)}")

    cell_counters: dict[tuple[str, str, int, str, str], Counter[str]] = {}
    normalized_rows: list[dict[str, Any]] = []
    for cell_id, group in cells.items():
        meta = group[0]
        key = (
            meta["model"],
            meta["dataset"],
            int(meta["target_k"]),
            meta["seed_stream"],
            meta["method"],
        )
        if key in cell_counters:
            raise ValueError(f"duplicate cell key: {key}")
        counter = Counter(
            {row["schedule"]: int(row["count"]) for row in group}
        )
        total = sum(counter.values())
        if total != int(meta["prompt_count"]):
            raise ValueError(f"{cell_id}: path counts do not sum to prompt_count")
        cell_counters[key] = counter
        for rank, (schedule, count) in enumerate(ranked_counter(counter), start=1):
            normalized_rows.append(
                {
                    **{field: meta[field] for field in (
                        "cell_id",
                        "model",
                        "dataset",
                        "seed_stream",
                        "base_seed",
                        "method",
                        "target_k",
                        "prompt_count",
                    )},
                    "unique_schedules": len(counter),
                    "rank": rank,
                    "schedule": schedule,
                    "cache_count": schedule.count("1"),
                    "count": count,
                    "mass": count / total,
                    "cache_steps": format_steps(cache_steps(schedule)),
                    "full_steps": format_steps(full_steps(schedule)),
                    "run_dir": meta["run_dir"],
                }
            )

    expected_keys = {
        (model, dataset, target_k, seed_stream, method)
        for model in MODELS
        for dataset in DATASETS
        for target_k in TARGET_K
        for seed_stream in SEED_STREAMS
        for method in METHODS
    }
    if set(cell_counters) != expected_keys:
        missing = sorted(expected_keys - set(cell_counters))
        raise ValueError(f"native path matrix is incomplete; missing={missing[:5]}")

    dataset_summaries: list[dict[str, Any]] = []
    dataset_paths: list[dict[str, Any]] = []
    dataset_top_groups: dict[
        tuple[str, str, int, str], list[dict[str, Any]]
    ] = {}
    dataset_counters: dict[tuple[str, str, int, str], Counter[str]] = {}
    for model in MODELS:
        for dataset in DATASETS:
            for target_k in TARGET_K:
                for method in METHODS:
                    key = (model, dataset, target_k, method)
                    counters = [
                        cell_counters[(model, dataset, target_k, stream, method)]
                        for stream in SEED_STREAMS
                    ]
                    pooled: Counter[str] = Counter()
                    for counter in counters:
                        pooled.update(counter)
                    dataset_counters[key] = pooled
                    total = sum(pooled.values())
                    ranked = ranked_counter(pooled)
                    records = [
                        path_record(
                            key=key,
                            rank=rank,
                            schedule=schedule,
                            count=count,
                            total=total,
                        )
                        for rank, (schedule, count) in enumerate(ranked, start=1)
                    ]
                    dataset_paths.extend(records)
                    dataset_top_groups[key] = records[:3]
                    dataset_summaries.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "target_k": target_k,
                            "method": method,
                            "prompt_seed_samples": total,
                            "unique_s0": len(counters[0]),
                            "unique_s1": len(counters[1]),
                            "unique_s2": len(counters[2]),
                            "unique_seed_mean": sum(map(len, counters)) / 3.0,
                            "unique_pooled": len(pooled),
                            "top1_mass": records[0]["mass"],
                            "top3_mass": sum(row["mass"] for row in records[:3]),
                        }
                    )

    dataset_detail, dataset_distance_summaries = build_distance_tables(
        dataset_top_groups, scope="dataset_pooled"
    )

    equal_paths: list[dict[str, Any]] = []
    equal_top_groups: dict[
        tuple[str, str, int, str], list[dict[str, Any]]
    ] = {}
    for model in MODELS:
        for target_k in TARGET_K:
            for method in METHODS:
                masses: dict[str, dict[str, float]] = defaultdict(dict)
                raw_counts: Counter[str] = Counter()
                for dataset in DATASETS:
                    counter = dataset_counters[(model, dataset, target_k, method)]
                    total = sum(counter.values())
                    raw_counts.update(counter)
                    for schedule, count in counter.items():
                        masses[schedule][dataset] = count / total
                ranked = sorted(
                    masses,
                    key=lambda schedule: (
                        -sum(masses[schedule].get(dataset, 0.0) for dataset in DATASETS)
                        / len(DATASETS),
                        schedule,
                    ),
                )
                records = []
                for rank, schedule in enumerate(ranked, start=1):
                    dataset_mass = {
                        dataset: masses[schedule].get(dataset, 0.0)
                        for dataset in DATASETS
                    }
                    equal_mass = sum(dataset_mass.values()) / len(DATASETS)
                    records.append(
                        {
                            "model": model,
                            "dataset": "three_datasets_equal",
                            "target_k": target_k,
                            "method": method,
                            "rank": rank,
                            "schedule": schedule,
                            "cache_count": schedule.count("1"),
                            "raw_count": raw_counts[schedule],
                            "mass": equal_mass,
                            "drawbench_mass": dataset_mass["drawbench_full"],
                            "parti_mass": dataset_mass["parti_full"],
                            "geneval_mass": dataset_mass["geneval_style"],
                            "cache_steps": format_steps(cache_steps(schedule)),
                            "full_steps": format_steps(full_steps(schedule)),
                        }
                    )
                equal_paths.extend(records)
                equal_top_groups[
                    (model, "three_datasets_equal", target_k, method)
                ] = records[:3]

    equal_detail, equal_distance_summaries = build_distance_tables(
        equal_top_groups, scope="three_datasets_equal"
    )

    cross_dataset_summaries: list[dict[str, Any]] = []
    for model in MODELS:
        for target_k in TARGET_K:
            for method in METHODS:
                for dataset_a, dataset_b in itertools.combinations(DATASETS, 2):
                    top_a = dataset_top_groups[(model, dataset_a, target_k, method)]
                    top_b = dataset_top_groups[(model, dataset_b, target_k, method)]
                    detail = top_distance_rows(
                        scope="cross_dataset_same_gate",
                        model=model,
                        dataset=f"{dataset_a}__{dataset_b}",
                        target_k=target_k,
                        method_a=method,
                        method_b=method,
                        top_a=top_a,
                        top_b=top_b,
                    )
                    cross_dataset_summaries.append(
                        {
                            "model": model,
                            "target_k": target_k,
                            "method": method,
                            "dataset_a": dataset_a,
                            "dataset_b": dataset_b,
                            **distance_summary(detail, top_a, top_b),
                        }
                    )

    output_dir = args.output_dir
    write_tsv(
        output_dir / "per_seed_path_counts.tsv",
        normalized_rows,
        (
            "cell_id",
            "model",
            "dataset",
            "seed_stream",
            "base_seed",
            "method",
            "target_k",
            "prompt_count",
            "unique_schedules",
            "rank",
            "schedule",
            "cache_count",
            "count",
            "mass",
            "cache_steps",
            "full_steps",
        ),
    )
    write_tsv(
        output_dir / "dataset_path_summary.tsv",
        dataset_summaries,
        (
            "model",
            "dataset",
            "target_k",
            "method",
            "prompt_seed_samples",
            "unique_s0",
            "unique_s1",
            "unique_s2",
            "unique_seed_mean",
            "unique_pooled",
            "top1_mass",
            "top3_mass",
        ),
    )
    write_tsv(
        output_dir / "dataset_path_counts.tsv",
        dataset_paths,
        (
            "model",
            "dataset",
            "target_k",
            "method",
            "rank",
            "schedule",
            "cache_count",
            "count",
            "mass",
            "cache_steps",
            "full_steps",
        ),
    )
    write_tsv(
        output_dir / "dataset_top3_distance_detail.tsv",
        dataset_detail,
        (
            "scope",
            "model",
            "dataset",
            "target_k",
            "method_a",
            "method_b",
            "rank_a",
            "rank_b",
            "schedule_a",
            "schedule_b",
            "mass_a",
            "mass_b",
            "cache_count_a",
            "cache_count_b",
            "hamming",
            "normalized_hamming",
            "cache_count_gap",
            "paired_swaps",
            "cache_jaccard_distance",
        ),
    )
    write_tsv(
        output_dir / "dataset_top3_distance_summary.tsv",
        dataset_distance_summaries,
        (
            "scope",
            "model",
            "dataset",
            "target_k",
            "method_a",
            "method_b",
            "top1_hamming",
            "top1_normalized_hamming",
            "top1_cache_count_a",
            "top1_cache_count_b",
            "top1_cache_count_gap",
            "top1_paired_swaps",
            "top1_cache_jaccard_distance",
            "min_top3_hamming",
            "optimal_matching_hamming_mean",
            "top3_conditional_weighted_hamming",
            "exact_shared_top3_paths",
        ),
    )
    write_tsv(
        output_dir / "equal_dataset_path_counts.tsv",
        equal_paths,
        (
            "model",
            "dataset",
            "target_k",
            "method",
            "rank",
            "schedule",
            "cache_count",
            "raw_count",
            "mass",
            "drawbench_mass",
            "parti_mass",
            "geneval_mass",
            "cache_steps",
            "full_steps",
        ),
    )
    write_tsv(
        output_dir / "equal_dataset_top3_distance_detail.tsv",
        equal_detail,
        (
            "scope",
            "model",
            "dataset",
            "target_k",
            "method_a",
            "method_b",
            "rank_a",
            "rank_b",
            "schedule_a",
            "schedule_b",
            "mass_a",
            "mass_b",
            "cache_count_a",
            "cache_count_b",
            "hamming",
            "normalized_hamming",
            "cache_count_gap",
            "paired_swaps",
            "cache_jaccard_distance",
        ),
    )
    write_tsv(
        output_dir / "equal_dataset_top3_distance_summary.tsv",
        equal_distance_summaries,
        (
            "scope",
            "model",
            "dataset",
            "target_k",
            "method_a",
            "method_b",
            "top1_hamming",
            "top1_normalized_hamming",
            "top1_cache_count_a",
            "top1_cache_count_b",
            "top1_cache_count_gap",
            "top1_paired_swaps",
            "top1_cache_jaccard_distance",
            "min_top3_hamming",
            "optimal_matching_hamming_mean",
            "top3_conditional_weighted_hamming",
            "exact_shared_top3_paths",
        ),
    )
    write_tsv(
        output_dir / "cross_dataset_top3_distance_summary.tsv",
        cross_dataset_summaries,
        (
            "model",
            "target_k",
            "method",
            "dataset_a",
            "dataset_b",
            "top1_hamming",
            "top1_normalized_hamming",
            "top1_cache_count_a",
            "top1_cache_count_b",
            "top1_cache_count_gap",
            "top1_paired_swaps",
            "top1_cache_jaccard_distance",
            "min_top3_hamming",
            "optimal_matching_hamming_mean",
            "top3_conditional_weighted_hamming",
            "exact_shared_top3_paths",
        ),
    )
    report = render_report(
        dataset_summaries=dataset_summaries,
        dataset_paths=dataset_paths,
        dataset_distances=dataset_distance_summaries,
        equal_paths=equal_paths,
        equal_distances=equal_distance_summaries,
        cross_dataset_distances=cross_dataset_summaries,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(
        f"[OK] native schedule paths: cells={len(cells)}, "
        f"dataset_paths={len(dataset_paths)}, report={args.report}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect", help="scan natural decision files on one cluster"
    )
    collect_parser.add_argument("--run-stats", type=Path, required=True)
    collect_parser.add_argument(
        "--cluster", choices=("site_a", "site_b"), required=True
    )
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.set_defaults(func=collect)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="merge cluster path counts and produce distance tables"
    )
    aggregate_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    aggregate_parser.add_argument("--report", type=Path, required=True)
    aggregate_parser.set_defaults(func=aggregate)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
