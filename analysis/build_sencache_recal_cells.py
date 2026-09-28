#!/usr/bin/env python3
"""Build the image-lane re-run cell table from the frozen frontier selection.

Every row is one generation cell: (model, dataset, budget, seed) together with
the (threshold_start, threshold_main) pair and the two ceiling knobs frozen for
that budget, the prompt file, the no-cache reference it will be evaluated
against, and the cluster that reference lives on.

Every budget lands on its target, so there is exactly one cell per
(model, dataset, budget, seed): 4 datasets x 3 budgets x 3 seeds = 36 per image
model, and 2 datasets x 3 budgets x 3 seeds = 18 for Wan2.1. The ceiling knobs
travel with the row because they differ per budget.

The `n` ablation of the plan's section 9.1.1 rides in the same table under its
own family names. It holds the budget at 29 and varies only `n`, on each model's
primary dataset and its three seed streams. `n = 10` at K29 is already a main
cell, so it adds no rows, and `n = 43` is the same gate as `n = 39` on K29's
39-step window -- the run limit refuses nothing at either -- so the ladder tops
out at one rung covering both main-cell values.

Plan: docs/sencache_recalibration_plan_zh.md S3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SEEDS = {"flux": (41, 42, 43), "qwen": (42, 100042, 200042)}
#: the dataset each model's matrix leads with, which is where the ablation sits
PRIMARY_DATASET = {"flux": "drawbench_full", "qwen": "drawbench_full"}
CANONICAL = ("k29", "k37", "k41")
FAMILIES = (("k29", "K29"), ("k37", "K37"), ("k41", "K41"),
            ("k29_n3", "K29"), ("k29_n20", "K29"), ("k29_n39", "K29"))

DATASETS = {
    ("flux", "drawbench_full"): ("resources/prompts/prompt.txt", 200, 1),
    ("flux", "parti_full"): ("resources/prompts/partiprompts_full_eval1632_seed42.txt", 1632, 2),
    ("flux", "geneval_style"): ("resources/prompts/geneval_seed43_n100.txt", 553, 1),
    ("flux", "diffusiondb_clean10k"): ("resources/prompts/diffusiondb_2m_clean_10000_seed42.txt", 10000, 2),
    ("qwen", "drawbench_full"): ("reference/hicache/code/models/qwen_image/prompts/DrawBench200.txt", 200, 1),
    ("qwen", "parti_full"): ("resources/prompts/partiprompts_full_eval1632_seed42.txt", 1632, 4),
    ("qwen", "geneval_style"): ("resources/prompts/geneval_seed43_n100.txt", 553, 2),
    ("qwen", "diffusiondb_clean10k"): ("resources/prompts/diffusiondb_2m_clean_10000_seed42.txt", 10000, 8),
}

_GP = "$DATA/cache_results/flux/golden_path_seed_fixed_site_a_6f39bf5"
_QFAM = "$DATA/cache_results/qwen_image/qwen_family_completion_v1/stage1"
_QXDS = "$DATA/cache_results/qwen_image/cross_dataset_qwen_xds_33fdaf0/originals"
_DDB = "$DATA/cache_results/cross_model_ddb_clean10k_formal_4d25951_v1"

# (model, dataset, seed) -> (no-cache reference dir, cluster holding it)
REFERENCES: dict[tuple[str, str, int], tuple[str, str]] = {}
for _seed in (41, 42, 43):
    REFERENCES[("flux", "drawbench_full", _seed)] = (
        f"{_GP}/drawbench_full_s{_seed}/original_gpseedfix_drawbench_full_n200_s{_seed}_50_site_a_6f39bf5", "site_a")
    REFERENCES[("flux", "parti_full", _seed)] = (
        f"{_GP}/parti_full_s{_seed}/original_gpseedfix_parti_full_n1632_s{_seed}_50_site_a_6f39bf5", "site_a")
    REFERENCES[("flux", "geneval_style", _seed)] = (
        f"{_GP}/geneval_style_s{_seed}/original_gpseedfix_geneval_style_n553_s{_seed}_50_site_a_6f39bf5", "site_a")
    REFERENCES[("flux", "diffusiondb_clean10k", _seed)] = (f"{_DDB}/flux/original/s{_seed}", "site_b")

REFERENCES[("qwen", "drawbench_full", 42)] = (
    "$DATA/cache_results/qwen_image/qwen_formal_23817f4_d200_original50", "site_a")
REFERENCES[("qwen", "parti_full", 42)] = (
    f"{_QXDS}/parti_full/qwen_parti_full_original50_s42_qwen_xds_33fdaf0", "site_a")
REFERENCES[("qwen", "geneval_style", 42)] = (
    f"{_QXDS}/geneval_style/qwen_geneval_style_original50_s42_qwen_xds_33fdaf0", "site_a")
for _seed, _h in ((100042, "H1"), (200042, "H2")):
    for _ds in ("drawbench_full", "parti_full", "geneval_style"):
        REFERENCES[("qwen", _ds, _seed)] = (f"{_QFAM}/{_h}/{_ds}/original_s{_seed}", "site_a")
for _seed in (42, 100042, 200042):
    REFERENCES[("qwen", "diffusiondb_clean10k", _seed)] = (f"{_DDB}/qwen/original/s{_seed}", "site_a")


WAN_SEEDS = {"penguin599": (54, 55, 56), "vbench944": (42, 43, 44)}
WAN_PROMPTS = {"penguin599": 599, "vbench944": 944}
WAN_ARRAY_TASKS = {"penguin599": 10, "vbench944": 16}
WAN_PRIMARY = "penguin599"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("image", "wan21"), default="image")
    parser.add_argument("--selection", type=Path, required=True,
                        help="frontier selection JSON (analysis/sencache_frontier.py select)")
    parser.add_argument("--out_tsv", type=Path, required=True)
    return parser.parse_args()


def build_wan(selection: dict, out_tsv: Path) -> int:
    header = ["dataset", "family", "budget", "seed", "start", "main",
              "max_skip", "switch_ratio", "n_prompts", "array_tasks"]
    lines = ["\t".join(header)]
    n_rows = 0
    for dataset in ("penguin599", "vbench944"):
        group = selection["groups"].get(dataset)
        if group is None:
            raise SystemExit(f"selection has no group for {dataset}")
        for family, budget_key in FAMILIES:
            if family not in CANONICAL and dataset != WAN_PRIMARY:
                continue
            entry = group["budgets"].get(family) or group["budgets"][budget_key]
            frozen = entry.get("frozen")
            if frozen is None:
                raise SystemExit(
                    f"{dataset} {family}: the frontier froze no pair; widen the "
                    f"threshold_main grid and re-run the sweep")
            for seed in WAN_SEEDS[dataset]:
                lines.append("\t".join((
                    dataset, family, str(entry["target"]), str(seed),
                    f"{frozen['start']:.12g}", f"{frozen['main']:.12g}",
                    str(entry["max_skip"]), f"{entry['switch_ratio']:.12g}",
                    str(WAN_PROMPTS[dataset]), str(WAN_ARRAY_TASKS[dataset]),
                )))
                n_rows += 1
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[cells] wrote {n_rows} rows -> {out_tsv}")
    return 0


def main() -> int:
    args = parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if args.lane == "wan21":
        return build_wan(selection, args.out_tsv)

    header = ["model", "family", "dataset", "budget", "seed", "start", "main",
              "max_skip", "switch_ratio",
              "prompt_file", "n_prompts", "array_tasks", "cluster", "gt_dir"]
    lines = ["\t".join(header)]
    n_rows = 0

    for model in ("flux", "qwen"):
        group = selection["groups"].get(model)
        if group is None:
            raise SystemExit(f"selection has no group for {model}")
        for family, budget_key in FAMILIES:
            entry = group["budgets"].get(family) or group["budgets"][budget_key]
            budget = entry["target"]
            frozen = entry.get("frozen")
            if frozen is None:
                raise SystemExit(
                    f"{model} {family}: the frontier froze no pair; widen the "
                    f"threshold_main grid and re-run the sweep")
            datasets = ((PRIMARY_DATASET[model],) if family not in CANONICAL
                        else ("drawbench_full", "parti_full", "geneval_style",
                              "diffusiondb_clean10k"))
            for dataset in datasets:
                prompt_file, n_prompts, array_tasks = DATASETS[(model, dataset)]
                for seed in SEEDS[model]:
                    gt_dir, cluster = REFERENCES[(model, dataset, seed)]
                    lines.append("\t".join((
                        model, family, dataset, str(budget), str(seed),
                        f"{frozen['start']:.12g}", f"{frozen['main']:.12g}",
                        str(entry["max_skip"]), f"{entry['switch_ratio']:.12g}",
                        prompt_file, str(n_prompts), str(array_tasks),
                        cluster, gt_dir,
                    )))
                    n_rows += 1

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[cells] wrote {n_rows} rows -> {args.out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
