#!/usr/bin/env python3
"""Full path census of the video baseline matrices' native dynamic gates.

The image-side counterpart is ``analysis/analyze_native_schedule_paths.py`` /
``docs/cross_model_native_gate_path_distribution_zh.md``; the statistics here
follow its definitions so the two sides read the same way:

* a *path* is the realized 50-bit cache/full string of one generation,
  rebuilt from ``decisions_*.json`` ``records[].action`` (never from summary
  counts);
* ``U(S0/S1/S2)`` are the per-seed-stream unique-path counts, ``pooled`` the
  union over the three streams of one (method, dataset, K) cell row;
* distances between two paths are reported in BOTH the ``hamming`` and the
  ``cache_count_gap`` + ``paired_swaps`` = (hamming - gap)/2 reading, because
  a native gate may realize K±1 and a plain Hamming then conflates budget
  difference with reordering;
* fixed-schedule anchors come from the frozen matrix config's schedule
  tables (budcache / meancache / shared triplet), so "how far is the modal
  gate path from the searched schedule at the same budget" is answerable.

One backbone per invocation; both write under
``resources/video_native_gate_paths/<backbone>/``.

    python analysis/video_native_gate_paths.py \
        --backbone hunyuan_video \
        --matrix-root $DATA/hunyuan_video/matrix \
        --out resources/video_native_gate_paths/hunyuan_video
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

GATES = ("seacache", "teacache", "sencache", "dicache")
DATASETS = {"penguin599": (599, (54, 55, 56)), "vbench944": (944, (42, 43, 44))}
BUDGETS = ("K29", "K37", "K41")
NUM_STEPS = 50

REPO = Path(__file__).resolve().parents[1]
CONFIGS = {
    "hunyuan_video": REPO / "resources/hunyuan_video/baseline_matrix_config.v2.json",
    "wan21": REPO / "resources/wan21/baseline_matrix_config.v1.json",
}


def schedule_of(payload: dict, source: Path) -> str:
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != NUM_STEPS:
        raise ValueError(f"{source}: expected {NUM_STEPS} records")
    ordered = sorted(records, key=lambda r: int(r["step"]))
    if [int(r["step"]) for r in ordered] != list(range(NUM_STEPS)):
        raise ValueError(f"{source}: steps are not exactly 0..{NUM_STEPS - 1}")
    bits = []
    for r in ordered:
        action = r.get("action")
        if action not in ("full", "cache"):
            raise ValueError(f"{source}: step {r['step']} action {action!r}")
        bits.append("1" if action == "cache" else "0")
    return "".join(bits)


def cache_steps(schedule: str) -> list[int]:
    return [i for i, b in enumerate(schedule) if b == "1"]


def fmt_steps(steps) -> str:
    return ",".join(str(s) for s in steps)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def dist_row(a: str, b: str) -> dict:
    h = hamming(a, b)
    gap = abs(len(cache_steps(a)) - len(cache_steps(b)))
    return {"hamming": h, "normalized_hamming": h / NUM_STEPS,
            "cache_count_gap": gap, "paired_swaps": (h - gap) / 2.0}


def write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in columns})


def fixed_anchors(backbone: str) -> dict[str, dict[str, str]]:
    """{budget: {anchor_name: 50-bit schedule}} from the frozen config."""
    cfg = json.loads(CONFIGS[backbone].read_text())
    tables = cfg["schedule_tables"]
    out: dict[str, dict[str, str]] = {b: {} for b in BUDGETS}
    for budget in BUDGETS:
        for name in ("budcache", "meancache", "triplet", "shared"):
            key = f"{name}_{budget}"
            if key in tables:
                steps = set(int(s) for s in tables[key]["cache_steps"])
                out[budget][name] = "".join(
                    "1" if i in steps else "0" for i in range(NUM_STEPS))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=tuple(CONFIGS))
    ap.add_argument("--matrix-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cells_root = args.matrix_root / "cells"
    anchors = fixed_anchors(args.backbone)

    per_seed_rows: list[dict] = []
    pooled_rows: list[dict] = []
    summary: dict = {"backbone": args.backbone, "gates": {}, "n_decisions": 0}

    for gate in GATES:
        summary["gates"][gate] = {}
        for dataset, (n_prompts, seeds) in DATASETS.items():
            for budget in BUDGETS:
                pooled: Counter[str] = Counter()
                per_stream: dict[int, Counter[str]] = {}
                for seed in seeds:
                    cell = cells_root / f"{gate}_{dataset}_{budget}_s{seed}"
                    files = sorted(cell.glob("decisions_*.json"))
                    if len(files) != n_prompts:
                        raise SystemExit(
                            f"{cell}: {len(files)} decisions, expected {n_prompts}")
                    ctr: Counter[str] = Counter()
                    for f in files:
                        ctr[schedule_of(json.loads(f.read_text()), f)] += 1
                    per_stream[seed] = ctr
                    pooled.update(ctr)
                    total = sum(ctr.values())
                    for sched, count in ctr.most_common():
                        per_seed_rows.append({
                            "method": gate, "dataset": dataset, "budget": budget,
                            "seed": seed, "count": count, "mass": count / total,
                            "n_cached": len(cache_steps(sched)),
                            "schedule": sched,
                            "cache_steps": fmt_steps(cache_steps(sched))})
                total = sum(pooled.values())
                summary["n_decisions"] += total
                for sched, count in pooled.most_common():
                    pooled_rows.append({
                        "method": gate, "dataset": dataset, "budget": budget,
                        "count": count, "mass": count / total,
                        "n_cached": len(cache_steps(sched)),
                        "schedule": sched,
                        "cache_steps": fmt_steps(cache_steps(sched))})

                top3 = pooled.most_common(3)
                modal, modal_n = top3[0]
                in_all = [s for s in pooled
                          if all(s in per_stream[seed] for seed in seeds)]
                mass_in_all = sum(pooled[s] for s in in_all) / total
                entry = {
                    "U_per_stream": {str(s): len(per_stream[s]) for s in seeds},
                    "pooled_unique": len(pooled),
                    "n_decisions": total,
                    "modal": {"schedule": modal, "count": modal_n,
                              "mass": modal_n / total,
                              "n_cached": len(cache_steps(modal))},
                    "top3": [{"schedule": s, "count": c, "mass": c / total,
                              "n_cached": len(cache_steps(s))}
                             for s, c in top3],
                    "paths_in_all_streams": len(in_all),
                    "mass_in_all_streams": mass_in_all,
                    "modal_vs_fixed": {
                        name: dist_row(modal, sched)
                        for name, sched in anchors[budget].items()},
                }
                summary["gates"][gate].setdefault(dataset, {})[budget] = entry

        # cross-dataset modal distance per budget
        xd = {}
        for budget in BUDGETS:
            pair = [summary["gates"][gate][ds][budget]["modal"]["schedule"]
                    for ds in DATASETS]
            xd[budget] = dist_row(pair[0], pair[1])
        summary["gates"][gate]["cross_dataset_modal"] = xd

    args.out.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out / "per_seed_path_counts.tsv", per_seed_rows,
              ["method", "dataset", "budget", "seed", "count", "mass",
               "n_cached", "schedule", "cache_steps"])
    write_tsv(args.out / "dataset_path_counts.tsv", pooled_rows,
              ["method", "dataset", "budget", "count", "mass",
               "n_cached", "schedule", "cache_steps"])
    (args.out / "path_census_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"[{args.backbone}] {summary['n_decisions']} decisions, "
          f"{len(pooled_rows)} pooled path rows, "
          f"{len(per_seed_rows)} per-seed rows -> {args.out}")
    for gate in GATES:
        for dataset in DATASETS:
            row = " ".join(
                f"{b}:U={summary['gates'][gate][dataset][b]['pooled_unique']}"
                f"/modal={summary['gates'][gate][dataset][b]['modal']['mass']:.2f}"
                for b in BUDGETS)
            print(f"  {gate:10s} {dataset:12s} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
