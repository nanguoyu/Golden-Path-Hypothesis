#!/usr/bin/env python3
"""Per-video realized cache paths of the video matrices' four dynamic gates.

`analysis/video_native_gate_paths.py` counts paths per cell; this script keeps
the *identity* of the generation that walked each path, so a path can be joined
to that video's own quality row in
`resources/video_full_results/pervideo_<backbone>.tsv.gz`.

One row per generated video: method, dataset, K, seed, prompt_idx, the realized
50-bit path (1 = cache, 0 = full, rebuilt from ``records[].action``), and its
cache count. Four gates x two evaluation sets x three budgets x three seed
streams = 72 cells per backbone, about 55.5k rows.

Runs on the cluster that holds the matrix cells (CPU/IO only, submit through
Slurm -- see `RUN/slurm_video_perprompt_paths.sh`):

    python analysis/video_perprompt_paths.py \
        --backbone hunyuan_video \
        --matrix-root $DATA/hunyuan_video/matrix \
        --out $DATA/video_perprompt_paths/perprompt_paths_hunyuan_video.tsv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

GATES = ("seacache", "teacache", "sencache", "dicache")
DATASETS = {"penguin599": (599, (54, 55, 56)), "vbench944": (944, (42, 43, 44))}
BUDGETS = ("K29", "K37", "K41")
NUM_STEPS = 50
COLUMNS = ["method", "dataset", "K", "seed", "prompt_idx", "path", "n_cached"]


def path_of(payload: dict, source: Path) -> str:
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    choices=("hunyuan_video", "wan21"))
    ap.add_argument("--matrix-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cells_root = args.matrix_root / "cells"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_cells = 0
    with gzip.open(args.out, "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        w.writeheader()
        for gate in GATES:
            for dataset, (n_prompts, seeds) in DATASETS.items():
                for budget in BUDGETS:
                    K = int(budget[1:])
                    for seed in seeds:
                        cell = cells_root / f"{gate}_{dataset}_{budget}_s{seed}"
                        files = sorted(cell.glob("decisions_*.json"))
                        if len(files) != n_prompts:
                            raise SystemExit(
                                f"{cell}: {len(files)} decisions, expected {n_prompts}")
                        seen: set[int] = set()
                        for f in files:
                            payload = json.loads(f.read_text())
                            # `seed` inside the payload is the per-video derived
                            # seed (base + prompt offset), not the stream's base
                            # seed, so the cell directory carries the stream id.
                            if payload.get("mode") != gate \
                                    or payload.get("dataset") != dataset \
                                    or payload.get("budget") != budget:
                                raise SystemExit(f"{f}: cell/payload identity mismatch")
                            idx = int(payload["prompt_idx"])
                            if idx in seen:
                                raise SystemExit(f"{f}: duplicate prompt_idx {idx}")
                            seen.add(idx)
                            path = path_of(payload, f)
                            n_cached = path.count("1")
                            if n_cached != int(payload["actual_cache_count"]):
                                raise SystemExit(
                                    f"{f}: path has {n_cached} cache steps, "
                                    f"actual_cache_count {payload['actual_cache_count']}")
                            w.writerow({"method": gate, "dataset": dataset, "K": K,
                                        "seed": seed, "prompt_idx": idx,
                                        "path": path, "n_cached": n_cached})
                            n_rows += 1
                        n_cells += 1
                        print(f"  {cell.name}: {len(files)}", flush=True)

    print(f"[{args.backbone}] {n_cells} cells, {n_rows} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
