#!/usr/bin/env python3
"""Stage the P4 per-image metrics of the schedule-search wave into two tables.

P4 evaluates each searched fixed schedule on four datasets x three seed streams
per model (`docs/schedule_search_plan_zh.md` S2 "评估"). Every cell leaves one
`metrics.json` with a `per_image` block; this brings them all down to one row
per (schedule, dataset, seed, prompt), the shape P5's renderer reads:

`perprompt_search_<model>.tsv.gz`
    schedule payload k dataset seed prompt_idx psnr ssim lpips image_reward clip

This is `resources/spx/perprompt_spx_<model>.tsv.gz` plus a `dataset` column.
The SPX wave ran on Parti alone, so its key (schedule, payload, k, seed) is
unique there; here the same schedule is run on four prompt sets whose indices
all start at 0, so the dataset has to be part of the key.

Cell directories are the layout `RUN/schedule_search_eval_cells.py` submits
into: `<root>/<dataset>/<model>/k<K>/<schedule>x<payload>_s<seed>`.

Read-only over the run directories, and a serial scan of a few hundred
`metrics.json` files. Run it as a batch job, not on a login node.

    python analysis/stage_schedule_search_perprompt.py \
        --out resources/schedule_search
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.sp_cross import parse_cell_name  # noqa: E402

METRICS = ("psnr", "ssim", "lpips", "image_reward", "clip")
MODELS = ("flux", "qwen")


def data_root() -> Path:
    return Path(os.environ.get("DATA", "outputs"))


def read_per_image(path: Path) -> tuple[list[int], dict[str, list]] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = payload.get("indices")
    per_image = payload.get("per_image") or {}
    if not indices or not per_image:
        return None
    columns = {}
    for metric in METRICS:
        values = per_image.get(metric)
        if values is None or len(values) != len(indices):
            return None
        columns[metric] = values
    return [int(i) for i in indices], columns


def fmt(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def stage(out_dir: Path, roots: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for model in MODELS:
        rows: list[str] = []
        cells = 0
        for root in roots:
            if not root.is_dir():
                print(f"[stage] no such root: {root}", flush=True)
                continue
            for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                model_dir = dataset_dir / model
                if not model_dir.is_dir():
                    continue
                for budget_dir in sorted(model_dir.iterdir()):
                    name = budget_dir.name
                    if not name.startswith("k") or not name[1:].isdigit():
                        continue
                    budget_k = int(name[1:])
                    for cell_dir in sorted(p for p in budget_dir.iterdir() if p.is_dir()):
                        parsed = parse_cell_name(cell_dir.name)
                        if parsed is None:
                            continue
                        schedule, payload, seed = parsed
                        metrics_path = cell_dir / "metrics.json"
                        if not metrics_path.is_file():
                            print(f"[stage] no metrics yet: {metrics_path}", flush=True)
                            continue
                        read = read_per_image(metrics_path)
                        if read is None:
                            print(f"[stage] SKIP (no per_image): {metrics_path}", flush=True)
                            continue
                        indices, columns = read
                        cells += 1
                        for position, prompt_idx in enumerate(indices):
                            rows.append(
                                "\t".join(
                                    [
                                        schedule,
                                        payload,
                                        str(budget_k),
                                        dataset_dir.name,
                                        str(seed),
                                        str(prompt_idx),
                                        *[fmt(columns[m][position]) for m in METRICS],
                                    ]
                                )
                            )
                        print(
                            f"[stage] {model} {dataset_dir.name} k{budget_k} "
                            f"{cell_dir.name}: {len(indices)}",
                            flush=True,
                        )
        target = out_dir / f"perprompt_search_{model}.tsv.gz"
        header = "\t".join(
            ["schedule", "payload", "k", "dataset", "seed", "prompt_idx", *METRICS]
        )
        with gzip.open(target, "wt", encoding="utf-8", newline="\n") as handle:
            handle.write(header + "\n")
            if rows:
                handle.write("\n".join(rows) + "\n")
        counts[model] = len(rows)
        print(f"[stage] wrote {target} ({len(rows)} rows, {cells} cells)", flush=True)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=_ROOT / "resources" / "schedule_search"
    )
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="repeatable; wave output root (default $DATA/schedule_search_eval)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    roots = args.root or [data_root() / "schedule_search_eval"]
    stage(args.out, [Path(p) for p in roots])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
