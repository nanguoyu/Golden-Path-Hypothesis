#!/usr/bin/env python3
"""Join video SPX or baseline metrics with the corresponding generation decisions."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RUN.video_spx.spx_cells import DATASETS, PAYLOADS, SPAN_VARIANT_PAYLOAD

METRICS = ("psnr", "ssim", "lpips", "temporal_lpips_delta")
FIELDS = (
    "row", "payload", "dataset", "K", "seed", "prompt_idx", "prompt_id",
    "actual_seed", "prompt_sha256", "psnr", "ssim", "lpips", "temporal_delta",
    "cache_count_realized",
)
BASELINE_FIELDS = ("method",) + FIELDS[2:]


def parse_cell_name(name: str):
    # Use the producer's allowed labels without importing numerical analyses.
    policies = "|".join(sorted(PAYLOADS + (SPAN_VARIANT_PAYLOAD,), key=len, reverse=True))
    datasets = "|".join(DATASETS)
    match = re.fullmatch(
        rf"(.+)x({policies})_({datasets})_K(\d+)_s(\d+)", name
    )
    if match is None:
        return None
    row, policy, dataset, budget, seed = match.groups()
    return row, policy, dataset, "K" + budget, int(seed)


def parse_baseline_name(name: str):
    datasets = "|".join(DATASETS)
    match = re.fullmatch(rf"(.+)_({datasets})_K(\d+)_s(\d+)", name)
    if match is None:
        return None
    method, dataset, budget, seed = match.groups()
    return method, None, dataset, "K" + budget, int(seed)


def required(record: dict, key: str, source: Path):
    if key not in record or record[key] is None:
        raise ValueError(f"{source}: missing required field {key!r}")
    return record[key]


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing required input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def stage(backbone: str, input_root: Path, output: Path,
          generation_root: Path | None = None, *, kind: str = "spx") -> int:
    """Write one row per evaluated video; no media decoding or metric recomputation."""
    files = sorted(input_root.glob("*.json"))
    if not files:
        raise ValueError(f"{input_root}: no evaluation JSON files")
    rows = []
    seen = set()
    prompts = {}
    for path in files:
        parsed = (parse_baseline_name(path.stem) if kind == "baseline"
                  else parse_cell_name(path.stem))
        if parsed is None:
            raise ValueError(f"{path}: expected a {kind} cell filename")
        name, policy, dataset, budget, base_seed = parsed
        metric_data = read_json(path)
        if metric_data.get("schema") != "video_pairwise_metrics.v2":
            raise ValueError(f"{path}: expected video_pairwise_metrics.v2")
        if required(metric_data, "backbone", path) != backbone:
            raise ValueError(f"{path}: backbone differs from --backbone {backbone}")
        if metric_data.get("frame_indices") != "all":
            raise ValueError(f"{path}: SPX temporal metrics require all frames")
        videos = required(metric_data, "per_video", path)
        if not isinstance(videos, list) or not videos:
            raise ValueError(f"{path}: per_video must be a nonempty list")
        if required(metric_data, "n_pairs", path) != len(videos):
            raise ValueError(f"{path}: n_pairs differs from the per_video length")
        acc = Path(required(metric_data, "acc", path))
        if acc.name != path.stem:
            raise ValueError(f"{path}: acc directory does not match the cell filename")
        if generation_root is not None:
            acc = generation_root / path.stem
        for entry in videos:
            idx = required(entry, "idx", path)
            if type(idx) is not int or idx < 0:
                raise ValueError(f"{path}: idx must be a nonnegative integer")
            if required(entry, "video", path) != f"video_{idx:05d}.mp4":
                raise ValueError(f"{path}: video filename differs from idx {idx}")
            key = (name, policy, dataset, budget, base_seed, idx)
            if key in seen:
                raise ValueError(f"{path}: duplicate evaluated video {key}")
            seen.add(key)
            decision_path = acc / f"decisions_{idx:05d}.json"
            decision = read_json(decision_path)
            if decision.get("schema") != f"{backbone}.baseline_screen_decisions.v1":
                raise ValueError(f"{decision_path}: wrong generation-decision schema")
            if kind == "baseline":
                for field, expected in (("mode", name), ("dataset", dataset),
                                        ("budget", budget)):
                    if required(decision, field, decision_path) != expected:
                        raise ValueError(f"{decision_path}: {field} differs from cell filename")
            if required(decision, "prompt_idx", decision_path) != idx:
                raise ValueError(f"{decision_path}: prompt_idx differs from metric idx")
            actual_seed = required(decision, "seed", decision_path)
            if actual_seed != base_seed + idx:
                raise ValueError(f"{decision_path}: seed is not cell base seed plus idx")
            prompt_id = required(decision, "prompt_id", decision_path)
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(f"{decision_path}: prompt_id must be a nonempty string")
            prompt = required(decision, "prompt", decision_path)
            if not isinstance(prompt, str):
                raise ValueError(f"{decision_path}: prompt must be a string")
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if required(entry, "prompt_sha256", path) != digest:
                raise ValueError(f"{path}: prompt differs from decisions for idx {idx}")
            identity = (prompt_id, digest)
            previous = prompts.setdefault((dataset, idx), identity)
            if previous != identity:
                raise ValueError(f"{path}: inconsistent prompt identity for {dataset}/{idx}")
            cache_count = required(decision, "actual_cache_count", decision_path)
            num_steps = required(decision, "num_steps", decision_path)
            if (type(cache_count) is not int or type(num_steps) is not int
                    or not 0 <= cache_count <= num_steps):
                raise ValueError(f"{decision_path}: invalid actual_cache_count or num_steps")
            values = {}
            for metric in METRICS:
                value = required(entry, metric, path)
                identical_psnr = kind == "baseline" and metric == "psnr" and value == math.inf
                if (not isinstance(value, (float, int))
                        or not (math.isfinite(value) or identical_psnr)):
                    raise ValueError(f"{path}: non-finite or nonnumeric {metric} at idx {idx}")
                values[metric] = value
            rows.append({
                "row": name, "payload": policy, "dataset": dataset,
                "K": int(budget.removeprefix("K")), "seed": base_seed,
                "prompt_idx": idx, "prompt_id": prompt_id, "actual_seed": actual_seed,
                "prompt_sha256": digest, "psnr": values["psnr"],
                "ssim": values["ssim"], "lpips": values["lpips"],
                "temporal_delta": values["temporal_lpips_delta"],
                "cache_count_realized": cache_count,
            })
            if kind == "baseline":
                rows[-1]["method"] = rows[-1].pop("row")
                del rows[-1]["payload"]
    fields = BASELINE_FIELDS if kind == "baseline" else FIELDS
    order = (("method",) if kind == "baseline" else ("row", "payload"))
    order += ("dataset", "K", "seed", "prompt_idx")
    rows.sort(key=lambda r: tuple(r[k] for k in order))
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("spx", "baseline"), default="spx",
                        help="baseline filenames: <method>_<dataset>_K<K>_s<base-seed>.json")
    parser.add_argument("--backbone", choices=("hunyuan_video", "wan21"), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path,
                        help="relocated parent of the cell generation directories; "
                             "otherwise use each metric JSON's acc directory")
    args = parser.parse_args()
    try:
        count = stage(args.backbone, args.input_root, args.output, args.generation_root,
                      kind=args.kind)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote {count} per-video rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
