#!/usr/bin/env python3
"""Freeze DiffusionDB native-gate operating points from count-only sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.select_native_gate_threshold import (
    select_candidate,
    summarize_candidate_directories,
)


TASK_NAMES = {
    "SeaCache": "native_sea",
    "TeaCache": "native_tea",
    "SenCache": "native_sen",
    "DiCache": "native_dicache",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_tsv", type=Path, required=True)
    parser.add_argument("--mean_tolerance", type=float, default=0.25)
    return parser.parse_args()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _threshold_tag(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _threshold_directory(
    task: dict[str, str],
    *,
    method: str,
    threshold: float,
) -> Path:
    mode = method.lower()
    return (
        Path(task["output_dir"])
        / TASK_NAMES[method]
        / f"{mode}_t{_threshold_tag(threshold)}"
    )


def _as_optional_float(raw: str) -> float | None:
    return None if raw in ("", "-") else float(raw)


def freeze(
    *,
    specs_path: Path,
    tasks_path: Path,
    mean_tolerance: float,
) -> dict[str, Any]:
    specs = _read_tsv(specs_path)
    tasks = _read_tsv(tasks_path)
    records: list[dict[str, Any]] = []

    for spec in specs:
        model = spec["model"]
        sweep = spec["sweep"]
        method = spec["method"]
        matching_tasks = [
            row
            for row in tasks
            if row["model"] == model and row["sweep"] == sweep
        ]
        if len(matching_tasks) != 3:
            raise ValueError(
                f"{model}/{sweep}: expected 3 calibration seed tasks, "
                f"found {len(matching_tasks)}"
            )
        limits = {int(row["limit"]) for row in matching_tasks}
        prompt_files = {row["prompt_file"] for row in matching_tasks}
        if len(limits) != 1 or len(prompt_files) != 1:
            raise ValueError(f"{model}/{sweep}: inconsistent prompt protocol")
        expected_samples = limits.pop() * len(matching_tasks)
        thresholds = [float(value) for value in spec["thresholds"].split(",")]

        for target in map(int, spec["target_cache_counts"].split(",")):
            candidates = []
            for threshold in thresholds:
                directories = [
                    _threshold_directory(
                        task,
                        method=method,
                        threshold=threshold,
                    )
                    for task in matching_tasks
                ]
                row = summarize_candidate_directories(
                    f"{threshold:.12g}",
                    directories,
                    target=target,
                )
                if row["prompt_count"] != expected_samples:
                    raise ValueError(
                        f"{model}/{sweep}/threshold={threshold:g}: expected "
                        f"{expected_samples} decisions, found {row['prompt_count']}"
                    )
                row["threshold"] = threshold
                candidates.append(row)

            selected, rule = select_candidate(
                candidates,
                mean_tolerance=mean_tolerance,
                selection_policy="mean-first",
            )
            records.append(
                {
                    "model": model,
                    "sweep": sweep,
                    "method": method,
                    "target_cache_count": target,
                    "target_cache_ratio": target / 50.0,
                    "selected_threshold": selected["threshold"],
                    "dicache_ret_ratio": _as_optional_float(
                        spec["dicache_ret_ratio"]
                    ),
                    "sencache_threshold_start": float(
                        spec["sencache_threshold_start"]
                    ),
                    "sample_count": selected["prompt_count"],
                    "base_seeds": sorted(
                        int(row["base_seed"]) for row in matching_tasks
                    ),
                    "prompt_file": next(iter(prompt_files)),
                    "cache_count_mean": selected["cache_count_mean"],
                    "cache_count_std": selected["cache_count_std"],
                    "cache_count_median": selected["cache_count_median"],
                    "cache_count_p10": selected["cache_count_p10"],
                    "cache_count_p90": selected["cache_count_p90"],
                    "cache_count_min": selected["cache_count_min"],
                    "cache_count_max": selected["cache_count_max"],
                    "exact_target_fraction": selected["exact_target_fraction"],
                    "within_one_fraction": selected["within_one_fraction"],
                    "mean_target_error": selected["mean_target_error"],
                    "needs_refinement": (
                        selected["mean_target_error"] > mean_tolerance
                    ),
                    "selection_rule": rule,
                    "candidates": candidates,
                }
            )

    records.sort(
        key=lambda row: (
            row["model"],
            row["target_cache_count"],
            row["method"],
        )
    )
    return {
        "format": "diffusiondb-native-gate-thresholds-v1",
        "selection_scope": (
            "one threshold per model/method/target K across three "
            "DiffusionDB-disjoint calibration seed streams"
        ),
        "selection_uses_image_quality": False,
        "selection_policy": "min mean |actual K-target K|, then exact mass, then std",
        "mean_tolerance": mean_tolerance,
        "operating_points": records,
    }


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "model",
        "method",
        "target_cache_count",
        "target_cache_ratio",
        "selected_threshold",
        "dicache_ret_ratio",
        "sencache_threshold_start",
        "sample_count",
        "cache_count_mean",
        "cache_count_std",
        "cache_count_median",
        "cache_count_p10",
        "cache_count_p90",
        "cache_count_min",
        "cache_count_max",
        "exact_target_fraction",
        "within_one_fraction",
        "mean_target_error",
        "needs_refinement",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    payload = freeze(
        specs_path=args.specs,
        tasks_path=args.tasks,
        mean_tolerance=args.mean_tolerance,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_tsv(args.out_tsv, payload["operating_points"])
    print(
        json.dumps(
            {
                "operating_points": len(payload["operating_points"]),
                "needs_refinement": sum(
                    row["needs_refinement"]
                    for row in payload["operating_points"]
                ),
                "out_json": str(args.out_json),
                "out_tsv": str(args.out_tsv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
