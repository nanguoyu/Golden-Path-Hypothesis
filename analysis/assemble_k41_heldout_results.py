#!/usr/bin/env python3
"""Assemble final K41 held-out metric tables from normal evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any


METRICS = ("psnr", "ssim", "lpips")
PRIMARY = ("mean_optimal", "robust_optimal")
ADAPTIVE = ("seacache", "teacache", "sencache", "dicache")
METHODS = (*PRIMARY, "budcache", *ADAPTIVE)
STANDARD_DATASETS = ("drawbench", "geneval", "parti")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_metrics(roots: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    loaded: dict[tuple[str, str], dict[str, Any]] = {}
    for root in roots:
        for path in sorted(root.rglob("metrics.json")):
            method = path.parent.name
            environment = path.parent.parent.name
            if method not in METHODS or not environment.endswith(("s41", "s42", "s43")):
                continue
            key = (environment, method)
            if key in loaded:
                raise ValueError(f"duplicate metrics for {environment}/{method}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            indices = [int(value) for value in payload["indices"]]
            if len(indices) != int(payload["n_pairs"]) or len(set(indices)) != len(indices):
                raise ValueError(f"invalid indices in {path}")
            for metric in METRICS:
                if len(payload["per_image"][metric]) != len(indices):
                    raise ValueError(f"incomplete {metric} values in {path}")
            loaded[key] = payload
    return loaded


def expected_pairs(environment: str) -> int:
    if environment.startswith("drawbench_"):
        return 200
    if environment.startswith("geneval_"):
        return 553
    if environment == "parti_s42":
        return 1628
    if environment.startswith("parti_"):
        return 1632
    if environment.startswith("diffusiondb_"):
        return 10000
    raise ValueError(f"unknown environment: {environment}")


def metric_rows(metrics: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    environments = sorted({environment for environment, _method in metrics})
    expected_keys = {(environment, method) for environment in environments for method in METHODS}
    if set(metrics) != expected_keys:
        missing = sorted(expected_keys - set(metrics))
        extra = sorted(set(metrics) - expected_keys)
        raise ValueError(f"metric matrix mismatch: missing={missing}, extra={extra}")
    for environment in environments:
        dataset, seed = environment.rsplit("_s", 1)
        for method in METHODS:
            payload = metrics[(environment, method)]
            n_pairs = int(payload["n_pairs"])
            if n_pairs != expected_pairs(environment):
                raise ValueError(f"{environment}/{method} has {n_pairs} pairs")
            row: dict[str, Any] = {
                "environment": environment,
                "dataset": dataset,
                "seed": int(seed),
                "method": method,
                "n_pairs": n_pairs,
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = float(payload["summary"][metric]["mean"])
                row[f"{metric}_std"] = float(payload["summary"][metric]["std"])
            rows.append(row)
    return rows


def comparison_rows(metrics: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    environments = sorted({environment for environment, _method in metrics})
    for environment in environments:
        for primary in PRIMARY:
            left = metrics[(environment, primary)]
            left_indices = [int(value) for value in left["indices"]]
            for comparator in ADAPTIVE:
                right = metrics[(environment, comparator)]
                right_indices = [int(value) for value in right["indices"]]
                if left_indices != right_indices:
                    raise ValueError(f"index mismatch for {environment}: {primary}/{comparator}")
                for metric in METRICS:
                    left_values = [float(value) for value in left["per_image"][metric]]
                    right_values = [float(value) for value in right["per_image"][metric]]
                    raw = [a - b for a, b in zip(left_values, right_values)]
                    favorable = [-value for value in raw] if metric == "lpips" else raw
                    rows.append(
                        {
                            "environment": environment,
                            "primary": primary,
                            "comparator": comparator,
                            "metric": metric,
                            "n_pairs": len(favorable),
                            "primary_mean": statistics.fmean(left_values),
                            "comparator_mean": statistics.fmean(right_values),
                            "favorable_mean_delta": statistics.fmean(favorable),
                            "favorable_median_delta": statistics.median(favorable),
                            "not_worse_fraction": sum(value >= 0.0 for value in favorable)
                            / len(favorable),
                        }
                    )
    return rows


def macro_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["environment"], row["method"]): row for row in rows}
    standard_envs = sorted(
        environment
        for environment, _method in by_key
        if environment.split("_s", 1)[0] in STANDARD_DATASETS
    )
    standard_envs = sorted(set(standard_envs))
    ddb_envs = sorted({environment for environment, _method in by_key if environment.startswith("diffusiondb_")})

    def aggregate(environments: list[str]) -> dict[str, Any]:
        return {
            method: {
                metric: statistics.fmean(
                    float(by_key[(environment, method)][f"{metric}_mean"])
                    for environment in environments
                )
                for metric in METRICS
            }
            for method in METHODS
        }

    return {
        "schema": "flux_k41_heldout_complete_summary.v1",
        "standard_environment_count": len(standard_envs),
        "diffusiondb_environment_count": len(ddb_envs),
        "standard_equal_environment_macro": aggregate(standard_envs),
        "diffusiondb_seed_macro": aggregate(ddb_envs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metrics = load_metrics(args.metric_root)
    rows = metric_rows(metrics)
    comparisons = comparison_rows(metrics)
    _write_tsv(args.output_dir / "complete_metrics.tsv", rows)
    _write_tsv(args.output_dir / "primary_vs_adaptive.tsv", comparisons)
    _atomic_json(args.output_dir / "complete_summary.json", macro_summary(rows))
    print(
        f"[k41-heldout-assemble] environments={len({row['environment'] for row in rows})} "
        f"metric_rows={len(rows)} comparisons={len(comparisons)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
