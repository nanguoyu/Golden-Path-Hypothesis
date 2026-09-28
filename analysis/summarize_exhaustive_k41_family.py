#!/usr/bin/env python3
"""Summarize frozen K41 family results across held-out environments.

The script consumes the pair-level JSON files written by
``flux/exhaustive_k41_family_runner.py``.  It reports family migration and the
predeclared mean/robust paths against the same-payload BudCache schedule.  Full
PSNR/SSIM/LPIPS comparisons against native adaptive methods remain in the
existing ``evaluation/eval_metrics.py`` path because those methods save normal
image directories rather than family scalar rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PAIR_SCHEMA = "flux_exhaustive_k41_family_pair.v1"
PRIMARY_RANKS = {
    "mean_optimal": 164762,
    "robust_optimal": 165962,
    "budcache": 176543,
}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _parse_mapping(values: list[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{option} requires NAME=VALUE, got {value!r}")
        name, raw = value.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise SystemExit(f"invalid or duplicate name for {option}: {name!r}")
        result[name] = raw.strip()
    return result


def load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    manifest: dict[int, dict[str, Any]] = {}
    for row in rows:
        rank = int(row["rank"])
        reasons = [reason for reason in row["reasons"].split(";") if reason]
        variable = tuple(
            int(piece) for piece in row["variable_full_steps"].split(",") if piece
        )
        manifest[rank] = {
            "rank": rank,
            "reasons": reasons,
            "variable_full_steps": variable,
            "full_steps": (0, 1, 2, *variable, 49),
            "discovery_mean_psnr_db": float(row["mean_psnr_db"]),
            "discovery_min_psnr_db": float(row["min_psnr_db"]),
        }
    if len(manifest) != 337:
        raise ValueError(f"manifest has {len(manifest)} unique ranks; expected 337")
    return manifest


def load_environment(path: Path, expected_ranks: list[int]) -> list[dict[str, Any]]:
    pair_files = sorted((path / "pairs").glob("pair_*.json"))
    pairs: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    protocol: dict[str, Any] | None = None
    for pair_file in pair_files:
        payload = json.loads(pair_file.read_text(encoding="utf-8"))
        if payload.get("schema") != PAIR_SCHEMA:
            raise ValueError(f"wrong pair schema: {pair_file}")
        identity = (int(payload["prompt_idx"]), int(payload["seed"]))
        if identity in identities:
            raise ValueError(f"duplicate prompt--seed identity: {identity}")
        identities.add(identity)
        if protocol is None:
            protocol = payload.get("protocol")
        elif payload.get("protocol") != protocol:
            raise ValueError(f"mixed protocols inside environment: {pair_file}")
        rows = payload.get("candidates")
        if not isinstance(rows, list) or [int(row["rank"]) for row in rows] != expected_ranks:
            raise ValueError(f"incomplete or reordered candidates: {pair_file}")
        pairs.append(payload)
    return pairs


def _rankdata(values: np.ndarray, *, descending: bool = False) -> np.ndarray:
    data = -values if descending else values
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(len(data), dtype=np.float64)
    start = 0
    while start < len(data):
        end = start + 1
        while end < len(data) and data[order[end]] == data[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    left_arr = np.asarray(list(left), dtype=np.float64)
    right_arr = np.asarray(list(right), dtype=np.float64)
    if len(left_arr) != len(right_arr) or len(left_arr) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    left_rank = _rankdata(left_arr)
    right_rank = _rankdata(right_arr)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_environment(
    name: str,
    pairs: list[dict[str, Any]],
    manifest: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ranks = list(manifest)
    if not pairs:
        raise ValueError(f"environment {name!r} has no pair files")
    matrix = np.asarray(
        [[float(row["psnr_db"]) for row in pair["candidates"]] for pair in pairs],
        dtype=np.float64,
    )
    means = matrix.mean(axis=0)
    discovery = np.asarray(
        [manifest[rank]["discovery_mean_psnr_db"] for rank in ranks], dtype=np.float64
    )
    discovery_order = list(np.argsort(-discovery, kind="mergesort"))
    heldout_order = list(np.argsort(-means, kind="mergesort"))
    discovery_top64 = {ranks[index] for index in discovery_order[:64]}
    heldout_top64 = {ranks[index] for index in heldout_order[:64]}

    candidate_rows: list[dict[str, Any]] = []
    heldout_rank_by_index = _rankdata(means, descending=True)
    discovery_rank_by_index = _rankdata(discovery, descending=True)
    for index, rank in enumerate(ranks):
        values = matrix[:, index]
        meta = manifest[rank]
        stats = _quantiles(values)
        candidate_rows.append(
            {
                "environment": name,
                "rank": rank,
                "n_pairs": len(pairs),
                "heldout_rank": float(heldout_rank_by_index[index]),
                "discovery_rank": float(discovery_rank_by_index[index]),
                "mean_psnr_db": stats["mean"],
                "median_psnr_db": stats["median"],
                "q25_psnr_db": stats["q25"],
                "q75_psnr_db": stats["q75"],
                "min_psnr_db": stats["min"],
                "max_psnr_db": stats["max"],
                "discovery_mean_psnr_db": meta["discovery_mean_psnr_db"],
                "discovery_min_psnr_db": meta["discovery_min_psnr_db"],
                "selection_count": len(meta["reasons"]),
                "reasons": ";".join(meta["reasons"]),
                "variable_full_steps": ",".join(map(str, meta["variable_full_steps"])),
            }
        )

    rank_to_col = {rank: index for index, rank in enumerate(ranks)}
    bud_col = rank_to_col[PRIMARY_RANKS["budcache"]]
    primary_rows: list[dict[str, Any]] = []
    for label in ("mean_optimal", "robust_optimal"):
        rank = PRIMARY_RANKS[label]
        delta = matrix[:, rank_to_col[rank]] - matrix[:, bud_col]
        stats = _quantiles(delta)
        primary_rows.append(
            {
                "environment": name,
                "path": label,
                "rank": rank,
                "comparator": "budcache",
                "n_pairs": len(pairs),
                "mean_delta_psnr_db": stats["mean"],
                "median_delta_psnr_db": stats["median"],
                "q25_delta_psnr_db": stats["q25"],
                "q75_delta_psnr_db": stats["q75"],
                "min_delta_psnr_db": stats["min"],
                "max_delta_psnr_db": stats["max"],
                "proportion_not_worse": float(np.mean(delta >= 0.0)),
            }
        )

    frequency_rows: list[dict[str, Any]] = []
    for step in range(50):
        count = sum(step in manifest[rank]["full_steps"] for rank in heldout_top64)
        frequency_rows.append(
            {
                "environment": name,
                "step": step,
                "top64_full_count": count,
                "top64_full_frequency": count / 64.0,
            }
        )

    best_discovery = float(np.max(discovery))
    family_gaps: dict[str, Any] = {}
    for gap in (0.10, 0.25, 0.50):
        indices = np.flatnonzero(discovery >= best_discovery - gap)
        family_gaps[f"{gap:.2f}"] = {
            "candidate_count": int(len(indices)),
            "heldout_mean_psnr_distribution": _quantiles(means[indices]),
        }

    random_indices = np.asarray(
        [
            index
            for index, rank in enumerate(ranks)
            if "uniform_random_128_seed20270826" in manifest[rank]["reasons"]
        ],
        dtype=int,
    )
    discovery_selected_indices = np.asarray(
        [
            index
            for index, rank in enumerate(ranks)
            if any(
                reason.startswith(("mean_top", "min_top", "prompt_", "subset_"))
                for reason in manifest[rank]["reasons"]
            )
        ],
        dtype=int,
    )
    top64_sources = Counter(
        reason
        for rank in heldout_top64
        for reason in manifest[rank]["reasons"]
    )
    summary = {
        "environment": name,
        "n_pairs": len(pairs),
        "spearman_discovery_vs_heldout": spearman(discovery, means),
        "discovery_top64_retained": len(discovery_top64 & heldout_top64),
        "discovery_top64_retention_rate": len(discovery_top64 & heldout_top64) / 64.0,
        "heldout_best_rank": ranks[heldout_order[0]],
        "heldout_best_mean_psnr_db": float(means[heldout_order[0]]),
        "family_gaps": family_gaps,
        "random_control_mean_distribution": _quantiles(means[random_indices]),
        "discovery_selected_mean_distribution": _quantiles(
            means[discovery_selected_indices]
        ),
        "heldout_top64_sources": dict(sorted(top64_sources.items())),
    }
    return summary, candidate_rows, primary_rows, frequency_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        action="append",
        required=True,
        help="NAME=family output directory; repeat once per dataset--seed environment",
    )
    parser.add_argument(
        "--expected_pairs",
        action="append",
        required=True,
        help="NAME=N completion requirement; repeat for every environment",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("resources/exhaustive_k41/formal_results/candidate_manifest.tsv"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environments = {
        name: Path(raw) for name, raw in _parse_mapping(args.environment, "--environment").items()
    }
    expected_raw = _parse_mapping(args.expected_pairs, "--expected_pairs")
    if set(expected_raw) != set(environments):
        raise SystemExit("--expected_pairs must name every supplied environment exactly once")
    expected = {name: int(raw) for name, raw in expected_raw.items()}

    manifest = load_manifest(args.manifest)
    ranks = list(manifest)
    summaries: list[dict[str, Any]] = []
    all_candidate_rows: list[dict[str, Any]] = []
    all_primary_rows: list[dict[str, Any]] = []
    all_frequency_rows: list[dict[str, Any]] = []
    environment_means: list[np.ndarray] = []

    for name, path in environments.items():
        pairs = load_environment(path, ranks)
        if name in expected and len(pairs) != expected[name]:
            raise ValueError(f"environment {name} has {len(pairs)} pairs; expected {expected[name]}")
        summary, candidate_rows, primary_rows, frequency_rows = summarize_environment(
            name, pairs, manifest
        )
        summaries.append(summary)
        all_candidate_rows.extend(candidate_rows)
        all_primary_rows.extend(primary_rows)
        all_frequency_rows.extend(frequency_rows)
        environment_means.append(
            np.asarray([float(row["mean_psnr_db"]) for row in candidate_rows])
        )

    macro = np.mean(np.stack(environment_means, axis=0), axis=0)
    macro_order = np.argsort(-macro, kind="mergesort")
    macro_rows = [
        {
            "macro_rank": order + 1,
            "rank": ranks[index],
            "equal_environment_mean_psnr_db": float(macro[index]),
            "reasons": ";".join(manifest[ranks[index]]["reasons"]),
            "variable_full_steps": ",".join(
                map(str, manifest[ranks[index]]["variable_full_steps"])
            ),
        }
        for order, index in enumerate(macro_order)
    ]

    candidate_fields = list(all_candidate_rows[0])
    primary_fields = list(all_primary_rows[0])
    frequency_fields = list(all_frequency_rows[0])
    macro_fields = list(macro_rows[0])
    _write_tsv(args.output_dir / "candidate_statistics.tsv", all_candidate_rows, candidate_fields)
    _write_tsv(args.output_dir / "primary_vs_budcache.tsv", all_primary_rows, primary_fields)
    _write_tsv(args.output_dir / "top64_full_step_frequency.tsv", all_frequency_rows, frequency_fields)
    _write_tsv(args.output_dir / "equal_environment_macro.tsv", macro_rows, macro_fields)
    _atomic_write_json(
        args.output_dir / "family_migration_summary.json",
        {
            "schema": "flux_exhaustive_k41_family_summary.v1",
            "manifest": str(args.manifest),
            "environment_count": len(environments),
            "candidate_count": len(manifest),
            "environments": summaries,
            "equal_environment_best_rank": macro_rows[0]["rank"],
            "equal_environment_best_mean_psnr_db": macro_rows[0][
                "equal_environment_mean_psnr_db"
            ],
        },
    )
    print(
        f"[k41-family-summary] environments={len(environments)} "
        f"candidate_rows={len(all_candidate_rows)} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
