#!/usr/bin/env python3
"""Validate and merge FLUX exhaustive-K41 part files.

The merge refuses gaps, overlaps, mixed experiment fingerprints, malformed row
intervals, or schedules whose rank does not match their bit string.  It then
writes the scalar table, near-optimal-set summaries, and the predeclared
candidate union used by the validation stage.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.exhaustive_schedule import (  # noqa: E402
    K41_SPACE,
    rank_for_full_steps,
    schedule_for_rank,
)


PART_SCHEMA = "flux_exhaustive_k41_part.v1"


def _read_part(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable part: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != PART_SCHEMA:
        raise ValueError(f"wrong part schema: {path}")
    return payload


def validate_parts(
    paths: Iterable[Path], *, rank_start: int, rank_end: int
) -> tuple[list[Path], dict[str, Any]]:
    """Return ordered parts after exact interval and identity validation."""

    candidates = [Path(path) for path in paths]
    if not candidates:
        raise ValueError("no part files found")
    intervals: list[tuple[int, int, Path]] = []
    fingerprint: str | None = None
    experiment: dict[str, Any] | None = None
    space: dict[str, Any] | None = None
    weights: dict[str, Any] | None = None
    for path in candidates:
        payload = _read_part(path)
        if fingerprint is None:
            fingerprint = payload.get("experiment_fingerprint")
            experiment = payload.get("experiment")
            space = payload.get("space")
            weights = payload.get("weights")
            if not isinstance(fingerprint, str) or not isinstance(experiment, dict):
                raise ValueError("first part lacks experiment identity")
            if (
                not isinstance(space, dict)
                or space.get("schema") != "exhaustive_schedule_space.v1"
            ):
                raise ValueError("first part lacks schedule-space identity")
            if not isinstance(weights, dict):
                raise ValueError("first part lacks resolved model-weight identity")
            if (
                space.get("total") != K41_SPACE.total
                or experiment.get("space_identity") != K41_SPACE.identity
            ):
                raise ValueError("parts do not use the frozen K41 universe")
        elif payload.get("experiment_fingerprint") != fingerprint:
            raise ValueError(f"mixed experiment fingerprint: {path}")
        elif payload.get("experiment") != experiment or payload.get("space") != space:
            raise ValueError(f"mixed experiment metadata: {path}")
        else:
            payload_weights = payload.get("weights")
            if not isinstance(payload_weights, dict) or (
                payload_weights.get("model_revision"),
                payload_weights.get("model_commit"),
            ) != (weights.get("model_revision"), weights.get("model_commit")):
                raise ValueError(f"mixed resolved model weights: {path}")
        start, end = payload.get("rank_start"), payload.get("rank_end")
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            raise ValueError(f"invalid part interval: {path}")
        if end > rank_end:
            raise ValueError(f"part extends beyond requested end {rank_end}: {path}")
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != end - start:
            raise ValueError(f"row count does not match part interval: {path}")
        if [row.get("rank") for row in rows] != list(range(start, end)):
            raise ValueError(f"row ranks are incomplete or out of order: {path}")
        intervals.append((start, end, path))

    intervals.sort(key=lambda item: (item[0], str(item[2])))
    expected = int(rank_start)
    ordered_paths: list[Path] = []
    for start, end, path in intervals:
        if start != expected:
            relation = "overlap/duplicate" if start < expected else "gap"
            raise ValueError(
                f"{relation} before {path}: expected rank {expected}, found {start}"
            )
        expected = end
        ordered_paths.append(path)
    if expected != rank_end:
        raise ValueError(
            f"trailing gap: expected coverage to {rank_end}, reached {expected}"
        )
    return ordered_paths, {
        "experiment_fingerprint": fingerprint,
        "experiment": experiment,
        "space": space,
        "weights": weights,
    }


def _iter_rows(parts: Iterable[Path]):
    for path in parts:
        payload = _read_part(path)
        yield from payload["rows"]


def _validate_row(row: dict[str, Any], prompt_indices: tuple[int, ...]) -> None:
    rank = int(row["rank"])
    full_steps, _cache_steps, bits = schedule_for_rank(rank)
    if row.get("bits") != bits:
        raise ValueError(f"rank {rank}: schedule bits do not match universe rank")
    variable = [step for step in full_steps if step not in K41_SPACE.forced_full_steps]
    if row.get("variable_full_steps") != variable:
        raise ValueError(f"rank {rank}: variable full steps do not match universe rank")
    records = row.get("per_prompt")
    if (
        not isinstance(records, list)
        or tuple(int(record.get("prompt_idx", -1)) for record in records)
        != prompt_indices
    ):
        raise ValueError(f"rank {rank}: per-prompt records have wrong identity/order")
    psnrs: list[float] = []
    for record in records:
        mse = float(record.get("mse", float("nan")))
        psnr = float(record.get("psnr_db", float("nan")))
        if not math.isfinite(mse) or mse < 0 or not math.isfinite(psnr):
            raise ValueError(f"rank {rank}: non-finite or invalid terminal metric")
        psnrs.append(psnr)
    mean_score = float(row.get("mean_psnr_db", float("nan")))
    min_score = float(row.get("min_psnr_db", float("nan")))
    wall_s = float(row.get("wall_s", float("nan")))
    if (
        not math.isfinite(mean_score)
        or not math.isfinite(min_score)
        or not math.isfinite(wall_s)
        or wall_s < 0
        or not math.isclose(mean_score, sum(psnrs) / len(psnrs), abs_tol=1e-9)
        or not math.isclose(min_score, min(psnrs), abs_tol=1e-9)
    ):
        raise ValueError(f"rank {rank}: aggregate metrics are invalid or inconsistent")


def _push_top(
    heap: list[tuple[float, int, int]], score: float, rank: int, limit: int
) -> None:
    item = (float(score), -int(rank), int(rank))
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _ordered_top(heap: list[tuple[float, int, int]]) -> list[tuple[float, int]]:
    return [
        (score, rank)
        for score, _negative_rank, rank in sorted(heap, key=lambda v: (-v[0], v[2]))
    ]


def _parse_named_schedule(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("schedule must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("schedule must be NAME=PATH")
    return name.strip(), Path(path)


def _rank_from_schedule_file(path: Path) -> int:
    bits = path.read_text(encoding="utf-8").strip()
    if len(bits) != K41_SPACE.num_steps or set(bits) - {"0", "1"}:
        raise ValueError(f"schedule must contain one 50-character bit string: {path}")
    full_steps = [idx for idx, bit in enumerate(bits) if bit == "0"]
    rank = rank_for_full_steps(full_steps)
    if schedule_for_rank(rank)[2] != bits:
        raise ValueError(f"schedule is outside the frozen universe: {path}")
    return rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rank_start", type=int, default=0)
    parser.add_argument("--rank_end", type=int, default=K41_SPACE.total)
    parser.add_argument("--top_n", type=int, default=64)
    parser.add_argument("--per_prompt_top_n", type=int, default=16)
    parser.add_argument(
        "--subset_top_n",
        type=int,
        default=16,
        help="top paths for every nonempty subset of discovery samples",
    )
    parser.add_argument("--random_count", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=20270826)
    parser.add_argument(
        "--allow_unresolved_weights",
        action="store_true",
        help="smoke only; formal merges require one resolved 40-hex model commit",
    )
    parser.add_argument(
        "--include_schedule",
        action="append",
        default=[],
        type=_parse_named_schedule,
        metavar="NAME=PATH",
        help="add an existing/geometry schedule if it belongs to the frozen universe",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0 <= args.rank_start < args.rank_end <= K41_SPACE.total):
        raise SystemExit("invalid --rank_start/--rank_end")
    if (
        args.top_n <= 0
        or args.per_prompt_top_n <= 0
        or args.subset_top_n <= 0
        or args.random_count < 0
    ):
        raise SystemExit("top counts must be positive and random_count non-negative")

    part_paths = sorted(args.parts_dir.glob("part_*.json"))
    parts, identity = validate_parts(
        part_paths, rank_start=args.rank_start, rank_end=args.rank_end
    )
    resolved_commit = identity["weights"].get("model_commit")
    commit_is_resolved = (
        isinstance(resolved_commit, str)
        and len(resolved_commit) == 40
        and not set(resolved_commit) - set("0123456789abcdef")
    )
    if not commit_is_resolved and not args.allow_unresolved_weights:
        raise ValueError(
            "formal merge requires a resolved 40-hex model commit; use "
            "--allow_unresolved_weights only for smoke data"
        )
    prompt_indices = tuple(int(v) for v in identity["experiment"]["prompt_indices"])
    if not prompt_indices:
        raise ValueError("experiment has no prompt indices")

    interval_size = args.rank_end - args.rank_start
    random_count = min(args.random_count, interval_size)
    random_ranks = set(
        random.Random(args.random_seed).sample(
            range(args.rank_start, args.rank_end), random_count
        )
    )
    top_mean: list[tuple[float, int, int]] = []
    top_min: list[tuple[float, int, int]] = []
    top_per_prompt: dict[int, list[tuple[float, int, int]]] = {
        idx: [] for idx in prompt_indices
    }
    subset_indices = [
        subset
        for size in range(2, len(prompt_indices))
        for subset in combinations(prompt_indices, size)
    ]
    top_per_subset: dict[tuple[int, ...], list[tuple[float, int, int]]] = {
        subset: [] for subset in subset_indices
    }
    scores_by_rank: dict[int, dict[str, Any]] = {}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / "merged.tsv.gz"
    with gzip.open(merged_path, "wt", encoding="utf-8", newline="") as handle:
        fields = [
            "rank",
            "variable_full_steps",
            "cache_mask_hex",
            "bits",
            *[f"mse_p{idx}" for idx in prompt_indices],
            *[f"psnr_p{idx}" for idx in prompt_indices],
            "mean_psnr_db",
            "min_psnr_db",
            "wall_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in _iter_rows(parts):
            _validate_row(row, prompt_indices)
            rank = int(row["rank"])
            per_prompt = {
                int(record["prompt_idx"]): record for record in row["per_prompt"]
            }
            mean_score = float(row["mean_psnr_db"])
            min_score = float(row["min_psnr_db"])
            _push_top(top_mean, mean_score, rank, args.top_n)
            _push_top(top_min, min_score, rank, args.top_n)
            for idx in prompt_indices:
                _push_top(
                    top_per_prompt[idx],
                    float(per_prompt[idx]["psnr_db"]),
                    rank,
                    args.per_prompt_top_n,
                )
            for subset in subset_indices:
                subset_score = sum(
                    float(per_prompt[idx]["psnr_db"]) for idx in subset
                ) / len(subset)
                _push_top(
                    top_per_subset[subset],
                    subset_score,
                    rank,
                    args.subset_top_n,
                )
            if rank in random_ranks:
                scores_by_rank[rank] = {
                    "mean_psnr_db": mean_score,
                    "min_psnr_db": min_score,
                    "per_prompt": {
                        idx: float(per_prompt[idx]["psnr_db"]) for idx in prompt_indices
                    },
                }
            writer.writerow(
                {
                    "rank": rank,
                    "variable_full_steps": ",".join(
                        str(v) for v in row["variable_full_steps"]
                    ),
                    "cache_mask_hex": row["cache_mask_hex"],
                    "bits": row["bits"],
                    **{f"mse_p{idx}": per_prompt[idx]["mse"] for idx in prompt_indices},
                    **{
                        f"psnr_p{idx}": per_prompt[idx]["psnr_db"]
                        for idx in prompt_indices
                    },
                    "mean_psnr_db": mean_score,
                    "min_psnr_db": min_score,
                    "wall_s": row["wall_s"],
                }
            )

    ordered_mean = _ordered_top(top_mean)
    ordered_min = _ordered_top(top_min)
    ordered_per_prompt = {
        idx: _ordered_top(heap) for idx, heap in top_per_prompt.items()
    }
    ordered_per_subset = {
        subset: _ordered_top(heap) for subset, heap in top_per_subset.items()
    }
    best_mean, best_rank = ordered_mean[0]
    thresholds = (0.1, 0.25, 0.5)
    family = {
        str(threshold): {
            "count": 0,
            "hamming_to_best_sum": 0,
            "hamming_to_best_min": None,
            "hamming_to_best_max": None,
            "variable_full_frequency": [0] * K41_SPACE.num_steps,
        }
        for threshold in thresholds
    }
    best_bits = schedule_for_rank(best_rank)[2]
    for row in _iter_rows(parts):
        score = float(row["mean_psnr_db"])
        for threshold in thresholds:
            if score + threshold < best_mean:
                continue
            record = family[str(threshold)]
            record["count"] += 1
            distance = sum(a != b for a, b in zip(best_bits, row["bits"]))
            record["hamming_to_best_sum"] += distance
            current_min = record["hamming_to_best_min"]
            current_max = record["hamming_to_best_max"]
            record["hamming_to_best_min"] = (
                distance if current_min is None else min(current_min, distance)
            )
            record["hamming_to_best_max"] = (
                distance if current_max is None else max(current_max, distance)
            )
            for step in row["variable_full_steps"]:
                record["variable_full_frequency"][int(step)] += 1
    for record in family.values():
        count = int(record["count"])
        record["mean_hamming_to_best"] = (
            float(record.pop("hamming_to_best_sum") / count) if count else None
        )
        record["variable_full_probability"] = [
            float(value / count) if count else 0.0
            for value in record.pop("variable_full_frequency")
        ]

    reasons: dict[int, set[str]] = defaultdict(set)
    for _score, rank in ordered_mean:
        reasons[rank].add(f"mean_top{args.top_n}")
    for _score, rank in ordered_min:
        reasons[rank].add(f"min_top{args.top_n}")
    for idx, ranked in ordered_per_prompt.items():
        for _score, rank in ranked:
            reasons[rank].add(f"prompt_{idx}_top{args.per_prompt_top_n}")
    for subset, ranked in ordered_per_subset.items():
        label = "_".join(str(idx) for idx in subset)
        for _score, rank in ranked:
            reasons[rank].add(f"subset_{label}_top{args.subset_top_n}")
    for rank in random_ranks:
        reasons[rank].add(f"uniform_random_{random_count}_seed{args.random_seed}")
    included: dict[str, int] = {}
    for name, path in args.include_schedule:
        rank = _rank_from_schedule_file(path)
        if not (args.rank_start <= rank < args.rank_end):
            raise ValueError(
                f"included schedule {name} rank {rank} outside merged interval"
            )
        reasons[rank].add(f"included:{name}")
        included[name] = rank

    # Recover scalar scores for top and included ranks not already held as random.
    needed = set(reasons) - set(scores_by_rank)
    for row in _iter_rows(parts):
        rank = int(row["rank"])
        if rank not in needed:
            continue
        scores_by_rank[rank] = {
            "mean_psnr_db": float(row["mean_psnr_db"]),
            "min_psnr_db": float(row["min_psnr_db"]),
            "per_prompt": {
                int(record["prompt_idx"]): float(record["psnr_db"])
                for record in row["per_prompt"]
            },
        }

    schedules_dir = args.output_dir / "candidates" / "schedules"
    schedules_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "candidates" / "candidate_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "rank",
            "reasons",
            "variable_full_steps",
            "mean_psnr_db",
            "min_psnr_db",
            *[f"psnr_p{idx}" for idx in prompt_indices],
            "schedule_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for rank in sorted(reasons):
            full_steps, _cache_steps, bits = schedule_for_rank(rank)
            variable = [
                step for step in full_steps if step not in K41_SPACE.forced_full_steps
            ]
            schedule_path = schedules_dir / f"rank_{rank:07d}.txt"
            schedule_path.write_text(bits + "\n", encoding="utf-8")
            score = scores_by_rank[rank]
            writer.writerow(
                {
                    "rank": rank,
                    "reasons": ";".join(sorted(reasons[rank])),
                    "variable_full_steps": ",".join(str(v) for v in variable),
                    "mean_psnr_db": score["mean_psnr_db"],
                    "min_psnr_db": score["min_psnr_db"],
                    **{
                        f"psnr_p{idx}": score["per_prompt"][idx]
                        for idx in prompt_indices
                    },
                    "schedule_file": str(schedule_path),
                }
            )

    summary = {
        "schema": "flux_exhaustive_k41_merge.v1",
        **identity,
        "rank_start": int(args.rank_start),
        "rank_end": int(args.rank_end),
        "n_rows": interval_size,
        "n_parts": len(parts),
        "merged_table": str(merged_path),
        "best_mean": {"rank": best_rank, "psnr_db": best_mean},
        "top_mean": [{"rank": rank, "psnr_db": score} for score, rank in ordered_mean],
        "top_min": [{"rank": rank, "psnr_db": score} for score, rank in ordered_min],
        "top_per_prompt": {
            str(idx): [{"rank": rank, "psnr_db": score} for score, rank in ranked]
            for idx, ranked in ordered_per_prompt.items()
        },
        "top_per_subset": {
            ",".join(str(idx) for idx in subset): [
                {"rank": rank, "psnr_db": score} for score, rank in ranked
            ]
            for subset, ranked in ordered_per_subset.items()
        },
        "near_optimal_mean_psnr": family,
        "candidate_count": len(reasons),
        "candidate_manifest": str(manifest_path),
        "included_schedules": included,
        "random": {"count": random_count, "seed": int(args.random_seed)},
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[merge-exhaustive-k41] rows={interval_size} parts={len(parts)} "
        f"candidates={len(reasons)} best_mean={best_mean:.4f} rank={best_rank}",
        flush=True,
    )
    print(f"[merge-exhaustive-k41] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
