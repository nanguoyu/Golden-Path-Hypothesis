#!/usr/bin/env python3
"""Select a native gate threshold from its realized cache-count distribution."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="Candidate label and directory containing decision JSON files.",
    )
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument(
        "--mean-tolerance",
        type=float,
        default=0.25,
        help="Prefer candidates within this distance of the target mean K.",
    )
    parser.add_argument(
        "--selection-policy",
        choices=("exact-within-tolerance", "mean-first"),
        default="exact-within-tolerance",
        help=(
            "Ranking rule. The default preserves the historical selector; "
            "mean-first minimizes aggregate budget error before exact-hit mass."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _decision_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("per_step", "steps", "records"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def summarize_candidate(
    label: str,
    directory: Path,
    *,
    target: int,
) -> dict[str, Any]:
    return summarize_candidate_directories(
        label,
        [directory],
        target=target,
    )


def summarize_candidate_directories(
    label: str,
    directories: list[Path],
    *,
    target: int,
) -> dict[str, Any]:
    paths = sorted(
        path
        for directory in directories
        for path in directory.rglob("*.json")
        if path.name.startswith(("decisions_", "actions_"))
    )
    if not paths:
        raise ValueError(
            f"{label}: no decision JSON files under "
            f"{', '.join(str(path) for path in directories)}"
        )

    counts: list[int] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("target_cache_count") is not None:
            raise ValueError(f"{label}: {path} is not a native-gate result")
        rows = _decision_rows(payload)
        if not rows:
            raise ValueError(f"{label}: no decision rows in {path}")
        if any(
            "closure_intervened" in row
            or (
                isinstance(row.get("gate"), dict)
                and "closure_intervened" in row["gate"]
            )
            for row in rows
        ):
            raise ValueError(f"{label}: {path} contains closure decisions")
        counts.append(sum(row.get("action") == "cache" for row in rows))

    ordered = sorted(counts)
    mean = statistics.fmean(counts)
    deviations = [abs(value - target) for value in counts]
    p10_index = max(0, int(0.1 * len(ordered) + 0.999999) - 1)
    p90_index = max(0, int(0.9 * len(ordered) + 0.999999) - 1)
    exact = sum(value == target for value in counts)
    within_one = sum(abs(value - target) <= 1 for value in counts)
    histogram = collections.Counter(counts)
    return {
        "label": label,
        "directory": str(directories[0]) if len(directories) == 1 else None,
        "directories": [str(directory) for directory in directories],
        "prompt_count": len(paths),
        "target_cache_count": int(target),
        "cache_count_mean": mean,
        "cache_count_std": statistics.pstdev(counts),
        "cache_count_median": statistics.median(counts),
        "cache_count_p10": ordered[p10_index],
        "cache_count_p90": ordered[p90_index],
        "cache_count_min": ordered[0],
        "cache_count_max": ordered[-1],
        "cache_count_modes": statistics.multimode(counts),
        "cache_count_histogram": {
            str(value): histogram[value] for value in sorted(histogram)
        },
        "mean_target_error": abs(mean - target),
        "mean_abs_prompt_error": statistics.fmean(deviations),
        "exact_target_count": exact,
        "exact_target_fraction": exact / len(counts),
        "within_one_count": within_one,
        "within_one_fraction": within_one / len(counts),
    }


def _parse_candidate(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError(f"candidate must be LABEL=DIR, got {value!r}")
    return label.strip(), Path(raw_path)


def select_candidate(
    rows: list[dict[str, Any]],
    *,
    mean_tolerance: float,
    selection_policy: str = "exact-within-tolerance",
) -> tuple[dict[str, Any], list[str]]:
    if selection_policy == "mean-first":
        ranked = sorted(
            rows,
            key=lambda row: (
                row["mean_target_error"],
                -row["exact_target_fraction"],
                row["cache_count_std"],
                row.get("threshold", row["label"]),
            ),
        )
        return ranked[0], [
            "min_mean_target_error",
            "max_exact_target_fraction",
            "min_cache_count_std",
            "threshold_or_label_tiebreak",
        ]
    if selection_policy != "exact-within-tolerance":
        raise ValueError(f"unsupported selection policy: {selection_policy}")

    near_target = [
        row for row in rows
        if row["mean_target_error"] <= float(mean_tolerance)
    ]
    if near_target:
        ranked = sorted(
            near_target,
            key=lambda row: (
                -row["exact_target_fraction"],
                row["mean_target_error"],
                row["cache_count_std"],
                row["label"],
            ),
        )
        rule = [
            f"require_mean_target_error_le_{mean_tolerance:g}",
            "max_exact_target_fraction",
            "min_mean_target_error",
            "min_cache_count_std",
            "lexical_label_tiebreak",
        ]
    else:
        ranked = sorted(
            rows,
            key=lambda row: (
                row["mean_target_error"],
                -row["exact_target_fraction"],
                row["cache_count_std"],
                row["label"],
            ),
        )
        rule = [
            "no_candidate_within_mean_tolerance",
            "min_mean_target_error",
            "max_exact_target_fraction",
            "min_cache_count_std",
            "lexical_label_tiebreak",
        ]
    return ranked[0], rule


def main() -> int:
    args = parse_args()
    grouped: dict[str, list[Path]] = collections.defaultdict(list)
    for label, directory in map(_parse_candidate, args.candidate):
        grouped[label].append(directory)
    rows = [
        summarize_candidate_directories(label, directories, target=args.target)
        for label, directories in grouped.items()
    ]
    selected, selection_rule = select_candidate(
        rows,
        mean_tolerance=args.mean_tolerance,
        selection_policy=args.selection_policy,
    )
    rows.sort(key=lambda row: row["label"])
    payload = {
        "format": "native-gate-threshold-selection-v1",
        "selection_rule": selection_rule,
        "target_cache_count": int(args.target),
        "mean_tolerance": float(args.mean_tolerance),
        "selection_policy": args.selection_policy,
        "selected": selected["label"],
        "candidates": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
