#!/usr/bin/env python3
"""Freeze image-model native-gate thresholds for several cache-count targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.select_native_gate_threshold import (
    select_candidate,
    summarize_candidate,
)


METHOD_ROOT_ARGS = {
    "SeaCache": "sea_root",
    "TeaCache": "tea_root",
    "SenCache": "sen_root",
    "DiCache": "dicache_root",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sea_root", type=Path, required=True)
    parser.add_argument("--tea_root", type=Path, required=True)
    parser.add_argument("--sen_root", type=Path, required=True)
    parser.add_argument("--dicache_root", type=Path, required=True)
    parser.add_argument("--targets", default="29,37,41")
    parser.add_argument("--mean_tolerance", type=float, default=0.25)
    parser.add_argument(
        "--backbone",
        choices=("flux", "qwen_image"),
        default="flux",
    )
    parser.add_argument("--sencache_threshold_start", type=float, default=0.005)
    parser.add_argument("--dicache_ret_ratio", type=float, default=0.2)
    parser.add_argument(
        "--dicache_target_variant",
        action="append",
        default=[],
        metavar="TARGET:RET_RATIO:ROOT",
        help=(
            "Use a separate native DiCache sweep and ret_ratio for one target. "
            "May be repeated, for example 41:0.1:/path/to/native_dicache_rr0p1."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _targets(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("--targets must contain positive comma-separated integers")
    if len(set(values)) != len(values):
        raise ValueError("--targets must not contain duplicates")
    return values


def _candidate_threshold(directory: Path) -> float:
    paths = sorted(directory.glob("decisions_*.json"))
    if not paths:
        raise ValueError(f"no decision files under {directory}")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    threshold = payload.get("native_threshold")
    if threshold is None:
        raise ValueError(f"{paths[0]} does not record native_threshold")
    return float(threshold)


def _dicache_target_variants(
    raw_values: list[str],
) -> dict[int, tuple[Path, float]]:
    variants: dict[int, tuple[Path, float]] = {}
    for raw in raw_values:
        parts = raw.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "--dicache_target_variant must be TARGET:RET_RATIO:ROOT"
            )
        target = int(parts[0])
        ret_ratio = float(parts[1])
        root = Path(parts[2])
        if target <= 0 or not 0.0 <= ret_ratio < 1.0:
            raise ValueError(
                "DiCache target must be positive and ret_ratio must be in [0,1)"
            )
        if target in variants:
            raise ValueError(f"duplicate DiCache target variant: {target}")
        variants[target] = (root, ret_ratio)
    return variants


def _discover(root: Path) -> list[tuple[str, float, Path]]:
    if not root.is_dir():
        raise ValueError(f"native-gate root does not exist: {root}")
    candidates = []
    seen: set[float] = set()
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if not any(directory.glob("decisions_*.json")):
            continue
        threshold = _candidate_threshold(directory)
        if threshold in seen:
            raise ValueError(f"duplicate threshold {threshold:g} under {root}")
        seen.add(threshold)
        candidates.append((f"t{threshold:.12g}", threshold, directory))
    if not candidates:
        raise ValueError(f"no native-gate candidates under {root}")
    return candidates


def build_selection(
    roots: dict[str, Path],
    *,
    targets: tuple[int, ...],
    mean_tolerance: float,
    backbone: str = "flux",
    sencache_threshold_start: float = 0.005,
    dicache_ret_ratio: float = 0.2,
    dicache_target_variants: dict[int, tuple[Path, float]] | None = None,
) -> dict[str, Any]:
    if sencache_threshold_start <= 0.0:
        raise ValueError("sencache_threshold_start must be positive")
    if not 0.0 <= dicache_ret_ratio < 1.0:
        raise ValueError("dicache_ret_ratio must be in [0,1)")
    target_variants = dicache_target_variants or {}
    unknown_targets = set(target_variants) - set(targets)
    if unknown_targets:
        raise ValueError(
            f"DiCache target variants are outside --targets: {sorted(unknown_targets)}"
        )
    discovered = {
        method: _discover(root)
        for method, root in roots.items()
    }
    dicache_discovered = {
        target: _discover(root)
        for target, (root, _ret_ratio) in target_variants.items()
    }
    output: dict[str, Any] = {
        "format": f"{backbone}-native-gate-thresholds-v1",
        "backbone": backbone,
        "selection_basis": "cache_count_only",
        "mean_tolerance": float(mean_tolerance),
        "targets": {},
    }
    for target in targets:
        methods: dict[str, Any] = {}
        for method, candidates in discovered.items():
            runtime_parameters: dict[str, Any] = {}
            if method == "SenCache":
                runtime_parameters["threshold_start"] = sencache_threshold_start
            elif method == "DiCache":
                if target in target_variants:
                    candidates = dicache_discovered[target]
                    runtime_parameters["ret_ratio"] = target_variants[target][1]
                else:
                    runtime_parameters["ret_ratio"] = dicache_ret_ratio
            rows = [
                {
                    **summarize_candidate(
                        label,
                        directory,
                        target=target,
                    ),
                    "threshold": threshold,
                }
                for label, threshold, directory in candidates
            ]
            selected, rule = select_candidate(
                rows,
                mean_tolerance=mean_tolerance,
            )
            methods[method] = {
                "threshold": selected["threshold"],
                "directory": selected["directory"],
                "selection_rule": rule,
                "statistics": selected,
                "candidates": sorted(rows, key=lambda row: row["threshold"]),
            }
            if runtime_parameters:
                methods[method]["runtime_parameters"] = runtime_parameters
        output["targets"][str(target)] = methods
    return output


def main() -> int:
    args = parse_args()
    roots = {
        method: getattr(args, argument)
        for method, argument in METHOD_ROOT_ARGS.items()
    }
    payload = build_selection(
        roots,
        targets=_targets(args.targets),
        mean_tolerance=args.mean_tolerance,
        backbone=args.backbone,
        sencache_threshold_start=args.sencache_threshold_start,
        dicache_ret_ratio=args.dicache_ret_ratio,
        dicache_target_variants=_dicache_target_variants(
            args.dicache_target_variant
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
