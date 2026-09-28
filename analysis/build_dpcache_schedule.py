#!/usr/bin/env python3
"""Merge DPCache calibration shards and build an exact-K schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.dpcache import cache_steps_from_full, select_full_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost_shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cache_count", type=int, default=29)
    parser.add_argument(
        "--model",
        choices=["flux", "qwen_image"],
        default="flux",
    )
    parser.add_argument("--first_full_steps", type=int, default=3)
    parser.add_argument("--last_full_steps", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sums = np.zeros(
        (args.num_steps, args.num_steps + 1, args.num_steps + 1),
        dtype=np.float64,
    )
    counts = np.zeros_like(sums, dtype=np.int64)
    for path in args.cost_shards:
        shard = np.load(path)
        if int(shard["num_steps"]) != args.num_steps or int(shard["order"]) != 2:
            raise ValueError(f"incompatible DPCache calibration shard: {path}")
        sums += np.asarray(shard["cost_sums"], dtype=np.float64)
        counts += np.asarray(shard["cost_counts"], dtype=np.int64)
    mean = np.full_like(sums, np.inf)
    valid = counts > 0
    mean[valid] = sums[valid] / counts[valid]

    full = select_full_steps(
        mean,
        total_steps=args.num_steps,
        full_count=args.num_steps - args.cache_count,
        first_full_steps=args.first_full_steps,
        last_full_steps=args.last_full_steps,
    )
    cache = cache_steps_from_full(full, total_steps=args.num_steps)
    payload = {
        "format": f"{args.model.replace('_', '-')}-dpcache-schedule-v1",
        "model": args.model,
        "role": "independent_calibration_exact_schedule",
        "num_steps": args.num_steps,
        "cache_count": args.cache_count,
        "full_count": len(full),
        "cache_steps": list(cache),
        "full_steps": list(full),
        "first_full_steps": args.first_full_steps,
        "last_full_steps": args.last_full_steps,
        "order": 2,
        "cost_shards": [str(path) for path in args.cost_shards],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
